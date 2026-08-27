"""GPT-OSS vLLM implementation using the openai-harmony SDK."""

from __future__ import annotations

import copy
from typing import Any

import torch
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    HarmonyError,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)
from vllm import LLM, SamplingParams

from src.llm.base import BaseLLM, shutdown_vllm_engine

_EFFORT_MAP: dict[str, ReasoningEffort] = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}


class GPTOSSClient(BaseLLM):
    """Client for GPT-OSS models via vLLM + openai-harmony."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int | None = None,
        reasoning_effort: str = "high",
        include_reasoning: bool = True,
        **kwargs,
    ):
        """Initialize GPT-OSS client.

        Args:
            model: Model identifier (e.g., "openai/gpt-oss-20b").
            tensor_parallel_size: Number of GPUs. If None, auto-detect.
            reasoning_effort: One of "low", "medium", or "high".
            include_reasoning: Return reasoning separately if available.
            **kwargs: Additional vLLM parameters.
        """
        self.set_reasoning_effort(reasoning_effort)  # validate before the expensive engine load
        super().__init__(model, **kwargs)
        self.include_reasoning = include_reasoning
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        if tensor_parallel_size is None:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                tensor_parallel_size = num_gpus

        if tensor_parallel_size is not None:
            kwargs["tensor_parallel_size"] = tensor_parallel_size

        kwargs.setdefault("trust_remote_code", True)
        self.llm = LLM(model=model, **kwargs)
        self.default_sampling_params = SamplingParams()

    def set_reasoning_effort(self, reasoning_effort: str) -> None:
        """Set the reasoning effort (validated) so one loaded engine can serve multiple efforts."""
        if reasoning_effort not in _EFFORT_MAP:
            raise ValueError(f"reasoning_effort must be one of {list(_EFFORT_MAP)}; got {reasoning_effort!r}")
        self.reasoning_effort = reasoning_effort

    def _build_harmony_conversation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> Conversation:
        """Convert standard message dicts to a Harmony Conversation.

        Handles OpenAI-style tool_calls and tool messages: assistant tool_calls
        render on the commentary channel with `functions.<name>` recipient and
        json content_type; tool results render as Role.TOOL messages routed
        back to the assistant on commentary.
        """
        effort = _EFFORT_MAP[self.reasoning_effort]

        harmony_messages = [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_reasoning_effort(effort),
            )
        ]

        system_instructions = "\n".join(
            m["content"] for m in messages if m.get("role") == "system" and m.get("content")
        )
        dev_content = DeveloperContent.new()
        if system_instructions:
            dev_content = dev_content.with_instructions(system_instructions)
        if tools:
            tool_descs = []
            for tool in tools:
                spec = tool.get("function", tool)
                tool_descs.append(
                    ToolDescription(
                        name=spec["name"],
                        description=spec.get("description", ""),
                        parameters=spec.get("parameters"),
                    )
                )
            dev_content = dev_content.with_function_tools(tool_descs)
        if system_instructions or tools:
            harmony_messages.append(Message.from_role_and_content(Role.DEVELOPER, dev_content))

        call_id_to_name: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue
            if role == "user":
                harmony_messages.append(Message.from_role_and_content(Role.USER, msg.get("content") or ""))
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for call in tool_calls:
                        fn = call.get("function") or {}
                        name = fn.get("name", "")
                        args = fn.get("arguments") or "{}"
                        call_id_to_name[call.get("id", "")] = name
                        harmony_messages.append(
                            Message.from_role_and_content(Role.ASSISTANT, args)
                            .with_channel("commentary")
                            .with_recipient(f"functions.{name}")
                            .with_content_type("<|constrain|> json")
                        )
                    continue
                content = msg.get("content") or ""
                if content:
                    harmony_messages.append(Message.from_role_and_content(Role.ASSISTANT, content))
            elif role == "tool":
                name = call_id_to_name.get(msg.get("tool_call_id", ""), "tool")
                harmony_messages.append(
                    Message.from_author_and_content(
                        Author.new(Role.TOOL, f"functions.{name}"),
                        msg.get("content") or "",
                    ).with_channel("commentary")
                )

        return Conversation.from_messages(harmony_messages)

    def chat(
        self,
        messages: list[dict[str, str]],
        sampling_params: SamplingParams | None = None,
        include_reasoning: bool | None = None,
        **kwargs,
    ) -> str | dict[str, str]:
        """Generate a single chat response."""
        return self.chat_batch(
            [messages], sampling_params=sampling_params, include_reasoning=include_reasoning, **kwargs
        )[0]

    def chat_batch(
        self,
        messages_list: list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        include_reasoning: bool | None = None,
        tools_list: list[list[dict] | None] | None = None,
        **kwargs,
    ) -> list[str | dict[str, str]]:
        """Generate responses for multiple conversations."""
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        if include_reasoning is None:
            include_reasoning = self.include_reasoning

        if tools_list is not None and len(tools_list) != len(messages_list):
            raise ValueError(
                f"tools_list length {len(tools_list)} does not match messages_list length {len(messages_list)}"
            )

        params = copy.copy(sampling_params)
        params.stop_token_ids = self.stop_token_ids

        prompts = []
        for i, messages in enumerate(messages_list):
            tools = tools_list[i] if tools_list is not None else None
            convo = self._build_harmony_conversation(messages, tools=tools)
            prefill_ids = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            prompts.append({"prompt_token_ids": prefill_ids})

        outputs = self.llm.generate(prompts=prompts, sampling_params=params)
        return self._process_outputs(outputs, include_reasoning)

    def _process_outputs(
        self,
        outputs,
        include_reasoning: bool,
    ) -> list[str | dict[str, Any]]:
        """Parse Harmony completion tokens back into text."""
        results = []
        for output in outputs:
            gen = output.outputs[0]
            try:
                entries = self.encoding.parse_messages_from_completion_tokens(
                    gen.token_ids,
                    Role.ASSISTANT,
                )
            except HarmonyError:
                if include_reasoning:
                    results.append(
                        {
                            "reasoning": "",
                            "content": "",
                            "raw_fallback": gen.text,
                            "harmony_parse_failed": True,
                        }
                    )
                else:
                    results.append(gen.text)
                continue

            reasoning_parts = []
            content_parts = []
            commentary_parts = []
            for entry in entries:
                entry_dict = entry.to_dict()
                raw_content = entry_dict.get("content", "")
                if isinstance(raw_content, list):
                    raw_content = "".join(
                        chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in raw_content
                    )
                if not raw_content:
                    continue

                channel = getattr(entry, "channel", None)
                if channel == "analysis":
                    reasoning_parts.append(raw_content)
                elif channel == "final":
                    content_parts.append(raw_content)
                elif channel == "commentary":
                    commentary_parts.append(raw_content)

            reasoning = "\n".join(reasoning_parts)
            content = "\n".join(content_parts)
            if not content and commentary_parts:
                content = "\n".join(commentary_parts)
            if not content:
                content = gen.text

            if include_reasoning:
                results.append({"reasoning": reasoning, "content": content})
            else:
                results.append(content)

        return results

    def close(self) -> None:
        """Shut down the underlying vLLM engine before interpreter teardown."""
        shutdown_vllm_engine(self.llm)

    def set_sampling_params(
        self,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 32768,
        seed: int | None = None,
        **kwargs,
    ) -> SamplingParams:
        """Create sampling parameters."""
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            **kwargs,
        )
