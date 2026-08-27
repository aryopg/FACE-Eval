"""vLLM implementation of the LLM interface."""

from __future__ import annotations

import json

import torch
from vllm import LLM, SamplingParams

from src.llm.base import BaseLLM, shutdown_vllm_engine


class VLLMClient(BaseLLM):
    """Client for vLLM models."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int | None = None,
        enable_thinking: bool = True,
        include_reasoning: bool = True,
        **kwargs,
    ):
        """Initialize vLLM client.

        Args:
            model: Model identifier (e.g., "Qwen/Qwen3.5-9B").
            tensor_parallel_size: Number of GPUs to use. If None, auto-detect all GPUs.
            enable_thinking: Enable thinking mode for models that support it.
            include_reasoning: Return reasoning separately if available.
            **kwargs: Additional vLLM parameters.
        """
        super().__init__(model, **kwargs)
        self.enable_thinking = enable_thinking
        self.include_reasoning = include_reasoning

        if tensor_parallel_size is None:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                tensor_parallel_size = num_gpus

        if tensor_parallel_size is not None:
            kwargs["tensor_parallel_size"] = tensor_parallel_size

        if "reasoning_parser" not in kwargs:
            model_lower = model.lower()
            # Each entry is (required_substrings, parser_name). All substrings must match.
            _REASONING_PARSERS = [
                (("qwen3",), "qwen3"),
                (("deepseek-v4",), "deepseek_v4"),
                (("deepseek-r1",), "deepseek_r1"),
                (("olmo", "think"), "olmo3"),
            ]
            for patterns, parser in _REASONING_PARSERS:
                if all(p in model_lower for p in patterns):
                    kwargs["reasoning_parser"] = parser
                    break

        self.llm = LLM(model=model, **kwargs)
        self.default_sampling_params = SamplingParams()

    def chat(
        self,
        messages: list[dict[str, str]],
        sampling_params: SamplingParams | None = None,
        enable_thinking: bool | None = None,
        include_reasoning: bool | None = None,
        **kwargs,
    ) -> str | dict[str, str]:
        """Generate a single chat response."""
        return self.chat_batch(
            [messages],
            sampling_params=sampling_params,
            enable_thinking=enable_thinking,
            include_reasoning=include_reasoning,
            **kwargs,
        )[0]

    def chat_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        sampling_params: SamplingParams | None = None,
        enable_thinking: bool | None = None,
        include_reasoning: bool | None = None,
        tools_list: list[list | None] | None = None,
        **kwargs,
    ) -> list[str | dict[str, str]]:
        """Generate responses for multiple conversations.

        Args:
            messages_list: List of message lists.
            sampling_params: vLLM SamplingParams. Uses defaults if None.
            enable_thinking: Override default thinking setting.
            include_reasoning: Override default reasoning inclusion.
            tools_list: Per-conversation tool definitions. If provided, must
                have same length as messages_list. Conversations are grouped
                by their tools and batched separately (vLLM requires a single
                tools value per batch call).
            **kwargs: Additional parameters.
        """
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        if enable_thinking is None:
            enable_thinking = self.enable_thinking

        if include_reasoning is None:
            include_reasoning = self.include_reasoning

        chat_template_kwargs = self._chat_template_kwargs(enable_thinking)

        if tools_list is None:
            outputs = self.llm.chat(
                messages=messages_list,
                sampling_params=sampling_params,
                chat_template_kwargs=chat_template_kwargs,
            )
            return self._process_outputs(outputs, include_reasoning)

        # Group conversations by tools (serialized) for separate batch calls.
        groups: dict[str, list[int]] = {}
        for i, tools in enumerate(tools_list):
            key = json.dumps(tools or [], sort_keys=True)
            groups.setdefault(key, []).append(i)

        results: list[str | dict[str, str] | None] = [None] * len(messages_list)
        for tools_key, indices in groups.items():
            tools = json.loads(tools_key) or None
            batch_messages = [messages_list[i] for i in indices]

            outputs = self.llm.chat(
                messages=batch_messages,
                sampling_params=sampling_params,
                chat_template_kwargs=chat_template_kwargs,
                tools=tools,
            )
            for idx, output in zip(indices, self._process_outputs(outputs, include_reasoning)):
                results[idx] = output

        return results  # type: ignore[return-value]  # all slots filled by grouping

    def _chat_template_kwargs(self, enable_thinking: bool) -> dict:
        """Build chat-template kwargs. Subclasses override to pass other controls."""
        return {"enable_thinking": enable_thinking}

    def _process_outputs(self, outputs, include_reasoning: bool) -> list[str | dict[str, str]]:
        """Extract text/reasoning from vLLM outputs."""
        results = []
        for output in outputs:
            completion_output = output.outputs[0]

            if include_reasoning and hasattr(completion_output, "reasoning"):
                reasoning = completion_output.reasoning or ""
                content = completion_output.text

                # vLLM's reasoning parser sometimes leaves the <think> block in content,
                # or leaks the closing delimiter into it. Repair both.
                if not reasoning.strip() and "</think>" in content:
                    parts = content.split("</think>", 1)
                    reasoning = parts[0].strip()
                    content = parts[1].strip()
                elif reasoning.strip() and content.lstrip().startswith("</think>"):
                    content = content.lstrip().removeprefix("</think>").strip()

                results.append({"reasoning": reasoning, "content": content})
            else:
                text = completion_output.text
                if include_reasoning and "</think>" in text:
                    parts = text.split("</think>", 1)
                    results.append({"reasoning": parts[0].strip(), "content": parts[1].strip()})
                else:
                    results.append(text)

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
