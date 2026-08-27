"""Google Gemini implementation of the LLM interface."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.llm.base import BaseLLM

load_dotenv()

log = logging.getLogger(__name__)

_THINKING_BUDGET: dict[str, int] = {"low": 512, "medium": 2_048, "high": 8_192}


def _to_genai_tool(tool_dict: dict) -> types.Tool:
    """Convert an OpenAI-style tool dict to a google-genai Tool.

    Handles both OpenAI format ({"type":"function","function":{...}}) and
    Anthropic format ({"name":...,"description":...,"input_schema":{...}}).
    """
    fn = tool_dict.get("function", tool_dict)
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters", fn.get("input_schema", {})),
            )
        ]
    )


def _openai_to_gemini_messages(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """Split system prompt and convert remaining messages to Gemini Content objects.

    Handles tool_calls (OpenAI assistant turns) and tool results (role:tool turns).
    """
    import json

    system: str | None = None
    contents: list[types.Content] = []
    # Maps tool_call_id → function_name for resolving tool result names.
    call_id_to_name: dict[str, str] = {}

    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
            continue
        if msg["role"] == "assistant":
            if "tool_calls" in msg:
                parts = []
                for tc in msg["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    call_id_to_name[tc["id"]] = fn_name
                    parts.append(
                        types.Part.from_function_call(
                            name=fn_name,
                            args=json.loads(tc["function"]["arguments"]),
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))
            else:
                contents.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=msg.get("content") or "")])
                )
        elif msg["role"] == "tool":
            fn_name = call_id_to_name.get(msg.get("tool_call_id", ""), "")
            part = types.Part.from_function_response(
                name=fn_name,
                response={"result": msg.get("content") or ""},
            )
            contents.append(types.Content(role="user", parts=[part]))
        else:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg.get("content") or "")]))
    return system, contents


def _extract_gemini_output(response) -> str | dict[str, str]:
    """Extract thinking parts and text parts from a Gemini response."""
    if not response.candidates:
        raise ValueError(
            f"Gemini returned no candidates — likely a safety or policy block. "
            f"prompt_feedback={getattr(response, 'prompt_feedback', None)!r}"
        )
    candidate = response.candidates[0]
    if candidate.content is None or not getattr(candidate.content, "parts", None):
        finish_reason = getattr(candidate, "finish_reason", "unknown")
        raise ValueError(
            f"Gemini candidate has no content parts (finish_reason={finish_reason!r}) — "
            "may be a safety, recitation, or quota block."
        )
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    for part in candidate.content.parts:
        if getattr(part, "thought", False):
            reasoning_parts.append(part.text)
        elif part.text:
            text_parts.append(part.text)
    content = " ".join(text_parts)
    if reasoning_parts:
        return {"reasoning": " ".join(reasoning_parts), "content": content}
    return content


class GeminiLLM(BaseLLM):
    """Client for Google Gemini models with async concurrent calls.

    For Gemini 2.5+ models, pass include_reasoning=True and reasoning_effort to
    enable thinking mode and extract reasoning traces.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        concurrency: int = 10,
        include_reasoning: bool = False,
        reasoning_effort: str | None = None,
        **kwargs,
    ):
        """Build the genai client and set the thinking mode.

        Thinking needs both `include_reasoning` and `reasoning_effort`. If either is unset, no
        thinking config is sent and the reply has no reasoning.

        Args:
            api_key: Falls back to the `GEMINI_API_KEY` environment variable.
            concurrency: Requests in flight at once.
            include_reasoning: Ask the model to return its thoughts.
            reasoning_effort: "low", "medium" or "high", mapped to a token budget by
                `_THINKING_BUDGET`. Any other non-empty value raises `KeyError` on the first
                request, not here.
            **kwargs: Stored on `self.kwargs` and never sent with a request.
        """
        super().__init__(model, **kwargs)
        self.concurrency = concurrency
        self.include_reasoning = include_reasoning
        self.reasoning_effort = reasoning_effort
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    def _thinking_config(self) -> types.ThinkingConfig | None:
        if not self.include_reasoning or not self.reasoning_effort:
            return None
        return types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=_THINKING_BUDGET[self.reasoning_effort],
        )

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> str | dict[str, str]:
        """Send one conversation through `chat_batch` and return its reply.

        `**kwargs` is accepted and dropped -- it is not passed on to `chat_batch`.
        """
        return self.chat_batch(
            [messages],
            max_tokens=max_tokens,
            temperature=temperature,
            tools_list=[tools] if tools else None,
        )[0]

    def chat_batch(
        self,
        messages_list: list[list[dict]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools_list: list[list | None] | None = None,
        on_result: Callable[[int, str | dict[str, str]], None] | None = None,
        **kwargs,
    ) -> list[str | dict[str, str]]:
        """Send every conversation to Gemini, `concurrency` at a time.

        Each conversation is converted from OpenAI format: the system message becomes
        `system_instruction` and the rest become Gemini `Content` objects.

        A failed request does not raise. It comes back in place as a dict with
        `harmony_parse_failed`, `raw_fallback` and `parse_error` keys. A safety or policy
        block counts as a failure and lands in that shape too, so the returned list always has
        one entry per conversation, in order.

        `**kwargs` is accepted and ignored.

        Args:
            on_result: Called as each request finishes, with its index and result. Requests
                finish out of order, so the calls do not follow the input order.
        """
        return asyncio.run(self._run_concurrent(messages_list, max_tokens, temperature, tools_list, on_result))

    async def _run_concurrent(
        self,
        messages_list: list[list[dict]],
        max_tokens: int,
        temperature: float,
        tools_list: list[list | None] | None,
        on_result: Callable[[int, str | dict[str, str]], None] | None = None,
    ) -> list[str | dict[str, str]]:
        sem = asyncio.Semaphore(self.concurrency)
        thinking_config = self._thinking_config()

        async def _one(i: int, messages: list[dict]) -> tuple[int, str | dict[str, str]]:
            raw_tools = tools_list[i] if tools_list else None
            system, contents = _openai_to_gemini_messages(messages)
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                **({"system_instruction": system} if system else {}),
                **({"tools": [_to_genai_tool(t) for t in raw_tools]} if raw_tools else {}),
                **({"thinking_config": thinking_config} if thinking_config else {}),
            )
            async with sem:
                resp = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            return i, _extract_gemini_output(resp)

        async def _one_safe(i: int, messages: list[dict]) -> tuple[int, str | dict[str, str]]:
            try:
                return await _one(i, messages)
            except Exception as exc:
                log.warning(f"Gemini request {i} failed: {exc!r}")
                return i, {"harmony_parse_failed": True, "raw_fallback": "", "parse_error": str(exc)}

        tasks = [asyncio.create_task(_one_safe(i, m)) for i, m in enumerate(messages_list)]
        results: dict[int, str | dict[str, str]] = {}
        for coro in asyncio.as_completed(tasks):
            i, result = await coro
            results[i] = result
            if on_result is not None:
                on_result(i, result)
        return [results[i] for i in range(len(messages_list))]
