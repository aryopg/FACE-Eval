"""OpenRouter implementation of the LLM interface."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable

import openai
from dotenv import load_dotenv

from src.llm.base import BaseLLM

load_dotenv()

log = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_DEFAULT_HEADERS = {"X-Data-Collection": "deny"}


def _extract_openrouter_response(message) -> str | dict[str, str]:
    """Extract text and optional reasoning field from an OpenRouter chat message."""
    reasoning = getattr(message, "reasoning", None)
    content = message.content or ""
    if reasoning:
        return {"reasoning": reasoning, "content": content}
    return content


class OpenRouterLLM(BaseLLM):
    """Client for models via OpenRouter using the OpenAI-compatible API.

    OpenRouter routes requests to many providers. Models that expose a reasoning
    trace (e.g. DeepSeek-R1) return it in the `reasoning` field of the message,
    which is captured and returned as `{"reasoning": ..., "content": ...}`.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        concurrency: int = 10,
        seed: int | None = None,
        **kwargs,
    ):
        """Build the client. The base URL is fixed to OpenRouter and is not a parameter.

        Every request carries the `X-Data-Collection: deny` header, which tells OpenRouter not
        to let the routed provider keep the prompt. There is no way to turn that off here.

        Args:
            api_key: Falls back to the `OPENROUTER_API_KEY` environment variable.
            concurrency: Requests in flight at once.
            seed: Sent with every request when set. Left None, no seed is sent. Whether a seed
                is honoured depends on the provider OpenRouter routes to.
            **kwargs: Stored on `self.kwargs` and never sent with a request.
        """
        super().__init__(model, **kwargs)
        self.concurrency = concurrency
        self.seed = seed
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=_OPENROUTER_BASE_URL,
            default_headers=_OPENROUTER_DEFAULT_HEADERS,
        )
        self._async_client: openai.AsyncOpenAI | None = None

    def _get_async_client(self) -> openai.AsyncOpenAI:
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                api_key=self.client.api_key,
                base_url=_OPENROUTER_BASE_URL,
                default_headers=_OPENROUTER_DEFAULT_HEADERS,
            )
        return self._async_client

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
        """Send every conversation to OpenRouter, `concurrency` at a time.

        A failed request does not raise. It comes back in place as a dict with
        `harmony_parse_failed`, `raw_fallback` and `parse_error` keys, so the returned list
        always has one entry per conversation, in order.

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
        client = self._get_async_client()

        async def _one(i: int, messages: list[dict]) -> tuple[int, str | dict[str, str]]:
            tools = tools_list[i] if tools_list else None
            create_kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **({"tools": tools} if tools else {}),
                **({"seed": self.seed} if self.seed is not None else {}),
            )
            async with sem:
                resp = await client.chat.completions.create(**create_kwargs)
            if not resp.choices:
                raise ValueError(
                    f"OpenRouter returned no choices for request {i} "
                    f"(model={self.model!r}). finish_reason may indicate a content block."
                )
            return i, _extract_openrouter_response(resp.choices[0].message)

        async def _one_safe(i: int, messages: list[dict]) -> tuple[int, str | dict[str, str]]:
            try:
                return await _one(i, messages)
            except Exception as exc:
                log.warning(f"OpenRouter request {i} failed: {exc!r}")
                return i, {"harmony_parse_failed": True, "raw_fallback": "", "parse_error": str(exc)}

        tasks = [asyncio.create_task(_one_safe(i, m)) for i, m in enumerate(messages_list)]
        results: dict[int, str | dict[str, str]] = {}
        for coro in asyncio.as_completed(tasks):
            i, result = await coro
            results[i] = result
            if on_result is not None:
                on_result(i, result)
        return [results[i] for i in range(len(messages_list))]

    def close(self) -> None:
        """Close the async client, if one was ever built.

        The async client is created on the first request, so this is a no-op on a client that
        never sent one. The sync client is not closed.
        """
        if self._async_client:
            asyncio.run(self._async_client.close())
