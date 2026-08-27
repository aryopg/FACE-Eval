"""Client for a persistent vLLM OpenAI-compatible server (`vllm serve`).

The model is loaded once by the server and stays resident across every run.py
invocation (all seeds, conventions, axes, think-modes), so no reload cost is paid
per run — the key win for frontier models that take ~an hour to load.

Reasoning/tool parsers are set at serve time (--reasoning-parser,
--tool-call-parser --enable-auto-tool-choice). This client only forwards
requests. `enable_thinking` is passed through chat_template_kwargs; run.py
currently restricts --no-think to the offline `vllm` backend, so the server
path runs in thinking mode.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import openai
from dotenv import load_dotenv

from src.llm.base import BaseLLM

load_dotenv()

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _extract_vllm_response(message) -> str | dict[str, str]:
    """Extract text and optional reasoning from a vLLM chat message.

    vLLM's OpenAI server returns the reasoning trace in `reasoning_content`.
    """
    reasoning = getattr(message, "reasoning_content", None)
    content = message.content or ""
    if reasoning:
        return {"reasoning": reasoning, "content": content}
    return content


class VLLMServerClient(BaseLLM):
    """Client for a vLLM server via its OpenAI-compatible chat/completions API."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        concurrency: int = 10,
        seed: int | None = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
        """Point the client at a running vLLM server.

        Args:
            base_url: Address of the server's OpenAI-compatible API. Defaults to
                `http://localhost:8000/v1`.
            api_key: Falls back to the `VLLM_API_KEY` environment variable, then to the literal
                "EMPTY", which is what an unauthenticated vLLM server expects.
            concurrency: Requests in flight at once.
            seed: Sent with every request when set. Left None, no seed is sent and the server
                samples freely.
            enable_thinking: Sent to the server in `chat_template_kwargs`. `run.py` allows
                `--no-think` only on the offline `vllm` backend, so this stays True in practice.
            **kwargs: Stored on `self.kwargs` and never sent with a request.
        """
        super().__init__(model, **kwargs)
        self.base_url = base_url or _DEFAULT_BASE_URL
        self.concurrency = concurrency
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=self.base_url,
        )
        self._async_client: openai.AsyncOpenAI | None = None

    def _get_async_client(self) -> openai.AsyncOpenAI:
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(api_key=self.client.api_key, base_url=self.base_url)
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
        **kwargs,
    ) -> list[str | dict[str, str]]:
        """Send every conversation to the server, `concurrency` at a time.

        A failed request does not raise. It comes back in place as a dict with
        `harmony_parse_failed`, `raw_fallback` and `parse_error` keys, so the returned list
        always has one entry per conversation, in order.

        `**kwargs` is accepted and ignored.
        """
        return asyncio.run(self._run_concurrent(messages_list, max_tokens, temperature, tools_list))

    async def _run_concurrent(
        self,
        messages_list: list[list[dict]],
        max_tokens: int,
        temperature: float,
        tools_list: list[list | None] | None,
    ) -> list[str | dict[str, str]]:
        sem = asyncio.Semaphore(self.concurrency)
        client = self._get_async_client()
        extra_body = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}

        async def _one(i: int, messages: list[dict]) -> tuple[int, str | dict[str, str]]:
            tools = tools_list[i] if tools_list else None
            create_kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
                **({"tools": tools} if tools else {}),
                **({"seed": self.seed} if self.seed is not None else {}),
            )
            async with sem:
                resp = await client.chat.completions.create(**create_kwargs)
            if not resp.choices:
                raise ValueError(f"vLLM server returned no choices for request {i} (model={self.model!r}).")
            return i, _extract_vllm_response(resp.choices[0].message)

        async def _one_safe(i: int, messages: list[dict]) -> tuple[int, str | dict[str, str]]:
            try:
                return await _one(i, messages)
            except Exception as exc:
                log.warning(f"vLLM server request {i} failed: {exc!r}")
                return i, {"harmony_parse_failed": True, "raw_fallback": "", "parse_error": str(exc)}

        tasks = [asyncio.create_task(_one_safe(i, m)) for i, m in enumerate(messages_list)]
        results: dict[int, str | dict[str, str]] = {}
        for coro in asyncio.as_completed(tasks):
            i, result = await coro
            results[i] = result
        return [results[i] for i in range(len(messages_list))]

    def close(self) -> None:
        """Close the async client, if one was ever built.

        The async client is created on the first request, so this is a no-op on a client that
        never sent one. The sync client is not closed. The server itself keeps running.
        """
        if self._async_client:
            asyncio.run(self._async_client.close())
