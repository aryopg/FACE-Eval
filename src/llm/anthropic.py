"""Anthropic implementation of the LLM interface."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import anthropic
from dotenv import load_dotenv

from src.llm.base import BaseLLM, chunk_batch_requests

log = logging.getLogger(__name__)

load_dotenv()

_THINKING_BUDGET: dict[str, int] = {"low": 5_000, "medium": 10_000, "high": 20_000}

# The Batch API rejects a submission over 256 MB with a 413. Long-CoT runs blow
# past that well before the request-count limit, so chunking chases bytes too.
# Headroom covers the SDK's own JSON framing around the request list.
MAX_BATCH_BYTES = 200_000_000


def _cached_system(system: str) -> list[dict]:
    """Wrap system string as a cached content block.

    Anthropic caches the prefix up to (and including) this block.
    Minimum ~1024 tokens for the cache to actually activate. Caching is
    unconditional — every caller gets it.
    """
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _cached_tools(tools: list[dict]) -> list[dict]:
    """Mark the last tool with cache_control to cache the full tool prefix.

    Placing the breakpoint on the last tool caches all tools in one shot.

    Under the Batch API, cache_control is supported but cross-request reuse
    within a single submission is not guaranteed: the first processed request
    writes the cache and later ones read it, so savings depend on the batch
    processing order, which Anthropic leaves unspecified.
    """
    if not tools:
        return tools
    return tools[:-1] + [{**tools[-1], "cache_control": {"type": "ephemeral"}}]


def _openai_to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool definitions to Anthropic format.

    OpenAI: {"type":"function","function":{"name":...,"description":...,"parameters":{...}}}
    Anthropic: {"name":...,"description":...,"input_schema":{...}}
    """
    result = []
    for tool in tools:
        fn = tool.get("function", tool)
        result.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", fn.get("input_schema", {})),
            }
        )
    return result


def _openai_to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Extract system prompt and convert OpenAI chat format to Anthropic format.

    Handles:
    - role:system → extracted as top-level system param (not a message)
    - assistant tool_calls → tool_use content blocks
    - role:tool → user message with tool_result content block
    """
    system: str | None = None
    result: list[dict] = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            system = msg["content"]
        elif role == "assistant":
            if "tool_calls" in msg:
                content = [
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    }
                    for tc in msg["tool_calls"]
                ]
                result.append({"role": "assistant", "content": content})
            else:
                result.append({"role": "assistant", "content": msg.get("content") or ""})
        elif role == "tool":
            result.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": msg.get("content") or "",
                        }
                    ],
                }
            )
        else:
            result.append({"role": role, "content": msg.get("content") or ""})
    return system, result


class AnthropicLLM(BaseLLM):
    """Client for Anthropic models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        use_batch: bool = True,
        include_reasoning: bool = False,
        reasoning_effort: str | None = None,
        max_batch_size: int = 10_000,
        **kwargs,
    ):
        """Initialize Anthropic client.

        Args:
            model: Model identifier.
            api_key: Anthropic API key. If None, reads from env.
            use_batch: Use batch API for normal priority.
            include_reasoning: Enable extended thinking and extract reasoning trace.
            reasoning_effort: Thinking budget level ("low", "medium", "high").
            **kwargs: Additional parameters.
        """
        super().__init__(model, **kwargs)

        if api_key is None:
            if use_batch:
                api_key = os.getenv("ANTHROPIC_API_KEY_NORMAL_BATCH")
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("Anthropic API key not found in environment")

        self._api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self._async_client: anthropic.AsyncAnthropic | None = None
        self.use_batch = use_batch
        self.include_reasoning = include_reasoning
        self.reasoning_effort = reasoning_effort
        self.max_batch_size = max_batch_size

    def _thinking_param(self) -> dict[str, Any] | None:
        if not self.include_reasoning or not self.reasoning_effort:
            return None
        return {"type": "enabled", "budget_tokens": _THINKING_BUDGET[self.reasoning_effort]}

    def _extract_response(self, content_blocks) -> str | dict[str, str]:
        """Extract text and optional reasoning from Anthropic content blocks."""
        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        for block in content_blocks:
            if block.type == "thinking":
                reasoning_parts.append(block.thinking)
            elif block.type == "text":
                text_parts.append(block.text)
        content = " ".join(text_parts)
        if reasoning_parts:
            return {"reasoning": " ".join(reasoning_parts), "content": content}
        return content

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: str | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> str | dict[str, str]:
        """Generate a single chat response."""
        extracted_system, converted_messages = _openai_to_anthropic_messages(messages)
        resolved_system = system or extracted_system
        converted_tools = _openai_to_anthropic_tools(tools) if tools else None

        thinking = self._thinking_param()
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": converted_messages,
            "temperature": 1.0 if thinking else temperature,
        }
        if resolved_system:
            request_kwargs["system"] = _cached_system(resolved_system)
        if converted_tools:
            request_kwargs["tools"] = _cached_tools(converted_tools)
        if thinking:
            request_kwargs["thinking"] = thinking
        request_kwargs.update(kwargs)

        response = self.client.messages.create(**request_kwargs)
        if not response.content:
            raise ValueError(f"Anthropic returned an empty content list for model {self.model!r}")
        return self._extract_response(response.content)

    async def async_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: str | None = None,
        **kwargs,
    ) -> str:
        """Generate a single chat response asynchronously (judge use; no thinking)."""
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            request_kwargs["system"] = system
        request_kwargs.update(kwargs)

        response = await self._async_client.messages.create(**request_kwargs)
        if not response.content:
            raise ValueError(f"Anthropic returned an empty content list for model {self.model!r}")
        return response.content[0].text

    def create_batch(
        self,
        requests: list[dict[str, Any]],
        max_retries: int = 5,
    ) -> str:
        """Submit a message batch and return the batch ID."""
        delay = 10.0
        for attempt in range(max_retries):
            try:
                batch = self.client.messages.batches.create(requests=requests)
                log.info(f"Created batch {batch.id} with {len(requests)} requests")
                return batch.id
            except anthropic.InternalServerError as exc:
                if attempt == max_retries - 1:
                    raise
                log.warning(f"create_batch attempt {attempt + 1} failed ({exc}); retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120.0)

    def poll_batch(
        self,
        batch_id: str,
        poll_interval: float = 60.0,
        log_interval: float = 900.0,
        progress_callback: Any | None = None,
    ) -> dict[str, str | dict[str, str]]:
        """Poll a batch until completion and return results keyed by custom_id.

        Failed requests map to an empty string so downstream parsers produce
        `parse_ok=False` rows that will be re-run on resume.
        """
        last_log = 0.0
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            succeeded = counts.succeeded
            errored = counts.errored
            total = succeeded + errored + counts.processing

            if progress_callback:
                progress_callback(succeeded + errored, total)

            if batch.processing_status == "ended":
                break

            now = time.time()
            if now - last_log >= log_interval:
                log.info(f"Batch {batch_id}: {succeeded} succeeded, {errored} errored, {counts.processing} processing")
                last_log = now
            time.sleep(poll_interval)

        results: dict[str, str | dict[str, str]] = {}
        for result in self.client.messages.batches.results(batch_id):
            cid = result.custom_id
            if result.result.type == "succeeded":
                content = result.result.message.content
                if not content:
                    log.warning(f"Batch request {cid} succeeded but returned empty content")
                    results[cid] = ""
                else:
                    results[cid] = self._extract_response(content)
            else:
                log.warning(f"Batch request {cid} failed with type={result.result.type!r}")
                results[cid] = ""

        return results

    def _poll_batches(
        self,
        batch_ids: list[str],
        poll_interval: float = 60.0,
        log_interval: float = 900.0,
        timeout: float = 24 * 3600,
    ) -> dict[str, str | dict[str, str]]:
        """Poll multiple batches until all complete. Returns results keyed by custom_id.

        Batches that time out contribute no entries; positions missing from the
        returned dict are filled with a harmony_parse_failed sentinel by chat_batch.
        """
        deadline = time.time() + timeout
        pending = set(batch_ids)
        all_results: dict[str, str | dict[str, str]] = {}
        last_log = 0.0

        while pending:
            if time.time() > deadline:
                log.warning(
                    f"Batch poll timed out ({timeout / 3600:.0f}h) with "
                    f"{len(pending)} batch(es) still pending: {sorted(pending)}"
                )
                break

            now = time.time()
            should_log = now - last_log >= log_interval
            if should_log:
                last_log = now
            for bid in list(pending):
                batch = self.client.messages.batches.retrieve(bid)
                counts = batch.request_counts
                if should_log:
                    log.info(
                        f"Batch {bid}: {counts.succeeded} succeeded, "
                        f"{counts.errored} errored, {counts.processing} processing"
                    )
                if batch.processing_status == "ended":
                    pending.discard(bid)
                    for result in self.client.messages.batches.results(bid):
                        cid = result.custom_id
                        if result.result.type == "succeeded":
                            content = result.result.message.content
                            if not content:
                                log.warning(f"Batch request {cid} succeeded but returned empty content")
                                all_results[cid] = ""
                            else:
                                all_results[cid] = self._extract_response(content)
                        else:
                            log.warning(f"Batch request {cid} failed with type={result.result.type!r}")
                            all_results[cid] = ""

            if pending:
                time.sleep(poll_interval)

        return all_results

    def chat_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: str | None = None,
        tools_list: list[list | None] | None = None,
        **kwargs,
    ) -> list[str | dict[str, str]]:
        """Generate responses for multiple conversations.

        Uses the Anthropic Message Batches API when use_batch=True,
        otherwise calls chat() sequentially.
        """
        if not self.use_batch:
            return [
                self.chat(
                    messages=m,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    tools=tools_list[i] if tools_list else None,
                )
                for i, m in enumerate(messages_list)
            ]

        thinking = self._thinking_param()
        requests: list[dict[str, Any]] = []
        for i, messages in enumerate(messages_list):
            raw_tools = tools_list[i] if tools_list else None
            extracted_system, converted_messages = _openai_to_anthropic_messages(messages)
            resolved_system = system or extracted_system
            converted_tools = _openai_to_anthropic_tools(raw_tools) if raw_tools else None
            params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 1.0 if thinking else temperature,
                "messages": converted_messages,
            }
            if resolved_system:
                params["system"] = _cached_system(resolved_system)
            if converted_tools:
                params["tools"] = _cached_tools(converted_tools)
            if thinking:
                params["thinking"] = thinking
            requests.append({"custom_id": str(i), "params": params})

        chunks = chunk_batch_requests(requests, self.max_batch_size, MAX_BATCH_BYTES)
        if len(chunks) > 1:
            log.info(f"Splitting {len(requests)} requests into {len(chunks)} batches of up to {self.max_batch_size}")
        batch_ids = [self.create_batch(chunk) for chunk in chunks]
        results_raw = self._poll_batches(batch_ids)
        _failed: dict[str, Any] = {
            "harmony_parse_failed": True,
            "raw_fallback": "",
            "parse_error": "batch chunk failed or timed out",
        }
        return [results_raw.get(str(i), _failed) for i in range(len(messages_list))]
