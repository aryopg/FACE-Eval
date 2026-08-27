"""OpenAI implementation of the LLM interface using the Responses API."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from typing import Any

import openai
from dotenv import load_dotenv

from src.llm.base import BaseLLM, chunk_batch_requests

load_dotenv()

log = logging.getLogger(__name__)

_POLL_INTERVAL = 60.0
_LOG_INTERVAL = 900.0
_MAX_POLL_WAIT = 24 * 3600
# "none" turns reasoning off on models that require the parameter. It still counts as
# an effort here because declaring any effort is what suppresses `temperature`, which
# reasoning models reject outright.
REASONING_EFFORTS = {"none", "low", "medium", "high"}

# A batch input file is capped at 200 MB, which long-CoT judge payloads reach well
# before the request-count limit. Headroom covers the newlines joining the lines.
MAX_BATCH_FILE_BYTES = 180_000_000


def _openai_to_responses_tools(tools: list[dict]) -> list[dict]:
    """Convert Chat Completions tool format to Responses API format.

    Chat Completions: {"type":"function","function":{"name":...,"description":...,"parameters":...}}
    Responses API:    {"type":"function","name":...,"description":...,"parameters":...}
    """
    result = []
    for tool in tools:
        fn = tool.get("function", tool)
        result.append(
            {
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", fn.get("input_schema", {})),
            }
        )
    return result


def _extract_responses_output(output_items) -> str | dict[str, str]:
    """Extract reasoning summary and content text from a Responses API output list.

    Handles both live API objects (attribute access) and batch API dicts.
    """
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    for item in output_items:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type == "reasoning":
            summaries = item.get("summary", []) if isinstance(item, dict) else getattr(item, "summary", [])
            for summary in summaries:
                text = summary.get("text") if isinstance(summary, dict) else getattr(summary, "text", None)
                if text:
                    reasoning_parts.append(text)
        elif item_type == "message":
            parts = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
            for part in parts:
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    text_parts.append(text)
    content = " ".join(text_parts)
    if reasoning_parts:
        return {"reasoning": " ".join(reasoning_parts), "content": content}
    return content


class OpenAILLM(BaseLLM):
    """Client for OpenAI models via the Responses API.

    Supports async concurrent calls (default) and the file-based OpenAI Batch API
    (use_batch=True). For o-series reasoning models, pass reasoning_effort to get
    reasoning summaries in the response.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        use_batch: bool = True,
        concurrency: int = 10,
        reasoning_effort: str | None = None,
        max_batch_size: int = 10_000,
        **kwargs,
    ):
        """Build the client and pick the request path.

        Args:
            api_key: OpenAI key. Falls back to the `OPENAI_API_KEY` environment variable.
            base_url: Override the API host. Left unset, the SDK default is used.
            use_batch: True (the default) sends every call through the file-based Batch API,
                including a single `chat`. False sends live async requests instead.
            concurrency: Live requests in flight at once. Used only when `use_batch` is False.
            reasoning_effort: One of `REASONING_EFFORTS` ("none", "low", "medium", "high"). Any
                value in that set -- "none" included -- makes the request ask for a reasoning
                summary and drops `temperature`, which reasoning models reject. A value outside
                the set is silently ignored and the request keeps `temperature`.
            max_batch_size: Requests per batch submission. Used only when `use_batch` is True, and
                applied alongside the separate `MAX_BATCH_FILE_BYTES` size cap.
            **kwargs: Stored on `self.kwargs` and never sent with a request.
        """
        super().__init__(model, **kwargs)
        self.use_batch = use_batch
        self.concurrency = concurrency
        self.reasoning_effort = reasoning_effort
        self.max_batch_size = max_batch_size
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            **({"base_url": base_url} if base_url else {}),
        )
        self._async_client: openai.AsyncOpenAI | None = None

    def _get_async_client(self) -> openai.AsyncOpenAI:
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                api_key=self.client.api_key,
                base_url=str(self.client.base_url),
            )
        return self._async_client

    def _reasoning_param(self) -> dict[str, str] | None:
        if self.reasoning_effort and self.reasoning_effort in REASONING_EFFORTS:
            return {"effort": self.reasoning_effort, "summary": "auto"}
        return None

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> str | dict[str, str]:
        """Send one conversation through `chat_batch` and return its reply.

        With `use_batch=True` this one call still goes through the Batch API: it submits a
        batch of one and blocks until that job finishes, up to 24 hours. Set `use_batch=False`
        for a live request.

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
        """Run every conversation, by batch job or by live requests.

        With `use_batch=True` the requests are split into batch jobs and polled until they
        finish, which can take hours. With `use_batch=False` they run as live async requests,
        `concurrency` at a time.

        Failures do not raise. They come back in two shapes: a single rejected entry becomes
        `""`, while a whole chunk that failed or timed out becomes a dict with
        `harmony_parse_failed`, `raw_fallback` and `parse_error` keys.

        `**kwargs` is accepted and ignored.

        Args:
            tools_list: One tool list (or None) per conversation, in Chat Completions format.
                Converted to Responses API format before sending.
        """
        if self.use_batch:
            return self._run_batch_api(messages_list, max_tokens, temperature, tools_list)
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
        reasoning = self._reasoning_param()

        async def _one(i: int, messages: list[dict]) -> tuple[int, str | dict[str, str]]:
            raw_tools = tools_list[i] if tools_list else None
            tools = _openai_to_responses_tools(raw_tools) if raw_tools else None
            create_kwargs: dict[str, Any] = dict(
                model=self.model,
                input=messages,
                max_output_tokens=max_tokens,
                **({"temperature": temperature} if reasoning is None else {}),
                **({"tools": tools} if tools else {}),
                **({"reasoning": reasoning} if reasoning else {}),
            )
            async with sem:
                resp = await client.responses.create(**create_kwargs)
            return i, _extract_responses_output(resp.output)

        results = await asyncio.gather(*(_one(i, m) for i, m in enumerate(messages_list)))
        return [v for _, v in sorted(results)]

    def build_batch_request(
        self,
        custom_id: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Build one /v1/responses batch line using this client's model and reasoning settings.

        Reasoning models reject an explicit temperature, so it is sent only when no
        reasoning effort is configured.
        """
        reasoning = self._reasoning_param()
        body: dict[str, Any] = dict(
            model=self.model,
            input=messages,
            max_output_tokens=max_tokens,
            **({"temperature": temperature} if reasoning is None else {}),
            **({"tools": tools} if tools else {}),
            **({"reasoning": reasoning} if reasoning else {}),
        )
        return {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}

    def create_batch(self, requests: list[dict[str, Any]]) -> str:
        """Upload a batch input file and submit it, returning the batch ID."""
        jsonl_bytes = "\n".join(json.dumps(r) for r in requests).encode()
        file_obj = self.client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        log.info(f"Created batch {batch.id} with {len(requests)} requests")
        return batch.id

    def poll_batch(self, batch_id: str) -> dict[str, str | dict[str, str]]:
        """Poll one batch until completion and return results keyed by custom_id."""
        return self._poll_all_batches([batch_id])

    def _run_batch_api(
        self,
        messages_list: list[list[dict]],
        max_tokens: int,
        temperature: float,
        tools_list: list[list | None] | None,
    ) -> list[str | dict[str, str]]:
        all_requests = []
        for i, messages in enumerate(messages_list):
            raw_tools = tools_list[i] if tools_list else None
            tools = _openai_to_responses_tools(raw_tools) if raw_tools else None
            all_requests.append(self.build_batch_request(str(i), messages, max_tokens, temperature, tools))

        chunks = chunk_batch_requests(all_requests, self.max_batch_size, MAX_BATCH_FILE_BYTES)
        if len(chunks) > 1:
            log.info(f"Splitting {len(all_requests)} requests into {len(chunks)} batches")

        batch_ids = [self.create_batch(c) for c in chunks]
        result_map = self._poll_all_batches(batch_ids)
        _failed: dict[str, Any] = {
            "harmony_parse_failed": True,
            "raw_fallback": "",
            "parse_error": "batch chunk failed or timed out",
        }
        return [result_map.get(str(i), _failed) for i in range(len(messages_list))]

    def _poll_all_batches(self, batch_ids: list[str]) -> dict[str, str | dict[str, str]]:
        """Poll multiple batches until all complete. Returns results keyed by custom_id.

        Failed/expired/cancelled batches are logged; their entries are absent from the
        returned dict and filled with a harmony_parse_failed sentinel by the caller.
        """
        deadline = time.time() + _MAX_POLL_WAIT
        pending = set(batch_ids)
        result_map: dict[str, str | dict[str, str]] = {}
        last_log = 0.0

        while pending:
            if time.time() > deadline:
                log.warning(
                    f"Batch poll timed out ({_MAX_POLL_WAIT / 3600:.0f}h) with "
                    f"{len(pending)} batch(es) still pending: {sorted(pending)}"
                )
                break

            for bid in list(pending):
                batch = self.client.batches.retrieve(bid)
                if batch.status not in ("completed", "failed", "expired", "cancelled"):
                    continue
                pending.discard(bid)
                if batch.status != "completed":
                    log.warning(
                        f"OpenAI batch {bid} ended with status={batch.status!r}; "
                        "its entries will be retried on resume."
                    )
                    continue
                # A batch whose every request was rejected still completes, but with an
                # error file and no output file. Quote the first error: the cause is
                # usually one malformed field shared by every request, and without it
                # the only symptom is a crash three frames deeper.
                if batch.output_file_id is None:
                    detail = ""
                    if batch.error_file_id:
                        lines = self.client.files.content(batch.error_file_id).text.splitlines()
                        if lines:
                            detail = f" First error: {lines[0]}"
                    log.warning(
                        f"OpenAI batch {bid} completed with no output file — every request in it "
                        f"failed.{detail} Its entries will be retried on resume."
                    )
                    continue
                output_file = self.client.files.content(batch.output_file_id)
                for line in output_file.text.splitlines():
                    entry = json.loads(line)
                    cid = entry["custom_id"]
                    if entry.get("error") or not entry.get("response"):
                        log.warning(
                            f"OpenAI batch entry {cid} failed: error={entry.get('error')!r}, "
                            f"status_code={entry.get('response', {}).get('status_code')!r}"
                        )
                        result_map[cid] = ""
                    else:
                        result_map[cid] = _extract_responses_output(entry["response"]["body"].get("output", []))

            if pending:
                now = time.time()
                if now - last_log >= _LOG_INTERVAL:
                    log.info(f"{len(pending)} batch(es) still pending")
                    last_log = now
                time.sleep(_POLL_INTERVAL)

        return result_map

    def close(self) -> None:
        """Close the async client, if one was ever built.

        The async client is created on the first live request, so this is a no-op on a
        batch-only run. The sync client used for file and batch calls is not closed.
        """
        if self._async_client:
            asyncio.run(self._async_client.close())
