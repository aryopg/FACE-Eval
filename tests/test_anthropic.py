"""Tests for src/llm/anthropic.py."""

from __future__ import annotations

import asyncio
import importlib
import importlib.machinery
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def _install_anthropic_stubs() -> None:
    """Install minimal stubs for anthropic and dotenv."""

    def _make_message_response(text: str = "hello", empty_content: bool = False):
        resp = MagicMock()
        if empty_content:
            resp.content = []
        else:
            block = MagicMock()
            block.type = "text"
            block.text = text
            resp.content = [block]
        return resp

    stub = types.ModuleType("anthropic")
    stub.__spec__ = importlib.machinery.ModuleSpec("anthropic", loader=None)

    class _Anthropic:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = MagicMock()
            self.messages.create = MagicMock(return_value=_make_message_response("sync response"))
            self.messages.batches = MagicMock()

    class _AsyncAnthropic:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = MagicMock()
            self.messages.create = AsyncMock(return_value=_make_message_response("async response"))

    stub.Anthropic = _Anthropic
    stub.AsyncAnthropic = _AsyncAnthropic
    sys.modules["anthropic"] = stub

    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.__spec__ = importlib.machinery.ModuleSpec("dotenv", loader=None)
    dotenv_stub.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv_stub


def _fresh_module():
    sys.modules.pop("src.llm.anthropic", None)
    _install_anthropic_stubs()
    return importlib.import_module("src.llm.anthropic")


class TestAnthropicLLMInit(unittest.TestCase):
    def setUp(self):
        self.mod = _fresh_module()

    def test_init_with_explicit_api_key(self):
        llm = self.mod.AnthropicLLM(model="claude-test", api_key="sk-test")
        self.assertEqual(llm.model, "claude-test")
        self.assertTrue(llm.use_batch)

    def test_init_reads_api_key_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-env"}):
            llm = self.mod.AnthropicLLM(model="claude-test", use_batch=False)
        self.assertEqual(llm.model, "claude-test")

    def test_init_reads_batch_api_key_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY_NORMAL_BATCH": "sk-batch"}):
            llm = self.mod.AnthropicLLM(model="claude-test", use_batch=True)
        self.assertTrue(llm.use_batch)

    def test_init_raises_when_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            # Remove both keys if present
            env = {k: v for k, v in __import__("os").environ.items() if "ANTHROPIC" not in k}
            with patch.dict("os.environ", env, clear=True):
                with self.assertRaises(ValueError, msg="Anthropic API key not found"):
                    self.mod.AnthropicLLM(model="claude-test")


class TestAnthropicLLMChat(unittest.TestCase):
    def setUp(self):
        self.mod = _fresh_module()
        self.llm = self.mod.AnthropicLLM(model="claude-test", api_key="sk-test")

    def test_chat_returns_text(self):
        result = self.llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "sync response")

    def test_chat_passes_system_prompt(self):
        self.llm.chat([{"role": "user", "content": "hi"}], system="be concise")
        call_kwargs = self.llm.client.messages.create.call_args.kwargs
        self.assertEqual(
            call_kwargs["system"],
            [{"type": "text", "text": "be concise", "cache_control": {"type": "ephemeral"}}],
        )

    def test_chat_no_system_when_not_provided(self):
        self.llm.chat([{"role": "user", "content": "hi"}])
        call_kwargs = self.llm.client.messages.create.call_args.kwargs
        self.assertNotIn("system", call_kwargs)

    def test_chat_passes_max_tokens_and_temperature(self):
        self.llm.chat([{"role": "user", "content": "hi"}], max_tokens=512, temperature=0.5)
        call_kwargs = self.llm.client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["max_tokens"], 512)
        self.assertEqual(call_kwargs["temperature"], 0.5)

    def test_chat_raises_on_empty_content(self):
        resp = MagicMock()
        resp.content = []
        self.llm.client.messages.create.return_value = resp
        with self.assertRaises(ValueError):
            self.llm.chat([{"role": "user", "content": "hi"}])

    def test_chat_passes_extra_kwargs(self):
        self.llm.chat([{"role": "user", "content": "hi"}], top_p=0.9)
        call_kwargs = self.llm.client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["top_p"], 0.9)


class TestAnthropicLLMAsyncChat(unittest.TestCase):
    def setUp(self):
        self.mod = _fresh_module()
        self.llm = self.mod.AnthropicLLM(model="claude-test", api_key="sk-test")

    def test_async_chat_returns_text(self):
        result = asyncio.get_event_loop().run_until_complete(self.llm.async_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(result, "async response")

    def test_async_chat_lazy_creates_async_client(self):
        self.assertIsNone(self.llm._async_client)
        asyncio.get_event_loop().run_until_complete(self.llm.async_chat([{"role": "user", "content": "hi"}]))
        self.assertIsNotNone(self.llm._async_client)

    def test_async_chat_reuses_async_client(self):
        asyncio.get_event_loop().run_until_complete(self.llm.async_chat([{"role": "user", "content": "hi"}]))
        client_first = self.llm._async_client
        asyncio.get_event_loop().run_until_complete(self.llm.async_chat([{"role": "user", "content": "hi again"}]))
        self.assertIs(self.llm._async_client, client_first)

    def test_async_chat_passes_system_prompt(self):
        asyncio.get_event_loop().run_until_complete(
            self.llm.async_chat([{"role": "user", "content": "hi"}], system="be brief")
        )
        call_kwargs = self.llm._async_client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["system"], "be brief")

    def test_async_chat_raises_on_empty_content(self):
        resp = MagicMock()
        resp.content = []
        async_client = MagicMock()
        async_client.messages.create = AsyncMock(return_value=resp)
        self.llm._async_client = async_client
        with self.assertRaises(ValueError):
            asyncio.get_event_loop().run_until_complete(self.llm.async_chat([{"role": "user", "content": "hi"}]))


class TestAnthropicLLMChatBatch(unittest.TestCase):
    def setUp(self):
        self.mod = _fresh_module()
        self.llm = self.mod.AnthropicLLM(model="claude-test", api_key="sk-test", use_batch=False)

    def test_chat_batch_single(self):
        results = self.llm.chat_batch([[{"role": "user", "content": "hi"}]])
        self.assertEqual(results, ["sync response"])

    def test_chat_batch_multiple(self):
        results = self.llm.chat_batch(
            [
                [{"role": "user", "content": "q1"}],
                [{"role": "user", "content": "q2"}],
            ]
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(self.llm.client.messages.create.call_count, 2)

    def test_chat_batch_empty(self):
        results = self.llm.chat_batch([])
        self.assertEqual(results, [])

    def test_chat_batch_passes_kwargs(self):
        self.llm.chat_batch(
            [[{"role": "user", "content": "hi"}]],
            max_tokens=128,
            temperature=0.0,
            system="sys",
        )
        call_kwargs = self.llm.client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["max_tokens"], 128)
        self.assertEqual(call_kwargs["temperature"], 0.0)
        self.assertEqual(
            call_kwargs["system"],
            [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        )


class TestAnthropicLLMBatch(unittest.TestCase):
    def setUp(self):
        self.mod = _fresh_module()
        self.llm = self.mod.AnthropicLLM(model="claude-test", api_key="sk-test")

    def test_create_batch_returns_batch_id(self):
        batch_mock = MagicMock()
        batch_mock.id = "batch_abc"
        self.llm.client.messages.batches.create.return_value = batch_mock
        result = self.llm.create_batch([{"custom_id": "r1", "params": {}}])
        self.assertEqual(result, "batch_abc")

    def test_poll_batch_ended_immediately(self):
        batch_ended = MagicMock()
        batch_ended.processing_status = "ended"
        batch_ended.request_counts = MagicMock(succeeded=2, errored=0, processing=0)
        self.llm.client.messages.batches.retrieve.return_value = batch_ended

        result_a = MagicMock()
        result_a.custom_id = "r1"
        result_a.result.type = "succeeded"
        content_block = MagicMock()
        content_block.type = "text"
        content_block.text = "answer1"
        result_a.result.message.content = [content_block]

        result_b = MagicMock()
        result_b.custom_id = "r2"
        result_b.result.type = "errored"

        self.llm.client.messages.batches.results.return_value = [result_a, result_b]

        results = self.llm.poll_batch("batch_abc")
        self.assertEqual(results["r1"], "answer1")
        self.assertEqual(results["r2"], "")

    def test_poll_batch_with_progress_callback(self):
        batch_ended = MagicMock()
        batch_ended.processing_status = "ended"
        batch_ended.request_counts = MagicMock(succeeded=1, errored=0, processing=0)
        self.llm.client.messages.batches.retrieve.return_value = batch_ended
        self.llm.client.messages.batches.results.return_value = []

        callback_calls = []
        self.llm.poll_batch("batch_abc", progress_callback=lambda done, total: callback_calls.append((done, total)))
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(callback_calls[0], (1, 1))

    def test_poll_batch_empty_content_logs_warning(self):
        batch_ended = MagicMock()
        batch_ended.processing_status = "ended"
        batch_ended.request_counts = MagicMock(succeeded=1, errored=0, processing=0)
        self.llm.client.messages.batches.retrieve.return_value = batch_ended

        result_empty = MagicMock()
        result_empty.custom_id = "r1"
        result_empty.result.type = "succeeded"
        result_empty.result.message.content = []
        self.llm.client.messages.batches.results.return_value = [result_empty]

        results = self.llm.poll_batch("batch_abc")
        self.assertEqual(results["r1"], "")


if __name__ == "__main__":
    unittest.main()
