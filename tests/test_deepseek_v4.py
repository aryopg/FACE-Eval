"""Unit tests for the DeepSeek-V4 backend.

vLLM and huggingface_hub are stubbed so the tests can run on CPU without GPUs
or network access. We also stub the encoder module — the real `encoding_dsv4`
lives in the model repo on HF and is downloaded at runtime.
"""

import importlib
import importlib.machinery
import sys
import types
import unittest


def _install_deepseek_v4_stubs() -> None:
    torch_stub = types.ModuleType("torch")
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    torch_stub.cuda = types.SimpleNamespace(device_count=lambda: 0)
    sys.modules["torch"] = torch_stub

    vllm_stub = types.ModuleType("vllm")
    vllm_stub.__spec__ = importlib.machinery.ModuleSpec("vllm", loader=None)

    class _SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    vllm_stub.LLM = object
    vllm_stub.SamplingParams = _SamplingParams
    sys.modules["vllm"] = vllm_stub

    hf_hub_stub = types.ModuleType("huggingface_hub")
    hf_hub_stub.__spec__ = importlib.machinery.ModuleSpec("huggingface_hub", loader=None)
    hf_hub_stub.hf_hub_download = lambda **_kw: "/tmp/_unused_encoding_dsv4.py"
    sys.modules["huggingface_hub"] = hf_hub_stub


class FakeEncoding:
    """Stand-in for the real encoding_dsv4 module."""

    def __init__(self):
        self.encode_calls: list[dict] = []
        self.parse_calls: list[dict] = []
        self._parse_result: dict | None = None
        self._parse_raises: Exception | None = None

    def encode_messages(self, messages, thinking_mode, reasoning_effort=None, **kwargs):
        self.encode_calls.append(
            {
                "messages": messages,
                "thinking_mode": thinking_mode,
                "reasoning_effort": reasoning_effort,
                "kwargs": kwargs,
            }
        )
        return f"<ENCODED:{thinking_mode}:{reasoning_effort}>"

    def parse_message_from_completion_text(self, text, thinking_mode):
        self.parse_calls.append({"text": text, "thinking_mode": thinking_mode})
        if self._parse_raises is not None:
            raise self._parse_raises
        return self._parse_result or {
            "role": "assistant",
            "reasoning_content": "stub-reasoning",
            "content": "stub-content",
            "tool_calls": [],
        }


class FakeGen:
    def __init__(self, text: str):
        self.text = text


class FakeOutput:
    def __init__(self, text: str):
        self.outputs = [FakeGen(text)]


def _new_client(monkeypatch_encoding: "FakeEncoding | None" = None):
    """Construct a DeepSeekV4Client without running __init__ (no real vLLM)."""
    sys.modules.pop("src.llm.deepseek_v4", None)
    mod = importlib.import_module("src.llm.deepseek_v4")
    client = mod.DeepSeekV4Client.__new__(mod.DeepSeekV4Client)
    client.model = "deepseek-ai/DeepSeek-V4-Flash"
    client.kwargs = {}
    client.include_reasoning = True
    client.thinking_mode = "thinking"
    client._dsml_reasoning_effort = None
    client.reasoning_effort = "high"
    client.encoding = monkeypatch_encoding or FakeEncoding()
    client.default_sampling_params = sys.modules["vllm"].SamplingParams()
    return client, mod


class DeepSeekV4Test(unittest.TestCase):
    def setUp(self):
        # Capture pre-existing modules so we can restore them after the test.
        # Without this, the stubbed `huggingface_hub` leaks into other test
        # files and breaks anything that imports the real `datasets` package.
        self._saved_modules = {
            name: sys.modules.get(name) for name in ("torch", "vllm", "huggingface_hub", "src.llm.deepseek_v4")
        }
        _install_deepseek_v4_stubs()
        sys.modules.pop("src.llm.deepseek_v4", None)

    def tearDown(self):
        for name, mod in self._saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def test_effort_map_covers_three_modes(self):
        mod = importlib.import_module("src.llm.deepseek_v4")
        self.assertEqual(mod._EFFORT_MAP["chat"], ("chat", None))
        self.assertEqual(mod._EFFORT_MAP["high"], ("thinking", "high"))
        self.assertEqual(mod._EFFORT_MAP["max"], ("thinking", "max"))

    def test_invalid_reasoning_effort_raises(self):
        mod = importlib.import_module("src.llm.deepseek_v4")
        with self.assertRaises(ValueError):
            mod.DeepSeekV4Client.__new__(mod.DeepSeekV4Client).__init__(
                model="deepseek-ai/DeepSeek-V4-Flash",
                reasoning_effort="ultra",
            )

    def test_set_reasoning_effort_recomputes_derived_pair(self):
        client, _ = _new_client()
        client.set_reasoning_effort("chat")
        self.assertEqual(client.reasoning_effort, "chat")
        self.assertEqual(client.thinking_mode, "chat")
        self.assertIsNone(client._dsml_reasoning_effort)

        client.set_reasoning_effort("max")
        self.assertEqual(client.thinking_mode, "thinking")
        self.assertEqual(client._dsml_reasoning_effort, "max")

    def test_set_reasoning_effort_rejects_invalid(self):
        client, _ = _new_client()
        with self.assertRaises(ValueError):
            client.set_reasoning_effort("ultra")

    def test_attach_tools_to_existing_system_message(self):
        client, _ = _new_client()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        out = client._attach_tools_to_system(messages, tools)

        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[0]["tools"], tools)
        self.assertEqual(out[0]["content"], "You are helpful.")
        # Original list is not mutated.
        self.assertNotIn("tools", messages[0])

    def test_attach_tools_synthesises_system_when_absent(self):
        client, _ = _new_client()
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        out = client._attach_tools_to_system(messages, tools)

        self.assertEqual(out[0], {"role": "system", "content": "", "tools": tools})
        self.assertEqual(out[1], {"role": "user", "content": "hi"})

    def test_attach_tools_noop_without_tools(self):
        client, _ = _new_client()
        messages = [{"role": "user", "content": "hi"}]
        out = client._attach_tools_to_system(messages, None)
        self.assertEqual(out, messages)
        # Returned list is a copy (defensive — caller shouldn't see mutations).
        self.assertIsNot(out, messages)

    def test_encode_prompt_passes_thinking_mode_and_effort(self):
        client, _ = _new_client()
        client.thinking_mode = "thinking"
        client._dsml_reasoning_effort = "max"
        client._encode_prompt([{"role": "user", "content": "hi"}], None)

        calls = client.encoding.encode_calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["thinking_mode"], "thinking")
        self.assertEqual(calls[0]["reasoning_effort"], "max")

    def test_encode_prompt_attaches_tools_before_encoding(self):
        client, _ = _new_client()
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        client._encode_prompt([{"role": "user", "content": "hi"}], tools)

        sent = client.encoding.encode_calls[0]["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[0]["tools"], tools)

    def test_process_outputs_returns_reasoning_and_content(self):
        encoding = FakeEncoding()
        encoding._parse_result = {
            "role": "assistant",
            "reasoning_content": "thinking step",
            "content": "final answer",
            "tool_calls": [],
        }
        client, _ = _new_client(encoding)

        result = client._process_outputs([FakeOutput("ignored-by-stub")], include_reasoning=True)
        self.assertEqual(result, [{"reasoning": "thinking step", "content": "final answer"}])

    def test_process_outputs_string_only_when_reasoning_disabled(self):
        encoding = FakeEncoding()
        encoding._parse_result = {
            "role": "assistant",
            "reasoning_content": "ignored",
            "content": "just the answer",
            "tool_calls": [],
        }
        client, _ = _new_client(encoding)
        result = client._process_outputs([FakeOutput("ignored")], include_reasoning=False)
        self.assertEqual(result, ["just the answer"])

    def test_process_outputs_handles_missing_keys(self):
        encoding = FakeEncoding()
        encoding._parse_result = {"role": "assistant"}  # no reasoning, no content
        client, _ = _new_client(encoding)
        result = client._process_outputs([FakeOutput("raw text")], include_reasoning=True)
        self.assertEqual(result, [{"reasoning": "", "content": ""}])

    def test_process_outputs_marks_parse_failures(self):
        encoding = FakeEncoding()
        encoding._parse_raises = ValueError("bad DSML")
        client, _ = _new_client(encoding)

        result = client._process_outputs([FakeOutput("garbled output")], include_reasoning=True)
        self.assertEqual(len(result), 1)
        r = result[0]
        self.assertEqual(r["reasoning"], "")
        self.assertEqual(r["content"], "")
        self.assertEqual(r["raw_fallback"], "garbled output<｜end▁of▁sentence｜>")
        self.assertTrue(r["harmony_parse_failed"])
        self.assertIn("ValueError", r["parse_error"])

    def test_process_outputs_parse_failure_without_reasoning_returns_raw_text(self):
        encoding = FakeEncoding()
        encoding._parse_raises = RuntimeError("nope")
        client, _ = _new_client(encoding)
        result = client._process_outputs([FakeOutput("raw")], include_reasoning=False)
        self.assertEqual(result, ["raw"])

    def test_chat_batch_validates_tools_list_length(self):
        client, _ = _new_client()
        with self.assertRaises(ValueError):
            client.chat_batch(
                [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
                tools_list=[None],  # length mismatch
            )

    def test_chat_batch_invokes_llm_generate_with_encoded_prompts(self):
        encoding = FakeEncoding()
        encoding._parse_result = {
            "role": "assistant",
            "reasoning_content": "r",
            "content": "c",
            "tool_calls": [],
        }
        client, _ = _new_client(encoding)

        seen = {}

        def fake_generate(prompts, sampling_params):
            seen["prompts"] = prompts
            return [FakeOutput("ignored") for _ in prompts]

        client.llm = types.SimpleNamespace(generate=fake_generate)

        result = client.chat_batch(
            [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ],
            tools_list=[None, [{"type": "function", "function": {"name": "f", "parameters": {}}}]],
        )

        self.assertEqual(len(seen["prompts"]), 2)
        self.assertTrue(all(isinstance(p, str) for p in seen["prompts"]))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"reasoning": "r", "content": "c"})

    def test_chat_delegates_to_chat_batch(self):
        client, _ = _new_client()
        client.chat_batch = lambda msgs, **_kw: [f"response to {msgs[0][0]['content']}"]
        result = client.chat([{"role": "user", "content": "ping"}])
        self.assertEqual(result, "response to ping")

    def test_set_sampling_params_uses_deepseek_defaults(self):
        client, _ = _new_client()
        sp = client.set_sampling_params()
        self.assertEqual(sp.temperature, 1.0)
        self.assertEqual(sp.top_p, 1.0)
        self.assertFalse(sp.skip_special_tokens)
        self.assertFalse(sp.spaces_between_special_tokens)


if __name__ == "__main__":
    unittest.main()
