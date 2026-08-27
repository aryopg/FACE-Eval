"""Unit tests for the Inkling backend.

vLLM and torch are stubbed at test-run time (not import time, so the stubs don't
leak into earlier-collected test modules) so the tests run on CPU without GPUs.
We only exercise the pure effort-parsing logic and the chat-template-kwargs
override, so the real engine is never constructed.
"""

import importlib.machinery
import sys
import types

import pytest


def _load():
    """Install torch/vllm stubs and return the Inkling symbols under test."""
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        torch_stub.cuda = types.SimpleNamespace(device_count=lambda: 0)
        sys.modules["torch"] = torch_stub
    if "vllm" not in sys.modules:
        vllm_stub = types.ModuleType("vllm")
        vllm_stub.__spec__ = importlib.machinery.ModuleSpec("vllm", loader=None)
        vllm_stub.LLM = object

        class _SamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        vllm_stub.SamplingParams = _SamplingParams
        sys.modules["vllm"] = vllm_stub

    from src.llm.inkling import InklingClient, _parse_effort

    return InklingClient, _parse_effort


@pytest.mark.parametrize(
    "raw,expected",
    [(None, None), ("0.0", 0.0), ("0.1", 0.1), ("0.5", 0.5), ("0.99", 0.99), ("medium", "medium"), ("max", "max")],
)
def test_parse_effort_valid(raw, expected):
    _, _parse_effort = _load()
    assert _parse_effort(raw) == expected


@pytest.mark.parametrize("raw", ["1.0", "1.5", "-0.1", "foo", "0.5x"])
def test_parse_effort_invalid(raw):
    _, _parse_effort = _load()
    with pytest.raises(ValueError):
        _parse_effort(raw)


def test_chat_template_kwargs_uses_parsed_effort():
    InklingClient, _parse_effort = _load()
    # Skip engine construction; set only the attribute the override reads.
    client = InklingClient.__new__(InklingClient)
    client.reasoning_effort = _parse_effort("0.5")
    assert client._chat_template_kwargs(enable_thinking=True) == {"reasoning_effort": 0.5}


@pytest.mark.parametrize(
    "extracted,expected",
    [
        (("Deliberating.", "The answer."), {"reasoning": "Deliberating.", "content": "The answer."}),
        # extract_reasoning returns Optionals; neither may leak into the JSONL as None.
        ((None, "The answer."), {"reasoning": "", "content": "The answer."}),
        (("Deliberating.", None), {"reasoning": "Deliberating.", "content": ""}),
        ((None, None), {"reasoning": "", "content": ""}),
    ],
)
def test_process_outputs_maps_parser_result(extracted, expected):
    InklingClient, _ = _load()
    client = InklingClient.__new__(InklingClient)
    client._reasoning_parser = types.SimpleNamespace(extract_reasoning=lambda *_args: extracted)
    outputs = [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="<|content_text|>x<|end_message|>")])]
    assert client._process_outputs(outputs, include_reasoning=True) == [expected]


def test_set_sampling_params_preserves_special_tokens():
    InklingClient, _ = _load()
    client = InklingClient.__new__(InklingClient)
    assert client.set_sampling_params(temperature=1.0).skip_special_tokens is False


def test_set_reasoning_effort_switches_without_reload():
    InklingClient, _ = _load()
    client = InklingClient.__new__(InklingClient)
    client.set_reasoning_effort("0.1")
    assert client._chat_template_kwargs(enable_thinking=True) == {"reasoning_effort": 0.1}
    client.set_reasoning_effort("0.99")
    assert client._chat_template_kwargs(enable_thinking=True) == {"reasoning_effort": 0.99}
    with pytest.raises(ValueError):
        client.set_reasoning_effort("1.5")
