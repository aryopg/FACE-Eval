"""Verifies pipeline routes models to the right LLM client.

The intent of this test file is purely backward-compatibility: when we add
DeepSeek-V4, the routing should also continue to send Qwen, Llama, Gemma,
Olmo, etc. through `VLLMClient`, and gpt-oss-* through `GPTOSSClient`.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_dataset():
    item = {
        "id": "item_0",
        "axis": "political",
        "condition": "explicit_A",
        "context_type": "rich",
        "scenario_id": "s_0",
        "question": "Q0",
    }

    class _Dataset:
        def __init__(self):
            self.dataset = [item]

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return item

        def get_messages_and_tools(self, idx):
            return [{"role": "user", "content": "hi"}], []

    return _Dataset()


def _stub_llm_client(name: str):
    """Return a MagicMock class whose instances record their constructor args."""
    cls = MagicMock(name=name)
    instance = MagicMock(name=f"{name}_instance")
    instance.set_sampling_params.return_value = types.SimpleNamespace()
    instance.chat_batch.return_value = ["plain answer"]
    cls.return_value = instance
    return cls, instance


def _ensure_llm_modules():
    """Ensure src.llm.* modules are importable as stubs even without vLLM."""
    for mod_name, class_name in (
        ("src.llm.vllm", "VLLMClient"),
        ("src.llm.gpt_oss", "GPTOSSClient"),
        ("src.llm.deepseek_v4", "DeepSeekV4Client"),
        ("src.llm.gemma4", "Gemma4Client"),
        ("src.llm.inkling", "InklingClient"),
    ):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            setattr(stub, class_name, MagicMock())
            sys.modules[mod_name] = stub


@pytest.fixture
def routing_environment(tmp_path, fake_dataset):
    """Pre-populate sys.modules with src.llm.* stubs and patch each client."""
    _ensure_llm_modules()
    vllm_cls, vllm_instance = _stub_llm_client("VLLMClient")
    gpt_oss_cls, gpt_oss_instance = _stub_llm_client("GPTOSSClient")
    v4_cls, v4_instance = _stub_llm_client("DeepSeekV4Client")
    gemma4_cls, gemma4_instance = _stub_llm_client("Gemma4Client")
    inkling_cls, inkling_instance = _stub_llm_client("InklingClient")

    # The pipeline imports these lazily inside run_inference, so patch the
    # attributes on the already-imported stub modules.
    with (
        patch.object(sys.modules["src.llm.vllm"], "VLLMClient", vllm_cls),
        patch.object(sys.modules["src.llm.gpt_oss"], "GPTOSSClient", gpt_oss_cls),
        patch.object(sys.modules["src.llm.deepseek_v4"], "DeepSeekV4Client", v4_cls),
        patch.object(sys.modules["src.llm.gemma4"], "Gemma4Client", gemma4_cls),
        patch.object(sys.modules["src.llm.inkling"], "InklingClient", inkling_cls),
    ):
        yield {
            "dataset": fake_dataset,
            "output_dir": str(tmp_path),
            "vllm_cls": vllm_cls,
            "gpt_oss_cls": gpt_oss_cls,
            "v4_cls": v4_cls,
            "gemma4_cls": gemma4_cls,
            "inkling_cls": inkling_cls,
        }


def _run(model: str, env, **extra):
    from src.pipeline import run_inference

    run_inference(
        model=model,
        seed=42,
        dataset=env["dataset"],
        output_dir=env["output_dir"],
        system_prompt="x",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_tokens=128,
        resume=False,
        backend="vllm",
        **extra,
    )


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3.5-9B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "allenai/OLMo-2-13B",
    ],
)
def test_other_models_route_to_vllm_client(model, routing_environment):
    _run(model, routing_environment)
    assert routing_environment["vllm_cls"].called, f"{model} did not route to VLLMClient"
    assert not routing_environment["gpt_oss_cls"].called
    assert not routing_environment["v4_cls"].called
    assert not routing_environment["gemma4_cls"].called


@pytest.mark.parametrize(
    "model",
    [
        "google/gemma-4-E4B-it",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it",
    ],
)
def test_gemma4_routes_to_gemma4_client(model, routing_environment):
    _run(model, routing_environment)
    assert routing_environment["gemma4_cls"].called, f"{model} did not route to Gemma4Client"
    assert not routing_environment["vllm_cls"].called
    assert not routing_environment["gpt_oss_cls"].called
    assert not routing_environment["v4_cls"].called


def test_gpt_oss_routes_to_gpt_oss_client(routing_environment):
    _run("openai/gpt-oss-20b", routing_environment, reasoning_effort="high")
    assert routing_environment["gpt_oss_cls"].called
    assert not routing_environment["vllm_cls"].called
    assert not routing_environment["v4_cls"].called


@pytest.mark.parametrize("model", ["deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Flash"])
def test_deepseek_v4_routes_to_deepseek_v4_client(model, routing_environment):
    _run(model, routing_environment, reasoning_effort="max")
    cls = routing_environment["v4_cls"]
    assert cls.called, f"{model} did not route to DeepSeekV4Client"
    # Confirm Think Max effort propagated through.
    kwargs = cls.call_args.kwargs
    assert kwargs["reasoning_effort"] == "max"
    assert kwargs["model"] == model


def test_deepseek_v4_rejects_low_medium_effort(routing_environment):
    for effort in ("low", "medium"):
        with pytest.raises(ValueError, match="not supported for DeepSeek V4"):
            _run("deepseek-ai/DeepSeek-V4-Flash", routing_environment, reasoning_effort=effort)


@pytest.mark.parametrize("effort", ["0.1", "0.5", "0.99"])
def test_inkling_routes_to_inkling_client(effort, routing_environment):
    _run("thinkingmachines/Inkling-NVFP4", routing_environment, reasoning_effort=effort)
    cls = routing_environment["inkling_cls"]
    assert cls.called, "Inkling did not route to InklingClient"
    # The raw effort string propagates; InklingClient parses it to a float.
    assert cls.call_args.kwargs["reasoning_effort"] == effort
    assert not routing_environment["vllm_cls"].called


def test_no_think_rejects_gpt_oss(routing_environment):
    with pytest.raises(ValueError, match="no_think is not supported"):
        _run("openai/gpt-oss-20b", routing_environment, no_think=True)


def test_no_think_rejects_non_vllm_backend(fake_dataset, tmp_path):
    from src.pipeline import run_inference

    with pytest.raises(ValueError, match="vLLM backend"):
        run_inference(
            model="claude-sonnet-4-20250514",
            seed=42,
            dataset=fake_dataset,
            output_dir=str(tmp_path),
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            max_tokens=128,
            resume=False,
            backend="anthropic",
            no_think=True,
        )


def test_no_think_vllm_disables_thinking_and_prefills_empty_think(routing_environment):
    _run("Qwen/Qwen3.5-9B", routing_environment, no_think=True)
    cls = routing_environment["vllm_cls"]
    instance = cls.return_value
    assert cls.call_args.kwargs["enable_thinking"] is False
    messages_sent = instance.chat_batch.call_args.args[0]
    assert messages_sent[0][-1] == {"role": "assistant", "content": "<think></think>"}


def test_no_think_gemma4_disables_thinking_without_prefill(routing_environment):
    _run("google/gemma-4-31B-it", routing_environment, no_think=True)
    cls = routing_environment["gemma4_cls"]
    instance = cls.return_value
    assert cls.call_args.kwargs["enable_thinking"] is False
    messages_sent = instance.chat_batch.call_args.args[0]
    assert messages_sent[0][-1] != {"role": "assistant", "content": "<think></think>"}


def test_no_think_deepseek_v4_forces_chat_effort(routing_environment):
    _run("deepseek-ai/DeepSeek-V4-Flash", routing_environment, no_think=True, reasoning_effort="max")
    cls = routing_environment["v4_cls"]
    assert cls.call_args.kwargs["reasoning_effort"] == "chat"


def test_preloaded_llm_reused_across_seeds_and_not_closed(routing_environment):
    """A caller-owned engine is reused across seeds and never torn down internally."""
    from src.pipeline import run_inference

    env = routing_environment
    preloaded = MagicMock(name="preloaded_llm")
    preloaded.set_sampling_params.return_value = types.SimpleNamespace()
    preloaded.chat_batch.return_value = ["plain answer"]

    for seed in (42, 123):
        run_inference(
            model="Qwen/Qwen3-4B",
            seed=seed,
            dataset=env["dataset"],
            output_dir=env["output_dir"],
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            max_tokens=128,
            resume=False,
            backend="vllm",
            preloaded_llm=preloaded,
        )

    assert not env["vllm_cls"].called  # no fresh engine constructed
    preloaded.close.assert_not_called()  # caller owns lifecycle
    assert preloaded.chat_batch.call_count == 2  # reused for both seeds
