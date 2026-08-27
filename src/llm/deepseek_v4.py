"""DeepSeek-V4 vLLM backend using the upstream `encoding_dsv4` reference encoder.

DeepSeek-V4 ships *no* Jinja chat template. Instead, the model repo includes
`encoding/encoding_dsv4.py` (a single Python file, ~28KB) which is the canonical
DSML encoder/decoder. We download it via `huggingface_hub.hf_hub_download` on
first use and import it dynamically — same spirit as `trust_remote_code=True`
elsewhere in the stack. This avoids vendoring (which drifts from upstream) and
re-implementing DSML by hand (which would diverge subtly).

Reasoning-effort mapping for the V4 family:

    "chat"  -> thinking_mode="chat"                      (Non-think)
    "high"  -> thinking_mode="thinking" + reasoning_effort="high"   (Think High)
    "max"   -> thinking_mode="thinking" + reasoning_effort="max"   (Think Max)

We deliberately reuse the existing `--reasoning-effort` CLI knob so callers can
toggle V4 thinking levels without learning a new flag.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams

from src.llm.base import BaseLLM, shutdown_vllm_engine
from src.utils.logging import get_logger

# vLLM strips the stop token that ends generation from output text even with
# skip_special_tokens=False. The DSML parser needs this token as a delimiter.
_DSML_EOS = "<｜end▁of▁sentence｜>"

# CLI-effort -> (thinking_mode, reasoning_effort) pair consumed by encoding_dsv4.
_EFFORT_MAP: dict[str, tuple[str, str | None]] = {
    "chat": ("chat", None),
    "high": ("thinking", "high"),
    "max": ("thinking", "max"),
}


def _load_encoding_module(model: str) -> Any:
    """Download `encoding/encoding_dsv4.py` from the model repo and import it.

    Cached in `sys.modules` under a model-specific key so multiple V4 variants
    can coexist in one process without collision.
    """
    cache_key = f"_encoding_dsv4__{model.replace('/', '__')}"
    if cache_key in sys.modules:
        return sys.modules[cache_key]

    local_path = hf_hub_download(repo_id=model, filename="encoding/encoding_dsv4.py")

    spec = importlib.util.spec_from_file_location(cache_key, Path(local_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load encoding_dsv4 spec from {local_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = module
    spec.loader.exec_module(module)
    return module


class DeepSeekV4Client(BaseLLM):
    """Client for DeepSeek-V4 (Pro / Flash) via vLLM + DSML encoding."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int | None = None,
        reasoning_effort: str = "high",
        include_reasoning: bool = True,
        **kwargs,
    ):
        """Initialize DeepSeek-V4 client.

        Args:
            model: Model identifier (e.g., "deepseek-ai/DeepSeek-V4-Flash").
            tensor_parallel_size: Number of GPUs. If None, auto-detect.
            reasoning_effort: One of "chat", "high", "max". See module docstring
                for the mapping to V4's three reasoning modes.
            include_reasoning: Return reasoning separately if available.
            **kwargs: Additional vLLM parameters.
        """
        self.set_reasoning_effort(reasoning_effort)  # validate before the expensive engine load

        super().__init__(model, **kwargs)
        self.include_reasoning = include_reasoning
        self.encoding = _load_encoding_module(model)

        if tensor_parallel_size is None:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                tensor_parallel_size = num_gpus

        if tensor_parallel_size is not None:
            kwargs["tensor_parallel_size"] = tensor_parallel_size

        # DeepSeek-V4's attention implementation in vLLM requires fp8 kv-cache;
        # the default "auto" triggers an assertion error inside vLLM.
        # Allow the caller to override via kwargs if a future vLLM version
        # relaxes this constraint.
        kwargs.setdefault("kv_cache_dtype", "fp8")
        kwargs.setdefault("trust_remote_code", True)
        self.llm = LLM(model=model, **kwargs)
        # DSML parser requires structural special tokens (<｜end▁of▁sentence｜>,
        # </think>, ｜DSML｜, etc.) to be present in gen.text. vLLM strips them
        # by default (skip_special_tokens=True), which causes 100% parse failures.
        self.default_sampling_params = SamplingParams(
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

    def set_reasoning_effort(self, reasoning_effort: str) -> None:
        """Set reasoning effort (validated), recomputing the derived thinking-mode pair.

        Lets one loaded engine serve multiple efforts without reloading weights.
        """
        if reasoning_effort not in _EFFORT_MAP:
            raise ValueError(f"reasoning_effort must be one of {list(_EFFORT_MAP)}; got {reasoning_effort!r}")
        self.reasoning_effort = reasoning_effort
        self.thinking_mode, self._dsml_reasoning_effort = _EFFORT_MAP[reasoning_effort]

    def _attach_tools_to_system(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
    ) -> list[dict[str, Any]]:
        """Return a copy of `messages` with `tools` attached to the system message.

        DeepSeek-V4's `encode_messages` expects tools as a `tools` field on the
        system (or developer) message. If no system message is present, prepend
        an empty one — the encoder is happy with empty content.
        """
        if not tools:
            return list(messages)

        out: list[dict[str, Any]] = []
        attached = False
        for msg in messages:
            if msg.get("role") == "system" and not attached:
                out.append({**msg, "tools": tools})
                attached = True
            else:
                out.append(dict(msg))
        if not attached:
            out.insert(0, {"role": "system", "content": "", "tools": tools})
        return out

    def _encode_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
    ) -> str:
        """Build the DSML prompt string for one conversation."""
        prepared = self._attach_tools_to_system(messages, tools)
        return self.encoding.encode_messages(
            prepared,
            thinking_mode=self.thinking_mode,
            reasoning_effort=self._dsml_reasoning_effort,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        sampling_params: SamplingParams | None = None,
        include_reasoning: bool | None = None,
        **kwargs,
    ) -> str | dict[str, str]:
        """Generate a single chat response."""
        return self.chat_batch(
            [messages], sampling_params=sampling_params, include_reasoning=include_reasoning, **kwargs
        )[0]

    def chat_batch(
        self,
        messages_list: list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
        include_reasoning: bool | None = None,
        tools_list: list[list[dict] | None] | None = None,
        **kwargs,
    ) -> list[str | dict[str, str]]:
        """Generate responses for multiple conversations."""
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        if include_reasoning is None:
            include_reasoning = self.include_reasoning

        if tools_list is not None and len(tools_list) != len(messages_list):
            raise ValueError(
                f"tools_list length {len(tools_list)} does not match messages_list length {len(messages_list)}"
            )

        prompts = [
            self._encode_prompt(messages, tools_list[i] if tools_list is not None else None)
            for i, messages in enumerate(messages_list)
        ]

        logger = get_logger()
        logger.info(f"DeepSeek V4 first prompt (first 400 chars): {prompts[0][:400]!r}")

        outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
        return self._process_outputs(outputs, include_reasoning)

    def _process_outputs(
        self,
        outputs,
        include_reasoning: bool,
    ) -> list[str | dict[str, Any]]:
        """Parse DSML completion text back into reasoning + content."""
        results = []
        for output in outputs:
            gen = output.outputs[0]
            # vLLM strips the EOS stop token from output text; the DSML parser
            # needs it as a delimiter. Always append if missing.
            text = gen.text
            if not text.endswith(_DSML_EOS):
                text += _DSML_EOS
            try:
                parsed = self.encoding.parse_message_from_completion_text(
                    text,
                    thinking_mode=self.thinking_mode,
                )
            except Exception as exc:  # noqa: BLE001 — upstream parser raises plain Exception
                logger = get_logger()
                logger.warning(
                    f"DSML parse failed [{type(exc).__name__}: {exc}] | "
                    f"raw[:300]: {text[:300]!r} | "
                    f"raw[-300:]: {text[-300:]!r}"
                )
                if include_reasoning:
                    results.append(
                        {
                            "reasoning": "",
                            "content": "",
                            "raw_fallback": text,
                            "parse_error": f"{type(exc).__name__}: {exc}",
                            "harmony_parse_failed": True,
                        }
                    )
                else:
                    results.append(gen.text)
                continue

            reasoning = parsed.get("reasoning_content") or parsed.get("reasoning") or ""
            content = parsed.get("content") or ""

            if include_reasoning:
                results.append({"reasoning": reasoning, "content": content})
            else:
                results.append(content)

        return results

    def close(self) -> None:
        """Shut down the underlying vLLM engine before interpreter teardown."""
        shutdown_vllm_engine(self.llm)

    def set_sampling_params(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 32768,
        seed: int | None = None,
        **kwargs,
    ) -> SamplingParams:
        """Create sampling parameters.

        Defaults follow DeepSeek's recommendation (temperature=1.0, top_p=1.0).
        skip_special_tokens and spaces_between_special_tokens are forced off so
        that DSML structural tokens survive decoding for the parser.
        """
        kwargs.setdefault("skip_special_tokens", False)
        kwargs.setdefault("spaces_between_special_tokens", False)
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            **kwargs,
        )
