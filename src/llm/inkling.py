"""Inkling (Thinking Machines) vLLM backend.

`tokenizer_mode="inkling"` selects vLLM's native `InklingRenderer` — Inkling has no
Jinja chat template, and the one on the HF repo is not what vLLM uses. The renderer
takes OpenAI-shaped messages directly (`render_inkling_messages` json.loads a
JSON-string `arguments` and resolves a tool result's name from the matching
`tool_call_id`), so unlike GPT-OSS/DeepSeek-V4 no custom encoder is needed — this
rides the base `VLLMClient.llm.chat()` path. Three things differ from a plain vLLM
model:

  1. `LLM()` needs `tokenizer_mode="inkling"` + `reasoning_parser="inkling"`.
  2. Reasoning effort is a *continuous float* in [0.0, 1.0), passed per request via
     `chat_template_kwargs={"reasoning_effort": <float>}` (not `enable_thinking`).
     The renderer maps preset names through its own table and silently resolves an
     unknown string to None, which emits no effort line at all — hence the
     up-front validation in `_parse_effort`, whose presets mirror that table.
  3. Reasoning is delimited by channel tokens, not `</think>`:

       <|message_model|><|content_thinking|>[reasoning]<|end_message|>
       <|message_model|><|content_text|>[answer]<|end_message|>

     Passing `reasoning_parser="inkling"` to `LLM()` does NOT split the output:
     that kwarg only feeds `StructuredOutputsConfig` for guided decoding, and
     `CompletionOutput` has no reasoning field. So the base client's `</think>`
     fallback finds no seam and drops everything into the answer. vLLM's actual
     Inkling parser is still reachable offline via `ParserManager`, so we call it
     directly on the generated text.

     It matches marker *strings* (Inkling sets `token_id_terminals={}`), and the
     markers are special tokens that the default skip_special_tokens=True deletes
     during decode. Hence the skip_special_tokens=False override below — without
     it the parser sees no markers and returns everything as content.

The CLI passes `--reasoning-effort` as a string, so we parse it to a float here
(named presets pass through unchanged).

Requires vLLM 0.26.0+ (nightly at time of writing) and these env vars set before
launch: VLLM_USE_V2_MODEL_RUNNER=1, FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1.
"""

from __future__ import annotations

from src.llm.vllm import VLLMClient

# Named presets accepted by the chat template (aliases for floats).
_PRESETS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def _parse_effort(effort: str | None) -> float | str | None:
    """Parse a CLI effort string into a float, a known preset, or None.

    None lets the template apply its own default (0.9 / "high").
    """
    if effort is None:
        return None
    try:
        val = float(effort)
    except ValueError:
        if effort not in _PRESETS:
            raise ValueError(
                f"reasoning_effort must be a float in [0.0, 1.0) or one of {sorted(_PRESETS)}; got {effort!r}"
            )
        return effort
    if not 0.0 <= val < 1.0:
        raise ValueError(f"reasoning_effort float must be in [0.0, 1.0); got {val}")
    return val


class InklingClient(VLLMClient):
    """Client for Thinking Machines Inkling via vLLM."""

    # Built lazily: needs the tokenizer from the engine, which does not exist
    # until VLLMClient.__init__ has run.
    _reasoning_parser = None

    def __init__(
        self,
        model: str,
        reasoning_effort: str | None = None,
        tensor_parallel_size: int | None = None,
        include_reasoning: bool = True,
        **kwargs,
    ):
        """Initialize Inkling client.

        Args:
            model: Model identifier (e.g., "thinkingmachines/Inkling-NVFP4").
            reasoning_effort: Float string in [0.0, 1.0) or a named preset
                (none/minimal/low/medium/high/xhigh/max). None → template default.
            tensor_parallel_size: Number of GPUs. If None, auto-detect.
            include_reasoning: Return reasoning separately if available.
            **kwargs: Additional vLLM parameters.
        """
        self.set_reasoning_effort(reasoning_effort)
        kwargs.setdefault("tokenizer_mode", "inkling")
        kwargs.setdefault("reasoning_parser", "inkling")
        kwargs.setdefault("trust_remote_code", True)
        super().__init__(
            model,
            tensor_parallel_size=tensor_parallel_size,
            include_reasoning=include_reasoning,
            **kwargs,
        )

    def set_reasoning_effort(self, reasoning_effort: str | None) -> None:
        """Parse and set the reasoning effort so one loaded engine can serve multiple efforts."""
        self.reasoning_effort = _parse_effort(reasoning_effort)

    def _chat_template_kwargs(self, enable_thinking: bool) -> dict:
        """Inject reasoning effort instead of the enable_thinking toggle."""
        return {"reasoning_effort": self.reasoning_effort}

    def set_sampling_params(self, **kwargs):
        """Create sampling params with special tokens preserved for channel parsing."""
        return super().set_sampling_params(**kwargs, skip_special_tokens=False)

    def _parser(self):
        """Return vLLM's registered Inkling reasoning parser, building it on first use."""
        if self._reasoning_parser is None:
            from vllm.parser import ParserManager

            parser_cls = ParserManager.get_reasoning_parser("inkling")
            self._reasoning_parser = parser_cls(self.llm.get_tokenizer())
        return self._reasoning_parser

    def _process_outputs(self, outputs, include_reasoning: bool) -> list:
        parser = self._parser()
        results = []
        for output in outputs:
            # Generation starts after the prompt's trailing <|message_model|>, which is
            # the state the Inkling parser seeds itself in. `request` is unused.
            reasoning, content = parser.extract_reasoning(output.outputs[0].text, None)
            if include_reasoning:
                results.append({"reasoning": reasoning or "", "content": content or ""})
            else:
                results.append(content or "")
        return results
