"""Gemma 4 vLLM client with channel-tag reasoning extraction.

Gemma 4 uses special channel tokens to delimit thinking:
  <|channel>thought
  [reasoning]
  <channel|>[final answer]

These are special tokens and are stripped during decode with the default
skip_special_tokens=True, which discards the separator between reasoning and
answer. This client forces skip_special_tokens=False and parses the tags.

Reference: https://huggingface.co/google/gemma-4-31B-it#thinking-mode-configuration
"""

from __future__ import annotations

try:
    from src.llm.vllm import VLLMClient
except (ImportError, ModuleNotFoundError):
    VLLMClient = object  # type: ignore[assignment,misc]

_CHANNEL_START = "<|channel>thought"
_CHANNEL_END = "<channel|>"
_EOS = "<eos>"
_THINK_TOKEN = "<|think|>"


def _inject_think_token(messages: list[dict]) -> list[dict]:
    """Prepend <|think|> to the system message to enable Gemma 4 thinking.

    If no system message exists, inserts one at the front. Returns a new list
    and new system-message dict — does not mutate the caller's messages.
    """
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        content = messages[0]["content"]
        if not content.startswith(_THINK_TOKEN):
            messages[0] = {**messages[0], "content": f"{_THINK_TOKEN}{content}"}
    else:
        messages.insert(0, {"role": "system", "content": _THINK_TOKEN})
    return messages


def _parse_gemma_output(text: str) -> dict[str, str]:
    """Parse Gemma 4 channel-tagged output into reasoning and content."""
    text = text.replace(_EOS, "").rstrip()

    if _CHANNEL_END not in text:
        return {"reasoning": "", "content": text.strip()}

    reasoning_raw, _, content = text.partition(_CHANNEL_END)

    if reasoning_raw.startswith(_CHANNEL_START):
        reasoning_raw = reasoning_raw[len(_CHANNEL_START) :]

    return {
        "reasoning": reasoning_raw.strip(),
        "content": content.strip(),
    }


class Gemma4Client(VLLMClient):
    """vLLM client for Gemma 4 models with channel-tag reasoning extraction."""

    def set_sampling_params(self, **kwargs):
        """Create sampling params with special tokens preserved for tag parsing."""
        return super().set_sampling_params(**kwargs, skip_special_tokens=False)

    def chat_batch(self, messages_list, *, enable_thinking=None, **kwargs):
        """Inject <|think|> into system prompts and enable template-level thinking.

        When enable_thinking is False (no-think mode), skips injection entirely.
        """
        if enable_thinking is None:
            enable_thinking = self.enable_thinking
        if not enable_thinking:
            return super().chat_batch(messages_list, enable_thinking=False, **kwargs)
        injected = [_inject_think_token(msgs) for msgs in messages_list]
        return super().chat_batch(injected, enable_thinking=True, **kwargs)

    def _process_outputs(self, outputs, include_reasoning: bool) -> list:
        results = []
        for output in outputs:
            text = output.outputs[0].text
            if include_reasoning:
                results.append(_parse_gemma_output(text))
            else:
                results.append(text.replace(_EOS, "").strip())
        return results
