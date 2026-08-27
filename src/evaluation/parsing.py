"""Parsing utilities for model outputs with CoT reasoning."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedOutput:
    reasoning: str
    raw_answer: str
    final_answer: str | None


def extract_answer(text: str) -> str | None:
    """Extract content from the last closed <answer></answer> tags."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def parse_model_output(output: str, reasoning_from_model: str = "") -> ParsedOutput:
    """Parse model output into reasoning and answer components.

    If `reasoning_from_model` is provided (e.g. from vLLM's reasoning field or
    the Harmony SDK), use it directly and treat `output` as the answer portion.
    Otherwise, split on the last `</think>` tag and strip any remaining open
    tags from the reasoning segment.
    """
    if reasoning_from_model:
        return ParsedOutput(
            reasoning=reasoning_from_model,
            raw_answer=output,
            final_answer=extract_answer(output),
        )

    # Split at the last </think> tag (case-insensitive).
    parts = re.split(r"</think>", output, flags=re.IGNORECASE)
    if len(parts) > 1:
        # Everything before the last </think> is reasoning.
        # Join intermediate segments (multiple <think> blocks) and strip open tags.
        reasoning_raw = "\n".join(parts[:-1])
        reasoning = re.sub(r"</?think>", "\n", reasoning_raw, flags=re.IGNORECASE).strip()
        reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
        raw_answer = parts[-1].strip()
    else:
        reasoning = ""
        raw_answer = output

    return ParsedOutput(
        reasoning=reasoning,
        raw_answer=raw_answer,
        final_answer=extract_answer(raw_answer),
    )
