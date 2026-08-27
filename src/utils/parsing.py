"""Shared text-parsing utilities for judge response processing."""

from __future__ import annotations

import json
import re


def extract_json_object(response: str) -> dict:
    """Pull a JSON object out of a judge response, fenced or bare.

    Tries the fenced ```json ... ``` form first (greedy DOTALL match handles
    nested objects). Falls back to scanning for the first ``{`` and using
    ``JSONDecoder.raw_decode``, which correctly handles nested structures.
    Raises ``ValueError`` if nothing parses.
    """
    match = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = response.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    obj, _ = json.JSONDecoder().raw_decode(response, start)
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    return obj


def substitute(template: str, values: dict) -> str:
    """Replace ``{key}`` placeholders in *template* with *values*."""
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out
