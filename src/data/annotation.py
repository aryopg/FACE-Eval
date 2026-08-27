"""Annotation helpers for the cue-salience annotation experiment (artifact-rating).

Three public functions:
- extract_artifact: pull the artifact text out of a dataset row
- parse_annotation_output: parse and validate a model's JSON output
- generate_ab_assignment: stable A/B label assignment per item stratum
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import yaml

# user_turn preamble always starts with "\n\nFor context," and ends with ":\n\n{artifact}"
_USER_TURN_PREFIX = "\n\nFor context,"
_USER_TURN_SEP = ":\n\n"

_VALID_SIDES = {"A", "B", "unclear", "refusal"}
_VALID_SCORES = {1, 2, 3, 4, 5, None}


def extract_artifact(row: dict) -> str:
    """Extract the artifact text from a dataset row.

    The extraction strategy depends on `context_type`:
    - explicit / implicit: artifact is the content of the tool message.
    - user_turn: artifact follows the profile marker in the user message.
    - user_turn_structured / user_turn_implicit: artifact is between
      <user_profile> and </user_profile> tags in the user message.

    Raises:
        ValueError: If context_type is unrecognised or the expected structure
            is absent.
    """
    context_type = row.get("context_type")
    messages = row.get("messages") or []

    if context_type in ("explicit", "implicit"):
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content") or ""
                if not content:
                    raise ValueError(f"Tool message is empty for context_type={context_type!r} (id={row.get('id')!r})")
                return content
        raise ValueError(f"No tool message found for context_type={context_type!r} (id={row.get('id')!r})")

    if context_type == "user_turn":
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                prefix_idx = content.find(_USER_TURN_PREFIX)
                if prefix_idx == -1:
                    raise ValueError(f"user_turn context prefix not found in user message (id={row.get('id')!r})")
                sep_idx = content.find(_USER_TURN_SEP, prefix_idx)
                if sep_idx == -1:
                    raise ValueError(f"user_turn separator ':\\n\\n' missing after prefix (id={row.get('id')!r})")
                return content[sep_idx + len(_USER_TURN_SEP) :]
        raise ValueError(f"No user message found for context_type={context_type!r} (id={row.get('id')!r})")

    if context_type in ("user_turn_structured", "user_turn_implicit"):
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                # Tag is source-specific: <user_profile>, <recent_emails>, <slack_summary>, etc.
                tag_match = re.search(r"\n\n<(\w+)>", content)
                if not tag_match:
                    raise ValueError(
                        f"No XML-tagged artifact block found in {context_type!r} user message "
                        f"(id={row.get('id')!r})"
                    )
                tag = tag_match.group(1)
                open_tag = f"<{tag}>"
                close_tag = f"</{tag}>"
                start = content.find(open_tag)
                end = content.find(close_tag)
                if start == -1 or end == -1:
                    raise ValueError(f"Malformed <{tag}> block in {context_type!r} user message (id={row.get('id')!r})")
                return content[start + len(open_tag) : end].strip()
        raise ValueError(f"No user message found for context_type={context_type!r} (id={row.get('id')!r})")

    raise ValueError(
        f"Unexpected context_type={context_type!r} (id={row.get('id')!r}). "
        "Expected one of: explicit, implicit, user_turn, user_turn_structured, user_turn_implicit"
    )


def parse_annotation_output(raw: str) -> tuple[dict | None, str]:
    """Parse and validate a model's annotation JSON output.

    Fallback chain (§12.7):
    1. Try json.loads(raw) directly.
    2. If raw contains </think>, take everything after it and try json.loads.
    3. Search for a ```json fenced block and try json.loads on its contents.
    4. Find the last balanced {...} substring in raw and try json.loads.
    5. Return (None, "parse_error").

    After parsing, validate the schema:
    - side ∈ {"A", "B", "unclear", "refusal"}
    - clarity_score ∈ {1, 2, 3, 4, 5, None}
    - clarity_score is None iff side == "refusal"
    - explanation is a str

    The thinking trace (if any) is already embedded in `raw` before the
    ``</think>`` tag; the caller stores it separately and does not need to
    pass it here.

    Returns:
        (parsed_dict, status) where status is "ok", "parse_error", or
        "schema_error".
    """

    def _try_loads(s: str) -> dict | None:
        try:
            obj = json.loads(s.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        return None

    # Step 1: whole output
    parsed = _try_loads(raw)

    # Step 2: after </think>
    if parsed is None and "</think>" in raw:
        after_think = raw.split("</think>", 1)[1]
        parsed = _try_loads(after_think)

    # Step 3: ```json fenced block
    if parsed is None:
        fence_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fence_match:
            parsed = _try_loads(fence_match.group(1))

    # Step 4: last balanced {...} substring
    if parsed is None:
        last_start = raw.rfind("{")
        if last_start != -1:
            depth = 0
            end_idx = None
            for i, ch in enumerate(raw[last_start:], start=last_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx is not None:
                parsed = _try_loads(raw[last_start : end_idx + 1])

    if parsed is None:
        return None, "parse_error"

    # Schema validation
    side = parsed.get("side")
    clarity_score = parsed.get("clarity_score")
    explanation = parsed.get("explanation")

    if side not in _VALID_SIDES:
        return None, "schema_error"
    if clarity_score not in _VALID_SCORES:
        return None, "schema_error"
    if (clarity_score is None) != (side == "refusal"):
        return None, "schema_error"
    if not isinstance(explanation, str):
        return None, "schema_error"

    return parsed, "ok"


def generate_ab_assignment(rows: list[dict], seed: int = 42) -> dict[str, dict]:
    """Generate a stable A/B label assignment for each cued item.

    Algorithm (§12.6):
    1. Group item IDs by (axis, source) stratum.
    2. For each stratum, sort IDs lexicographically, then shuffle with
       random.Random(seed).
    3. Assign the first ceil(N/2) items a_is_gt=True (A = ground-truth side),
       the rest a_is_gt=False.
    4. Derive side labels from condition: ground_truth_side is the last
       "_"-delimited token (e.g. "liberal", "utilitarian"). The opposing
       side is derived from config/side_definitions.yaml.

    Returns:
        {item_id: {"A": side_label, "B": side_label, "a_is_gt": bool}}
    """
    # This file is src/data/annotation.py, so the repo root is two levels up.
    side_defs: dict[str, Any] = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "config" / "side_definitions.yaml").read_text()
    )
    # Build axis -> [side_a, side_b] map (preserving YAML insertion order)
    axis_sides: dict[str, list[str]] = {axis: list(sides.keys()) for axis, sides in side_defs.items()}

    # Group rows by stratum
    strata: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        axis = row.get("axis") or "unknown"
        source = row.get("source") or "unknown"
        key = (axis, source)
        strata.setdefault(key, []).append(row)

    assignment: dict[str, dict] = {}
    rng = random.Random(seed)

    for (axis, _source), stratum_rows in sorted(strata.items()):
        sorted_ids = sorted(row["id"] for row in stratum_rows)
        rng.shuffle(sorted_ids)

        n = len(sorted_ids)
        n_gt_is_a = math.ceil(n / 2)

        # Build id -> row lookup for this stratum
        id_to_row = {row["id"]: row for row in stratum_rows}

        for rank, item_id in enumerate(sorted_ids):
            a_is_gt = rank < n_gt_is_a
            row = id_to_row[item_id]
            condition = row.get("condition") or ""
            # condition suffix is the named side label (e.g. "liberal", "utilitarian")
            gt_label = condition.split("_")[-1]

            sides = axis_sides.get(axis, [])
            if len(sides) == 2:
                other_label = sides[1] if sides[0] == gt_label else sides[0]
            else:
                other_label = "other"

            if a_is_gt:
                a_label = gt_label
                b_label = other_label
            else:
                a_label = other_label
                b_label = gt_label

            assignment[item_id] = {"A": a_label, "B": b_label, "a_is_gt": a_is_gt}

    # ceil(n/2) split over a shuffled stratum keeps A/B counts balanced by
    # construction, to within one item on odd-sized strata.
    return assignment
