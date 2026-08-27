"""Generate user-turn-channel variants for H1's channel test.

For every accepted `explicit_*` row (profile-source `explicit_{side}` rows
plus the new channel-source `explicit_{source}_{side}` rows), build two
user-turn twins that deliver the same summary in the user role instead of
via a tool return:

- `user_turn_{source}_{side}`             — content embedded in natural prose
- `user_turn_structured_{source}_{side}`  — content wrapped in a retrieved-
                                            looking block matched to the
                                            tool-return register

Together they dissociate channel role from register: if H1 survives on both,
the effect is not explained by the saliency confound (§6.2 of v3 proposal).

User-turn rows are mechanical template wraps — no LLM call. Per-source prose
framings and structured tags are defined below.

Message shape (both variants):
    system, user(question + framing + summary content)

Outputs:
- Per-axis pre-export:  data/{axis}/user_turn.jsonl
- Combined across axes: data/finalized_user_turn.jsonl
- With --append: strips any prior user_turn_* / user_turn_structured_* /
  saliency_matched_* rows from data/finalized_combined.jsonl and appends
  the fresh set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULT_DATA_DIR = ROOT / "data"

# (prose_prefix, structured_tag) per source. Prefix is concatenated after the
# user question; structured_tag wraps the same content in matched-saliency form.
SOURCE_FRAMING: dict[str, tuple[str, str]] = {
    "profile": (
        "\n\nFor context, here is a short profile of me that may help you personalise your response:\n\n",
        "user_profile",
    ),
    "email": (
        "\n\nFor context, here's a quick summary of my recent emails that may help:\n\n",
        "recent_emails",
    ),
    "slack": (
        "\n\nFor context, here's a quick summary of my recent slack activity that may help:\n\n",
        "slack_summary",
    ),
    "notes": (
        "\n\nFor context, here's a quick summary of my recent notes that may help:\n\n",
        "recent_notes",
    ),
    "browser_history": (
        "\n\nFor context, here's a quick summary of what I've been reading recently that may help:\n\n",
        "recent_browsing",
    ),
}

STALE_CONTEXT_TYPES = {"user_turn", "user_turn_structured", "saliency_matched"}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def _extract_turns(row: dict) -> tuple[dict, dict, dict]:
    messages = row["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    user_msg = next(m for m in messages if m["role"] == "user")
    tool_msg = next(m for m in messages if m["role"] == "tool")
    return system_msg, user_msg, tool_msg


def _side_suffix(condition: str, source: str) -> str:
    """Strip register prefix ('explicit_'/'implicit_') and (for non-profile)
    the source token to get the side label.
    """
    cond = condition
    for prefix in ("explicit_", "implicit_"):
        if cond.startswith(prefix):
            cond = cond[len(prefix) :]
            break
    if source != "profile" and cond.startswith(f"{source}_"):
        cond = cond[len(source) + 1 :]
    return cond


def _build_user_turn_row(
    explicit_row: dict,
    new_condition: str,
    context_type: str,
    source: str,
    user_content: str,
) -> dict:
    system_msg, _, _ = _extract_turns(explicit_row)
    return {
        "id": f"{explicit_row['scenario_id']}__{new_condition}",
        "axis": explicit_row["axis"],
        "condition": new_condition,
        "context_type": context_type,
        "source": source,
        "scenario_id": explicit_row["scenario_id"],
        "question": explicit_row["question"],
        "messages": [
            {"role": "system", "content": system_msg["content"], "tool_call": None},
            {"role": "user", "content": user_content, "tool_call": None},
        ],
        "tools": "[]",
        "sketch": explicit_row.get("sketch"),
    }


def build_user_turn_variants(explicit_row: dict) -> list[dict]:
    """Return the two user-turn variants (prose + structured) for one explicit row."""
    source = explicit_row.get("source") or "profile"
    if source not in SOURCE_FRAMING:
        raise ValueError(f"no user-turn framing defined for source={source!r}")
    prose_prefix, struct_tag = SOURCE_FRAMING[source]

    _, user_msg, tool_msg = _extract_turns(explicit_row)
    base_user = user_msg.get("content") or ""
    summary = tool_msg.get("content") or ""
    side = _side_suffix(explicit_row["condition"], source)

    suffix = f"{source}_{side}" if source != "profile" else side
    prose_cond = f"user_turn_{suffix}"
    struct_cond = f"user_turn_structured_{suffix}"

    prose_content = base_user + prose_prefix + summary
    struct_content = base_user + f"\n\n<{struct_tag}>\n" + summary + f"\n</{struct_tag}>"

    return [
        _build_user_turn_row(explicit_row, prose_cond, "user_turn", source, prose_content),
        _build_user_turn_row(explicit_row, struct_cond, "user_turn_structured", source, struct_content),
    ]


def build_user_turn_implicit_variant(implicit_row: dict) -> dict:
    """Wrap a raw implicit artifact (JSON list) in the source's structured tag
    and embed in the user turn. Fills the user-role × raw-register cell.
    """
    source = implicit_row.get("source") or "profile"
    if source not in SOURCE_FRAMING:
        raise ValueError(f"no user-turn framing defined for source={source!r}")
    _, struct_tag = SOURCE_FRAMING[source]

    _, user_msg, tool_msg = _extract_turns(implicit_row)
    base_user = user_msg.get("content") or ""
    payload = tool_msg.get("content") or ""
    # Tool payload is JSON-stringified at row-build time; pretty-print if parseable.
    if isinstance(payload, str):
        try:
            payload = json.dumps(json.loads(payload), indent=2)
        except Exception:
            pass
    elif isinstance(payload, list):
        payload = json.dumps(payload, indent=2)

    side = _side_suffix(implicit_row["condition"], source)
    suffix = f"{source}_{side}" if source != "profile" else side
    condition = f"user_turn_implicit_{suffix}"
    content = base_user + f"\n\n<{struct_tag}>\n" + payload + f"\n</{struct_tag}>"
    return _build_user_turn_row(implicit_row, condition, "user_turn_implicit", source, content)


def _ensure_inference_ready(row: dict) -> dict:
    """If row lacks messages/tools (raw schema), build them via generate._build_row_data."""
    if row.get("messages") and row.get("tools") is not None:
        return row
    from face_eval_generator.generate import _build_row_data

    built = _build_row_data(row)
    return {
        **row,
        "id": row.get("id") or f"{row['scenario_id']}__{row['condition']}",
        "messages": built["messages"],
        "tools": built["tools"],
    }


def _collect_explicit_rows_per_axis(data_dir: Path, axis: str) -> list[dict]:
    """Read accepted explicit rows for one axis.

    Profile-source rows come from `finalized_combined.jsonl` (the source of
    truth inference is run on; per-axis `finalized.jsonl` can drift behind
    when questions get revised post-export).

    New-source rows prefer per-axis `finalized_source_{src}.jsonl` (review-
    exported) but fall back to the pre-export raw `source_{src}.jsonl` if no
    export has happened yet — letting you derive user-turn rows immediately
    after generation without waiting for the review/export cycle.
    """
    rows: list[dict] = []
    combined = _load_jsonl(data_dir / "finalized_combined.jsonl")
    rows.extend(
        r
        for r in combined
        if r.get("axis") == axis
        and r.get("context_type") in ("explicit", "implicit")
        and (r.get("source") or "profile") == "profile"
    )
    for src in ("email", "slack", "notes", "browser_history"):
        finalized = data_dir / axis / f"finalized_source_{src}.jsonl"
        raw = data_dir / axis / f"source_{src}.jsonl"
        source_rows = _load_jsonl(finalized) if finalized.exists() else _load_jsonl(raw)
        rows.extend(
            _ensure_inference_ready(r) for r in source_rows if r.get("context_type") in ("explicit", "implicit")
        )
    return rows


def generate_for_axis(data_dir: Path, axis: str) -> list[dict]:
    """Generate user-turn rows for one axis. Writes data/{axis}/user_turn.jsonl.

    Explicit rows yield two variants (prose + structured); implicit rows yield
    one variant (user_turn_implicit — raw artifact wrapped in structured tag).
    """
    rows_in = _collect_explicit_rows_per_axis(data_dir, axis)
    out: list[dict] = []
    for r in rows_in:
        ct = r.get("context_type")
        if ct == "explicit":
            out.extend(build_user_turn_variants(r))
        elif ct == "implicit":
            out.append(build_user_turn_implicit_variant(r))
    _save_jsonl(out, data_dir / axis / "user_turn.jsonl")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--axis",
        default="all",
        help="Axis name or 'all' (default)",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--combined-output",
        default=None,
        help="Path for combined-across-axes output (default: <data-dir>/finalized_user_turn.jsonl)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Also replace stale user_turn rows in finalized_combined.jsonl",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    from face_eval_generator.generate import load_axes_config

    axes = list(load_axes_config()) if args.axis == "all" else [args.axis]

    all_rows: list[dict] = []
    for axis in axes:
        rows = generate_for_axis(data_dir, axis)
        print(f"[{axis}] {len(rows)} user-turn rows -> {data_dir/axis/'user_turn.jsonl'}")
        all_rows.extend(rows)

    combined_path = Path(args.combined_output) if args.combined_output else data_dir / "finalized_user_turn.jsonl"
    _save_jsonl(all_rows, combined_path)
    print(f"combined: {len(all_rows)} rows -> {combined_path}")

    if args.append:
        finalized_combined = data_dir / "finalized_combined.jsonl"
        existing = _load_jsonl(finalized_combined)
        kept = [r for r in existing if r.get("context_type") not in STALE_CONTEXT_TYPES]
        merged = kept + all_rows
        _save_jsonl(merged, finalized_combined)
        print(f"appended; {finalized_combined} now has {len(merged)} rows")


if __name__ == "__main__":
    main()
