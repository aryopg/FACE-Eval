"""Assemble the full agentic-sycophancy dataset into finalized_combined.jsonl.

This script is drift-safe: it preserves profile-source rows (which carry the
revised questions) and `no_context` controls from the existing
finalized_combined.jsonl, then rebuilds the new-source tool-channel rows
from per-axis pre-export files, then merges in user-turn rows from
finalized_user_turn.jsonl.

Why not just call export_all_finalized()? That helper re-exports profile
rows from per-axis scenarios.jsonl, which carries an older snapshot of the
questions (revisions were applied to finalized_combined.jsonl but never
back-propagated to scenarios.jsonl). This script avoids that regression.

Output: data/finalized_combined.jsonl (overwritten in place; a .bak is
written first).
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter

from face_eval_generator.generate import (
    DEFAULT_DATA_DIR,
    SOURCE_VARIANT_NAMES,
    _build_row_data,
    load_axes_config,
    load_jsonl,
    save_jsonl,
)


def _build_inference_ready(raw: dict) -> dict:
    """Take a raw pre-export row and produce a fully inference-ready row."""
    built = _build_row_data(raw)
    return {
        "id": f"{raw['scenario_id']}__{raw['condition']}",
        "axis": raw["axis"],
        "condition": raw["condition"],
        "context_type": raw["context_type"],
        "source": raw.get("source"),
        "scenario_id": raw["scenario_id"],
        "question": raw["question"],
        "messages": built["messages"],
        "tools": built["tools"],
        "sketch": raw.get("sketch"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing")
    args = parser.parse_args()

    combined_path = DEFAULT_DATA_DIR / "finalized_combined.jsonl"

    # 1. Preserve profile-source rows + no_context from existing combined.
    existing = load_jsonl(combined_path)
    preserved: list[dict] = []
    for r in existing:
        ct = r.get("context_type")
        src = r.get("source")
        if ct == "none":
            preserved.append(r)
        elif ct in ("explicit", "implicit") and (src or "profile") == "profile":
            preserved.append(r)
    print(f"preserved from existing combined: {len(preserved)} rows (profile tool-channel + no_context)")

    # 2. Rebuild new-source tool-channel rows from per-axis source_{src}.jsonl.
    new_source_rows: list[dict] = []
    for axis in load_axes_config():
        for src in SOURCE_VARIANT_NAMES:
            raw_path = DEFAULT_DATA_DIR / axis / f"source_{src}.jsonl"
            if not raw_path.exists():
                continue
            for raw in load_jsonl(raw_path):
                # Defensive: ensure source field set (was always set by generator)
                raw.setdefault("source", src)
                new_source_rows.append(_build_inference_ready(raw))
    print(f"new-source tool-channel rows (built from source_*.jsonl): {len(new_source_rows)}")

    # 3. Pull user-turn rows from finalized_user_turn.jsonl (already inference-ready).
    user_turn_rows = load_jsonl(DEFAULT_DATA_DIR / "finalized_user_turn.jsonl")
    print(f"user-turn rows (from finalized_user_turn.jsonl): {len(user_turn_rows)}")

    merged = preserved + new_source_rows + user_turn_rows
    print()
    print(f"TOTAL merged rows: {len(merged)}")

    # Breakdown
    breakdown = Counter((r.get("context_type"), r.get("source")) for r in merged)
    print("breakdown:")
    for k, v in sorted(breakdown.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")):
        print(f"  {k}: {v}")

    # ID uniqueness check
    ids = [r["id"] for r in merged]
    dup = [iid for iid, n in Counter(ids).items() if n > 1]
    if dup:
        print(f"WARNING: {len(dup)} duplicate ids; first few: {dup[:5]}")

    if args.dry_run:
        print()
        print("(dry-run — not writing)")
        return

    if combined_path.exists():
        bak = combined_path.with_suffix(combined_path.suffix + ".bak")
        shutil.copy2(combined_path, bak)
        print(f"backup -> {bak}")

    save_jsonl(merged, combined_path)
    print(f"wrote {len(merged)} rows -> {combined_path}")


if __name__ == "__main__":
    main()
