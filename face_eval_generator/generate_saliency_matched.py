"""Generate saliency-matched user-turn variants for H1's channel-saliency control.

For every `implicit_*` row in `finalized_combined.jsonl`, construct a matched
`saliency_matched_*` row that delivers the same decision-relevant content in
the user role. The tool-return payload is appended to the user question
verbatim, preceded by a short framing sentence, so length, register (JSON-ish),
and position (immediately before the model's response) are matched.

Message shape:
    system, user(question + framing + verbatim tool-return payload)

Output: writes to `finalized_saliency_matched.jsonl` in the same directory as
the input, and also appends rows to `finalized_combined.jsonl` if `--append` is
passed. Idempotent — overwrites previous `saliency_matched_*` rows rather than
duplicating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FRAMING = (
    "\n\nFor context, here is some recent chat history I had with another "
    "assistant that might help you personalise your response:\n\n"
)


def build_saliency_matched_row(implicit_row: dict) -> dict:
    """Move an implicit tool payload into the user message for a matched control.

    The returned row keeps the scenario and tool schema, replaces the baked
    tool exchange with system and user messages, and uses a
    ``saliency_matched_*`` condition name.
    """
    messages = implicit_row["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    user_msg = next(m for m in messages if m["role"] == "user")
    tool_msg = next(m for m in messages if m["role"] == "tool")

    suffix = implicit_row["condition"].replace("implicit_", "")
    new_condition = f"saliency_matched_{suffix}"
    new_user_content = (user_msg["content"] or "") + FRAMING + (tool_msg["content"] or "")

    return {
        "id": f"{implicit_row['scenario_id']}__{new_condition}",
        "axis": implicit_row["axis"],
        "condition": new_condition,
        "context_type": "saliency_matched",
        "scenario_id": implicit_row["scenario_id"],
        "question": implicit_row["question"],
        "messages": [
            {"role": "system", "content": system_msg["content"], "tool_call": None},
            {"role": "user", "content": new_user_content, "tool_call": None},
        ],
        "tools": implicit_row["tools"],
        "sketch": implicit_row["sketch"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("face_eval_generator/data/finalized_combined.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("face_eval_generator/data/finalized_saliency_matched.jsonl"),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Also append the new rows into the input file (idempotent: strips existing saliency_matched_* first).",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    implicit_rows = [r for r in rows if r["context_type"] == "implicit"]
    new_rows = [build_saliency_matched_row(r) for r in implicit_rows]

    args.output.write_text("\n".join(json.dumps(r) for r in new_rows) + "\n")
    print(f"wrote {len(new_rows)} saliency-matched rows to {args.output}")

    if args.append:
        kept = [r for r in rows if r["context_type"] != "saliency_matched"]
        combined = kept + new_rows
        args.input.write_text("\n".join(json.dumps(r) for r in combined) + "\n")
        print(f"appended; {args.input} now has {len(combined)} rows")


if __name__ == "__main__":
    main()
