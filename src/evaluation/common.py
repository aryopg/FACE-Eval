"""Helpers shared by the evaluation modules.

Helpers that the evaluation modules and their scripts share. They live here, and not
inside one of those modules, so that no `src.evaluation` module has to import another
one to reach them. This module imports only `src.data`.
"""

from __future__ import annotations

from src.data.face_eval import FaceEval, convert_flat_to_openai_messages


def preferred_side(condition: str) -> str:
    """Preference label carried by a condition string, or "none" for the control.

    A condition is `{context_type prefix}_{source channel}_{side}`, so the side is the
    last token. This is the single definition; the judge prompts, the trace-form coder
    and the inspect-ai task all use it.

    An earlier version stripped only `explicit_`/`implicit_`, which left the source token
    on tool rows and the whole string on user rows. Every verdict written before
    2026-08-17 used that version, and those verdicts were deliberately not regenerated:
    the side is the last token either way, so the judges resolved it correctly. If you do
    ever re-judge, do not pass --resume, or one file ends up holding both prompt versions
    with nothing recording which row came from which.
    """
    if condition == "no_context":
        return "none"
    return condition.rsplit("_", 1)[-1]


def load_dataset_index(dataset: FaceEval) -> dict[str, dict]:
    """Index the dataset rows by their `id`, so a row can be found by id."""
    return {dataset[i]["id"]: dataset[i] for i in range(len(dataset))}


def render_action_context(dataset_row: dict) -> str:
    """Render the full observable action context: system prompt + user message + tool call/return.

    Includes the system message (unlike the old trajectory renderer) because a realistic deployment
    monitor has access to the system prompt. Returns "" for rows with no messages.
    """
    msgs = convert_flat_to_openai_messages(dataset_row["messages"])
    parts: list[str] = []
    for msg in msgs:
        role = msg["role"]
        if role == "system":
            parts.append(f"[System prompt]\n{msg['content']}")
        elif role == "user":
            parts.append(f"[User message]\n{msg['content']}")
        elif role == "assistant" and "tool_calls" in msg:
            tc = msg["tool_calls"][0]
            fn_name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            parts.append(f"[Assistant tool call]\n{fn_name}({args})")
        elif role == "tool":
            parts.append(f"[Tool return]\n{msg['content']}")
    return "\n\n".join(parts)
