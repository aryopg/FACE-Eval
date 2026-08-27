"""Dataset loader for agentic sycophancy evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import datasets


def convert_flat_to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert flat tool_call messages to OpenAI-compatible format.

    The dataset stores tool calls in a flat format for Parquet compatibility:
        {"role": "assistant", "content": null, "tool_call": "func_name"}
        {"role": "tool", "content": "...", "tool_call": "func_name"}

    This converts them to OpenAI format that vLLM chat templates expect:
        {"role": "assistant", "tool_calls": [{"id": "call_0", ...}]}
        {"role": "tool", "tool_call_id": "call_0", "content": "..."}
    """
    converted = []
    call_counter = 0

    for msg in messages:
        tool_call = msg.get("tool_call")
        if not tool_call:
            # Normalise content: None → "" so APIs always receive a string.
            converted.append({"role": msg["role"], "content": msg.get("content") or ""})
            continue

        if msg["role"] == "assistant":
            call_id = f"call_{call_counter}"
            call_counter += 1
            converted.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_call, "arguments": "{}"},
                        }
                    ],
                }
            )
        elif msg["role"] == "tool":
            if call_counter == 0:
                # Tool result appeared before any assistant tool_call — the
                # dataset is malformed. Raise immediately rather than silently
                # emitting call_id="call_-1" which the API will reject.
                raise ValueError(
                    f"tool message at position {len(converted)} appeared before any "
                    "assistant tool_call message. Dataset row is malformed."
                )
            call_id = f"call_{call_counter - 1}"
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": msg.get("content") or "",
                }
            )

    return converted


class FaceEval:
    """Loader for the agentic sycophancy evaluation dataset.

    Loads multi-turn conversations with tool calls for evaluating
    sycophancy as CoT unfaithfulness.
    """

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        dataset_name: str = "edinburgh-dawg/face-eval",
        split: str = "train",
        axis: str | None = None,
        condition: str | None = None,
        use_auth_token: str | None = None,
    ):
        """Initialize dataset loader.

        Args:
            dataset_path: Local path to HF dataset directory.
            dataset_name: HuggingFace dataset identifier.
            split: Dataset split to load.
            axis: Filter by axis (e.g., "political", "ethics").
            condition: Filter by condition (e.g., "explicit_liberal", "no_context").
            use_auth_token: HuggingFace token. Falls back to HF_TOKEN env var.
        """
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.dataset_name = dataset_name

        if use_auth_token is None:
            use_auth_token = os.getenv("HF_TOKEN")

        if self.dataset_path:
            ds = datasets.load_from_disk(str(self.dataset_path))
        else:
            ds = datasets.load_dataset(
                dataset_name,
                split=split,
                token=use_auth_token,
            )

        if axis:
            ds = ds.filter(lambda x: x["axis"] == axis)
        if condition:
            ds = ds.filter(lambda x: x["condition"] == condition)

        self.dataset = ds
        # Cache axes at construction time — used in __repr__ and avoids a full
        # column scan on every print/log.
        self._axes: list[str] = sorted(set(self.dataset["axis"])) if len(ds) > 0 else []

    def get_messages_and_tools(self, idx: int) -> tuple[list[dict], list[dict]]:
        """Get OpenAI-formatted messages and parsed tools for a row.

        Returns:
            (messages, tools) tuple ready for vLLM/OpenAI API calls.
        """
        row = self.dataset[idx]
        messages = convert_flat_to_openai_messages(row["messages"])
        try:
            tools: list[dict] = json.loads(row["tools"]) if row["tools"] else []
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse tools JSON for row {idx} (id={row.get('id', '?')}): {e}") from e
        # user_turn conditions embed preference in the user message, not via tool call.
        # Expose no tools so the model cannot infer it is in an agentic/memory context.
        if row.get("context_type") in ("user_turn", "user_turn_structured", "user_turn_implicit"):
            tools = []
        return messages, tools

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset[idx]

    def __repr__(self) -> str:
        source = self.dataset_path or self.dataset_name
        return f"FaceEval(source={source}, axes={self._axes}, total={len(self)})"
