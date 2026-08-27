"""Tests for src/data/face_eval.convert_flat_to_openai_messages."""

from __future__ import annotations

import pytest

from src.data.face_eval import convert_flat_to_openai_messages

# ---------------------------------------------------------------------------
# Basic messages (no tool calls)
# ---------------------------------------------------------------------------


class TestBasicMessages:
    def test_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = convert_flat_to_openai_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message(self):
        messages = [{"role": "assistant", "content": "Hi there"}]
        result = convert_flat_to_openai_messages(messages)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_system_message(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        result = convert_flat_to_openai_messages(messages)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Good, thanks!"},
        ]
        result = convert_flat_to_openai_messages(messages)
        assert len(result) == 4
        assert result[0] == {"role": "user", "content": "Hi"}
        assert result[1] == {"role": "assistant", "content": "Hello!"}
        assert result[2] == {"role": "user", "content": "How are you?"}
        assert result[3] == {"role": "assistant", "content": "Good, thanks!"}

    def test_empty_input(self):
        assert convert_flat_to_openai_messages([]) == []

    def test_missing_content_defaults_to_empty_string(self):
        messages = [{"role": "user"}]
        result = convert_flat_to_openai_messages(messages)
        assert result == [{"role": "user", "content": ""}]

    def test_none_content_defaults_to_empty_string(self):
        messages = [{"role": "user", "content": None}]
        result = convert_flat_to_openai_messages(messages)
        assert result == [{"role": "user", "content": ""}]


# ---------------------------------------------------------------------------
# Tool call messages
# ---------------------------------------------------------------------------


class TestToolCallMessages:
    def test_single_tool_call(self):
        messages = [
            {"role": "assistant", "content": None, "tool_call": "get_weather"},
        ]
        result = convert_flat_to_openai_messages(messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_0"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == "{}"

    def test_tool_call_has_no_content_key(self):
        messages = [
            {"role": "assistant", "content": None, "tool_call": "search"},
        ]
        result = convert_flat_to_openai_messages(messages)
        assert "content" not in result[0]

    def test_tool_call_ids_increment(self):
        messages = [
            {"role": "assistant", "content": None, "tool_call": "func_a"},
            {"role": "tool", "content": "result_a", "tool_call": "func_a"},
            {"role": "assistant", "content": None, "tool_call": "func_b"},
            {"role": "tool", "content": "result_b", "tool_call": "func_b"},
        ]
        result = convert_flat_to_openai_messages(messages)
        assert result[0]["tool_calls"][0]["id"] == "call_0"
        assert result[2]["tool_calls"][0]["id"] == "call_1"


# ---------------------------------------------------------------------------
# Tool response messages
# ---------------------------------------------------------------------------


class TestToolResponseMessages:
    def test_single_tool_response(self):
        messages = [
            {"role": "assistant", "content": None, "tool_call": "get_weather"},
            {"role": "tool", "content": '{"temp": 72}', "tool_call": "get_weather"},
        ]
        result = convert_flat_to_openai_messages(messages)
        assert len(result) == 2
        resp = result[1]
        assert resp["role"] == "tool"
        assert resp["tool_call_id"] == "call_0"
        assert resp["content"] == '{"temp": 72}'

    def test_tool_response_references_previous_call_id(self):
        """Tool response uses call_counter - 1, linking to the most recent call."""
        messages = [
            {"role": "assistant", "content": None, "tool_call": "alpha"},
            {"role": "tool", "content": "res_alpha", "tool_call": "alpha"},
            {"role": "assistant", "content": None, "tool_call": "beta"},
            {"role": "tool", "content": "res_beta", "tool_call": "beta"},
        ]
        result = convert_flat_to_openai_messages(messages)
        # First tool response references call_0
        assert result[1]["tool_call_id"] == "call_0"
        # Second tool response references call_1
        assert result[3]["tool_call_id"] == "call_1"

    def test_tool_response_missing_content(self):
        messages = [
            {"role": "assistant", "content": None, "tool_call": "noop"},
            {"role": "tool", "tool_call": "noop"},
        ]
        result = convert_flat_to_openai_messages(messages)
        assert result[1]["content"] == ""


# ---------------------------------------------------------------------------
# Full conversation flows
# ---------------------------------------------------------------------------


class TestFullConversation:
    def test_user_then_tool_call_then_response_then_assistant(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": None, "tool_call": "get_weather"},
            {"role": "tool", "content": '{"temp": 72}', "tool_call": "get_weather"},
            {"role": "assistant", "content": "It's 72 degrees."},
        ]
        result = convert_flat_to_openai_messages(messages)
        assert len(result) == 4
        assert result[0] == {"role": "user", "content": "What's the weather?"}
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_0"
        assert result[3] == {"role": "assistant", "content": "It's 72 degrees."}

    def test_multiple_tool_calls_in_sequence(self):
        messages = [
            {"role": "user", "content": "Compare weather in NYC and LA"},
            {"role": "assistant", "content": None, "tool_call": "get_weather_nyc"},
            {"role": "tool", "content": "cold", "tool_call": "get_weather_nyc"},
            {"role": "assistant", "content": None, "tool_call": "get_weather_la"},
            {"role": "tool", "content": "warm", "tool_call": "get_weather_la"},
            {"role": "assistant", "content": "NYC is cold, LA is warm."},
        ]
        result = convert_flat_to_openai_messages(messages)
        assert len(result) == 6
        # First tool call
        assert result[1]["tool_calls"][0]["function"]["name"] == "get_weather_nyc"
        assert result[1]["tool_calls"][0]["id"] == "call_0"
        assert result[2]["tool_call_id"] == "call_0"
        # Second tool call
        assert result[3]["tool_calls"][0]["function"]["name"] == "get_weather_la"
        assert result[3]["tool_calls"][0]["id"] == "call_1"
        assert result[4]["tool_call_id"] == "call_1"

    def test_messages_with_extra_metadata_fields(self):
        """Extra fields in the input dicts should not cause errors."""
        messages = [
            {"role": "user", "content": "Hi", "timestamp": "2024-01-01"},
            {"role": "assistant", "content": "Hello", "model": "gpt-4"},
        ]
        result = convert_flat_to_openai_messages(messages)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hi"}
        assert result[1] == {"role": "assistant", "content": "Hello"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_tool_call_field_is_none(self):
        """tool_call=None should be treated as no tool call (falsy)."""
        messages = [{"role": "assistant", "content": "plain", "tool_call": None}]
        result = convert_flat_to_openai_messages(messages)
        assert result == [{"role": "assistant", "content": "plain"}]

    def test_tool_call_field_is_empty_string(self):
        """tool_call='' is falsy, should be treated as normal message."""
        messages = [{"role": "assistant", "content": "plain", "tool_call": ""}]
        result = convert_flat_to_openai_messages(messages)
        assert result == [{"role": "assistant", "content": "plain"}]

    @pytest.mark.parametrize(
        "role,content",
        [
            ("user", "question"),
            ("assistant", "answer"),
            ("system", "instruction"),
        ],
    )
    def test_roles_preserved_for_plain_messages(self, role, content):
        messages = [{"role": role, "content": content}]
        result = convert_flat_to_openai_messages(messages)
        assert result[0]["role"] == role
        assert result[0]["content"] == content


# ---------------------------------------------------------------------------
# FaceEval class: loading, filtering, accessors
# ---------------------------------------------------------------------------


def _write_local_dataset(tmp_path, rows):
    """Materialize a HuggingFace-loadable dataset to disk for tests."""
    import datasets

    ds = datasets.Dataset.from_list(rows)
    save_dir = tmp_path / "ds"
    ds.save_to_disk(str(save_dir))
    return save_dir


def _sample_rows():
    import json as _json

    tools = _json.dumps([{"type": "function", "function": {"name": "get_profile", "parameters": {}}}])
    return [
        {
            "id": "political-0",
            "axis": "political",
            "condition": "explicit_liberal",
            "context_type": "explicit",
            "scenario_id": "pol-0",
            "question": "policy?",
            "messages": [{"role": "user", "content": "policy?"}],
            "tools": tools,
        },
        {
            "id": "political-1",
            "axis": "political",
            "condition": "no_context",
            "context_type": "none",
            "scenario_id": "pol-1",
            "question": "policy?",
            "messages": [{"role": "user", "content": "policy?"}],
            "tools": tools,
        },
        {
            "id": "ethics-0",
            "axis": "ethics",
            "condition": "explicit_utilitarian",
            "context_type": "explicit",
            "scenario_id": "eth-0",
            "question": "trolley?",
            "messages": [{"role": "user", "content": "trolley?"}],
            "tools": tools,
        },
    ]


class TestFaceEvalClass:
    def test_loads_full_local_dataset(self, tmp_path):
        from src.data.face_eval import FaceEval

        path = _write_local_dataset(tmp_path, _sample_rows())
        ds = FaceEval(dataset_path=path)
        assert len(ds) == 3

    def test_filters_by_axis(self, tmp_path):
        from src.data.face_eval import FaceEval

        path = _write_local_dataset(tmp_path, _sample_rows())
        ds = FaceEval(dataset_path=path, axis="political")
        assert len(ds) == 2
        assert all(row["axis"] == "political" for row in ds.dataset)

    def test_filters_by_condition(self, tmp_path):
        from src.data.face_eval import FaceEval

        path = _write_local_dataset(tmp_path, _sample_rows())
        ds = FaceEval(dataset_path=path, condition="no_context")
        assert len(ds) == 1
        assert ds[0]["condition"] == "no_context"

    def test_filters_by_axis_and_condition_combined(self, tmp_path):
        from src.data.face_eval import FaceEval

        path = _write_local_dataset(tmp_path, _sample_rows())
        ds = FaceEval(dataset_path=path, axis="political", condition="explicit_liberal")
        assert len(ds) == 1
        assert ds[0]["id"] == "political-0"

    def test_get_messages_and_tools_parses_tools_json(self, tmp_path):
        from src.data.face_eval import FaceEval

        path = _write_local_dataset(tmp_path, _sample_rows())
        ds = FaceEval(dataset_path=path)
        messages, tools = ds.get_messages_and_tools(0)
        assert isinstance(messages, list) and messages[0]["role"] == "user"
        assert isinstance(tools, list)
        assert tools[0]["function"]["name"] == "get_profile"

    def test_repr_includes_source_and_axes(self, tmp_path):
        from src.data.face_eval import FaceEval

        path = _write_local_dataset(tmp_path, _sample_rows())
        ds = FaceEval(dataset_path=path)
        r = repr(ds)
        assert "total=3" in r
        assert "political" in r and "ethics" in r

    def test_getitem_returns_row_dict(self, tmp_path):
        from src.data.face_eval import FaceEval

        path = _write_local_dataset(tmp_path, _sample_rows())
        ds = FaceEval(dataset_path=path)
        row = ds[0]
        assert row["id"] == "political-0"
        assert row["axis"] == "political"

    def test_filter_with_no_matches_yields_empty_dataset(self, tmp_path):
        from src.data.face_eval import FaceEval

        path = _write_local_dataset(tmp_path, _sample_rows())
        ds = FaceEval(dataset_path=path, axis="nonexistent_axis")
        assert len(ds) == 0
        # __repr__ must not crash when there are no axes to collect
        assert "total=0" in repr(ds)
