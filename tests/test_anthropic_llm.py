"""Unit tests for AnthropicLLM."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.llm.anthropic import (
    _THINKING_BUDGET,
    AnthropicLLM,
    _cached_system,
    _cached_tools,
    _openai_to_anthropic_messages,
    _openai_to_anthropic_tools,
)


def _mock_blocks(text: str, thinking: str | None = None):
    blocks = []
    if thinking:
        b = MagicMock()
        b.type = "thinking"
        b.thinking = thinking
        blocks.append(b)
    b2 = MagicMock()
    b2.type = "text"
    b2.text = text
    blocks.append(b2)
    return blocks


def test_extract_response_text_only():
    llm = AnthropicLLM.__new__(AnthropicLLM)
    result = llm._extract_response(_mock_blocks("hello"))
    assert result == "hello"


def test_extract_response_with_thinking():
    llm = AnthropicLLM.__new__(AnthropicLLM)
    result = llm._extract_response(_mock_blocks("answer", thinking="reasoning here"))
    assert isinstance(result, dict)
    assert result["reasoning"] == "reasoning here"
    assert result["content"] == "answer"


def test_thinking_param_none_when_no_effort():
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.include_reasoning = True
    llm.reasoning_effort = None
    assert llm._thinking_param() is None


def test_thinking_param_returns_budget():
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.include_reasoning = True
    llm.reasoning_effort = "medium"
    result = llm._thinking_param()
    assert result == {"type": "enabled", "budget_tokens": _THINKING_BUDGET["medium"]}


def test_openai_to_anthropic_tools_converts_format():
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_user_memory",
                "description": "Get memory",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    result = _openai_to_anthropic_tools(openai_tools)
    assert result == [
        {
            "name": "get_user_memory",
            "description": "Get memory",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_openai_to_anthropic_messages_extracts_system():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    system, converted = _openai_to_anthropic_messages(messages)
    assert system == "You are helpful."
    assert converted == [{"role": "user", "content": "Hello"}]


def test_openai_to_anthropic_messages_converts_tool_calls():
    messages = [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_memory", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "memory result"},
    ]
    system, converted = _openai_to_anthropic_messages(messages)
    assert system is None
    assert converted[0] == {"role": "user", "content": "Question"}
    assert converted[1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "call_0", "name": "get_memory", "input": {}}],
    }
    assert converted[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_0", "content": "memory result"}],
    }


def test_cached_system_wraps_as_content_block():
    result = _cached_system("You are helpful.")
    assert result == [{"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}]


def test_cached_tools_marks_last_tool():
    tools = [
        {"name": "tool_a", "description": "a", "input_schema": {}},
        {"name": "tool_b", "description": "b", "input_schema": {}},
    ]
    result = _cached_tools(tools)
    assert result[0] == tools[0]
    assert result[1] == {**tools[1], "cache_control": {"type": "ephemeral"}}


def test_cached_tools_empty_passthrough():
    assert _cached_tools([]) == []


def test_chat_batch_sequential_no_batch():
    msg_block = MagicMock()
    msg_block.type = "text"
    msg_block.text = "hi"

    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model = "claude-haiku-4-5-20251001"
    llm.use_batch = False
    llm.include_reasoning = False
    llm.reasoning_effort = None
    llm._async_client = None
    llm.client = MagicMock()
    llm.client.messages.create.return_value.content = [msg_block]

    results = llm.chat_batch([[{"role": "user", "content": "hey"}]] * 2)
    assert results == ["hi", "hi"]
    assert llm.client.messages.create.call_count == 2
