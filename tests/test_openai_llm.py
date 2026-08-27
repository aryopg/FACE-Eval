"""Unit tests for OpenAILLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.openai_llm import OpenAILLM, _extract_responses_output, _openai_to_responses_tools


def _make_output(text: str, reasoning: str | None = None):
    items = []
    if reasoning:
        r_item = MagicMock()
        r_item.type = "reasoning"
        summary = MagicMock()
        summary.text = reasoning
        r_item.summary = [summary]
        items.append(r_item)
    msg_item = MagicMock()
    msg_item.type = "message"
    part = MagicMock()
    part.text = text
    msg_item.content = [part]
    items.append(msg_item)
    return items


def test_openai_to_responses_tools_converts_format():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_memory",
                "description": "Retrieve memory",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    result = _openai_to_responses_tools(tools)
    assert result == [
        {
            "type": "function",
            "name": "get_memory",
            "description": "Retrieve memory",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_extract_text_only():
    assert _extract_responses_output(_make_output("hello")) == "hello"


def test_extract_with_reasoning():
    result = _extract_responses_output(_make_output("answer", reasoning="step by step"))
    assert isinstance(result, dict)
    assert result["reasoning"] == "step by step"
    assert result["content"] == "answer"


@patch("src.llm.openai_llm.openai.AsyncOpenAI")
@patch("src.llm.openai_llm.openai.OpenAI")
def test_chat_batch_concurrent_ordered(_mock_sync, mock_async_cls):
    mock_async = AsyncMock()
    mock_async_cls.return_value = mock_async
    fake_resp = MagicMock()
    fake_resp.output = _make_output("hi")
    mock_async.responses.create = AsyncMock(return_value=fake_resp)

    llm = OpenAILLM(model="gpt-4o-mini", api_key="fake", use_batch=False)
    results = llm.chat_batch([[{"role": "user", "content": "hey"}]] * 3)
    assert results == ["hi", "hi", "hi"]


def _batch(status: str, output_file_id: str | None = None, error_file_id: str | None = None):
    b = MagicMock()
    b.status = status
    b.output_file_id = output_file_id
    b.error_file_id = error_file_id
    return b


def test_poll_skips_batch_that_completed_with_no_output_file(caplog):
    """Every request rejected → status 'completed', error file only, no output file."""
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.client = MagicMock()
    llm.client.batches.retrieve.return_value = _batch("completed", error_file_id="file-err")
    llm.client.files.content.return_value = MagicMock(text='{"error": "Unsupported parameter: reasoning"}\nsecond')

    assert llm._poll_all_batches(["batch-1"]) == {}
    assert "Unsupported parameter: reasoning" in caplog.text
    assert "second" not in caplog.text


def test_poll_handles_no_output_and_no_error_file(caplog):
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.client = MagicMock()
    llm.client.batches.retrieve.return_value = _batch("completed")

    assert llm._poll_all_batches(["batch-1"]) == {}
    llm.client.files.content.assert_not_called()
    assert "every request in it failed" in caplog.text


def test_chat_delegates_to_chat_batch():
    llm = OpenAILLM.__new__(OpenAILLM)
    with patch.object(llm, "chat_batch", return_value=["hi"]) as mock_batch:
        result = llm.chat([{"role": "user", "content": "hey"}])
    assert result == "hi"
    mock_batch.assert_called_once()
