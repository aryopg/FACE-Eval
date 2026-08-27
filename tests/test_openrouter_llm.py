"""Unit tests for OpenRouterLLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.openrouter import OpenRouterLLM, _extract_openrouter_response


def test_extract_text_only():
    msg = MagicMock()
    msg.reasoning = None
    msg.content = "hello"
    assert _extract_openrouter_response(msg) == "hello"


def test_extract_with_reasoning():
    msg = MagicMock()
    msg.reasoning = "chain of thought"
    msg.content = "conclusion"
    result = _extract_openrouter_response(msg)
    assert result == {"reasoning": "chain of thought", "content": "conclusion"}


@patch("src.llm.openrouter.openai.AsyncOpenAI")
@patch("src.llm.openrouter.openai.OpenAI")
def test_chat_batch_concurrent(_mock_sync, mock_async_cls):
    mock_async = AsyncMock()
    mock_async_cls.return_value = mock_async
    msg = MagicMock()
    msg.reasoning = None
    msg.content = "hi"
    fake_choice = MagicMock()
    fake_choice.message = msg
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]
    mock_async.chat.completions.create = AsyncMock(return_value=fake_resp)

    llm = OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", api_key="fake")
    results = llm.chat_batch([[{"role": "user", "content": "hey"}]] * 2)
    assert results == ["hi", "hi"]


@patch("src.llm.openrouter.openai.AsyncOpenAI")
@patch("src.llm.openrouter.openai.OpenAI")
def test_per_task_failure_returns_sentinel_not_raises(_mock_sync, mock_async_cls):
    mock_async = AsyncMock()
    mock_async_cls.return_value = mock_async
    msg = MagicMock()
    msg.reasoning = None
    msg.content = "ok"
    fake_choice = MagicMock()
    fake_choice.message = msg
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]
    # First request raises; second and third succeed.
    mock_async.chat.completions.create = AsyncMock(side_effect=[RuntimeError("upstream error"), fake_resp, fake_resp])

    llm = OpenRouterLLM(model="test-model", api_key="fake")
    results = llm.chat_batch([[{"role": "user", "content": f"q{i}"}] for i in range(3)])

    failed = [r for r in results if isinstance(r, dict) and r.get("harmony_parse_failed")]
    succeeded = [r for r in results if r == "ok"]
    assert len(failed) == 1
    assert len(succeeded) == 2


@patch("src.llm.openrouter.openai.AsyncOpenAI")
@patch("src.llm.openrouter.openai.OpenAI")
def test_on_result_callback_called_for_each(_mock_sync, mock_async_cls):
    mock_async = AsyncMock()
    mock_async_cls.return_value = mock_async
    msg = MagicMock()
    msg.reasoning = None
    msg.content = "hi"
    fake_choice = MagicMock()
    fake_choice.message = msg
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]
    mock_async.chat.completions.create = AsyncMock(return_value=fake_resp)

    calls: list[tuple[int, object]] = []
    llm = OpenRouterLLM(model="test-model", api_key="fake")
    llm.chat_batch(
        [[{"role": "user", "content": f"q{i}"}] for i in range(3)],
        on_result=lambda i, r: calls.append((i, r)),
    )

    assert sorted(i for i, _ in calls) == [0, 1, 2]
    assert all(r == "hi" for _, r in calls)
