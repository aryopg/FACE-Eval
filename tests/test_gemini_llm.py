"""Unit tests for GeminiLLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.gemini import GeminiLLM, _extract_gemini_output


def _make_gemini_response(text: str, thought: str | None = None):
    parts = []
    if thought:
        p = MagicMock()
        p.thought = True
        p.text = thought
        parts.append(p)
    p2 = MagicMock()
    p2.thought = False
    p2.text = text
    parts.append(p2)
    candidate = MagicMock()
    candidate.content.parts = parts
    resp = MagicMock()
    resp.candidates = [candidate]
    return resp


def test_extract_text_only():
    assert _extract_gemini_output(_make_gemini_response("hello")) == "hello"


def test_extract_with_thought():
    result = _extract_gemini_output(_make_gemini_response("answer", thought="reasoning here"))
    assert isinstance(result, dict)
    assert result["reasoning"] == "reasoning here"
    assert result["content"] == "answer"


@patch("src.llm.gemini.genai.Client")
def test_chat_batch_concurrent(mock_client_cls):
    mock_aio = MagicMock()
    mock_aio.models.generate_content = AsyncMock(return_value=_make_gemini_response("hi"))
    mock_client_cls.return_value.aio = mock_aio

    llm = GeminiLLM(model="gemini-2.0-flash", api_key="fake")
    results = llm.chat_batch([[{"role": "user", "content": "hey"}]] * 2)
    assert results == ["hi", "hi"]


@patch("src.llm.gemini.genai.Client")
def test_per_task_failure_returns_sentinel_not_raises(mock_client_cls):
    mock_aio = MagicMock()
    mock_client_cls.return_value.aio = mock_aio
    # First request raises; second and third succeed.
    mock_aio.models.generate_content = AsyncMock(
        side_effect=[RuntimeError("quota error"), _make_gemini_response("ok"), _make_gemini_response("ok")]
    )

    llm = GeminiLLM(model="gemini-2.0-flash", api_key="fake")
    results = llm.chat_batch([[{"role": "user", "content": f"q{i}"}] for i in range(3)])

    failed = [r for r in results if isinstance(r, dict) and r.get("harmony_parse_failed")]
    succeeded = [r for r in results if r == "ok"]
    assert len(failed) == 1
    assert len(succeeded) == 2


@patch("src.llm.gemini.genai.Client")
def test_on_result_callback_called_for_each(mock_client_cls):
    mock_aio = MagicMock()
    mock_client_cls.return_value.aio = mock_aio
    mock_aio.models.generate_content = AsyncMock(return_value=_make_gemini_response("hi"))

    calls: list[tuple[int, object]] = []
    llm = GeminiLLM(model="gemini-2.0-flash", api_key="fake")
    llm.chat_batch(
        [[{"role": "user", "content": f"q{i}"}] for i in range(3)],
        on_result=lambda i, r: calls.append((i, r)),
    )

    assert sorted(i for i, _ in calls) == [0, 1, 2]
    assert all(r == "hi" for _, r in calls)
