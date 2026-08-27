"""Tests for Gemma4Client output parsing."""

from __future__ import annotations

from src.llm.gemma4 import _inject_think_token, _parse_gemma_output


class TestParseGemma4Output:
    """Unit tests for the Gemma4 channel-tag parser."""

    def test_full_channel_tags_splits_reasoning_and_content(self):
        text = "<|channel>thought\nI should reason carefully.\n<channel|>Here is my answer."
        result = _parse_gemma_output(text)
        assert result["reasoning"] == "I should reason carefully."
        assert result["content"] == "Here is my answer."

    def test_multiline_reasoning_preserved(self):
        text = "<|channel>thought\nStep 1.\nStep 2.\nStep 3.\n<channel|>Final answer."
        result = _parse_gemma_output(text)
        assert "Step 1." in result["reasoning"]
        assert "Step 2." in result["reasoning"]
        assert result["content"] == "Final answer."

    def test_strips_eos_from_content(self):
        text = "<|channel>thought\nReasoning.\n<channel|>The answer.<eos>"
        result = _parse_gemma_output(text)
        assert result["content"] == "The answer."
        assert "<eos>" not in result["content"]

    def test_strips_eos_when_no_channel_tags(self):
        text = "Plain answer with no thinking.<eos>"
        result = _parse_gemma_output(text)
        assert result["content"] == "Plain answer with no thinking."
        assert result["reasoning"] == ""

    def test_no_channel_tags_returns_empty_reasoning(self):
        text = "This model output has no thinking block."
        result = _parse_gemma_output(text)
        assert result["reasoning"] == ""
        assert result["content"] == "This model output has no thinking block."

    def test_empty_thinking_block(self):
        text = "<|channel>thought\n<channel|>Just the answer."
        result = _parse_gemma_output(text)
        assert result["reasoning"] == ""
        assert result["content"] == "Just the answer."

    def test_channel_start_prefix_stripped_from_reasoning(self):
        text = "<|channel>thought\nSome reasoning.<channel|>Answer."
        result = _parse_gemma_output(text)
        assert not result["reasoning"].startswith("<|channel>")
        assert "Some reasoning." in result["reasoning"]

    def test_content_with_answer_tags_preserved(self):
        text = "<|channel>thought\nReasoning.\n<channel|><answer>42</answer>"
        result = _parse_gemma_output(text)
        assert result["content"] == "<answer>42</answer>"

    def test_empty_input(self):
        result = _parse_gemma_output("")
        assert result["reasoning"] == ""
        assert result["content"] == ""


class TestInjectThinkToken:
    """Unit tests for <|think|> system-prompt injection."""

    def test_prepends_to_system_message(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        result = _inject_think_token(messages)
        assert result[0]["content"].startswith("<|think|>")
        assert "You are helpful." in result[0]["content"]

    def test_does_not_double_inject(self):
        messages = [{"role": "system", "content": "<|think|>Already has it."}]
        result = _inject_think_token(messages)
        assert result[0]["content"].count("<|think|>") == 1

    def test_adds_system_message_when_none_present(self):
        messages = [{"role": "user", "content": "Hello."}]
        result = _inject_think_token(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "<|think|>"
        assert result[1]["role"] == "user"

    def test_non_system_messages_unchanged(self):
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "A."},
        ]
        result = _inject_think_token(messages)
        assert result[1] == {"role": "user", "content": "Q?"}
        assert result[2] == {"role": "assistant", "content": "A."}

    def test_returns_new_list_does_not_mutate(self):
        original = [{"role": "system", "content": "Original."}]
        result = _inject_think_token(original)
        assert original[0]["content"] == "Original."
        assert result is not original
