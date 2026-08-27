"""Tests for src/evaluation/parsing.py."""

from __future__ import annotations

from src.evaluation.parsing import ParsedOutput, extract_answer, parse_model_output

# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    def test_simple_answer(self):
        assert extract_answer("The answer is <answer>42</answer>.") == "42"

    def test_whitespace_inside_tags(self):
        assert extract_answer("<answer>  42  </answer>") == "42"

    def test_no_answer_tag(self):
        assert extract_answer("There is no tag here") is None

    def test_multiple_answer_tags_returns_last(self):
        text = "<answer>1</answer> then <answer>2</answer>"
        assert extract_answer(text) == "2"

    def test_empty_answer_tag(self):
        assert extract_answer("<answer></answer>") == ""

    def test_newlines_inside_tag(self):
        text = "<answer>\nline1\nline2\n</answer>"
        assert extract_answer(text) == "line1\nline2"

    def test_nested_content(self):
        text = "<answer><b>42</b></answer>"
        assert extract_answer(text) == "<b>42</b>"

    def test_case_insensitive(self):
        assert extract_answer("<ANSWER>42</ANSWER>") == "42"
        assert extract_answer("<Answer>42</Answer>") == "42"

    def test_unclosed_tag_returns_none(self):
        assert extract_answer("<answer>42") is None

    def test_answer_with_surrounding_text(self):
        text = "I think the result is <answer>7</answer> because of math."
        assert extract_answer(text) == "7"

    def test_empty_string(self):
        assert extract_answer("") is None


# ---------------------------------------------------------------------------
# parse_model_output
# ---------------------------------------------------------------------------


class TestParseModelOutput:
    def test_with_reasoning_from_model(self):
        result = parse_model_output(
            output="<answer>42</answer>",
            reasoning_from_model="I thought about it carefully.",
        )
        assert result.reasoning == "I thought about it carefully."
        assert result.raw_answer == "<answer>42</answer>"
        assert result.final_answer == "42"

    def test_with_reasoning_from_model_no_answer_tag(self):
        result = parse_model_output(
            output="The answer is 42.",
            reasoning_from_model="My reasoning.",
        )
        assert result.reasoning == "My reasoning."
        assert result.raw_answer == "The answer is 42."
        assert result.final_answer is None

    def test_with_think_tags(self):
        text = "<think>step 1\nstep 2</think>The answer is <answer>42</answer>"
        result = parse_model_output(text)
        assert "step 1" in result.reasoning
        assert "step 2" in result.reasoning
        assert result.raw_answer == "The answer is <answer>42</answer>"
        assert result.final_answer == "42"

    def test_without_think_tags(self):
        text = "Just a plain answer <answer>7</answer>"
        result = parse_model_output(text)
        assert result.reasoning == ""
        assert result.raw_answer == "Just a plain answer <answer>7</answer>"
        assert result.final_answer == "7"

    def test_empty_output(self):
        result = parse_model_output("")
        assert result.reasoning == ""
        assert result.raw_answer == ""
        assert result.final_answer is None

    def test_only_think_tags_no_content_after(self):
        text = "<think>some reasoning</think>"
        result = parse_model_output(text)
        assert "some reasoning" in result.reasoning
        assert result.raw_answer == ""
        assert result.final_answer is None

    def test_multiple_think_blocks(self):
        text = "<think>block 1</think>middle<think>block 2</think><answer>99</answer>"
        result = parse_model_output(text)
        # Split on last </think>, so reasoning includes everything before it
        assert "block 1" in result.reasoning
        assert "block 2" in result.reasoning
        assert result.final_answer == "99"

    def test_think_tag_case_insensitive(self):
        text = "<THINK>reasoning here</THINK><answer>5</answer>"
        result = parse_model_output(text)
        assert "reasoning here" in result.reasoning
        assert result.final_answer == "5"

    def test_reasoning_strips_think_tags(self):
        text = "<think>hello world</think>done"
        result = parse_model_output(text)
        # <think> and </think> are replaced with newlines, then stripped
        assert "<think>" not in result.reasoning
        assert "</think>" not in result.reasoning

    def test_answer_only_extracted_from_raw_answer(self):
        # Answer tag in reasoning (before last </think>) should NOT be extracted
        text = "<think><answer>wrong</answer></think><answer>right</answer>"
        result = parse_model_output(text)
        assert result.final_answer == "right"

    def test_no_opening_think_tag_but_has_closing(self):
        # When <think> is prefilled (not part of generated text), the output
        # starts with reasoning content directly and only has </think>.
        text = "step 1\nstep 2\n</think>\n<answer>42</answer>"
        result = parse_model_output(text)
        assert "step 1" in result.reasoning
        assert "step 2" in result.reasoning
        assert result.final_answer == "42"
        assert "<think>" not in result.reasoning
        assert "</think>" not in result.reasoning

    def test_empty_reasoning_from_model_treated_as_falsy(self):
        # Empty string is falsy, so it should fall through to the split logic
        text = "<think>reasoning</think><answer>10</answer>"
        result = parse_model_output(text, reasoning_from_model="")
        assert "reasoning" in result.reasoning
        assert result.final_answer == "10"

    def test_reasoning_collapses_excessive_newlines(self):
        text = "<think>\n\n\n\nline1\n\n\n\nline2\n\n\n\n</think>end"
        result = parse_model_output(text)
        assert "\n\n\n" not in result.reasoning

    def test_raw_answer_is_stripped(self):
        text = "<think>stuff</think>   spaced answer   "
        result = parse_model_output(text)
        assert result.raw_answer == "spaced answer"


# ---------------------------------------------------------------------------
# ParsedOutput dataclass
# ---------------------------------------------------------------------------


class TestParsedOutput:
    def test_fields(self):
        p = ParsedOutput(reasoning="r", raw_answer="ra", final_answer="fa")
        assert p.reasoning == "r"
        assert p.raw_answer == "ra"
        assert p.final_answer == "fa"

    def test_final_answer_none(self):
        p = ParsedOutput(reasoning="", raw_answer="", final_answer=None)
        assert p.final_answer is None
