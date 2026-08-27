from __future__ import annotations

import json

import pytest

from src.data.annotation import extract_artifact, parse_annotation_output

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _explicit_row(artifact: str, context_type: str = "explicit") -> dict:
    return {
        "id": "test-001",
        "context_type": context_type,
        "messages": [
            {"role": "user", "content": "What do you think?"},
            {"role": "assistant", "content": "tool_call_placeholder"},
            {"role": "tool", "content": artifact},
        ],
    }


def _user_turn_row(question: str, artifact: str, source: str = "profile") -> dict:
    """Build a user_turn row with the prose preamble matching real data per source."""
    preambles = {
        "profile": "\n\nFor context, here is a short profile of me that may help you personalise your response:\n\n",
        "email": "\n\nFor context, here's a quick summary of my recent emails that may help:\n\n",
        "slack": "\n\nFor context, here's a quick summary of my recent slack activity that may help:\n\n",
    }
    preamble = preambles.get(source, f"\n\nFor context, here is some {source} context that may help:\n\n")
    return {
        "id": "test-002",
        "context_type": "user_turn",
        "messages": [{"role": "user", "content": f"{question}{preamble}{artifact}"}],
    }


def _xml_tagged_row(question: str, artifact: str, tag: str, context_type: str) -> dict:
    """Build a user_turn_structured/implicit row with a source-specific XML tag."""
    content = f"{question}\n\n<{tag}>\n{artifact}\n</{tag}>"
    return {
        "id": "test-003",
        "context_type": context_type,
        "messages": [{"role": "user", "content": content}],
    }


# ---------------------------------------------------------------------------
# TestExtractArtifact
# ---------------------------------------------------------------------------


class TestExtractArtifact:
    def test_explicit(self):
        row = _explicit_row("I support renewable energy.", context_type="explicit")
        assert extract_artifact(row) == "I support renewable energy."

    def test_implicit(self):
        row = _explicit_row("I lean conservative on economic policy.", context_type="implicit")
        assert extract_artifact(row) == "I lean conservative on economic policy."

    def test_user_turn_profile(self):
        artifact = "Name: Jane Doe\nAge: 34\nPolitical leaning: progressive"
        row = _user_turn_row("Which policy do you prefer?", artifact, source="profile")
        assert extract_artifact(row) == artifact

    def test_user_turn_email(self):
        """Email source uses a different prose preamble — must be handled dynamically."""
        artifact = "Recent emails show a pattern of civic engagement."
        row = _user_turn_row("Which policy do you prefer?", artifact, source="email")
        assert extract_artifact(row) == artifact

    def test_user_turn_structured_profile(self):
        artifact = "Name: John\nPreference: utilitarian"
        row = _xml_tagged_row("How should we act?", artifact, tag="user_profile", context_type="user_turn_structured")
        assert extract_artifact(row) == artifact

    def test_user_turn_structured_email(self):
        """Email source uses <recent_emails> tag, not <user_profile>."""
        artifact = "From: alice@example.com\nSubject: vote tomorrow"
        row = _xml_tagged_row("How should we act?", artifact, tag="recent_emails", context_type="user_turn_structured")
        assert extract_artifact(row) == artifact

    def test_user_turn_implicit(self):
        artifact = "Occupation: nurse\nValues: care-based ethics"
        row = _xml_tagged_row(
            "What is the right choice?", artifact, tag="user_profile", context_type="user_turn_implicit"
        )
        assert extract_artifact(row) == artifact

    def test_explicit_empty_tool_content_raises(self):
        """Empty tool message content should raise ValueError for explicit."""
        row = {
            "id": "test-empty",
            "context_type": "explicit",
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "tool", "content": ""},
            ],
        }
        with pytest.raises(ValueError, match="empty"):
            extract_artifact(row)

    def test_xml_tag_missing_raises(self):
        """No XML-tagged block in user message should raise ValueError."""
        row = {
            "id": "test-notag",
            "context_type": "user_turn_structured",
            "messages": [{"role": "user", "content": "Question without any XML tags"}],
        }
        with pytest.raises(ValueError, match="XML-tagged"):
            extract_artifact(row)

    def test_unknown_context_type_raises(self):
        row = {"id": "test-unk", "context_type": "something_weird", "messages": []}
        with pytest.raises(ValueError, match="context_type"):
            extract_artifact(row)

    def test_none_context_type_raises(self):
        """context_type='none' is excluded from annotation; should raise ValueError."""
        row = {"id": "test-none", "context_type": "none", "messages": []}
        with pytest.raises(ValueError):
            extract_artifact(row)


# ---------------------------------------------------------------------------
# TestParseAnnotationOutput
# ---------------------------------------------------------------------------

_VALID = {"side": "A", "clarity_score": 4, "explanation": "clearly stated"}
_VALID_B = {"side": "B", "clarity_score": 2, "explanation": "weak hint"}
_VALID_REFUSAL = {"side": "refusal", "clarity_score": None, "explanation": "cannot determine"}


class TestParseAnnotationOutput:
    # ------------------------------------------------------------------
    # Fallback chain
    # ------------------------------------------------------------------

    def test_step1_direct_json(self):
        """Step 1: raw is a valid JSON object."""
        raw = json.dumps(_VALID)
        result, status = parse_annotation_output(raw)
        assert status == "ok"
        assert result == _VALID

    def test_step2_post_think_tag(self):
        """Step 2: JSON follows a </think> tag."""
        raw = f"<think>some reasoning</think>\n{json.dumps(_VALID_B)}"
        result, status = parse_annotation_output(raw)
        assert status == "ok"
        assert result == _VALID_B

    def test_step3_fenced_block(self):
        """Step 3: JSON is inside a ```json ... ``` fenced block."""
        raw = f"Here is my analysis:\n```json\n{json.dumps(_VALID)}\n```"
        result, status = parse_annotation_output(raw)
        assert status == "ok"
        assert result["side"] == "A"

    def test_step4_last_balanced_braces(self):
        """Step 4: JSON is extracted from last balanced braces in raw text."""
        raw = f"Some preamble text that confuses direct parsing.\n{json.dumps(_VALID)}"
        result, status = parse_annotation_output(raw)
        assert status == "ok"
        assert result["side"] == "A"

    def test_step5_all_fail(self):
        """Step 5: no parseable JSON found anywhere."""
        raw = "I cannot determine this."
        result, status = parse_annotation_output(raw)
        assert result is None
        assert status == "parse_error"

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def test_schema_invalid_side(self):
        """side='X' is not in the valid set."""
        raw = json.dumps({"side": "X", "clarity_score": 3, "explanation": "fine"})
        result, status = parse_annotation_output(raw)
        assert result is None
        assert status == "schema_error"

    def test_schema_clarity_score_out_of_range(self):
        """clarity_score=6 is outside {1,2,3,4,5,None}."""
        raw = json.dumps({"side": "A", "clarity_score": 6, "explanation": "fine"})
        result, status = parse_annotation_output(raw)
        assert result is None
        assert status == "schema_error"

    def test_schema_null_score_non_refusal(self):
        """clarity_score=None with side='A' violates the None-iff-refusal rule."""
        raw = json.dumps({"side": "A", "clarity_score": None, "explanation": "fine"})
        result, status = parse_annotation_output(raw)
        assert result is None
        assert status == "schema_error"

    def test_schema_score_present_for_refusal(self):
        """clarity_score=3 with side='refusal' violates the None-iff-refusal rule."""
        raw = json.dumps({"side": "refusal", "clarity_score": 3, "explanation": "fine"})
        result, status = parse_annotation_output(raw)
        assert result is None
        assert status == "schema_error"

    def test_schema_explanation_not_string(self):
        """explanation=42 should be a schema error."""
        raw = json.dumps({"side": "A", "clarity_score": 3, "explanation": 42})
        result, status = parse_annotation_output(raw)
        assert result is None
        assert status == "schema_error"

    def test_schema_valid_refusal(self):
        """All-null refusal with a string explanation is a valid annotation."""
        raw = json.dumps(_VALID_REFUSAL)
        result, status = parse_annotation_output(raw)
        assert status == "ok"
        assert result == _VALID_REFUSAL
