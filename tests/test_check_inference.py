"""Tests for inference output checking logic."""

from __future__ import annotations

import pytest

from src.results.check import MIN_LEGIBLE_RATE, RunCheckResult, check_run, summarize_by_model

REQUIRED_FIELDS = ["id", "question", "condition", "axis"]


def _make_record(**overrides) -> dict:
    base = {
        "id": "political_001__explicit_liberal",
        "axis": "political",
        "condition": "explicit_liberal",
        "question": "What do you think about X?",
        "reasoning": "I think this is a fair question...",
        "raw_answer": "<answer>A</answer>",
    }
    base.update(overrides)
    return base


def _make_records(n: int, **overrides) -> list[dict]:
    return [_make_record(id=f"item_{i}", **overrides) for i in range(n)]


# ---------------------------------------------------------------------------
# Completeness / legibility
# ---------------------------------------------------------------------------


def test_complete_run_passes():
    records = _make_records(500)
    result = check_run(records)
    assert result.count == 500
    assert result.legible_count == 500
    assert result.legible_rate == pytest.approx(1.0)
    assert result.empty_reasoning == 0
    assert result.empty_raw_answer == 0
    assert result.missing_fields == 0
    assert result.ok is True


def test_short_run_at_legible_threshold_passes():
    # 450 / 500 = 90% — at the threshold, still ok
    records = _make_records(450)
    result = check_run(records, expected_count=500)
    assert result.count == 450
    assert result.legible_rate == pytest.approx(0.90)
    assert result.ok is True


def test_short_run_below_legible_threshold_fails():
    # 449 / 500 = 89.8% — below threshold
    records = _make_records(449)
    result = check_run(records, expected_count=500)
    assert result.legible_rate < MIN_LEGIBLE_RATE
    assert result.ok is False


def test_custom_expected_count():
    records = _make_records(10)
    result = check_run(records, expected_count=10)
    assert result.ok is True


# ---------------------------------------------------------------------------
# Empty reasoning — counts toward illegibility
# ---------------------------------------------------------------------------


def test_empty_reasoning_below_threshold_passes():
    # 2 illegible / 500 expected = 99.6% legible — above threshold
    records = _make_records(500)
    records[0]["reasoning"] = ""
    records[5]["reasoning"] = ""
    result = check_run(records)
    assert result.empty_reasoning == 2
    assert result.legible_count == 498
    assert result.ok is True


def test_none_reasoning_below_threshold_passes():
    records = _make_records(500)
    records[0]["reasoning"] = None
    result = check_run(records)
    assert result.empty_reasoning == 1
    assert result.legible_count == 499
    assert result.ok is True


def test_whitespace_reasoning_below_threshold_passes():
    records = _make_records(500)
    records[0]["reasoning"] = "   \n  "
    result = check_run(records)
    assert result.empty_reasoning == 1
    assert result.ok is True


def test_missing_reasoning_key_below_threshold_passes():
    records = _make_records(500)
    del records[0]["reasoning"]
    result = check_run(records)
    assert result.empty_reasoning == 1
    assert result.ok is True


def test_illegible_above_threshold_fails():
    # 51 illegible / 500 = 89.8% legible — below 90% threshold
    records = _make_records(500)
    for i in range(51):
        records[i]["reasoning"] = ""
    result = check_run(records)
    assert result.empty_reasoning == 51
    assert result.legible_count == 449
    assert result.legible_rate < MIN_LEGIBLE_RATE
    assert result.ok is False


def test_reasoning_missing_by_condition():
    records = _make_records(500)
    records[0] = _make_record(id="item_0", reasoning="", condition="no_context")
    records[1] = _make_record(id="item_1", reasoning="", condition="no_context")
    records[2] = _make_record(id="item_2", reasoning="", condition="user_turn_liberal")
    result = check_run(records)
    assert result.reasoning_missing_by_condition == {"no_context": 2, "user_turn_liberal": 1}
    assert result.empty_reasoning == 3


# ---------------------------------------------------------------------------
# Empty raw_answer — counts toward illegibility but no longer a hard fail
# ---------------------------------------------------------------------------


def test_few_empty_raw_answers_still_pass():
    records = _make_records(500)
    records[3]["raw_answer"] = ""
    result = check_run(records)
    assert result.empty_raw_answer == 1
    assert result.legible_count == 499
    assert result.ok is True


def test_none_raw_answer_counts_as_illegible():
    records = _make_records(500)
    records[3]["raw_answer"] = None
    result = check_run(records)
    assert result.empty_raw_answer == 1
    assert result.legible_count == 499
    assert result.ok is True


def test_missing_raw_answer_key_counts_as_illegible():
    records = _make_records(500)
    del records[3]["raw_answer"]
    result = check_run(records)
    assert result.empty_raw_answer == 1
    assert result.legible_count == 499
    assert result.ok is True


def test_many_empty_raw_answers_fail():
    records = _make_records(500)
    for i in range(60):
        records[i]["raw_answer"] = ""
    result = check_run(records)
    assert result.empty_raw_answer == 60
    assert result.legible_count == 440
    assert result.ok is False


def test_reasoning_and_raw_answer_missing_on_same_record_counts_once():
    records = _make_records(500)
    records[0]["reasoning"] = ""
    records[0]["raw_answer"] = ""
    result = check_run(records)
    assert result.empty_reasoning == 1
    assert result.empty_raw_answer == 1
    assert result.legible_count == 499  # only one illegible record


# ---------------------------------------------------------------------------
# Missing required fields — still a hard fail
# ---------------------------------------------------------------------------


def test_missing_id_flagged():
    records = _make_records(500)
    del records[0]["id"]
    result = check_run(records)
    assert result.missing_fields == 1
    assert result.ok is False


def test_missing_axis_flagged():
    records = _make_records(500)
    del records[0]["axis"]
    result = check_run(records)
    assert result.missing_fields == 1
    assert result.ok is False


def test_multiple_missing_fields_in_one_record_counts_once():
    records = _make_records(500)
    del records[0]["id"]
    del records[0]["axis"]
    result = check_run(records)
    assert result.missing_fields == 1  # one record, not two


def test_missing_fields_across_records():
    records = _make_records(500)
    del records[0]["id"]
    del records[7]["question"]
    result = check_run(records)
    assert result.missing_fields == 2


# ---------------------------------------------------------------------------
# Multiple issues
# ---------------------------------------------------------------------------


def test_multiple_issue_types_all_counted():
    records = _make_records(498)  # short count
    records[0]["reasoning"] = ""
    records[1]["raw_answer"] = None
    del records[2]["axis"]
    result = check_run(records)
    assert result.count == 498
    assert result.empty_reasoning == 1
    assert result.empty_raw_answer == 1
    assert result.missing_fields == 1
    assert result.ok is False  # missing_fields is a hard fail


def test_empty_records_list():
    result = check_run([])
    assert result.count == 0
    assert result.legible_count == 0
    assert result.ok is False


# ---------------------------------------------------------------------------
# Per-model rollup
# ---------------------------------------------------------------------------


def _run(count: int, expected: int = 100, empty_reasoning: int = 0, empty_raw_answer: int = 0) -> RunCheckResult:
    records = _make_records(count)
    for r in records[:empty_reasoning]:
        r["reasoning"] = ""
    for r in records[count - empty_raw_answer :]:
        r["raw_answer"] = ""
    return check_run(records, expected_count=expected)


def test_summary_sums_across_runs_of_one_model():
    runs = [("model_a", _run(100, empty_reasoning=5)), ("model_a", _run(100, empty_reasoning=15))]
    (summary,) = summarize_by_model(runs)
    assert summary.model == "model_a"
    assert summary.runs == 2
    assert summary.rows == 200
    assert summary.expected == 200
    assert summary.empty_reasoning == 20
    assert summary.missing_reasoning_rate == pytest.approx(0.10)


def test_summary_keeps_short_runs_separate_from_missing_content():
    # 20/100 rows, all legible: short, not illegible. check_run calls this run not ok.
    runs = [("model_a", _run(20))]
    (summary,) = summarize_by_model(runs)
    assert summary.short_runs == 1
    assert summary.rows == 20
    assert summary.expected == 100
    assert summary.missing_reasoning_rate == 0.0
    assert summary.missing_answer_rate == 0.0


def test_summary_sorted_by_missing_reasoning_rate():
    runs = [
        ("clean", _run(100)),
        ("worst", _run(100, empty_reasoning=40)),
        ("middle", _run(100, empty_reasoning=10)),
    ]
    assert [s.model for s in summarize_by_model(runs)] == ["worst", "middle", "clean"]


def test_summary_worst_run_rate_survives_averaging():
    runs = [("model_a", _run(100)), ("model_a", _run(100)), ("model_a", _run(100, empty_reasoning=60))]
    (summary,) = summarize_by_model(runs)
    assert summary.missing_reasoning_rate == pytest.approx(0.20)
    assert summary.worst_reasoning_rate == pytest.approx(0.60)


def test_summary_counts_missing_answers():
    runs = [("model_a", _run(100, empty_raw_answer=25))]
    (summary,) = summarize_by_model(runs)
    assert summary.empty_raw_answer == 25
    assert summary.missing_answer_rate == pytest.approx(0.25)
