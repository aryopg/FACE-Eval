"""Inference output quality checks, run between download and judge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_REQUIRED_FIELDS = ("id", "question", "condition", "axis")

# A sample is "legible" if it has both reasoning and raw_answer. Runs with at
# least this fraction of expected samples legible are judge-ready — the judge
# can still produce meaningful statistics from a slightly incomplete run.
MIN_LEGIBLE_RATE = 0.90


@dataclass
class RunCheckResult:
    """One run's inference records, counted.

    `count` is the rows actually found; `expected` is the rows the run should have
    had. A row is legible when it has both a non-empty `reasoning` and a non-empty
    `raw_answer`. `empty_reasoning` and `empty_raw_answer` are counted separately,
    so one bad row can add to both. `missing_fields` counts rows missing any of
    `id`, `question`, `condition`, `axis`. `reasoning_missing_by_condition` splits
    `empty_reasoning` by the row's `condition`, under the key `"unknown"` when the
    row has no condition.
    """

    count: int
    expected: int
    legible_count: int
    empty_reasoning: int
    empty_raw_answer: int
    missing_fields: int
    reasoning_missing_by_condition: dict[str, int] = field(default_factory=dict)

    @property
    def reasoning_missing_rate(self) -> float:
        """Share of the rows found that have no reasoning.

        The denominator is `count`, the rows actually found -- not `expected`. A run
        that stopped early therefore reads clean here if the rows it did write are
        clean. Returns 0.0 for an empty run.
        """
        return self.empty_reasoning / self.count if self.count > 0 else 0.0

    @property
    def legible_rate(self) -> float:
        """Share of the rows the run should have had that are legible.

        The denominator is `expected`, not `count`, so a missing row counts the same
        as an illegible one. This is the only place a short run is penalised.
        Returns 0.0 when `expected` is 0.
        """
        return self.legible_count / self.expected if self.expected > 0 else 0.0

    @property
    def ok(self) -> bool:
        """Whether this run may go to the judge.

        Two conditions, both required:

        1. `legible_rate` is at least `MIN_LEGIBLE_RATE` (90%). Up to 10% of the
           expected rows may be missing or illegible; the judge can still produce
           meaningful statistics from the rest. Exactly 90% passes.
        2. `missing_fields` is 0. One row missing `id`, `question`, `condition` or
           `axis` fails the whole run -- those fields key the analysis, so a run
           with any of them absent cannot be scored or grouped.

        A model x convention unit is judgeable only if every one of its seeds is ok.
        """
        return self.legible_rate >= MIN_LEGIBLE_RATE and self.missing_fields == 0


@dataclass
class ModelSummary:
    """One model's runs rolled up, for the per-model view of a whole sweep."""

    model: str
    runs: int
    rows: int
    expected: int
    short_runs: int
    empty_reasoning: int
    empty_raw_answer: int
    worst_reasoning_rate: float

    @property
    def missing_reasoning_rate(self) -> float:
        """Share of this model's rows that have no reasoning, pooled over its runs.

        The denominator is `rows` (rows actually found), not `expected`, so a short
        run does not read as an illegible one. Returns 0.0 when there are no rows.
        """
        return self.empty_reasoning / self.rows if self.rows > 0 else 0.0

    @property
    def missing_answer_rate(self) -> float:
        """Share of this model's rows that have no raw answer, pooled over its runs.

        Same denominator as `missing_reasoning_rate`. Returns 0.0 when there are no
        rows.
        """
        return self.empty_raw_answer / self.rows if self.rows > 0 else 0.0


def summarize_by_model(runs: list[tuple[str, RunCheckResult]]) -> list[ModelSummary]:
    """Roll up per-run checks by model, worst missing-content rate first.

    Rates use actual rows as the denominator, so a short run does not read as an
    illegible one -- `short_runs` carries that signal separately.
    """
    by_model: dict[str, list[RunCheckResult]] = {}
    for model, result in runs:
        by_model.setdefault(model, []).append(result)

    summaries = [
        ModelSummary(
            model=model,
            runs=len(results),
            rows=sum(r.count for r in results),
            expected=sum(r.expected for r in results),
            short_runs=sum(1 for r in results if r.count < r.expected),
            empty_reasoning=sum(r.empty_reasoning for r in results),
            empty_raw_answer=sum(r.empty_raw_answer for r in results),
            worst_reasoning_rate=max(r.reasoning_missing_rate for r in results),
        )
        for model, results in by_model.items()
    ]
    summaries.sort(key=lambda s: (-s.missing_reasoning_rate, -s.missing_answer_rate, s.model))
    return summaries


def check_run(records: list[dict[str, Any]], expected_count: int = 500) -> RunCheckResult:
    """Check a single run's inference records for judge-readiness."""
    legible_count = 0
    empty_reasoning = 0
    empty_raw_answer = 0
    missing_fields = 0
    reasoning_missing_by_condition: dict[str, int] = {}

    for r in records:
        if any(f not in r for f in _REQUIRED_FIELDS):
            missing_fields += 1

        reasoning = r.get("reasoning")
        has_reasoning = isinstance(reasoning, str) and bool(reasoning.strip())
        if not has_reasoning:
            empty_reasoning += 1
            cond = r.get("condition", "unknown")
            reasoning_missing_by_condition[cond] = reasoning_missing_by_condition.get(cond, 0) + 1

        raw_answer = r.get("raw_answer")
        has_raw_answer = isinstance(raw_answer, str) and bool(raw_answer.strip())
        if not has_raw_answer:
            empty_raw_answer += 1

        if has_reasoning and has_raw_answer:
            legible_count += 1

    return RunCheckResult(
        count=len(records),
        expected=expected_count,
        legible_count=legible_count,
        empty_reasoning=empty_reasoning,
        empty_raw_answer=empty_raw_answer,
        missing_fields=missing_fields,
        reasoning_missing_by_condition=reasoning_missing_by_condition,
    )
