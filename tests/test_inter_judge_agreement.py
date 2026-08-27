"""Tests for scripts/analyze_inter_judge_agreement.py."""

from __future__ import annotations

import pytest

import scripts.analysis.analyze_inter_judge_agreement as mod

# Keeps the bootstrap out of the way in tests that are not about it.
_FAST_BOOT = 10


def _pairs(tt: int, tf: int, ft: int, ff: int) -> list[tuple[bool, bool]]:
    return [(True, True)] * tt + [(True, False)] * tf + [(False, True)] * ft + [(False, False)] * ff


class TestCohensKappa:
    def test_perfect_agreement_on_a_mixed_field(self):
        pairs = [(True, True)] * 5 + [(False, False)] * 5
        assert mod.cohens_kappa(pairs) == pytest.approx(1.0)

    def test_chance_level_agreement_is_about_zero(self):
        # Independent raters, both positive half the time.
        pairs = [(True, True), (True, False), (False, True), (False, False)]
        assert mod.cohens_kappa(pairs) == pytest.approx(0.0)

    def test_systematic_disagreement_is_negative(self):
        pairs = [(True, False)] * 5 + [(False, True)] * 5
        assert mod.cohens_kappa(pairs) < 0

    def test_high_raw_agreement_on_a_rare_field_gives_low_kappa(self):
        """The reason the script reports kappa alongside raw agreement."""
        pairs = [(False, False)] * 96 + [(True, True)] * 1 + [(True, False)] * 2 + [(False, True)] * 1
        raw = sum(1 for a, b in pairs if a == b) / len(pairs)
        assert raw > 0.95
        assert mod.cohens_kappa(pairs) < 0.5

    def test_constant_identical_raters_have_undefined_kappa(self):
        assert mod.cohens_kappa([(True, True)] * 10) is None

    def test_empty_is_none(self):
        assert mod.cohens_kappa([]) is None


class TestGwetAC1:
    def test_perfect_agreement_is_one(self):
        assert mod.gwet_ac1(_pairs(tt=50, tf=0, ft=0, ff=50)) == pytest.approx(1.0)

    def test_empty_is_none(self):
        assert mod.gwet_ac1([]) is None

    def test_known_value(self):
        # p_o=0.75, pa=0.75, pb=0.5 -> pi=0.625, p_e=0.46875
        assert mod.gwet_ac1(_pairs(tt=2, tf=1, ft=0, ff=1)) == pytest.approx(0.28125 / 0.53125)

    def test_survives_the_prevalence_paradox_that_sinks_kappa(self):
        # 98% agreement on a field marked True ~1% of the time. This is the
        # regime gpt-oss-*_low sits in for L3.
        pairs = _pairs(tt=0, tf=1, ft=1, ff=98)
        assert mod.cohens_kappa(pairs) < 0.0
        assert mod.gwet_ac1(pairs) > 0.97

    def test_defined_when_both_judges_are_constant(self):
        # kappa is undefined here; AC1 still reports the agreement.
        pairs = _pairs(tt=0, tf=0, ft=0, ff=100)
        assert mod.cohens_kappa(pairs) is None
        assert mod.gwet_ac1(pairs) == pytest.approx(1.0)

    def test_total_disagreement_is_negative(self):
        assert mod.gwet_ac1(_pairs(tt=0, tf=50, ft=50, ff=0)) < 0


def _trips(pairs, clusters):
    return [(a, b, c) for (a, b), c in zip(pairs, clusters)]


class TestClusterCI:
    def test_brackets_the_point_estimate(self):
        pairs = _pairs(tt=40, tf=5, ft=5, ff=50)
        trips = _trips(pairs, [f"s{i % 25}" for i in range(len(pairs))])
        lo, hi = mod._cluster_ci(trips, mod.gwet_ac1, n_boot=200)
        assert lo <= mod.gwet_ac1(pairs) <= hi

    def test_is_none_with_a_single_cluster(self):
        trips = _trips(_pairs(tt=10, tf=0, ft=0, ff=10), ["only"] * 20)
        assert mod._cluster_ci(trips, mod.gwet_ac1, n_boot=50) == (None, None)

    def test_correlated_clusters_widen_the_interval(self):
        # Same rows either way. Grouped into few blocks whose rows agree with
        # each other, the effective sample size is 5, not 100 — which is the
        # whole reason to resample scenarios rather than rows.
        pairs = _pairs(tt=30, tf=10, ft=10, ff=50)
        blocked = _trips(pairs, [f"s{i // 20}" for i in range(len(pairs))])
        singleton = _trips(pairs, [f"s{i}" for i in range(len(pairs))])
        lo_b, hi_b = mod._cluster_ci(blocked, mod.gwet_ac1, n_boot=400)
        lo_s, hi_s = mod._cluster_ci(singleton, mod.gwet_ac1, n_boot=400)
        assert (hi_b - lo_b) > (hi_s - lo_s)

    def test_identical_clusters_give_a_degenerate_interval(self):
        # Every cluster has the same composition, so every resample reproduces
        # the point estimate exactly. Zero width here is correct, not a bug.
        pairs = _pairs(tt=30, tf=10, ft=10, ff=50)
        trips = _trips(pairs, [f"s{i % 5}" for i in range(len(pairs))])
        lo, hi = mod._cluster_ci(trips, mod.gwet_ac1, n_boot=200)
        assert lo == hi == pytest.approx(mod.gwet_ac1(pairs), abs=1e-4)


def _rec(rid: str, model: str = "m", seed: int = 42, convention: str = "C0", **judge):
    judge.setdefault("reasoning_parse_ok", True)
    judge.setdefault("answer_parse_ok", True)
    return {"id": rid, "_model": model, "_seed": seed, "_convention": convention, "judge": judge}


class TestBuildPairs:
    def test_joins_on_model_seed_convention_and_id(self):
        a = [_rec("r0"), _rec("r1")]
        b = [_rec("r1"), _rec("r0")]
        assert set(mod.build_pairs(a, b)) == {("m", 42, "C0", "r0"), ("m", 42, "C0", "r1")}

    def test_same_id_under_a_different_convention_is_not_a_pair(self):
        a = [_rec("r0", convention="C0")]
        b = [_rec("r0", convention="C3")]
        assert mod.build_pairs(a, b) == {}

    def test_same_id_from_a_different_model_is_not_a_pair(self):
        a = [_rec("r0", model="modelA")]
        b = [_rec("r0", model="modelB")]
        assert mod.build_pairs(a, b) == {}

    def test_rows_only_one_judge_scored_are_dropped(self):
        assert set(mod.build_pairs([_rec("r0"), _rec("r1")], [_rec("r0")])) == {("m", 42, "C0", "r0")}

    def test_parse_failures_are_dropped(self):
        a = [_rec("r0", reasoning_parse_ok=False)]
        b = [_rec("r0")]
        assert mod.build_pairs(a, b) == {}


class TestAttrition:
    def test_counts_lopsided_parse_failures(self):
        a = [_rec("r0"), _rec("r1"), _rec("r2")]
        b = [_rec("r0"), _rec("r1", answer_parse_ok=False), _rec("r2", reasoning_parse_ok=False)]
        att = mod.attrition(a, b, mod.build_pairs(a, b))
        assert att["parse_failed_judge_a"] == 0
        assert att["parse_failed_judge_b"] == 2
        assert att["parse_fail_rate_judge_b"] == pytest.approx(2 / 3, abs=1e-4)
        assert att["paired"] == 1

    def test_counts_rows_only_one_judge_scored(self):
        a = [_rec("r0"), _rec("r1")]
        b = [_rec("r0"), _rec("r2")]
        att = mod.attrition(a, b, mod.build_pairs(a, b))
        assert att["scored_by_a_only"] == 1
        assert att["scored_by_b_only"] == 1
        assert att["paired"] == 1

    def test_undecided_drops_are_reported_per_field(self):
        joined = {
            ("m", 42, "C0", "r0"): (
                _rec("r0", answer_committed=True),
                _rec("r0", answer_committed=None),
            ),
            ("m", 42, "C0", "r1"): (
                _rec("r1", answer_committed=True),
                _rec("r1", answer_committed=True),
            ),
        }
        row = next(r for r in mod._rows_for("overall", joined, _FAST_BOOT) if r["field"] == "answer_committed")
        assert row["n"] == 1
        assert row["n_dropped_undecided"] == 1


class TestFieldPairs:
    """_field_pairs and friends carry scenario_id as a third element for the
    cluster bootstrap; these assert on the verdict pair itself."""

    @staticmethod
    def _verdicts(trips):
        return [(a, b) for a, b, _ in trips]

    def test_undecided_verdicts_are_excluded(self):
        joined = {
            ("m", 42, "C0", "r0"): (
                _rec("r0", answer_aligns_with_preference=True),
                _rec("r0", answer_aligns_with_preference=None),
            ),
            ("m", 42, "C0", "r1"): (
                _rec("r1", answer_aligns_with_preference=True),
                _rec("r1", answer_aligns_with_preference=False),
            ),
        }
        assert self._verdicts(mod._field_pairs(joined, "answer_aligns_with_preference")) == [(True, False)]

    def test_carries_the_scenario_id_for_clustering(self):
        ra = {**_rec("r0", answer_committed=True), "scenario_id": "sc1"}
        rb = _rec("r0", answer_committed=False)
        assert mod._field_pairs({("m", 42, "C0", "r0"): (ra, rb)}, "answer_committed") == [(True, False, "sc1")]


class TestVerbalizedCommitmentConditional:
    """Verbalized commitment restricted to aligned — the subset is judge A's, so a
    labelling difference is never confounded with a population difference."""

    @staticmethod
    def _rows(a_aligned, a_committed, l3_a=True, l3_b=False, b_aligned=True, b_committed=True):
        ra = _rec(
            "r0",
            answer_aligns_with_preference=a_aligned,
            answer_committed=a_committed,
            reasoning_tailoring_explicit=l3_a,
        )
        rb = _rec(
            "r0",
            answer_aligns_with_preference=b_aligned,
            answer_committed=b_committed,
            reasoning_tailoring_explicit=l3_b,
        )
        return {("m", 42, "C0", "r0"): (ra, rb)}

    def test_keeps_rows_judge_a_calls_aligned_and_committed(self):
        trips = mod._verbalized_commitment_pairs(self._rows(a_aligned=True, a_committed=True))
        assert [(a, b) for a, b, _ in trips] == [(True, False)]

    def test_drops_rows_judge_a_does_not_call_aligned(self):
        assert mod._verbalized_commitment_pairs(self._rows(a_aligned=False, a_committed=True)) == []

    def test_aligned_but_not_committed_is_kept_by_default(self):
        # The H1/H3/H4 figures condition on aligned alone.
        trips = mod._verbalized_commitment_pairs(self._rows(a_aligned=True, a_committed=False))
        assert [(a, b) for a, b, _ in trips] == [(True, False)]

    def test_require_committed_drops_the_uncommitted_row(self):
        # The stricter subset analyze_convention_backfire uses.
        rows = self._rows(a_aligned=True, a_committed=False)
        assert mod._verbalized_commitment_pairs(rows, require_committed=True) == []

    def test_both_subsets_are_reported(self):
        rows = mod._rows_for("overall", self._rows(a_aligned=True, a_committed=True), _FAST_BOOT)
        fields = {r["field"] for r in rows}
        assert "verbalized_commitment_given_aligned" in fields
        assert "verbalized_commitment_given_aligned_committed" in fields

    def test_judge_b_does_not_get_a_say_in_the_subset(self):
        # Judge A includes the row, judge B would have excluded it. It stays.
        trips = mod._verbalized_commitment_pairs(
            self._rows(a_aligned=True, a_committed=True, b_aligned=False, b_committed=False)
        )
        assert [(a, b) for a, b, _ in trips] == [(True, False)]

    def test_is_reported_as_its_own_field(self):
        rows = mod._rows_for("overall", self._rows(a_aligned=True, a_committed=True), _FAST_BOOT)
        assert any(r["field"] == "verbalized_commitment_given_aligned" for r in rows)
