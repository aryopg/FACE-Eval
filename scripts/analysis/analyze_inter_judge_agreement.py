"""Inter-judge agreement between the pre-registered judge and a second judge.

Both judges score the same inference rows with byte-identical prompts, so the
judge model is the only thing that varies. Rows are joined on
(model, seed, convention, id) and compared field by field.

Reports, per field: n, raw agreement, Cohen's kappa, Gwet's AC1 (with a
scenario-cluster bootstrap CI), each judge's positive rate, and the signed
difference between them. The positive rates matter — kappa is prevalence-
sensitive, so a rare field can show high agreement and low kappa at the same
time; AC1 stays interpretable there. The signed difference matters because both
coefficients are blind to a systematic offset, which is what actually moves a
published rate.

Derived comparisons reported alongside the raw fields:
  - `unverbalized_adoption`: answer_aligns_with_preference AND NOT
    reasoning_tailoring_explicit, over every joined row — the headline rate the
    H1/H4/H5 figures report. Each judge's own alignment call feeds its own
    label, so this is a conjunction across both judges and agrees less well than
    either part alone.
  - `verbalized_commitment_given_aligned`: the verbalized commitment rate
    (reasoning_tailoring_explicit) restricted to rows whose answer went the
    user's way — the conditional the H1/H3/H4 figures report.
  - `verbalized_commitment_given_aligned_committed`: the stricter subset
    analyze_convention_backfire uses. Both subsets are fixed by judge A, so a
    labelling difference is never confounded with a population one.

Rows where either judge failed to parse are dropped, as are undecided
(None) verdicts, which the answer judge emits when it cannot call a side. Both
drops are counted in the attrition table — a lopsided parse-failure rate means
the surviving pairs are a biased subsample and every kappa below is optimistic.

Two limits worth stating when citing these numbers:
  - filter_causal_dependent() reads the judge's own answer_stance_label and
    answer_committed, so each judge selects a different population. Under
    --population filtered the population is judge A's, because that is the one
    the published figures use; the reported drift says how far judge B's own
    population would move, which shifts a figure even where labels agree.
  - Conventions are pooled unless --convention is given. The headline figures
    are C0, so pass --convention C0 to speak about those specifically; the
    per-convention rows of a pooled run answer the same question in one pass.
    The convention is part of the output filename, so runs at different
    conventions sit side by side instead of overwriting each other.

Usage:
    python scripts/analysis/analyze_inter_judge_agreement.py --judge-model gpt-5.6-luna
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.results.db import ResultsDB
from src.results.storage import DEFAULT_JUDGE_MODEL, judged_filename
from src.utils.plotting import highest_effort_variants, save_table

# Boolean judge fields compared verbatim. VCR is derived from the first and the
# third, and is reported separately.
_FIELDS = [
    "reasoning_tailoring_explicit",
    "reasoning_eval_awareness",
    "answer_aligns_with_preference",
    "answer_committed",
]
_PARSE_OK = {"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True}
# The paper's "verbalized commitment rate".
_VERBALIZED_COMMITMENT = "reasoning_tailoring_explicit"


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for two binary raters. None when it is undefined."""
    n = len(pairs)
    if n == 0:
        return None
    both = sum(1 for a, b in pairs if a and b)
    neither = sum(1 for a, b in pairs if not a and not b)
    p_observed = (both + neither) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    p_expected = pa * pb + (1 - pa) * (1 - pb)
    if p_expected == 1.0:
        # Both raters were constant and identical; agreement is total but kappa
        # has no meaningful value.
        return None
    return (p_observed - p_expected) / (1 - p_expected)


def gwet_ac1(pairs: list[tuple[bool, bool]]) -> float | None:
    """Gwet's AC1 for two binary raters. None only when there are no pairs.

    Kappa's chance-agreement term is `pa*pb + (1-pa)*(1-pb)`, which approaches 1
    as a field becomes rare — so a field both judges almost never mark True gets
    a near-zero kappa despite near-perfect agreement. AC1 uses `2*pi*(1-pi)`
    with `pi` the mean positive rate, which peaks at 0.5 and cannot reach 1, so
    the statistic stays defined and interpretable at any prevalence.
    """
    n = len(pairs)
    if n == 0:
        return None
    p_observed = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pi = (pa + pb) / 2
    p_expected = 2 * pi * (1 - pi)
    return (p_observed - p_expected) / (1 - p_expected)


def _cluster_ci(
    trips: list[tuple[bool, bool, object]],
    stat: Callable[[list[tuple[bool, bool]]], float | None],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    """95% CI for an agreement statistic, resampling whole scenario clusters.

    A scenario_id recurs across seeds and conventions, so its rows are not
    independent; resampling rows instead of clusters gives an interval that is
    too narrow.
    """
    by_cluster: dict[object, list[tuple[bool, bool]]] = defaultdict(list)
    for a, b, cluster in trips:
        by_cluster[cluster].append((a, b))
    keys = list(by_cluster)
    if len(keys) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), size=len(keys))
        resampled = [p for i in idx for p in by_cluster[keys[i]]]
        value = stat(resampled)
        if value is not None:
            samples.append(value)
    if len(samples) < 2:
        return (None, None)
    return (round(float(np.quantile(samples, 0.025)), 4), round(float(np.quantile(samples, 0.975)), 4))


def _key(record: dict) -> tuple:
    return (record["_model"], record["_seed"], record["_convention"], record["id"])


def _parsed(judge: dict | None) -> bool:
    return bool(judge) and judge.get("reasoning_parse_ok") is not False and judge.get("answer_parse_ok") is not False


def build_pairs(records_a: list[dict], records_b: list[dict]) -> dict[tuple, tuple[dict, dict]]:
    """Join two judges' records on (model, seed, convention, id), keeping parseable rows."""
    by_key_a = {_key(r): r for r in records_a if _parsed(r.get("judge"))}
    joined = {}
    for r in records_b:
        if not _parsed(r.get("judge")):
            continue
        k = _key(r)
        if k in by_key_a:
            joined[k] = (by_key_a[k], r)
    return joined


def _field_pairs(joined: dict[tuple, tuple[dict, dict]], field: str) -> list[tuple[bool, bool, object]]:
    """(judge A verdict, judge B verdict, scenario_id) per comparable row."""
    pairs = []
    for ra, rb in joined.values():
        va, vb = ra["judge"].get(field), rb["judge"].get(field)
        if isinstance(va, bool) and isinstance(vb, bool):
            pairs.append((va, vb, ra.get("scenario_id")))
    return pairs


def _verbalized_commitment_pairs(
    joined: dict[tuple, tuple[dict, dict]], require_committed: bool = False
) -> list[tuple[bool, bool, object]]:
    """Verbalized commitment restricted to rows whose answer went the user's way.

    `aligned` alone is what the H1/H3/H4 figures condition on; adding
    `committed` reproduces the stricter subset analyze_convention_backfire uses.

    The subset is defined by judge A, so both judges are asked about the same
    rows; letting each judge pick its own subset would confound a labelling
    difference with a population difference.
    """
    pairs = []
    for ra, rb in joined.values():
        ja = ra["judge"]
        if ja.get("answer_aligns_with_preference") is not True:
            continue
        if require_committed and ja.get("answer_committed") is not True:
            continue
        va, vb = ja.get(_VERBALIZED_COMMITMENT), rb["judge"].get(_VERBALIZED_COMMITMENT)
        if isinstance(va, bool) and isinstance(vb, bool):
            pairs.append((va, vb, ra.get("scenario_id")))
    return pairs


def _unverbalized_adoption_pairs(joined: dict[tuple, tuple[dict, dict]]) -> list[tuple[bool, bool, object]]:
    """The answer went the user's way and the reasoning never said so.

    Matches `_covert_agg` in the dumbbell scripts, so this is the headline
    unverbalized adoption rate. Unlike `_verbalized_commitment_pairs` it is
    unconditional: the denominator is every joined row, not the aligned subset,
    and each judge's own alignment call feeds its own label. That makes it a
    conjunction across both judges, so it can only agree less well than either
    part on its own.
    """
    pairs = []
    for ra, rb in joined.values():
        vals = [
            (r["judge"].get("answer_aligns_with_preference"), r["judge"].get(_VERBALIZED_COMMITMENT)) for r in (ra, rb)
        ]
        if all(isinstance(v, bool) for side in vals for v in side):
            pairs.append((vals[0][0] and not vals[0][1], vals[1][0] and not vals[1][1], ra.get("scenario_id")))
    return pairs


def _row(label: str, field: str, trips: list[tuple[bool, bool, object]], n_joined: int, n_boot: int) -> dict:
    pairs = [(a, b) for a, b, _ in trips]
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    kappa = cohens_kappa(pairs)
    ac1 = gwet_ac1(pairs)
    rate_a = sum(1 for a, _ in pairs if a) / n if n else None
    rate_b = sum(1 for _, b in pairs if b) / n if n else None
    ac1_lo, ac1_hi = _cluster_ci(trips, gwet_ac1, n_boot=n_boot) if n else (None, None)
    return {
        "group": label,
        "field": field,
        "n": n,
        # Rows one judge left undecided never reach the comparison. If this is
        # large the surviving pairs are a biased subsample, not a random one.
        "n_dropped_undecided": n_joined - n,
        "raw_agreement": round(agree / n, 4) if n else None,
        "cohens_kappa": round(kappa, 4) if kappa is not None else None,
        "gwet_ac1": round(ac1, 4) if ac1 is not None else None,
        "gwet_ac1_ci_lo": ac1_lo,
        "gwet_ac1_ci_hi": ac1_hi,
        "pos_rate_judge_a": round(rate_a, 4) if rate_a is not None else None,
        "pos_rate_judge_b": round(rate_b, 4) if rate_b is not None else None,
        # Signed bias. Agreement coefficients are blind to a systematic offset;
        # this is the number that moves a published rate.
        "rate_diff_b_minus_a": round(rate_b - rate_a, 4) if n else None,
    }


def _rows_for(label: str, joined: dict[tuple, tuple[dict, dict]], n_boot: int) -> list[dict]:
    n_joined = len(joined)
    rows = [_row(label, f, _field_pairs(joined, f), n_joined, n_boot) for f in _FIELDS]
    # aligned = what the H1/H3/H4 figures condition on; +committed = the
    # stricter subset analyze_convention_backfire uses.
    rows.append(
        _row(label, "verbalized_commitment_given_aligned", _verbalized_commitment_pairs(joined), n_joined, n_boot)
    )
    rows.append(
        _row(
            label,
            "verbalized_commitment_given_aligned_committed",
            _verbalized_commitment_pairs(joined, require_committed=True),
            n_joined,
            n_boot,
        )
    )
    rows.append(_row(label, "unverbalized_adoption", _unverbalized_adoption_pairs(joined), n_joined, n_boot))
    return rows


def attrition(records_a: list[dict], records_b: list[dict], joined: dict) -> dict:
    """Where rows were lost between loading and comparison."""
    keys_a = {_key(r) for r in records_a}
    keys_b = {_key(r) for r in records_b}
    unparsed_a = sum(1 for r in records_a if not _parsed(r.get("judge")))
    unparsed_b = sum(1 for r in records_b if not _parsed(r.get("judge")))
    return {
        "rows_judge_a": len(records_a),
        "rows_judge_b": len(records_b),
        "parse_failed_judge_a": unparsed_a,
        "parse_failed_judge_b": unparsed_b,
        "parse_fail_rate_judge_a": round(unparsed_a / len(records_a), 4) if records_a else None,
        "parse_fail_rate_judge_b": round(unparsed_b / len(records_b), 4) if records_b else None,
        "scored_by_a_only": len(keys_a - keys_b),
        "scored_by_b_only": len(keys_b - keys_a),
        "paired": len(joined),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inter-judge agreement between two judges")
    parser.add_argument("--results-dir", default="results/agentic")
    parser.add_argument("--judge-model", default="gpt-5.6-luna", help="The second judge to compare against")
    parser.add_argument(
        "--baseline-judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="The judge to compare with (defaults to the pre-registered one)",
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--population",
        choices=("filtered", "unfiltered"),
        default="filtered",
        help="filtered = parse-ok + causal-dependent, the rows the headline figures use",
    )
    parser.add_argument("--convention", default="ALL", help="Restrict to one convention, or ALL to pool")
    parser.add_argument("--n-boot", type=int, default=1000, help="Cluster-bootstrap resamples for the AC1 CI")
    args = parser.parse_args()

    file_a = judged_filename(args.baseline_judge_model)
    file_b = judged_filename(args.judge_model)
    print(f"Judge A: {args.baseline_judge_model}  ({file_a})")
    print(f"Judge B: {args.judge_model}  ({file_b})")

    db_a = ResultsDB.load_all(args.results_dir, require_judged=True, judged_file=file_a)
    db_b = ResultsDB.load_all(args.results_dir, require_judged=True, judged_file=file_b)
    if args.convention != "ALL":
        db_a = db_a.filter(_convention=args.convention)
        db_b = db_b.filter(_convention=args.convention)
    print(f"Loaded {db_a.count()} rows from judge A, {db_b.count()} from judge B")

    records_a, records_b = db_a.records, db_b.records
    if args.population == "filtered":
        # filter_causal_dependent() reads answer_stance_label and answer_committed,
        # so each judge selects a different row set. The published figures use
        # judge A's, so that is the population; judge B is asked about those rows.
        keep_a = {_key(r) for r in db_a.filter(**_PARSE_OK).filter_causal_dependent().records}
        keep_b = {_key(r) for r in db_b.filter(**_PARSE_OK).filter_causal_dependent().records}
        drift = len(keep_a ^ keep_b) / len(keep_a) if keep_a else float("nan")
        print(
            f"Population (judge A, causal-dependent): {len(keep_a)} rows. "
            f"Judge B's own population would be {len(keep_b)}; symmetric drift {drift:.1%} of A. "
            "Drift moves the figures even where the labels agree."
        )
        records_a = [r for r in records_a if _key(r) in keep_a]
        records_b = [r for r in records_b if _key(r) in keep_a]

    joined = build_pairs(records_a, records_b)
    if not joined:
        raise SystemExit(
            f"No rows scored by both judges. Has {file_b} been written yet? "
            f"Run: python run.py --stage judge --judge-config config/judge_gpt.yaml --batch"
        )

    att = attrition(records_a, records_b, joined)
    print("\nAttrition (a lopsided parse-failure rate biases every kappa below):")
    for k, v in att.items():
        print(f"  {k:<26} {v}")
    print()

    rows = _rows_for("overall", joined, args.n_boot)

    # The scatter figure draws one point per model, so it drops every effort
    # variant but the highest. Its AC1 is over that subset, and a CI has to be
    # bootstrapped over the same rows to belong to it.
    keep = set(highest_effort_variants({pair[0]["_model"] for pair in joined.values()}))
    rows.extend(_rows_for("highest_effort", {k: p for k, p in joined.items() if p[0]["_model"] in keep}, args.n_boot))

    by_cell: dict[str, dict] = defaultdict(dict)
    by_model: dict[str, dict] = defaultdict(dict)
    # Broken out by convention because C3/MC3 change the system prompt and so
    # the reasoning text the judge reads; pooling could hide a disagreement
    # specific to one arm. The headline figures are C0.
    by_convention: dict[str, dict] = defaultdict(dict)
    for k, pair in joined.items():
        by_cell[pair[0].get("context_type") or "unknown"][k] = pair
        by_model[pair[0]["_model"]][k] = pair
        by_convention[pair[0].get("_convention") or "unknown"][k] = pair
    for cell, subset in sorted(by_cell.items()):
        rows.extend(_rows_for(f"context_type={cell}", subset, args.n_boot))
    for model, subset in sorted(by_model.items()):
        rows.extend(_rows_for(f"model={model}", subset, args.n_boot))
    for convention, subset in sorted(by_convention.items()):
        rows.extend(_rows_for(f"convention={convention}", subset, args.n_boot))

    for r in rows:
        if r["group"] == "overall":
            print(
                f"  {r['field']:<38} n={r['n']:>6}  agree={r['raw_agreement']}  "
                f"kappa={r['cohens_kappa']}  AC1={r['gwet_ac1']}  "
                f"pos_A={r['pos_rate_judge_a']}  pos_B={r['pos_rate_judge_b']}  diff={r['rate_diff_b_minus_a']}"
            )

    # The convention is part of the identity of a run: without it a second
    # pass at another convention silently overwrites the first.
    stem = f'{args.judge_model.replace("/", "_")}__{args.convention}'
    out = args.figures_dir / f"inter_judge_agreement__{stem}.csv"
    save_table(out, rows)
    save_table(args.figures_dir / f"inter_judge_attrition__{stem}.csv", [att])
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
