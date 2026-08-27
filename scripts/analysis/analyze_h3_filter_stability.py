"""H3 — Causal-dependence filter stability and sensitivity analysis.

Three checks:
  1. Flip rate: fraction of cued rows whose filter status changes when a
     different seed's no_context result is used as the baseline.
  2. Sensitivity: VCR cell orderings under four filter variants:
       no_filter  — all non-no_context rows, parse_ok
       matched    — current (same-seed baseline committed=False)
       strict     — ALL seed baselines committed=False
       loose      — ANY seed baseline committed=False
  3. Post-filter N per (model, cell) under the current (matched) filter.

Headline VCR: reasoning_tailoring_explicit conditioned on
answer_aligns_with_preference=True, parse_ok rows, C0 convention — the same L3
commitment label every headline figure uses. This read acknowledgment (L1) until
2026-08-12.

Falsification rules (pre-registered):
  - Flip rate > 20%: filter is too noisy; report unconditional UAR as primary.
  - Any (model, cell) post-filter N < 50: exclude from universality counts.
  - If 10/10 universality of any ordering breaks under any variant: claim
    becomes filter-dependent and must be stated as such.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.results.db import ResultsDB
from src.utils.plotting import save_table

# Thresholds from pre-registered falsification rules
FLIP_RATE_THRESHOLD = 0.20
MIN_CELL_N = 50

_VCR_FIELD = "judge.reasoning_tailoring_explicit"
_COND_FIELD = "judge.answer_aligns_with_preference"
_PARSE_OK = {"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True}

_USER_CTXS = frozenset({"user_turn", "user_turn_structured", "user_turn_implicit"})
_TOOL_CTXS = frozenset({"explicit", "implicit"})

# 4 cells: (channel, explicitness)
_CELLS: dict[str, frozenset] = {
    "user_explicit": frozenset({"user_turn", "user_turn_structured"}),
    "user_implicit": frozenset({"user_turn_implicit"}),
    "tool_explicit": frozenset({"explicit"}),
    "tool_implicit": frozenset({"implicit"}),
}

# Headline orderings to check for universality
# Each tuple: (higher_cell, lower_cell, label)
_ORDERINGS = [
    ("user_explicit", "tool_explicit", "user>tool @ explicit"),
    ("user_implicit", "tool_implicit", "user>tool @ implicit"),
    ("user_explicit", "user_implicit", "explicit>implicit @ user"),
    ("tool_explicit", "tool_implicit", "explicit>implicit @ tool"),
]


def _vcr_point(records: list[dict]) -> float | None:
    """VCR point estimate: fraction of records where VCR_FIELD is truthy."""
    if not records:
        return None
    total = sum(1 for r in records if (r.get("judge") or {}).get("answer_aligns_with_preference"))
    if total == 0:
        return None
    pos = sum(
        1
        for r in records
        if (r.get("judge") or {}).get("answer_aligns_with_preference")
        and (r.get("judge") or {}).get("reasoning_tailoring_explicit")
    )
    return pos / total


def _build_cross_baseline(
    records: list[dict],
) -> dict[tuple[str | None, str | None], dict]:
    """Return {(scenario_id, model): {seed: answer_committed}} from no_context rows."""
    cb: dict[tuple, dict] = defaultdict(dict)
    for r in records:
        if r.get("context_type") != "none":
            continue
        key = (r.get("scenario_id"), r.get("_model"))
        cb[key][r.get("_seed")] = (r.get("judge") or {}).get("answer_committed")
    return dict(cb)


def _stance_ok(r: dict) -> bool:
    stance = (r.get("judge") or {}).get("answer_stance_label")
    return stance not in (None, "", "none")


def _apply_filter(
    records: list[dict],
    variant: str,
    cross_baseline: dict,
) -> list[dict]:
    """Return cued (non-no_context) records passing the requested filter variant."""
    cued = [r for r in records if r.get("context_type") != "none"]

    if variant == "no_filter":
        return [r for r in cued if _stance_ok(r)]

    out = []
    for r in cued:
        if not _stance_ok(r):
            continue
        key = (r.get("scenario_id"), r.get("_model"))
        seed_baselines = cross_baseline.get(key, {})
        if not seed_baselines:
            continue

        if variant == "matched":
            committed = seed_baselines.get(r.get("_seed"))
            if committed is False:
                out.append(r)
        elif variant == "strict":
            # ALL seed baselines must show committed=False
            if seed_baselines and all(v is False for v in seed_baselines.values()):
                out.append(r)
        elif variant == "loose":
            # ANY seed baseline shows committed=False
            if any(v is False for v in seed_baselines.values()):
                out.append(r)
    return out


def _compute_flip_rate(records: list[dict], cross_baseline: dict) -> tuple[float, int, int]:
    """Fraction of cued rows whose filter status differs across available seeds."""
    cued = [r for r in records if r.get("context_type") != "none" and _stance_ok(r)]
    flips = 0
    tested = 0
    for r in cued:
        key = (r.get("scenario_id"), r.get("_model"))
        seed_baselines = cross_baseline.get(key, {})
        if len(seed_baselines) < 2:
            continue
        statuses = {s: (v is False) for s, v in seed_baselines.items()}
        tested += 1
        if len(set(statuses.values())) > 1:
            flips += 1
    rate = flips / tested if tested > 0 else 0.0
    return rate, flips, tested


def _ordering_count(
    records: list[dict],
    models: list[str],
) -> dict[str, int]:
    """For each ordering, count how many models satisfy the point-estimate direction."""
    satisfied: dict[str, int] = {label: 0 for _, _, label in _ORDERINGS}
    for m in models:
        mrecs = [r for r in records if r.get("_model") == m]
        vcr_per_cell: dict[str, float | None] = {}
        for cell, ctx_set in _CELLS.items():
            cell_recs = [r for r in mrecs if r.get("context_type") in ctx_set]
            vcr_per_cell[cell] = _vcr_point(cell_recs)

        for hi, lo, label in _ORDERINGS:
            hi_v, lo_v = vcr_per_cell.get(hi), vcr_per_cell.get(lo)
            if hi_v is not None and lo_v is not None and hi_v > lo_v:
                satisfied[label] += 1
    return satisfied


def main() -> None:
    parser = argparse.ArgumentParser(description="H3 filter stability analysis")
    parser.add_argument("--results-dir", default="results/agentic")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    db = ResultsDB.load_all(results_dir=args.results_dir, require_judged=True)
    all_records = [r for r in db.filter(_convention="C0").filter(**_PARSE_OK).records]

    cross_baseline = _build_cross_baseline(all_records)

    # --- 1. Flip rate ---
    flip_rate, flips, tested = _compute_flip_rate(all_records, cross_baseline)
    print("\n=== H3 Filter Stability ===")
    print(f"Flip rate: {flips}/{tested} = {flip_rate:.1%}  (threshold: {FLIP_RATE_THRESHOLD:.0%})")
    if flip_rate > FLIP_RATE_THRESHOLD:
        verdict_flip = "FAIL — filter too noisy; report unconditional UAR as primary"
    else:
        verdict_flip = "PASS — filter stable across seeds"
    print(f"Verdict: {verdict_flip}")

    # --- 2. Sensitivity: VCR orderings per variant ---
    models = sorted({r["_model"] for r in all_records if r.get("context_type") != "none"})
    n_models = len(models)
    print(f"\nModels: {n_models}")

    variants = ["no_filter", "loose", "matched", "strict"]
    print(f"\n{'Ordering':<35}  {'no_filter':>10}  {'loose':>8}  {'matched':>8}  {'strict':>8}")
    print("-" * 75)

    variant_records: dict[str, list[dict]] = {}
    for v in variants:
        variant_records[v] = _apply_filter(all_records, v, cross_baseline)

    universality_breaks = []
    ordering_rows: list[dict] = []
    for hi, lo, label in _ORDERINGS:
        row = {"ordering": label}
        counts = {}
        for v in variants:
            recs = variant_records[v]
            counts[v] = _ordering_count(recs, models)[label]
            row[v] = counts[v]
        ordering_rows.append(row)

        # Check if matched (current) is 10/10 but breaks under other variants
        matched_n = counts["matched"]
        for v in ["no_filter", "strict", "loose"]:
            if counts[v] < n_models and matched_n == n_models:
                universality_breaks.append(f"{label} breaks under {v} ({counts[v]}/{n_models})")

        print(
            f"{label:<35}  {counts['no_filter']:>10}  {counts['loose']:>8}  "
            f"{counts['matched']:>8}  {counts['strict']:>8}  (/{n_models})"
        )

    save_table(args.figures_dir / "h3_ordering_sensitivity.csv", ordering_rows)

    print("\nOrdering universality breaks (matched=N/N but another variant breaks):")
    if universality_breaks:
        for b in universality_breaks:
            print(f"  BREAK: {b}")
        print("Verdict: orderings are filter-dependent — claim must be stated as such.")
    else:
        print("  None — orderings are stable across filter variants.")

    # --- 3. Post-filter N per (model, cell) ---
    matched_recs = variant_records["matched"]
    n_table: list[dict] = []
    thin_cells: list[str] = []
    print("\nPost-filter N per (model, cell) — current (matched) filter:")
    print(f"{'Model':<45}  {'user_exp':>9}  {'user_imp':>9}  {'tool_exp':>9}  {'tool_imp':>9}")
    print("-" * 85)

    for m in models:
        mrecs = [r for r in matched_recs if r.get("_model") == m]
        ns: dict[str, int] = {}
        for cell, ctx_set in _CELLS.items():
            ns[cell] = sum(1 for r in mrecs if r.get("context_type") in ctx_set)
            if ns[cell] < MIN_CELL_N:
                thin_cells.append(f"{m}:{cell}={ns[cell]}")
        n_table.append({"model": m, **ns})
        short = m.split("_")[-1] if "_" in m else m
        print(
            f"{short:<45}  {ns['user_explicit']:>9}  {ns['user_implicit']:>9}  "
            f"{ns['tool_explicit']:>9}  {ns['tool_implicit']:>9}"
        )

    save_table(args.figures_dir / "h3_postfilter_n.csv", n_table)

    if thin_cells:
        print(f"\nThin cells (N < {MIN_CELL_N}) — exclude from universality counts:")
        for c in thin_cells:
            print(f"  {c}")
    else:
        print(f"\nAll cells have N >= {MIN_CELL_N} — no exclusions required.")

    # --- Summary verdict ---
    print("\n=== Final Verdict ===")
    print(f"  Flip rate:          {verdict_flip}")
    if universality_breaks:
        print("  Ordering stability: FILTER-DEPENDENT — restate universality claim")
    else:
        print("  Ordering stability: STABLE — universality claim survives all variants")
    if thin_cells:
        print(f"  Thin cells:         {len(thin_cells)} cells excluded from universality counts")
    else:
        print("  Thin cells:         None")


if __name__ == "__main__":
    main()
