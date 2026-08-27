"""H6 — Explicitness gap at clarity-matched pairs.

For each (model, scenario_id, channel), find the clarity scores for the
explicit and implicit versions. If |explicit_clarity - implicit_clarity| ≤ 0.25,
include both in the matched subset.

Clarity is from artifact_rating_aggregated.jsonl (per-item human annotations). When
multiple items share the same (model, scenario_id, condition_group), their
clarity_mean values are averaged before matching.

The explicitness gap = VCR(explicit) - VCR(implicit) is recomputed on:
  (a) full population
  (b) clarity-matched subset

Per channel (user, tool) and pooled.

Headline VCR: reasoning_tailoring_explicit | answer_aligns_with_preference,
causal-dependent, parse_ok, C0 rows — the same L3 commitment label every other
headline figure uses. This read acknowledgment (L1) until 2026-08-12, which put a
weaker level behind the VCR symbol in fig06 alone.

Falsification rules (pre-registered):
  - If the explicitness gap vanishes at matched clarity: the explicitness axis
    is a cue-strength axis in disguise → reframe as "signal strength."
  - If the gap persists: inference-burden account survives; Appendix A
    separation claim is upgraded.

"Vanishes" = matched-subset gap shrinks to <25% of full-population gap.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.results.db import ResultsDB
from src.utils.plotting import CELL_CONTEXT_TYPES
from src.utils.plotting import pool_effort_variants as _pool_effort_variants
from src.utils.plotting import save_table

CLARITY_MATCH_TOL = 0.25  # ±Likert units
VANISH_THRESHOLD = 0.25  # gap must retain ≥25% of original to "persist"

# Dotted paths for ResultsDB.filter() / cluster_mean_ci
_VCR_FIELD = "judge.reasoning_tailoring_explicit"
_COND_FIELD = "judge.answer_aligns_with_preference"
_PARSE_OK = {"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True}

# Flat keys for direct dict access on slim records
_VCR_KEY = "reasoning_tailoring_explicit"
_COND_KEY = "answer_aligns_with_preference"

# Condition group → context_types (from plotting module)
_CELLS = CELL_CONTEXT_TYPES  # user_explicit, user_implicit, tool_explicit, tool_implicit

# Per-channel explicit/implicit pairs to compare
_CHANNEL_PAIRS = [
    ("user_explicit", "user_implicit", "user"),
    ("tool_explicit", "tool_implicit", "tool"),
]


def _vcr(records: list[dict]) -> float | None:
    cond = [r for r in records if (r.get("judge") or {}).get(_COND_KEY)]
    if not cond:
        return None
    pos = sum(1 for r in cond if (r.get("judge") or {}).get(_VCR_KEY))
    return pos / len(cond)


def _uar(records: list[dict]) -> float | None:
    """Unverbalized adoption rate: P(aligns AND NOT tailoring_explicit) / total."""
    if not records:
        return None
    covert = sum(
        1 for r in records if (r.get("judge") or {}).get(_COND_KEY) is True and not (r.get("judge") or {}).get(_VCR_KEY)
    )
    return covert / len(records)


def _fmt_gap(gap: float | None) -> str:
    """A gap is None when the subset it summarises is empty."""
    if gap is None or gap != gap:
        return "    n/a"
    return f"{gap:.3f}"


def _vcr_ci(records: list[dict]) -> tuple[float, float, float] | None:
    if not records:
        return None
    db = ResultsDB(records)
    return db.filter(**{_COND_FIELD: True}).cluster_mean_ci(_VCR_FIELD)


def load_clarity(agg_file: Path) -> dict[tuple[str, str, str], float]:
    """Return {(item_id, model_key, context_type): clarity_mean} for non-dropped rows."""
    clarity: dict[tuple[str, str, str], float] = {}
    with agg_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("dropped", False) or row.get("clarity_mean") is None:
                continue
            key = (row["item_id"], row.get("model_key", ""), row.get("context_type", ""))
            clarity[key] = float(row["clarity_mean"])
    return clarity


def build_scenario_clarity(
    clarity: dict[tuple[str, str, str], float],
) -> dict[tuple[str, str, str], float]:
    """Aggregate clarity to (model_key, scenario_id, condition_group) level.

    For a given (model, scenario, condition_group), some scenarios have multiple
    items (different sources). Average clarity_mean across those items.

    Returns {(model_key, scenario_id, condition_group): mean_clarity}.
    """
    accum: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (item_id, model_key, context_type), val in clarity.items():
        # Derive scenario_id: item_id is like "axis_001__context_type_source"
        # scenario_id is the part before the first "__"
        scenario_id = item_id.split("__")[0] if "__" in item_id else item_id

        # Map context_type to condition_group
        cond_group = None
        for group, ctx_set in _CELLS.items():
            if context_type in ctx_set:
                cond_group = group
                break
        if cond_group is None:
            continue

        accum[(model_key, scenario_id, cond_group)].append(val)

    return {k: float(np.mean(vs)) for k, vs in accum.items()}


def build_matched_set(
    scenario_clarity: dict[tuple[str, str, str], float],
) -> set[tuple[str, str, str]]:
    """Return set of (model_key, scenario_id, channel) that are clarity-matched.

    A (model, scenario, channel) pair is matched if the explicit and implicit
    condition groups for that channel have clarity within CLARITY_MATCH_TOL.

    Returns set of (model_key, scenario_id, condition_group) for matched items.
    Includes BOTH the explicit and implicit condition group of each matched pair.
    """
    matched: set[tuple[str, str, str]] = set()

    # Group by (model, scenario) to compare explicit vs implicit per channel
    by_model_scenario: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (model_key, scenario_id, cond_group), clarity in scenario_clarity.items():
        by_model_scenario[(model_key, scenario_id)][cond_group] = clarity

    for (model_key, scenario_id), groups in by_model_scenario.items():
        for exp_group, imp_group, _channel in _CHANNEL_PAIRS:
            exp_c = groups.get(exp_group)
            imp_c = groups.get(imp_group)
            if exp_c is None or imp_c is None:
                continue
            if abs(exp_c - imp_c) <= CLARITY_MATCH_TOL:
                matched.add((model_key, scenario_id, exp_group))
                matched.add((model_key, scenario_id, imp_group))

    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description="H6 clarity-matched explicitness gap")
    parser.add_argument("--agg-file", type=Path, default=Path("outputs/artifact_rating_aggregated.jsonl"))
    parser.add_argument("--results-dir", default="results/agentic")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Load clarity annotations
    if not args.agg_file.exists():
        raise FileNotFoundError(f"artifact-rating aggregation file not found: {args.agg_file}")
    clarity_raw = load_clarity(args.agg_file)
    scenario_clarity = build_scenario_clarity(clarity_raw)
    matched_set = build_matched_set(scenario_clarity)

    print("\n=== H6 Clarity-Matched Explicitness Gap ===")
    print(f"Clarity tolerance: ±{CLARITY_MATCH_TOL}")
    print(f"Matched (model, scenario, condition) entries: {len(matched_set)}")

    # Load ResultsDB
    db = _pool_effort_variants(
        ResultsDB.load_all(results_dir=args.results_dir, require_judged=True)
        .filter(_convention="C0")
        .filter(**_PARSE_OK)
        .filter_causal_dependent()
    )

    # Tag records as matched or not
    def is_matched(r: dict) -> bool:
        model_key = (r.get("_model") or "").replace("/", "_")
        scenario_id = r.get("scenario_id", "")
        ctx = r.get("context_type", "")
        cond_group = None
        for group, ctx_set in _CELLS.items():
            if ctx in ctx_set:
                cond_group = group
                break
        if cond_group is None:
            return False
        return (model_key, scenario_id, cond_group) in matched_set

    tagged = [{**r, "_matched": is_matched(r)} for r in db.records]

    models = sorted({r["_model"] for r in tagged})
    print(f"Models: {len(models)}")

    total_recs = len(tagged)
    matched_recs = sum(1 for r in tagged if r["_matched"])
    print(f"Records: {total_recs} total, {matched_recs} matched ({matched_recs/total_recs:.1%})\n")
    if matched_recs == 0:
        # Every clarity-matched number below would be vacuous. Usually the join
        # found nothing: model_key in the annotation file has to match _model in
        # the results dir, and models with no artifact-rating run never match at all.
        annotated = sorted({k[1] for k in clarity_raw})
        raise SystemExit(
            f"No records matched the clarity subset. The annotation file covers {len(annotated)} model(s):\n"
            f"  {annotated}\n"
            f"but the results dir holds {len(models)} model(s):\n"
            f"  {models}\n"
            "Check that artifact-rating was aggregated for these models and that the keys agree."
        )

    # Compute gaps per model and per channel
    all_rows: list[dict] = []

    print(
        f"{'Model':<42}  {'channel':>8}  {'gap_full':>9}  {'gap_matched':>12}  "
        f"{'n_full':>7}  {'n_matched':>10}  verdict"
    )
    print("-" * 110)

    any_vanished = False
    any_persisted = False

    for m in models:
        mrecs = [r for r in tagged if r.get("_model") == m]
        short = m.split("_")[-1] if "_" in m else m

        for exp_group, imp_group, channel in _CHANNEL_PAIRS:
            exp_ctx = _CELLS[exp_group]
            imp_ctx = _CELLS[imp_group]

            # Full population
            exp_full = [r for r in mrecs if r.get("context_type") in exp_ctx]
            imp_full = [r for r in mrecs if r.get("context_type") in imp_ctx]
            vcr_exp_full = _vcr(exp_full)
            vcr_imp_full = _vcr(imp_full)
            gap_full = (
                (vcr_exp_full - vcr_imp_full) if vcr_exp_full is not None and vcr_imp_full is not None else float("nan")
            )

            # Clarity-matched subset
            exp_match = [r for r in exp_full if r["_matched"]]
            imp_match = [r for r in imp_full if r["_matched"]]
            vcr_exp_match = _vcr(exp_match)
            vcr_imp_match = _vcr(imp_match)
            gap_match = (
                (vcr_exp_match - vcr_imp_match)
                if vcr_exp_match is not None and vcr_imp_match is not None
                else float("nan")
            )

            # Verdict: does the gap persist or vanish?
            if not np.isnan(gap_full) and gap_full != 0 and not np.isnan(gap_match):
                retention = gap_match / gap_full
                if retention < VANISH_THRESHOLD:
                    verdict = "VANISHED"
                    any_vanished = True
                else:
                    verdict = "persists"
                    any_persisted = True
            else:
                retention = float("nan")
                verdict = "insufficient data"

            n_full_exp = sum(1 for r in exp_full if (r.get("judge") or {}).get(_COND_KEY))
            n_match_exp = sum(1 for r in exp_match if (r.get("judge") or {}).get(_COND_KEY))

            print(
                f"{short:<42}  {channel:>8}  {gap_full:>9.3f}  {gap_match:>12.3f}  "
                f"{n_full_exp:>7}  {n_match_exp:>10}  {verdict}"
            )
            all_rows.append(
                {
                    "model": m,
                    "channel": channel,
                    "gap_full": gap_full,
                    "gap_matched": gap_match,
                    "retention_ratio": retention,
                    "n_full_exp_cond": n_full_exp,
                    "n_matched_exp_cond": n_match_exp,
                    "verdict": verdict,
                }
            )

    # Pooled across all models
    print("\n  Pooled:")
    for exp_group, imp_group, channel in _CHANNEL_PAIRS:
        exp_ctx = _CELLS[exp_group]
        imp_ctx = _CELLS[imp_group]

        exp_full = [r for r in tagged if r.get("context_type") in exp_ctx]
        imp_full = [r for r in tagged if r.get("context_type") in imp_ctx]
        exp_match = [r for r in exp_full if r["_matched"]]
        imp_match = [r for r in imp_full if r["_matched"]]

        gap_full = None
        gap_match = None
        vcr_exp_f = _vcr_ci(exp_full)
        vcr_imp_f = _vcr_ci(imp_full)
        if vcr_exp_f and vcr_imp_f:
            gap_full = vcr_exp_f[0] - vcr_imp_f[0]

        vcr_exp_m = _vcr_ci(exp_match)
        vcr_imp_m = _vcr_ci(imp_match)
        if vcr_exp_m and vcr_imp_m:
            gap_match = vcr_exp_m[0] - vcr_imp_m[0]

        retention = (gap_match / gap_full) if gap_full and gap_match and gap_full != 0 else float("nan")
        # A gap is None when its subset is empty — routine for a channel with no
        # clarity-matched rows, so it must not be formatted as a float.
        print(
            f"  {channel}: gap_full={_fmt_gap(gap_full)}  gap_matched={_fmt_gap(gap_match)}  "
            f"retention={retention:.1%}  "
            f"n_full={len(exp_full)}  n_matched={len(exp_match)}"
        )

    save_table(args.figures_dir / "h6_clarity_matched_gap.csv", all_rows)

    # --- Per-scenario CSV for scatter plot ---
    by_m_s_ctx: dict[tuple, list] = defaultdict(list)
    for r in tagged:
        by_m_s_ctx[(r.get("_model", ""), r.get("scenario_id", ""), r.get("context_type", ""))].append(r)

    per_scenario_rows: list[dict] = []
    all_sids = sorted({r.get("scenario_id", "") for r in tagged if r.get("scenario_id")})

    for m in models:
        model_key = m.replace("/", "_")
        for exp_group, imp_group, channel in _CHANNEL_PAIRS:
            for sid in all_sids:
                exp_recs: list[dict] = []
                for ctx in _CELLS[exp_group]:
                    exp_recs.extend(by_m_s_ctx.get((m, sid, ctx), []))
                imp_recs: list[dict] = []
                for ctx in _CELLS[imp_group]:
                    imp_recs.extend(by_m_s_ctx.get((m, sid, ctx), []))

                vcr_exp = _vcr(exp_recs)
                vcr_imp = _vcr(imp_recs)
                if vcr_exp is None or vcr_imp is None:
                    continue

                c_exp = scenario_clarity.get((model_key, sid, exp_group))
                c_imp = scenario_clarity.get((model_key, sid, imp_group))
                if c_exp is None or c_imp is None:
                    continue

                uar_exp = _uar(exp_recs)
                uar_imp = _uar(imp_recs)

                per_scenario_rows.append(
                    {
                        "model": m,
                        "scenario_id": sid,
                        "channel": channel,
                        "clarity_exp": c_exp,
                        "clarity_imp": c_imp,
                        "clarity_diff": c_exp - c_imp,
                        "vcr_exp": vcr_exp,
                        "vcr_imp": vcr_imp,
                        "vcr_gap": vcr_exp - vcr_imp,
                        "uar_exp": uar_exp if uar_exp is not None else float("nan"),
                        "uar_imp": uar_imp if uar_imp is not None else float("nan"),
                        "uar_gap": (uar_exp - uar_imp) if uar_exp is not None and uar_imp is not None else float("nan"),
                    }
                )

    save_table(args.figures_dir / "h6_per_scenario.csv", per_scenario_rows)
    print(f"\nPer-scenario: {len(per_scenario_rows)} points → h6_per_scenario.csv")

    # --- Verdict ---
    print("\n=== Final Verdict ===")
    if any_vanished and not any_persisted:
        print("  REFRAME — gap vanishes at matched clarity across all channels/models.")
        print("  Action: relabel 'explicitness' axis as 'signal strength'; Appendix A framing confirmed.")
    elif any_vanished and any_persisted:
        print("  MIXED — gap vanishes for some (model, channel) cells but persists for others.")
        print("  Action: report retention ratios; qualify universality of inference-burden claim.")
    else:
        print("  UPGRADE — gap persists at matched clarity.")
        print("  Action: inference-burden account survives; Appendix A separation is a strict upgrade.")


if __name__ == "__main__":
    main()
