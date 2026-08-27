"""H5 — System-prompt nulls: joint bootstrap delta-of-deltas.

For each convention contrast (C0 vs C3, C0 vs MC0), compute the CI on:
  delta_of_deltas = (VCR_user_C0 - VCR_tool_C0) - (VCR_user_Cx - VCR_tool_Cx)

Using a JOINT bootstrap over scenario_id: scenario_ids are resampled once per
iteration and both gap_C0 and gap_Cx are computed from the same resample. This
preserves the C0/Cx correlation and avoids inflated CIs.

Metric: VCR = reasoning_tailoring_explicit | answer_aligns_with_preference,
causal-dependent, parse_ok rows — the same L3 commitment label every headline
figure uses. This read acknowledgment (L1) until 2026-08-12.

Results are reported per model and per family.

Falsification rules (pre-registered):
  - If any family's delta-of-deltas CI excludes zero: convention matters;
    Qwen-only claim is wrong → rewrite Takeaway 4.
  - If CIs include effects of ±0.15: too wide to claim null → report as
    "we do not detect" with MDE stated.
  - Otherwise: null is defensible.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.results.db import ResultsDB
from src.utils.plotting import DIR_FAMILY as _DIR_FAMILY
from src.utils.plotting import pool_effort_variants as _pool_effort_variants
from src.utils.plotting import save_table, short_model_name

MDE_THRESHOLD = 0.15  # half-CI exceeding this → cannot claim null

_VCR_FIELD = "judge.reasoning_tailoring_explicit"
_COND_FIELD = "judge.answer_aligns_with_preference"
_PARSE_OK = {"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True}

_USER_CTXS = frozenset({"user_turn", "user_turn_structured", "user_turn_implicit"})
_TOOL_CTXS = frozenset({"explicit", "implicit"})

CONTRASTS = [("C0", "C3"), ("C0", "MC0")]


def _gap(records: list[dict]) -> float:
    """VCR(user) - VCR(tool) on the given records."""
    cond_recs = [r for r in records if (r.get("judge") or {}).get("answer_aligns_with_preference")]
    user_cond = [r for r in cond_recs if r.get("context_type") in _USER_CTXS]
    tool_cond = [r for r in cond_recs if r.get("context_type") in _TOOL_CTXS]
    f_user = (
        sum(1 for r in user_cond if (r.get("judge") or {}).get("reasoning_tailoring_explicit")) / len(user_cond)
        if user_cond
        else float("nan")
    )
    f_tool = (
        sum(1 for r in tool_cond if (r.get("judge") or {}).get("reasoning_tailoring_explicit")) / len(tool_cond)
        if tool_cond
        else float("nan")
    )
    if np.isnan(f_user) or np.isnan(f_tool):
        return float("nan")
    return f_user - f_tool


def _precompute_scenario_stats(
    clusters: dict[str, list[dict]],
    scenario_order: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (u_vcr, u_cond, t_vcr, t_cond) arrays aligned on scenario_order.

    Each element i accumulates counts over all records for scenario_order[i].
    u = user channel, t = tool channel.
    vcr = both aligns AND acknowledges; cond = aligns only.
    """
    n = len(scenario_order)
    sid_idx = {sid: i for i, sid in enumerate(scenario_order)}
    u_vcr = np.zeros(n, dtype=float)
    u_cond = np.zeros(n, dtype=float)
    t_vcr = np.zeros(n, dtype=float)
    t_cond = np.zeros(n, dtype=float)
    for sid, recs in clusters.items():
        i = sid_idx.get(sid)
        if i is None:
            continue
        for r in recs:
            ctx = r.get("context_type")
            j = r.get("judge") or {}
            aligns = j.get("answer_aligns_with_preference")
            commits = j.get("reasoning_tailoring_explicit")
            if ctx in _USER_CTXS:
                if aligns:
                    u_cond[i] += 1
                    if commits:
                        u_vcr[i] += 1
            elif ctx in _TOOL_CTXS:
                if aligns:
                    t_cond[i] += 1
                    if commits:
                        t_vcr[i] += 1
    return u_vcr, u_cond, t_vcr, t_cond


def _joint_bootstrap(
    clusters_c0: dict[str, list[dict]],
    clusters_cx: dict[str, list[dict]],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict | None:
    """Joint bootstrap over scenario_id for delta-of-deltas (vectorized).

    Precomputes per-scenario (n_vcr, n_cond) arrays for each (convention,
    channel), then resamples with a single (n_boot × n_scenarios) index
    matrix — no Python loop over bootstrap iterations.

    Returns dict with point, ci_lo, ci_hi, mde (CI half-width), n_scenarios.
    """
    common = sorted(set(clusters_c0.keys()) & set(clusters_cx.keys()))
    if len(common) < 4:
        return None

    u0_vcr, u0_cond, t0_vcr, t0_cond = _precompute_scenario_stats(clusters_c0, common)
    ux_vcr, ux_cond, tx_vcr, tx_cond = _precompute_scenario_stats(clusters_cx, common)

    def _gap_from_arrays(u_v, u_c, t_v, t_c) -> float:
        u_c_sum, t_c_sum = u_c.sum(), t_c.sum()
        if u_c_sum == 0 or t_c_sum == 0:
            return float("nan")
        return float(u_v.sum() / u_c_sum - t_v.sum() / t_c_sum)

    gap_c0 = _gap_from_arrays(u0_vcr, u0_cond, t0_vcr, t0_cond)
    gap_cx = _gap_from_arrays(ux_vcr, ux_cond, tx_vcr, tx_cond)
    if np.isnan(gap_c0) or np.isnan(gap_cx):
        return None
    point = gap_c0 - gap_cx

    # Vectorized bootstrap: one (n_boot × n) index matrix resamples all arrays
    rng = np.random.default_rng(seed)
    n = len(common)
    idx = rng.integers(0, n, size=(n_boot, n))  # (n_boot, n_scenarios)

    u0_v_b = u0_vcr[idx].sum(axis=1)
    u0_c_b = u0_cond[idx].sum(axis=1)
    t0_v_b = t0_vcr[idx].sum(axis=1)
    t0_c_b = t0_cond[idx].sum(axis=1)
    ux_v_b = ux_vcr[idx].sum(axis=1)
    ux_c_b = ux_cond[idx].sum(axis=1)
    tx_v_b = tx_vcr[idx].sum(axis=1)
    tx_c_b = tx_cond[idx].sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_u0 = np.where(u0_c_b > 0, u0_v_b / u0_c_b, np.nan)
        f_t0 = np.where(t0_c_b > 0, t0_v_b / t0_c_b, np.nan)
        f_ux = np.where(ux_c_b > 0, ux_v_b / ux_c_b, np.nan)
        f_tx = np.where(tx_c_b > 0, tx_v_b / tx_c_b, np.nan)

    boot_deltas = (f_u0 - f_t0) - (f_ux - f_tx)
    valid = ~np.isnan(boot_deltas)
    if valid.sum() < 10:
        return None
    boot_deltas = boot_deltas[valid]

    ci_lo = float(np.quantile(boot_deltas, 0.025))
    ci_hi = float(np.quantile(boot_deltas, 0.975))
    return {
        "point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mde": (ci_hi - ci_lo) / 2,
        "n_scenarios": len(common),
        "gap_c0": gap_c0,
        "gap_cx": gap_cx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H5 convention power analysis")
    parser.add_argument("--results-dir", default="results/agentic")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    db = _pool_effort_variants(
        ResultsDB.load_all(results_dir=args.results_dir, require_judged=True)
        .filter(**_PARSE_OK)
        .filter_causal_dependent()
    )

    models = sorted({r["_model"] for r in db.records})
    print(f"\n=== H5 Convention Power Analysis ({len(models)} models) ===")

    all_rows: list[dict] = []

    for conv_a, conv_b in CONTRASTS:
        print(f"\n--- Contrast: {conv_a} vs {conv_b} ---")
        print(
            f"{'Model':<45}  {'gap_c0':>7}  {'gap_cx':>7}  {'delta':>7}  "
            f"{'ci_lo':>7}  {'ci_hi':>7}  {'mde':>6}  {'n_scen':>7}  verdict"
        )
        print("-" * 120)

        family_clusters_c0: dict[str, dict[str, list[dict]]] = defaultdict(dict)
        family_clusters_cx: dict[str, dict[str, list[dict]]] = defaultdict(dict)

        for m in models:
            mdb = db.filter(_model=m)
            recs_c0 = mdb.filter(_convention=conv_a).records
            recs_cx = mdb.filter(_convention=conv_b).records

            if not recs_c0 or not recs_cx:
                continue

            # Per-model bootstrap
            clust_c0: dict[str, list[dict]] = defaultdict(list)
            for r in recs_c0:
                clust_c0[r.get("scenario_id", "")].append(r)
            clust_cx: dict[str, list[dict]] = defaultdict(list)
            for r in recs_cx:
                clust_cx[r.get("scenario_id", "")].append(r)

            result = _joint_bootstrap(clust_c0, clust_cx, n_boot=args.n_boot)
            if result is None:
                continue

            # Family accumulation for family-level bootstrap
            fam = _DIR_FAMILY.get(m, "Other")
            for sid, recs in clust_c0.items():
                family_clusters_c0[fam].setdefault(sid, []).extend(recs)
            for sid, recs in clust_cx.items():
                family_clusters_cx[fam].setdefault(sid, []).extend(recs)

            excludes_zero = result["ci_lo"] > 0 or result["ci_hi"] < 0
            mde_wide = result["mde"] > MDE_THRESHOLD
            if excludes_zero:
                verdict = "CONVENTION MATTERS"
            elif mde_wide:
                verdict = f"UNDERPOWERED (MDE={result['mde']:.2f})"
            else:
                verdict = "null defensible"

            short = short_model_name(m)
            print(
                f"{short:<45}  {result['gap_c0']:>7.3f}  {result['gap_cx']:>7.3f}  "
                f"{result['point']:>7.3f}  {result['ci_lo']:>7.3f}  {result['ci_hi']:>7.3f}  "
                f"{result['mde']:>6.3f}  {result['n_scenarios']:>7}  {verdict}"
            )
            all_rows.append(
                {
                    "contrast": f"{conv_a}_vs_{conv_b}",
                    "level": "model",
                    "model": m,
                    "family": fam,
                    **result,
                    "excludes_zero": excludes_zero,
                    "verdict": verdict,
                }
            )

        # Family-level bootstrap
        print("\n  Family-level:")
        families = sorted(set(_DIR_FAMILY.get(m, "Other") for m in models))
        for fam in families:
            fc0 = family_clusters_c0.get(fam, {})
            fcx = family_clusters_cx.get(fam, {})
            result = _joint_bootstrap(fc0, fcx, n_boot=args.n_boot)
            if result is None:
                print(f"  {fam}: insufficient data")
                continue

            excludes_zero = result["ci_lo"] > 0 or result["ci_hi"] < 0
            mde_wide = result["mde"] > MDE_THRESHOLD
            if excludes_zero:
                verdict = "CONVENTION MATTERS"
            elif mde_wide:
                verdict = f"UNDERPOWERED (MDE={result['mde']:.2f})"
            else:
                verdict = "null defensible"

            print(
                f"  {fam:<20}  delta={result['point']:.3f}  "
                f"CI=[{result['ci_lo']:.3f},{result['ci_hi']:.3f}]  "
                f"MDE={result['mde']:.3f}  n={result['n_scenarios']}  {verdict}"
            )
            all_rows.append(
                {
                    "contrast": f"{conv_a}_vs_{conv_b}",
                    "level": "family",
                    "model": fam,
                    "family": fam,
                    **result,
                    "excludes_zero": excludes_zero,
                    "verdict": verdict,
                }
            )

    save_table(args.figures_dir / "h5_convention_power.csv", all_rows)

    # --- Verdict ---
    print("\n=== Final Verdict ===")
    convention_matters = [r for r in all_rows if r.get("excludes_zero") and r.get("level") == "family"]
    underpowered = [r for r in all_rows if "UNDERPOWERED" in str(r.get("verdict", "")) and r.get("level") == "family"]

    if convention_matters:
        families = sorted({r["model"] for r in convention_matters})
        print(f"  FAIL — convention moves the gap for families: {families}")
        print("  Action: Qwen-only claim is wrong; rewrite Takeaway 4.")
    elif underpowered:
        families = sorted({r["model"] for r in underpowered})
        mdes = [r["mde"] for r in underpowered]
        print(f"  UNDERPOWERED — CIs too wide for families: {families}  (MDEs: {[f'{m:.2f}' for m in mdes]})")
        print("  Action: report as 'we do not detect' with MDE stated; cannot claim null.")
    else:
        print("  PASS — null is defensible for all families across all contrasts.")
        print("  Action: one robustness sentence; convention has no detectable effect.")


if __name__ == "__main__":
    main()
