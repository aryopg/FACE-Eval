"""Inter-judge agreement scatter, judge A vs judge B. One point per model.

--metric picks what is plotted:
  vcr  verbalized commitment (reasoning_tailoring_explicit) among the rows whose
       answer went the user's way — the "given aligned" conditional the H1/H3/H4
       figures report.
  uar  unverbalized adoption (aligned and NOT tailoring_explicit) over every row,
       the headline rate. It is a conjunction across both judges, so it agrees
       less well than either part alone; that is a result, not an artefact.
(The marginals and the +committed variant are in the CSV too, unplotted.)

The identity line is the reference: a point above it means judge B marks it more
often than the pre-registered judge for that model. Spearman rho is annotated but
deliberately secondary — it measures only whether the two judges *rank* models
alike, which a wide spread of rates across models inflates almost by
construction.
Gwet's AC1 carries the pooled agreement. The number that actually moves a
published rate is the mean signed offset, which stays in the summary CSV rather
than on the figure, where the gap to y = x already shows it.

Only the highest-effort variant of each effort-swept model is drawn, one point
per model. Note what that costs: the dropped gpt-oss low-effort runs were the
only low-prevalence points in the figure, and they are where AC1 and kappa
diverge hardest, so the figure no longer carries its own case for preferring
AC1. Every annotated statistic is computed over the models drawn.

Input: figures/inter_judge_agreement__<judge>__<convention>.csv
(analyze_inter_judge_agreement.py, run at the same --convention)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.utils.plotting import (
    DIR_FAMILY,
    FAMILY_COLORS,
    FAMILY_ORDER,
    MODEL_LABEL_INLINE,
    assign_model_colors,
    highest_effort_variants,
    save_figure,
    save_legend,
    save_table,
    select_models,
    setup_plot_style,
)

X_LABEL = "Judge 1 (Claude Haiku 4.5)"
Y_LABEL = "Judge 2 (GPT-5.6-Luna)"
# Field, panel name and axis window per metric. The windows are zoomed on where
# the highest-effort models actually sit; any point outside is clipped away
# silently, so the draw checks and says so. Unverbalized adoption is a rate over
# every row rather than over the aligned subset, so it sits far lower.
METRICS = {
    "vcr": ("verbalized_commitment_given_aligned", "verbalized commitment given aligned", (0.5, 1.0)),
    "uar": ("unverbalized_adoption", "unverbalized adoption", (0.0, 0.5)),
}


def _num(value: str) -> float | None:
    return float(value) if value not in ("", "None", None) else None


def load_rows(csv_path: Path, field: str) -> dict[str, dict]:
    """Per-model rows for one judge field, keyed by model dir name."""
    out: dict[str, dict] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row["field"] != field or not row["group"].startswith("model="):
                continue
            model = row["group"].removeprefix("model=")
            a, b = _num(row["pos_rate_judge_a"]), _num(row["pos_rate_judge_b"])
            if a is None or b is None:
                continue
            out[model] = {"a": a, "b": b, "n": int(row["n"]), "raw": _num(row["raw_agreement"])}
    return out


def _ci_for(csv_path: Path, field: str, n_drawn: int) -> tuple[float | None, float | None]:
    """The bootstrap CI of whichever group was computed over the rows drawn.

    Matched on n, not on the group name: a CI belongs to this figure only if the
    analysis bootstrapped it over the same rows. "overall" covers every model,
    "highest_effort" covers the one-point-per-model subset; an older CSV predates
    the second group and simply yields no CI.
    """
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row["field"] == field and row["group"] in ("overall", "highest_effort") and int(row["n"]) == n_drawn:
                return _num(row["gwet_ac1_ci_lo"]), _num(row["gwet_ac1_ci_hi"])
    return None, None


def _pooled_ac1(data: dict[str, dict]) -> float:
    """Gwet's AC1 over the models drawn, from their per-model pooled quantities.

    Exact, not an approximation: AC1 is a function of the pooled observed
    agreement and the two pooled marginals, so n-weighted means of the per-model
    rows reproduce the coefficient the analysis script computes row by row.
    """
    w = np.array([d["n"] for d in data.values()], dtype=float)
    w /= w.sum()
    p_observed = float(w @ np.array([d["raw"] for d in data.values()]))
    pi = float(w @ np.array([(d["a"] + d["b"]) / 2 for d in data.values()]))
    p_expected = 2 * pi * (1 - pi)
    return (p_observed - p_expected) / (1 - p_expected)


def _draw(
    ax,
    data: dict[str, dict],
    colors: dict[str, tuple],
    ci: tuple[float | None, float | None],
    panel_name: str,
    axis_limits: tuple[float, float],
) -> dict:
    xs = np.array([d["a"] for d in data.values()])
    ys = np.array([d["b"] for d in data.values()])

    lo_ax, hi_ax = axis_limits
    clipped = [m for m, d in data.items() if not (lo_ax <= d["a"] <= hi_ax and lo_ax <= d["b"] <= hi_ax)]
    if clipped:
        print(f"  WARNING: outside {axis_limits}, not drawn: {clipped}")

    ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="grey", alpha=0.7, zorder=1)
    for model, d in data.items():
        ax.scatter(d["a"], d["b"], s=42, color=colors.get(model, (0.5, 0.5, 0.5)), edgecolor="black", lw=0.5, zorder=3)

    # rho is undefined if either judge gave the same rate to every model.
    rho = float(spearmanr(xs, ys).statistic) if len(xs) > 1 and xs.std() > 0 and ys.std() > 0 else float("nan")
    bias = float(np.mean(ys - xs))
    ac1 = _pooled_ac1(data)
    lo, hi = ci
    # Bias stays in the summary CSV but off the figure: "bias" alone does not say
    # whose rate is higher, and the offset is already visible as the gap to y = x.
    ac1_str = f"AC1 = {ac1:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else "")
    ax.text(
        0.03,
        0.97,
        f"{ac1_str}\nSpearman $\\rho$ = {rho:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
    )

    ax.patch.set_alpha(0)
    ax.set_xlim(*axis_limits)
    ax.set_ylim(*axis_limits)
    ax.set_aspect("equal")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    return {"panel": panel_name, "n_models": len(data), "spearman_rho": rho, "bias_b_minus_a": bias, "pooled_ac1": ac1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inter-judge agreement scatter, judge A vs judge B")
    parser.add_argument("--metric", choices=sorted(METRICS), default="vcr")
    parser.add_argument("--judge-model", default="gpt-5.6-luna")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--convention",
        default="ALL",
        help="Which analyze run to plot; must match the --convention it was run at",
    )
    args = parser.parse_args()

    stem = f'{args.judge_model.replace("/", "_")}__{args.convention}'
    csv_path = args.figures_dir / f"inter_judge_agreement__{stem}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Run analyze_inter_judge_agreement.py at --convention {args.convention} first: {csv_path}"
        )

    field, panel_name, axis_limits = METRICS[args.metric]
    setup_plot_style(wide=True)
    data = load_rows(csv_path, field)
    if not data:
        raise SystemExit(f"No rows for field {field!r} in {csv_path} — re-run analyze_inter_judge_agreement.py")
    models = highest_effort_variants(select_models(data))
    if not models:
        raise SystemExit(f"No registered models in {csv_path}")
    colors = assign_model_colors(models)

    ci = _ci_for(csv_path, field, sum(data[m]["n"] for m in models))
    if ci == (None, None):
        print("  No CI: re-run analyze_inter_judge_agreement.py to bootstrap the highest_effort group")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    summary = [_draw(ax, {m: data[m] for m in models}, colors, ci, panel_name, axis_limits)]
    fig.tight_layout()

    out = args.figures_dir / f"inter_judge_agreement_{args.metric}__{stem}.svg"
    save_figure(fig, out)
    save_table(out.with_suffix(".csv"), summary)

    drawn = {DIR_FAMILY[m] for m in models}
    families = [f for f in FAMILY_ORDER if f in drawn]
    handles = [
        mlines.Line2D([], [], marker="o", ls="", markerfacecolor=FAMILY_COLORS[f], markeredgecolor="black", label=f)
        for f in families
    ]
    save_legend(handles, [h.get_label() for h in handles], out.with_name(out.stem + "_legend.svg"), ncol=4)

    for s in summary:
        print(f"  {s['panel']:<32} n={s['n_models']:>3}  rho={s['spearman_rho']:.3f}  bias={s['bias_b_minus_a']:+.3f}")
    print(f"Saved {out}")
    print(f"Models plotted: {[MODEL_LABEL_INLINE.get(m, m) for m in models]}")


if __name__ == "__main__":
    main()
