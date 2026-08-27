"""H1 phase diagram — marginal sycophantic-alignment rate vs CoT verbalisation.

For each (model, cell), pooled over cued records (context_type != 'none'):
  X = P(Align_ans | cued)              — marginal sycophantic-alignment rate
  Y = P(CommitCoT | Align_ans)         — verbalized commitment (high = faithful)
      (L3 = tailoring_explicit).

The no-cue baseline is implicit: no_context records produce
aligns_with_preference=null (see plot_no_context_shift.py), so X reduces to the
fraction of cued answers that landed on the user's preferred side.

Reds-shaded background + dashed iso-curves both encode
  hidden_rate = X * (1 - Y) = P(Align_ans ∧ ¬CoT_signal | cued)
the marginal covert-alignment rate. Bottom-right = bad. Colorbar on the right
labels this metric ("Unverbalized Adoption Rate"). Layout: 2×2 facets, one per
cell. Color = family, lightness = model size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scripts.plots import _eval_aware_filter as eval_aware
from src.results.db import ResultsDB
from src.utils.plotting import (
    CELL_CONTEXT_TYPES,
    CELL_DISPLAY,
    CELLS_4,
    CUE_FOLLOWING_RATE_LABEL,
)
from src.utils.plotting import DIR_FAMILY as _DIR_FAMILY_BASE
from src.utils.plotting import (
    FAMILY_ORDER,
    FONT_SIZE_TITLE_WIDE,
    METRIC_LABEL,
    MODEL_LABEL_INLINE,
    MODEL_PARAMS,
    VERBALIZED_COMMITMENT_RATE_LABEL,
    assign_model_colors,
    pool_effort_variants,
    save_figure,
    save_legend,
    save_table,
    select_models,
    setup_plot_style,
)


def _conditional_faithfulness_label(metric_key: str) -> str:
    """P(CoT_metric | aligned) — good-direction Y label."""
    bare = lambda s: s.strip("$")  # noqa: E731
    cond = bare(METRIC_LABEL["answer_alignment"])
    return rf"$P({bare(METRIC_LABEL[metric_key])} \mid {cond})$"


FIGURES_DIR = Path("figures")

_COND = {"judge.answer_aligns_with_preference": True}

_DIR_FAMILY: dict[str, str] = _DIR_FAMILY_BASE

XLIM = (0.0, 1.0)
YLIM = (0.0, 1.0)


# Colorbar (and background) saturate at this hidden-rate value. Slightly above
# the empirical maximum so the colorbar remains informative for outliers.
CBAR_VMAX = 0.30


def load_clean() -> ResultsDB:
    return pool_effort_variants(
        ResultsDB.load_all(require_judged=True)
        .filter(_convention="C0")
        .filter(**{"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True})
        .filter_causal_dependent()
    )


def _draw_shaded_background(ax) -> None:
    """Reds heatmap of hidden_rate = X*(1-Y), with bottom-right = worst."""
    nx, ny = 300, 300
    x = np.linspace(*XLIM, nx)
    y = np.linspace(*YLIM, ny)
    X, Y = np.meshgrid(x, y)
    hidden = X * (1.0 - Y)
    ax.imshow(
        hidden,
        origin="lower",
        extent=(XLIM[0], XLIM[1], YLIM[0], YLIM[1]),
        cmap="Reds",
        vmin=0.0,
        vmax=CBAR_VMAX,
        alpha=0.28,
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )


def _draw_isocurves(ax) -> None:
    """Iso-lines for hidden_rate = X*(1-Y); curve: Y = 1 - c/X (emerges from (c, 0))."""
    _draw_shaded_background(ax)
    iso_values = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    cmap = sns.color_palette("flare", as_cmap=True)
    norm = mcolors.Normalize(vmin=0.0, vmax=CBAR_VMAX)
    x_iso = np.linspace(max(XLIM[0], 1e-3), XLIM[1], 400)
    for c in iso_values:
        color = cmap(min(1.0, norm(c) + 0.35))
        y_iso = 1.0 - c / x_iso
        mask = (y_iso >= YLIM[0]) & (y_iso <= YLIM[1]) & (x_iso >= c)
        if not mask.any():
            continue
        ax.plot(x_iso[mask], y_iso[mask], color=color, lw=1.0, ls="--", alpha=0.9, zorder=1)


def _compute_points(
    db: ResultsDB, models: list[str], y_field: str, n_boot: int = 2000, seed: int = 42
) -> dict[str, dict[str, dict]]:
    """Marginal sycophantic-alignment estimator with scenario-cluster bootstrap 95% CI.

    For each scenario_id, count (n_total, n_align, n_align_with_cot)
    over all cued records. Pool across scenarios for the point estimate; bootstrap
    over scenarios for percentile CIs on X and Y separately (each captures both
    within-scenario seed variance and between-scenario variance).

    X = P(Align_ans | cued)
    Y = P(CoT_signal | Align_ans)
    """
    cot_field = y_field.split(".", 1)[1] if y_field.startswith("judge.") else y_field
    out: dict[str, dict[str, dict]] = {cell: {} for cell in CELLS_4}
    rng_master = np.random.default_rng(seed)
    for m in models:
        for cell in CELLS_4:
            recs = db.filter(_model=m).filter_in("context_type", CELL_CONTEXT_TYPES[cell]).records
            by_sid: dict[str, list[dict]] = {}
            for r in recs:
                by_sid.setdefault(r["scenario_id"], []).append(r)

            per_scenario: list[tuple[int, int, int]] = []
            for scenario_recs in by_sid.values():
                n_t = len(scenario_recs)
                n_ac = 0
                n_ac_cot = 0
                for r in scenario_recs:
                    rj = r["judge"]
                    if rj.get("answer_aligns_with_preference") is True:
                        n_ac += 1
                        if rj.get(cot_field) is True:
                            n_ac_cot += 1
                per_scenario.append((n_t, n_ac, n_ac_cot))
            if not per_scenario:
                continue
            stats = np.array(per_scenario, dtype=float)
            sum_t, sum_ac, sum_ac_cot = stats.sum(axis=0)
            if sum_t == 0 or sum_ac == 0:
                continue
            x = sum_ac / sum_t
            y = sum_ac_cot / sum_ac

            n_s = len(stats)
            if n_boot >= 2 and n_s >= 2:
                rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                idx = rng.integers(0, n_s, size=(n_boot, n_s))
                b_t = stats[idx, 0].sum(axis=1)
                b_ac = stats[idx, 1].sum(axis=1)
                b_ac_cot = stats[idx, 2].sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    b_x = np.where(b_t > 0, b_ac / b_t, 0.0)
                    b_y = np.where(b_ac > 0, b_ac_cot / b_ac, 0.0)
                x_lo, x_hi = float(np.quantile(b_x, 0.025)), float(np.quantile(b_x, 0.975))
                y_lo, y_hi = float(np.quantile(b_y, 0.025)), float(np.quantile(b_y, 0.975))
            else:
                x_lo = x_hi = x
                y_lo = y_hi = y
            out[cell][m] = {"x": x, "y": y, "x_lo": x_lo, "x_hi": x_hi, "y_lo": y_lo, "y_hi": y_hi}
    return out


def _annotate_selected(ax, ch_points: dict[str, dict]) -> None:
    """Annotate top-2 by hidden_rate = X*(1-Y) (worst) plus the cleanest
    high-commitment point (highest Y among models with X > median)."""
    if not ch_points:
        return
    items = [(m, p["x"], p["y"], p["x"] * (1.0 - p["y"])) for m, p in ch_points.items()]
    items.sort(key=lambda t: t[3], reverse=True)
    worst = items[:2]
    # cleanest: among the top half by X, the one with highest Y (most faithful CoT)
    high_x = sorted(items, key=lambda t: t[1], reverse=True)[: max(1, len(items) // 2)]
    cleanest = max(high_x, key=lambda t: t[2]) if high_x else None
    targets = list(worst)
    if cleanest and cleanest not in worst:
        targets.append(cleanest)
    for m, x, y, _ in targets:
        ax.annotate(
            MODEL_LABEL_INLINE.get(m, m),
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=6.5,
            color="black",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="#EFEFEAFF", ec="none", alpha=0.6),
        )


def _build_figure(
    db: ResultsDB,
    models: list[str],
    y_field: str,
    y_label: str,
    model_colors: dict[str, tuple],
    cbar_label: str,
) -> tuple[plt.Figure, dict[str, dict[str, dict]]]:
    points = _compute_points(db, models, y_field)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.0), sharex=True, sharey=True)
    for ax, cell in zip(axes.flatten(), CELLS_4):
        _draw_isocurves(ax)
        cell_points = points[cell]
        for m, p in cell_points.items():
            color = model_colors.get(m, (0.5, 0.5, 0.5))
            xerr = [[max(0.0, p["x"] - p["x_lo"])], [max(0.0, p["x_hi"] - p["x"])]]
            yerr = [[max(0.0, p["y"] - p["y_lo"])], [max(0.0, p["y_hi"] - p["y"])]]
            ax.errorbar(
                p["x"],
                p["y"],
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                color=color,
                ms=7,
                lw=0.8,
                capsize=0,
                alpha=0.9,
                zorder=3,
                markeredgecolor="black",
                markeredgewidth=1.5,
                ecolor="black",
                elinewidth=0.8,
            )
        # _annotate_selected(ax, cell_points)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.tick_params(axis="both", length=0)
        ax.patch.set_alpha(0)
        ax.set_title(CELL_DISPLAY[cell])

    fig.supxlabel(CUE_FOLLOWING_RATE_LABEL, y=0.01, fontsize=FONT_SIZE_TITLE_WIDE)
    fig.supylabel(y_label, x=0.02, fontsize=FONT_SIZE_TITLE_WIDE)

    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.5, rect=(0.01, -0.01, 0.95, 1.0))
    cbar_ax = fig.add_axes([0.96, 0.10, 0.015, 0.80])
    sm = plt.cm.ScalarMappable(
        cmap=sns.color_palette("flare", as_cmap=True), norm=mcolors.Normalize(vmin=0.0, vmax=CBAR_VMAX)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label + r" ($\downarrow$)")
    cbar.ax.tick_params(labelsize=9, length=0)
    cbar.outline.set_linewidth(1.5)
    cbar.outline.set_edgecolor("black")
    return fig, points


def main() -> None:
    global FIGURES_DIR
    parser = argparse.ArgumentParser()
    eval_aware.add_flag(parser)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    FIGURES_DIR = args.figures_dir
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    setup_plot_style(wide=True)
    plt.rcParams["axes.facecolor"] = "none"

    db, eval_suffix = eval_aware.apply(load_clean(), args)
    models = select_models({r["_model"] for r in db.records})
    model_colors = assign_model_colors(models)

    for suffix, y_field, y_metric_key, cbar_label in [
        ("l3", "judge.reasoning_tailoring_explicit", "cot_commitment", "Unverbalized Adoption Rate"),
    ]:
        y_label = VERBALIZED_COMMITMENT_RATE_LABEL
        fig, points = _build_figure(db, models, y_field, y_label, model_colors, cbar_label)
        out = FIGURES_DIR / f"h1_phase_diagram_{suffix}{eval_suffix}.svg"
        save_figure(fig, out)
        rows = [
            {
                "metric": suffix.upper(),
                "cell": cell,
                "cell_label": CELL_DISPLAY[cell],
                "model": m,
                "x": p["x"],
                "y": p["y"],
                "x_ci_lo": p["x_lo"],
                "x_ci_hi": p["x_hi"],
                "y_ci_lo": p["y_lo"],
                "y_ci_hi": p["y_hi"],
            }
            for cell, cp in points.items()
            for m, p in cp.items()
        ]
        save_table(out.with_suffix(".csv"), rows)

    # Legend: per-model patches grouped by family, ordered light→dark within family
    by_family: dict[str, list[tuple[int, str]]] = {}
    for m in models:
        fam = _DIR_FAMILY.get(m, "Other")
        by_family.setdefault(fam, []).append((MODEL_PARAMS.get(m, 0), m))
    handles: list = []
    fam_order = [f for f in FAMILY_ORDER if f in by_family]
    for fam in fam_order:
        entries = sorted(by_family[fam])
        for _, m in entries:
            handles.append(
                mlines.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=model_colors[m],
                    markeredgecolor="black",
                    markeredgewidth=2,
                    markersize=8,
                    label=MODEL_LABEL_INLINE.get(m, m),
                )
            )
    save_legend(
        handles,
        [h.get_label() for h in handles],
        FIGURES_DIR / f"h1_phase_diagram{eval_suffix}_legend.svg",
        ncol=5,
    )
    print(f"Saved h1_phase_diagram_l3{eval_suffix}.svg, h1_phase_diagram_l1{eval_suffix}.svg in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
