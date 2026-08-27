"""H1 — channel asymmetry under default convention (C0).

Bar charts (x = model, grouped bars per model):
  2-bar version: user channel avg vs tool channel avg
  4-bar version: all 4 channels separately
Both versions generated for L3, the verbalized-commitment label the paper reports.

Color = family; shade within family = lighter (smallest) → full color (largest).
Models ordered family by family (FAMILY_ORDER), ascending params within.
Effort variants are pooled into one entry per base model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from scripts.plots import _eval_aware_filter as eval_aware
from scripts.plots.plot_register_matched_dumbbell import paired_delta
from src.results.db import ResultsDB
from src.utils.plotting import (
    CELL_CONTEXT_TYPES,
    CELL_DISPLAY,
    CELLS_4,
    DIR_FAMILY,
    MODEL_LABEL,
    VERBALIZED_COMMITMENT_RATE_LABEL,
    assign_model_colors,
    pool_effort_variants,
    save_figure,
    save_legend,
    save_table,
    select_models,
    setup_plot_style,
    tight_ylim,
    yerrs_from_cis,
)

FIGURES_DIR = Path("figures")

CELL_HATCH: dict[str, str] = {
    "tool_explicit": "///",
    "tool_implicit": "xxx",
    "user_explicit": "",
    "user_implicit": "..",
}
_COND = {"judge.answer_aligns_with_preference": True}


def cond_rate(db: ResultsDB, field: str) -> tuple[float, float, float]:
    """Conditional rate with 95% scenario-cluster bootstrap CI: (point, ci_lo, ci_hi)."""
    return db.filter(**_COND).cluster_mean_ci(field)


def load_clean() -> ResultsDB:
    return (
        ResultsDB.load_all(require_judged=True)
        .filter(_convention="C0")
        .filter(**{"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True})
        .filter_causal_dependent()
    )


def _build_bar_figure(
    models: list[str],
    rates: dict[str, dict[str, tuple[float, float, float]]],
    bar_configs: list[tuple[str, str, str]],  # (rate_key, hatch, display_label)
    model_colors: dict[str, tuple],
    y_label: str,
    fig_width: float,
    outpath: Path,
) -> None:
    n_bars = len(bar_configs)
    bar_w = min(0.75 / n_bars, 0.32)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w

    fig, ax = plt.subplots(figsize=(fig_width, 3.0))
    ax.patch.set_alpha(0)

    all_vals, all_lo, all_hi = [], [], []
    for xi, model in enumerate(models):
        color = model_colors.get(model, (0.5, 0.5, 0.5, 1.0))
        for offset, (rate_key, hatch, _) in zip(offsets, bar_configs):
            rate, lo, hi = rates.get(model, {}).get(rate_key, (0.0, 0.0, 0.0))
            ax.bar(
                xi + offset,
                rate,
                bar_w,
                color=color,
                hatch=hatch,
                yerr=yerrs_from_cis([rate], [lo], [hi]),
                capsize=2,
                error_kw={"ecolor": "black", "lw": 0.8},
                zorder=3,
            )
            all_vals.append(rate)
            all_lo.append(lo)
            all_hi.append(hi)

    # Family separator lines
    prev_fam = None
    for xi, model in enumerate(models):
        fam = DIR_FAMILY.get(model, "Other")
        if prev_fam is not None and fam != prev_fam:
            ax.axvline(xi - 0.5, color="grey", lw=0.8, ls=":", alpha=0.6, zorder=1)
        prev_fam = fam

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=7.5)
    ax.tick_params(axis="both", length=0)
    ax.set_ylabel(y_label)
    # tight_ylim uses upper-CI half-widths so the axis respects the visible bars
    half = [max(hi - v, v - lo) for v, lo, hi in zip(all_vals, all_lo, all_hi)]
    ax.set_ylim(*tight_ylim(all_vals, half, zero_floor=True))
    ax.set_xlim(-0.6, len(models) - 0.4)
    fig.tight_layout()
    save_figure(fig, outpath)

    # Companion CSV: one row per (model, bar).
    table_rows: list[dict] = []
    for model in models:
        for rate_key, _hatch, label in bar_configs:
            rate, lo, hi = rates.get(model, {}).get(rate_key, (0.0, 0.0, 0.0))
            table_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABEL.get(model, model),
                    "bar": label,
                    "rate_key": rate_key,
                    "rate": rate,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    save_table(outpath.with_suffix(".csv"), table_rows)

    # Legend: channel hatch patches
    channel_handles = [
        mpatches.Patch(facecolor="#cccccc", hatch=h, edgecolor="black", label=lbl) for _, h, lbl in bar_configs
    ]
    all_handles = channel_handles
    save_legend(
        all_handles,
        [h.get_label() for h in all_handles],
        outpath.with_name(outpath.stem + "_legend.svg"),
        ncol=len(all_handles),
    )
    print(f"Saved {outpath}")


def _save_middle_comparison(db: ResultsDB, models: list[str], eval_suffix: str) -> None:
    """UserImplicit minus ToolExplicit per model, with a paired interval.

    The bar CSV carries only marginal per-cell CIs, and the middle-comparison sentence is
    about the difference. Overlapping marginals are a conservative stand-in for this, not
    a test of it, so the two readings are kept in separate files on purpose.
    """
    rows = []
    for model in models:
        mdb = db.filter(_model=model)
        tool_explicit = mdb.filter_in("context_type", CELL_CONTEXT_TYPES["tool_explicit"])
        user_implicit = mdb.filter_in("context_type", CELL_CONTEXT_TYPES["user_implicit"])
        point, lo, hi = paired_delta(tool_explicit, user_implicit, "verbalized")
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABEL.get(model, model),
                "user_implicit": cond_rate(user_implicit, "judge.reasoning_tailoring_explicit")[0],
                "tool_explicit": cond_rate(tool_explicit, "judge.reasoning_tailoring_explicit")[0],
                "delta": point,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "excludes_zero": not (lo <= 0 <= hi),
            }
        )
    out = FIGURES_DIR / f"h1_middle_comparison_l3{eval_suffix}.csv"
    save_table(out, rows)
    print(f"Saved {out}")


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
    db = pool_effort_variants(db)

    models = select_models({r["_model"] for r in db.records})
    model_colors = assign_model_colors(models)

    # Build per-cell rates
    rates: dict[str, dict[str, tuple[float, float]]] = {}
    user_ctxs = CELL_CONTEXT_TYPES["user_explicit"] + CELL_CONTEXT_TYPES["user_implicit"]
    tool_ctxs = CELL_CONTEXT_TYPES["tool_explicit"] + CELL_CONTEXT_TYPES["tool_implicit"]
    for model in models:
        mdb = db.filter(_model=model)
        r: dict[str, tuple[float, float]] = {}
        for cell in CELLS_4:
            cdb = mdb.filter_in("context_type", CELL_CONTEXT_TYPES[cell])
            r[f"{cell}_l3"] = cond_rate(cdb, "judge.reasoning_tailoring_explicit")
        user_db = mdb.filter_in("context_type", user_ctxs)
        tool_db = mdb.filter_in("context_type", tool_ctxs)
        r["user_l3"] = cond_rate(user_db, "judge.reasoning_tailoring_explicit")
        r["tool_l3"] = cond_rate(tool_db, "judge.reasoning_tailoring_explicit")
        rates[model] = r

    l3_2bar = [("user_l3", "", "user channel"), ("tool_l3", "///", "tool channel")]
    l3_4bar = [(f"{cell}_l3", CELL_HATCH[cell], CELL_DISPLAY[cell]) for cell in CELLS_4]

    l3_label = VERBALIZED_COMMITMENT_RATE_LABEL

    _build_bar_figure(
        models,
        rates,
        l3_2bar,
        model_colors,
        l3_label,
        6.5,
        FIGURES_DIR / f"h1_channel_rates_l3_2bar{eval_suffix}.svg",
    )
    _build_bar_figure(
        models,
        rates,
        l3_4bar,
        model_colors,
        l3_label,
        8.5,
        FIGURES_DIR / f"h1_channel_rates_l3_4bar{eval_suffix}.svg",
    )

    _save_middle_comparison(db, models, eval_suffix)


if __name__ == "__main__":
    main()
