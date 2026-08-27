"""Position-bias check: side-ID accuracy by which option is ground truth (appendix).

Side-by-side bar plot: for each model, side-ID rate on items where A=ground-truth
vs. items where B=ground-truth (pooled across all 5 conditions).

If |rate_A_half - rate_B_half| > 0.05 on any model, that model bar pair is
annotated with a warning marker (*).

Input: outputs/artifact_rating_aggregated.jsonl

Outputs:
  figures/position_bias.svg
  figures/position_bias.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting import (
    DIR_FAMILY,
    ERROR_KW_BAR,
    MODEL_LABEL,
    SIDE_ID_LABEL,
    save_figure,
    save_table,
    select_models,
    setup_plot_style,
)

FIGURES_DIR = Path("figures")


BIAS_THRESHOLD = 0.05
COLOR_A = "#4A90D9"  # steel blue
COLOR_B = "#E8907A"  # salmon

BAR_W = 0.35


_RNG = np.random.default_rng(42)
_N_BOOT = 1000


def _side_id_ci(items: list[dict]) -> tuple[float, float, float] | None:
    """Bootstrap 95% CI for side-ID accuracy on non-dropped items.

    Returns (rate, ci_lo, ci_hi) or None if no valid items.
    """
    valid = np.array([1 if r.get("majority_side_is_gt", False) else 0 for r in items if not r.get("dropped", False)])
    if len(valid) == 0:
        return None
    rate = float(valid.mean())
    boots = _RNG.choice(valid, size=(_N_BOOT, len(valid)), replace=True).mean(axis=1)
    return rate, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the position-bias check.")
    parser.add_argument("--agg-file", default="outputs/artifact_rating_aggregated.jsonl")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    FIGURES_DIR_OUT = Path(args.figures_dir)
    FIGURES_DIR_OUT.mkdir(parents=True, exist_ok=True)

    setup_plot_style(wide=True)
    plt.rcParams["axes.facecolor"] = "none"

    # Load aggregated JSONL
    by_model: dict[str, list[dict]] = {}
    with open(args.agg_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mk = rec.get("model_key", "")
            by_model.setdefault(mk, []).append(rec)

    present = set(by_model.keys())
    models = select_models(present)
    if not models:
        raise ValueError(f"No known models found in {args.agg_file}. Keys: {sorted(present)}")
    unplotted = [m for m in present if m not in models]
    if unplotted:
        print(f"Warning: {len(unplotted)} model(s) in input but not in the registry, skipping: {unplotted}")

    # Compute per-model rates + bootstrap CIs split by a_is_gt
    records: list[dict] = []
    for model in models:
        items = by_model[model]
        a_half = [r for r in items if r.get("a_is_gt") is True]
        b_half = [r for r in items if r.get("a_is_gt") is False]
        ci_a = _side_id_ci(a_half)
        ci_b = _side_id_ci(b_half)
        rate_a = ci_a[0] if ci_a else None
        rate_b = ci_b[0] if ci_b else None
        records.append(
            {
                "model": model,
                "label": MODEL_LABEL.get(model, model),
                "family": DIR_FAMILY.get(model, "Other"),
                "rate_a": rate_a,
                "ci_lo_a": ci_a[1] if ci_a else None,
                "ci_hi_a": ci_a[2] if ci_a else None,
                "rate_b": rate_b,
                "ci_lo_b": ci_b[1] if ci_b else None,
                "ci_hi_b": ci_b[2] if ci_b else None,
                "n_a": len([r for r in a_half if not r.get("dropped", False)]),
                "n_b": len([r for r in b_half if not r.get("dropped", False)]),
                "warn": rate_a is not None and rate_b is not None and abs(rate_a - rate_b) > BIAS_THRESHOLD,
            }
        )

    n = len(records)
    xs = np.arange(n)
    offsets = np.array([-BAR_W / 2, BAR_W / 2])

    fig, ax = plt.subplots(figsize=(max(6.5, n * 0.85), 3.0))
    ax.patch.set_alpha(0)

    # Family separator lines
    prev_fam = None
    for xi, row in enumerate(records):
        if prev_fam is not None and row["family"] != prev_fam:
            ax.axvline(xi - 0.5, color="grey", lw=0.8, ls=":", alpha=0.6, zorder=1)
        prev_fam = row["family"]

    for xi, row in enumerate(records):
        for half_idx, (rate, ci_lo, ci_hi, color) in enumerate(
            [
                (row["rate_a"], row["ci_lo_a"], row["ci_hi_a"], COLOR_A),
                (row["rate_b"], row["ci_lo_b"], row["ci_hi_b"], COLOR_B),
            ]
        ):
            x = xi + offsets[half_idx]
            if rate is None:
                ax.bar(x, 0, BAR_W, color="grey", alpha=0.3, zorder=2)
                continue
            yerr = [[max(0.0, rate - ci_lo)], [max(0.0, ci_hi - rate)]]
            ax.bar(
                x,
                rate,
                BAR_W,
                color=color,
                yerr=yerr,
                zorder=2,
                error_kw=ERROR_KW_BAR,
                capsize=3,
            )

    # Stretch y-axis to data range so clustering near the top is visible
    all_lo = [r["ci_lo_a"] for r in records if r["ci_lo_a"] is not None]
    all_lo += [r["ci_lo_b"] for r in records if r["ci_lo_b"] is not None]
    ylo = max(0.0, min(all_lo) - 0.02) if all_lo else 0.0

    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in records], fontsize=7.5)
    ax.tick_params(axis="both", length=0)
    ax.set_ylabel(SIDE_ID_LABEL)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(ylo, 1.02)

    fig.tight_layout()
    out_svg = FIGURES_DIR_OUT / "position_bias.svg"
    save_figure(fig, out_svg)
    print(f"Saved {out_svg}")

    # CSV table
    table_rows = [
        {
            "model": r["model"],
            "model_label": r["label"],
            "family": r["family"],
            "rate_a_half": r["rate_a"],
            "ci_lo_a": r["ci_lo_a"],
            "ci_hi_a": r["ci_hi_a"],
            "rate_b_half": r["rate_b"],
            "ci_lo_b": r["ci_lo_b"],
            "ci_hi_b": r["ci_hi_b"],
            "n_a_half": r["n_a"],
            "n_b_half": r["n_b"],
            "abs_diff": abs(r["rate_a"] - r["rate_b"]) if r["rate_a"] is not None and r["rate_b"] is not None else None,
            "warn": r["warn"],
        }
        for r in records
    ]
    save_table(FIGURES_DIR_OUT / "position_bias.csv", table_rows)
    print(f"Saved {FIGURES_DIR_OUT / 'position_bias.csv'}")

    from src.utils.plotting import save_legend

    leg_handles = [
        mpatches.Patch(facecolor=COLOR_A, edgecolor="black", label="A"),
        mpatches.Patch(facecolor=COLOR_B, edgecolor="black", label="B"),
    ]
    save_legend(
        leg_handles,
        [h.get_label() for h in leg_handles],
        FIGURES_DIR_OUT / "position_bias_legend.svg",
        ncol=2,
        title="Ground Truth Position",
    )
    print(f"Saved {FIGURES_DIR_OUT / 'position_bias_legend.svg'}")


if __name__ == "__main__":
    main()
