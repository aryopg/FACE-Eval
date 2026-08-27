"""H4 — Eval-awareness rates by model and cell.

Bar chart (4 bars per model, one per cell) on a truncated y-axis (0 to 0.2, or
higher if a bar or its error bar needs it) so the few non-zero rates are readable.
Error bars are the 95% scenario-cluster bootstrap CI over the three seeds, read from
the CSV. A dashed reference line at 5% anchors the scale; the pre-registered H4 rule is about a
>10 pp *difference between channels*, not about the level, and is checked in
analyze_h4_eval_awareness.py rather than read off this figure.

Color = model family/shade (same convention as Fig. 1); hatch = cell.
The figure should look nearly empty — that is the finding.

Reads: figures/h4_awareness_by_cell.csv  (written by analyze_h4_eval_awareness.py)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting import DIR_FAMILY as _DIR_FAMILY
from src.utils.plotting import (
    ERROR_KW_BAR,
    MODEL_LABEL,
    MODEL_PARAMS,
)
from src.utils.plotting import VARIANT_TO_BASE as _VARIANT_TO_BASE
from src.utils.plotting import assign_model_colors as _assign_model_colors
from src.utils.plotting import (
    save_figure,
    save_legend,
    setup_plot_style,
)
from src.utils.plotting import sort_models as _sort_models
from src.utils.plotting import (
    yerrs_from_cis,
)

FIGURES_DIR = Path("figures")
# Reference line and default axis top. The bars sit far below 1.0, so a 0-1 axis
# renders them as invisible slivers; the axis is truncated instead, and grown past
# Y_TOP_DEFAULT if any bar would otherwise be clipped.
AWARENESS_REF = 0.05
Y_TOP_DEFAULT = 0.20

CELLS = ["user_explicit", "user_implicit", "tool_explicit", "tool_implicit"]
CELL_HATCH = {
    "user_explicit": "",
    "user_implicit": "..",
    "tool_explicit": "///",
    "tool_implicit": "xxx",
}
CELL_DISPLAY = {
    "user_explicit": "User-message (explicit)",
    "user_implicit": "User-message (implicit)",
    "tool_explicit": "Tool-return (explicit)",
    "tool_implicit": "Tool-return (implicit)",
}


def _cell_value(row: dict[str, str], column: str) -> float:
    return float(row[column]) if row.get(column) not in ("nan", "", "None", None) else float("nan")


def load_csv(csv_path: Path) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Load h4_awareness_by_cell.csv → {model: {cell: (rate, ci_lo, ci_hi)}}, pooling variants.

    The CI is the scenario-cluster bootstrap written by analyze_h4_eval_awareness.py.
    Effort variants pool by averaging the three numbers, which is what the rate alone
    already did — the bounds follow their own point, so the pooled bar keeps an interval.
    """
    raw: dict[str, list[dict[str, tuple[float, float, float]]]] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        if any(f"awareness_{cell}_ci_lo" not in (reader.fieldnames or []) for cell in CELLS):
            raise ValueError(f"{csv_path} has no CI columns — rerun analyze_h4_eval_awareness.py (make analyze-h4)")
        for row in reader:
            base = _VARIANT_TO_BASE.get(row["model"], row["model"])
            entry = {
                cell: (
                    _cell_value(row, f"awareness_{cell}"),
                    _cell_value(row, f"awareness_{cell}_ci_lo"),
                    _cell_value(row, f"awareness_{cell}_ci_hi"),
                )
                for cell in CELLS
            }
            raw.setdefault(base, []).append(entry)

    result: dict[str, dict[str, tuple[float, float, float]]] = {}
    for base, entries in raw.items():
        result[base] = {}
        for cell in CELLS:
            present = [e[cell] for e in entries if not math.isnan(e[cell][0])]
            if not present:
                result[base][cell] = (float("nan"),) * 3
                continue
            result[base][cell] = tuple(float(np.mean([p[i] for p in present])) for i in range(3))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="H4 eval-awareness bar plot")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    csv_path = args.figures_dir / "h4_awareness_by_cell.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run analyze_h4_eval_awareness.py first: {csv_path}")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style(wide=True)
    plt.rcParams["axes.facecolor"] = "none"

    data = load_csv(csv_path)
    # Qwen 3 is a partial sweep (3 of 12 runs) and is not shown in this figure.
    models = _sort_models([m for m in data if m in MODEL_PARAMS and _DIR_FAMILY.get(m) != "Qwen 3"])
    model_colors = _assign_model_colors(models)

    n_bars = len(CELLS)
    bar_w = min(0.75 / n_bars, 0.18)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w

    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.patch.set_alpha(0)

    for xi, m in enumerate(models):
        color = model_colors.get(m, (0.5, 0.5, 0.5))
        for offset, cell in zip(offsets, CELLS):
            rate, lo, hi = data[m][cell]
            if math.isnan(rate):
                continue
            ax.bar(
                xi + offset,
                rate,
                bar_w,
                color=color,
                hatch=CELL_HATCH[cell],
                edgecolor="black",
                linewidth=0.5,
                yerr=yerrs_from_cis([rate], [lo], [hi]),
                capsize=2,
                error_kw=ERROR_KW_BAR,
                zorder=3,
            )

    # Family separator lines
    prev_fam = None
    for xi, m in enumerate(models):
        fam = _DIR_FAMILY.get(m, "Other")
        if prev_fam is not None and fam != prev_fam:
            ax.axvline(xi - 0.5, color="grey", lw=0.8, ls=":", alpha=0.6, zorder=1)
        prev_fam = fam

    ax.axhline(AWARENESS_REF, color="#d62728", lw=1.0, ls="--", alpha=0.8, zorder=2)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=7.5)
    ax.tick_params(axis="both", length=0)
    ax.set_ylabel("Eval-awareness rate")
    # The upper CI, not the bar, is the highest ink: clipping a whisker hides the same
    # thing clipping a bar would.
    max_ink = max((max(v[0], v[2]) for m in models for v in data[m].values() if not math.isnan(v[0])), default=0.0)
    y_top = max(Y_TOP_DEFAULT, max_ink * 1.08)
    if y_top > Y_TOP_DEFAULT:
        print(f"WARNING: highest eval-awareness bar or CI is {max_ink:.3f}; y-axis grown to {y_top:.3f}")
    else:
        print(f"Highest eval-awareness bar or CI: {max_ink:.3f} (y-axis 0-{y_top:.2f})")
    ax.set_ylim(0, y_top)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

    fig.tight_layout()
    out = args.figures_dir / "h4_eval_awareness.svg"
    save_figure(fig, out)
    print(f"Saved {out}")

    legend_handles = [
        mpatches.Patch(facecolor="#cccccc", hatch=CELL_HATCH[cell], edgecolor="black", label=CELL_DISPLAY[cell])
        for cell in CELLS
    ]
    legend_handles.append(mlines.Line2D([], [], color="#d62728", lw=1.0, ls="--", label=f"{AWARENESS_REF:.0%}"))
    save_legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        args.figures_dir / "h4_eval_awareness_legend.svg",
        ncol=len(legend_handles),
    )


if __name__ == "__main__":
    main()
