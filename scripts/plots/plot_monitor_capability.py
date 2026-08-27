"""H2 main-text Panel A — monitor AUROC capability by condition.

Reads figures/h2_increment_bprime_no_context.csv and plots action-only vs
action+reasoning AUROC with 95% scenario-cluster bootstrap CIs. This is distinct from
the increment plot: it shows absolute monitor capability and whether reasoning access
changes that capability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.plot_h2_increment import _monitor_order
from scripts.plots.plot_no_context_shift import CELL_COLORS
from src.utils.logging import get_logger
from src.utils.plotting import (
    CELL_DISPLAY,
    CELLS_4,
    figure_suffix,
    load_metric_rows,
    save_figure,
    save_legend,
    setup_plot_style,
)

VIEW_SPECS = (
    ("auc_cot_only", "CoT only", "\\\\\\"),
    ("auc_action", "Action only", "///"),
    ("auc_cot", "Action + CoT", ""),
)


def _legend_handles():
    return [
        mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatch,
            label=label,
        )
        for _key, label, hatch in VIEW_SPECS
    ]


def _plot(rows, mm: str, out: Path, logger) -> Path:
    setup_plot_style()
    plt.rcParams["axes.facecolor"] = "none"
    fig, ax = plt.subplots(figsize=(6, 4.6))

    x_positions = np.arange(len(CELLS_4), dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    for view_i, (key, _label, hatch) in enumerate(VIEW_SPECS):
        for x, cell in zip(x_positions, CELLS_4):
            r = rows.get((mm, cell))
            if r is None:
                continue
            pt, lo, hi = r[key], r[f"{key}_lo"], r[f"{key}_hi"]
            yerr = [[max(0.0, pt - lo)], [max(0.0, hi - pt)]]
            xpos = x + offsets[view_i]
            ax.bar(
                xpos,
                pt,
                width=width,
                color=CELL_COLORS[cell],
                edgecolor="black",
                linewidth=1.5,
                hatch=hatch,
                alpha=0.9,
                zorder=2,
            )
            ax.errorbar(
                xpos,
                pt,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=2,
                capthick=1.5,
                zorder=3,
            )
    ax.axhline(0.5, color="black", lw=0.9, ls=":", alpha=0.55, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([CELL_DISPLAY[c].replace(" (", "\n(") for c in CELLS_4])
    ax.set_ylabel("AUROC")
    ax.set_title("")
    ax.set_ylim(0.48, 0.79)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.tick_params(axis="both", length=0)

    fig.tight_layout()
    save_figure(fig, out)
    handles = _legend_handles()
    save_legend(
        handles,
        [h.get_label() for h in handles],
        out.with_name(out.stem + "_legend.svg"),
        ncol=3,
    )
    logger.info(f"Saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 monitor capability AUROC plot")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--monitor-model", default=None, help="Default: strongest monitor in the CSV.")
    args = parser.parse_args()
    logger = get_logger()

    csv_path = args.figures_dir / "h2_increment_bprime_no_context.csv"
    if not csv_path.exists():
        logger.warning(f"No {csv_path} — run analyze_monitor_increment.py first")
        return

    rows = load_metric_rows(csv_path, index_by=("monitor_model", "cell"))
    monitors = _monitor_order(rows.keys())
    mm = args.monitor_model or monitors[-1]
    _plot(rows, mm, args.figures_dir / f"h2_monitor_capability_bprime{figure_suffix(mm)}.svg", logger)


if __name__ == "__main__":
    main()
