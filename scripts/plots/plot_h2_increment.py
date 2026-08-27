"""H2 Plot 1 — CoT − action detection increment.

Reads figures/h2_increment_pooled.csv (analyze_monitor_increment.py).
Per-cell dot-with-CI, x = increment, rows = the 4 cells; one figure per monitor
(suffixed __{monitor}). Pre-registered zero line + ε threshold. Reads: does the
increment shrink on Tool/Implicit, and does it survive on the strong monitor?

Also exposes `dot_ci_panels` + shared constants used by the attribution-gap plot.

Usage:
    python scripts/plots/plot_h2_increment.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from src.utils.logging import get_logger
from src.utils.plotting import (
    CELL_DISPLAY,
    CELLS_4,
    FONT_SIZE_TITLE_WIDE,
    figure_suffix,
    load_metric_rows,
    save_figure,
    save_legend,
    setup_plot_style,
    short_monitor_name,
)

CELL_COLORS = {
    "user_explicit": "#1f77b4",
    "user_implicit": "#6baed6",
    "tool_explicit": "#d62728",
    "tool_implicit": "#fb9a99",
}


def _monitor_order(keys) -> list[str]:
    """Weak first, strong after: 4o-mini is the weak baseline, the rest sort by name."""
    mms = sorted({mm for mm, _ in keys})

    def rank(m):
        return (0 if "4o-mini" in m else 1, m)

    return sorted(mms, key=rank)


def dot_ci_panels(data, monitors, value_key, xlabel, out, logger, eps_line=None, xlim=None) -> Path:
    """One panel per monitor; per-cell dot-with-CI at value_key{,_lo,_hi}. Returns out path."""
    n = len(monitors)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 0.6, 3.0), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, mm in zip(axes, monitors):
        for y, cell in enumerate(CELLS_4):
            r = data.get((mm, cell))
            if r is None:
                continue
            pt, lo, hi = r[value_key], r[f"{value_key}_lo"], r[f"{value_key}_hi"]
            xerr = [[max(0.0, pt - lo)], [max(0.0, hi - pt)]]
            ax.errorbar(
                pt,
                y,
                xerr=xerr,
                fmt="o",
                color=CELL_COLORS[cell],
                ms=8,
                capsize=0,
                alpha=0.95,
                zorder=3,
                markeredgecolor="black",
                markeredgewidth=1.5,
                ecolor="black",
                elinewidth=0.8,
            )
        ax.axvline(0.0, color="black", lw=1.0, alpha=0.5, zorder=1)
        if eps_line is not None:
            ax.axvline(eps_line, color="black", lw=0.9, ls="--", alpha=0.4, zorder=1)
        ax.set_title(short_monitor_name(mm), fontsize=FONT_SIZE_TITLE_WIDE)
        ax.set_ylim(len(CELLS_4) - 0.5, -0.5)
        ax.set_yticks(range(len(CELLS_4)))
        ax.set_yticklabels([CELL_DISPLAY[c] for c in CELLS_4])
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.tick_params(axis="both", length=0)
        ax.patch.set_alpha(0)
        if xlim:
            ax.set_xlim(*xlim)

    fig.supxlabel(xlabel, y=0.01, fontsize=FONT_SIZE_TITLE_WIDE)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    save_figure(fig, out)
    logger.info(f"Saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 increment dot-CI plot (Plot 1)")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--monitor-model", default=None, help="Plot only this monitor; default = one file per monitor.")
    parser.add_argument("--label-scheme", choices=("bprime", "strict"), default="bprime")
    args = parser.parse_args()
    logger = get_logger()

    csv_path = (
        args.figures_dir / "h2_increment_bprime_no_context.csv"
        if args.label_scheme == "bprime"
        else args.figures_dir / "h2_increment_pooled.csv"
    )
    if not csv_path.exists():
        logger.warning(f"No {csv_path} — run analyze_monitor_increment.py first")
        return

    setup_plot_style(wide=True)
    plt.rcParams["axes.facecolor"] = "none"
    data = load_metric_rows(csv_path, index_by=("monitor_model", "cell"))
    monitors = [args.monitor_model] if args.monitor_model else _monitor_order(data.keys())
    logger.info(f"Monitors: {monitors}")

    handles = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CELL_COLORS[c],
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=9,
            label=CELL_DISPLAY[c],
        )
        for c in CELLS_4
    ]
    for mm in monitors:
        value_key = "increment" if args.label_scheme == "bprime" else "incr_b"
        infix = "_bprime" if args.label_scheme == "bprime" else ""
        out = dot_ci_panels(
            data,
            [mm],
            value_key,
            "Added reasoning value\nAUROC(Action+reasoning) - AUROC(Action only)",
            args.figures_dir / f"h2_increment_dumbbell{infix}{figure_suffix(mm)}.svg",
            logger,
            eps_line=0.03,
        )
        save_legend(handles, [h.get_label() for h in handles], out.with_name(out.stem + "_legend.svg"), ncol=4)


if __name__ == "__main__":
    main()
