"""H2 Plot 2 — thesis-in-one: Unverbalized Adoption Rate vs CoT-monitor detection.

Reads figures/h2_increment_bymodel.csv (analyze_monitor_increment.py).
One point per (run_model, cell): x = marginal Unverbalized Adoption Rate
P(adoption and not verbalized | cued), y = a CoT-monitor detection metric under causal
(B) labels. Colour = model; marker = cell (channel).

Three y-metrics are emitted (TPR@FPR sits near the chance floor, so AUROC / AUPRC give
a less floor-bound read):
  auroc  -> auc_cot_b
  auprc  -> ap_cot_b      (baseline = positive prevalence, not 0.5)
  tpr    -> tpr_cot_b     (TPR @ FPR=0.05; random baseline = 0.05)

Read: the point cloud tests whether model-cells with more marginal unverbalized adoption are
harder for the monitor to detect.

Default: strong monitor only (cleanest); pass --monitor-model for others.

Every checkpoint the monitor run covered is plotted. For the AUROC metric the pooled
r and its band come from the scenario-cluster bootstrap in
h2_thesis_pooled_fit_bprime__{monitor}.csv; the other two y-metrics have no cluster
bootstrap and keep the point-resample band.

Usage:
    python scripts/plots/plot_h2_thesis_scatter.py
    python scripts/plots/plot_h2_thesis_scatter.py --monitor-model gpt-4o-mini-2024-07-18
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.plot_h2_increment import _monitor_order
from src.utils.logging import get_logger
from src.utils.plotting import (
    CELL_DISPLAY,
    CELL_MARKERS,
    CELLS_4,
    assign_model_colors,
    figure_suffix,
    load_metric_rows,
    save_figure,
    save_legend,
    setup_plot_style,
    short_model_name,
)

# y-metric -> dict(key, label, baseline reference line, lift-over-prevalence?, fixed y-floor).
# AUPRC is plotted as lift over the positive prevalence, whose no-skill baseline differs
# per point, so it stays comparable across cells. AUROC chance = 0.5, TPR baseline = 0.05.

# Per-checkpoint lines sit back so the full-strength markers and the opaque pooled fit read.
_MODEL_LINE_ALPHA = 0.5

Y_METRICS = {
    "auroc": dict(
        key="auc_cot_b",
        label="Action+CoT monitor AUROC\nfor strict causal attribution",
        baseline=0.5,
        lift=False,
        floor=0.45,
    ),
    "auprc": dict(
        key="ap_cot_b",
        label="Action+CoT monitor AUPRC lift\nfor strict causal attribution",
        baseline=0.0,
        lift=True,
        floor=None,
    ),
    "tpr": dict(
        key="tpr_cot_b",
        label="Action+CoT monitor recall\nat 5% false-positive rate",
        baseline=0.05,
        lift=False,
        floor=-0.02,
    ),
}


def _scatter(rows, models, colors, spec, mm, out, logger, boot=None) -> Path:
    setup_plot_style()
    plt.rcParams["axes.facecolor"] = "none"
    fig, ax = plt.subplots(figsize=(5.2, 4.6))

    xs, ys, plotted = [], [], []
    for r in rows:
        x, y = r.get("unverbalized_adoption_rate"), r.get(spec["key"])
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            continue
        if spec["lift"]:
            y = y - r["prevalence_b"]
        _draw_point(ax, r, x, y, colors, spec, zorder=3)
        xs.append(x)
        ys.append(y)
        plotted.append((r, x, y))

    _draw_pooled_trend(
        ax, xs, ys, with_band=True, color="black", alpha=0.9, label_prefix="Pearson r", boot=boot, zorder=6
    )

    _finish_axes(ax, xs, ys, spec, mm)
    fig.tight_layout()
    save_figure(fig, out)
    logger.info(f"Saved {out}")
    return out


def _draw_pooled_trend(
    ax,
    xs: list[float],
    ys: list[float],
    *,
    with_band: bool,
    color: str,
    alpha: float,
    label_prefix: str,
    boot: list[dict] | None = None,
    zorder: int = 2,
) -> None:
    """Pooled fit line (at `zorder`), its bootstrap band (always behind the cloud), and the r label.

    `boot` carries scenario-cluster replicates (r, slope, intercept) when the analysis
    script produced them; without it the band and CI fall back to resampling the
    (model, cell) points, which treats the points as independent draws.
    """
    if len(set(xs)) >= 2:
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        a, b = np.polyfit(xs_arr, ys_arr, 1)
        xr = np.linspace(float(xs_arr.min()), float(xs_arr.max()), 100)
        if boot:
            slopes = np.asarray([r["slope"] for r in boot], dtype=float)
            intercepts = np.asarray([r["intercept"] for r in boot], dtype=float)
            boot_lines = np.outer(slopes, xr) + intercepts[:, None]
            boot_corrs = [r["r"] for r in boot]
        else:
            lines, boot_corrs = _bootstrap_lines_and_corrs(xs_arr, ys_arr, xr, seed=42)
            boot_lines = np.asarray(lines) if lines else np.empty((0, len(xr)))
        if with_band and len(boot_lines):
            ax.fill_between(
                xr,
                np.quantile(boot_lines, 0.025, axis=0),
                np.quantile(boot_lines, 0.975, axis=0),
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
        corr = float(np.corrcoef(xs_arr, ys_arr)[0, 1])
        if boot_corrs:
            lo, hi = np.quantile(boot_corrs, [0.025, 0.975])
            label = f"{label_prefix} = {corr:.2f} [{lo:.2f}, {hi:.2f}]"
        else:
            label = f"{label_prefix} = {corr:.2f}"
        ax.plot(xr, a * xr + b, color=color, lw=1.8, ls="--", alpha=alpha, zorder=zorder, label=label)
        leg = ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(0.8)


def _bootstrap_lines_and_corrs(
    xs: np.ndarray,
    ys: np.ndarray,
    xr: np.ndarray,
    *,
    seed: int,
    n_boot: int = 5000,
) -> tuple[list[np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    boot_lines = []
    boot_corrs = []
    n = len(xs)
    for _ in range(n_boot):
        sample = rng.integers(0, n, size=n)
        if len(set(xs[sample])) < 2:
            continue
        ba, bb = np.polyfit(xs[sample], ys[sample], 1)
        if np.std(xs[sample]) > 0 and np.std(ys[sample]) > 0:
            boot_corrs.append(float(np.corrcoef(xs[sample], ys[sample])[0, 1]))
        boot_lines.append(ba * xr + bb)
    return boot_lines, boot_corrs


def _draw_point(ax, r, x: float, y: float, colors: dict, spec, *, zorder: int) -> None:
    """One (checkpoint, cell) marker.

    No per-point CI bars: the panel's claim is the pooled slope, whose uncertainty the
    bootstrap band already carries, and 32 crossed bars invited a pairwise reading of
    points the figure does not support. The per-point intervals stay in the CSV
    (unverbalized_adoption_rate_lo/hi, auc_cot_b_lo/hi) for the appendix table.
    """
    ax.plot(
        x,
        y,
        CELL_MARKERS[r["cell"]],
        ms=8,
        markerfacecolor=colors[r["run_model"]],
        markeredgecolor="black",
        markeredgewidth=1.2,
        zorder=zorder,
    )


def _finish_axes(ax, xs: list[float], ys: list[float], spec, mm) -> None:
    ax.axhline(spec["baseline"], color="grey", lw=0.9, ls=":", alpha=0.6, zorder=1)
    ax.set_xlabel("Unverbalized adoption rate")
    ax.set_ylabel(spec["label"])
    # ax.set_title(short_monitor_name(mm))
    xmax = max(xs) if xs else 1.0
    xhi = min(1.0, max(0.4, np.ceil((xmax + 0.02) * 10.0) / 10.0))
    ax.set_xlim(-0.01, xhi + 0.01)
    # y-floor fixed per metric (so points are not crammed at the top over empty space);
    # AUPRC lift floors just below the data (and the 0 line).
    lo = spec["floor"] if spec["floor"] is not None else min([0.0, *ys]) - 0.02
    upper = (max(ys) if ys else 0.3) * 1.10
    ax.set_ylim(lo, upper + 0.05 * (upper - lo))
    ax.grid(linestyle=":", alpha=0.35)
    ax.tick_params(axis="both", length=0)


def _scatter_model_lines(rows, models, colors, spec, mm, out, logger, boot=None) -> Path:
    setup_plot_style()
    plt.rcParams["axes.facecolor"] = "none"
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    xs, ys = [], []
    by_model: dict[str, list[tuple[float, float]]] = {m: [] for m in models}
    for r in rows:
        x, y = r.get("unverbalized_adoption_rate"), r.get(spec["key"])
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            continue
        if spec["lift"]:
            y = y - r["prevalence_b"]
        xs.append(x)
        ys.append(y)
        by_model[r["run_model"]].append((x, y))
        _draw_point(ax, r, x, y, colors, spec, zorder=4)

    _draw_pooled_trend(
        ax,
        xs,
        ys,
        with_band=True,
        color="0.15",
        alpha=0.95,
        label_prefix="pooled r",
        boot=boot,
        zorder=6,
    )
    pooled_handles, pooled_labels = ax.get_legend_handles_labels()

    handles = list(pooled_handles)
    labels = list(pooled_labels)
    for m in models:
        pts = by_model.get(m, [])
        if len(pts) < 2:
            continue
        x = np.asarray([p[0] for p in pts], dtype=float)
        y = np.asarray([p[1] for p in pts], dtype=float)
        if len(set(x)) < 2:
            continue
        a, b = np.polyfit(x, y, 1)
        xr = np.linspace(float(x.min()), float(x.max()), 40)
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else float("nan")
        (line,) = ax.plot(
            xr,
            a * xr + b,
            color=colors[m],
            lw=1.7,
            alpha=_MODEL_LINE_ALPHA,
            zorder=3,
        )
        handles.append(line)
        labels.append(f"{short_model_name(m)} r={corr:.2f}")
    if handles:
        leg = ax.legend(handles, labels, fontsize=8, loc="upper right", ncol=3, framealpha=0.9)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(0.8)
        save_legend(
            handles,
            labels,
            out.with_name(out.stem + "_legend.svg"),
            ncol=3,
        )

    _finish_axes(ax, xs, ys, spec, mm)
    fig.tight_layout()
    save_figure(fig, out)
    logger.info(f"Saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 Unverbalized-Adoption-Rate-vs-detection scatter (Plot 2)")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--monitor-model", default=None, help="Default: strong monitor (last in weak->strong order).")
    parser.add_argument("--label-scheme", choices=("bprime", "strict"), default="bprime")
    args = parser.parse_args()
    logger = get_logger()

    csv_path = (
        args.figures_dir / "h2_increment_bprime_no_context_bymodel.csv"
        if args.label_scheme == "bprime"
        else args.figures_dir / "h2_increment_bymodel.csv"
    )
    if not csv_path.exists():
        logger.warning(f"No {csv_path} — run analyze_monitor_increment.py first")
        return

    rows = load_metric_rows(csv_path, str_cols=("monitor_model", "run_model", "cell"))
    monitors = _monitor_order({(r["monitor_model"], r["cell"]) for r in rows})
    mm = args.monitor_model or monitors[-1]
    rows = [r for r in rows if r["monitor_model"] == mm]
    suffix = figure_suffix(mm)
    infix = "_bprime" if args.label_scheme == "bprime" else ""
    models = sorted({r["run_model"] for r in rows})
    colors = assign_model_colors(models)
    logger.info(f"Scatter for monitor: {mm} ({len(rows)} points, {len(models)} models)")
    logger.info(f"Checkpoints: {', '.join(models)}")

    # Cluster-bootstrap replicates for the pooled fit (AUROC only — the analysis script
    # bootstraps that metric alone). Missing file => the point-resample fallback.
    boot_path = args.figures_dir / f"h2_thesis_pooled_fit{infix}__{mm}.csv"
    boot = load_metric_rows(boot_path) if boot_path.exists() else None
    if boot and boot[0].get("n_points") != len(rows):
        over = boot[0].get("n_points")
        raise SystemExit(
            f"{boot_path} was bootstrapped over {'an unrecorded number of' if over is None else f'{over:.0f}'} points "
            f"but this figure draws {len(rows)} — rerun scripts/analysis/analyze_monitor_increment.py."
        )
    if boot is None:
        logger.warning(f"No {boot_path} — pooled r falls back to resampling the (model, cell) points")

    out = None
    for metric, spec in Y_METRICS.items():
        spec = dict(spec)
        if args.label_scheme == "bprime":
            if metric == "auroc":
                spec["label"] = "Action+CoT monitor AUROC"
            elif metric == "auprc":
                spec["label"] = "Action+CoT monitor AUPRC lift"
        out = _scatter(
            rows,
            models,
            colors,
            spec,
            mm,
            args.figures_dir / f"h2_thesis_scatter_{metric}{infix}{suffix}.svg",
            logger,
            boot=boot if metric == "auroc" else None,
        )
        _scatter_model_lines(
            rows,
            models,
            colors,
            spec,
            mm,
            args.figures_dir / f"h2_thesis_scatter_{metric}_bymodel_lines{infix}{suffix}.svg",
            logger,
            boot=boot if metric == "auroc" else None,
        )

    # Colour and marker are two independent keys, so they get a box each: one box that
    # mixes them reads as one key with an arbitrary break in it.
    model_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[m],
            markeredgecolor="black",
            markersize=9,
            label=short_model_name(m),
        )
        for m in models
    ]
    cell_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker=CELL_MARKERS[c],
            color="w",
            markerfacecolor="#999999",
            markeredgecolor="black",
            markersize=9,
            label=CELL_DISPLAY[c],
        )
        for c in CELLS_4
    ]
    stem = f"h2_thesis_scatter{infix}{suffix}"
    save_legend(
        model_handles,
        [h.get_label() for h in model_handles],
        out.with_name(f"{stem}_legend_models.svg"),
        ncol=4,
    )
    save_legend(
        cell_handles,
        [h.get_label() for h in cell_handles],
        out.with_name(f"{stem}_legend_channels.svg"),
        ncol=len(cell_handles),
    )


if __name__ == "__main__":
    main()
