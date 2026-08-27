"""H6 — Explicitness gap vs clarity: scatter + regression.

Two styles (--style):
  gap  (default) — x = within-pair clarity difference, y = within-pair metric gap.
                   Intercept at x=0 is the effect at matched clarity.
  raw            — x = absolute clarity score, y = raw metric value.
                   Two regression lines (explicit solid, implicit dashed); a
                   persistent vertical offset between them at all clarity levels
                   rules out the pure cue-strength account.

Two metrics (--metric): vcr (verbalized commitment rate) or uar (unverbalized adoption).

Reads: figures/h6_per_scenario.csv  (written by analyze_h6_clarity_matched.py)
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
from src.utils.plotting import FAMILY_COLORS, save_figure, save_legend, setup_plot_style

FIGURES_DIR = Path("figures")
CLARITY_MATCH_TOL = 0.25
N_BINS = 8
N_BOOT = 2000

METRIC_GAP_COL = {"vcr": "vcr_gap", "uar": "uar_gap"}
METRIC_YLABEL = {
    "vcr": r"$\Delta$" + "VCR (explicit" + r"$-$" + "implicit)",
    "uar": r"$\Delta$" + "Unverbalized adoption rate (explicit" + r"$-$" + "implicit)",
}

METRIC_RAW_COLS = {"vcr": ("vcr_exp", "vcr_imp"), "uar": ("uar_exp", "uar_imp")}
METRIC_RAW_YLABEL = {"vcr": "VCR", "uar": "Unverbalized adoption rate"}

_EXP_COLOR = "#d62728"  # red for explicit
_IMP_COLOR = "#1f77b4"  # blue for implicit


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            entry: dict = {
                "model": row["model"],
                "channel": row["channel"],
                "clarity_diff": float(row["clarity_diff"]),
                "clarity_exp": float(row["clarity_exp"]),
                "clarity_imp": float(row["clarity_imp"]),
                "vcr_gap": float(row["vcr_gap"]),
                "vcr_exp": float(row["vcr_exp"]),
                "vcr_imp": float(row["vcr_imp"]),
            }
            for col in ("uar_gap", "uar_exp", "uar_imp"):
                if col in row and row[col] not in ("nan", ""):
                    entry[col] = float(row[col])
            rows.append(entry)
    return rows


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _fit_logistic(x: np.ndarray, y: np.ndarray, n_iter: int = 30) -> tuple[float, float]:
    """Fit p = expit(a + b*x) via Newton-Raphson IRLS. Returns (a, b)."""
    a, b = 0.0, float(np.clip((y.mean() - 0.5) / max(x.std(), 1e-9), -2, 2))
    for _ in range(n_iter):
        p = _expit(a + b * x)
        w = np.maximum(p * (1.0 - p), 1e-10)
        g_a = np.sum(p - y)
        g_b = np.sum((p - y) * x)
        h_aa = np.sum(w)
        h_ab = np.sum(w * x)
        h_bb = np.sum(w * x * x)
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        a -= (h_bb * g_a - h_ab * g_b) / det
        b -= (h_aa * g_b - h_ab * g_a) / det
    return float(a), float(b)


def _bootstrap_regression(
    x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """OLS regression with bootstrap CI.

    Returns (x_line, ci_lo, ci_hi, intercept, intercept_ci_half).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    coeffs = np.polyfit(x, y, 1)
    y_line = np.polyval(coeffs, x_line)

    boot_lines = np.empty((n_boot, len(x_line)))
    boot_intercepts = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        c = np.polyfit(x[idx], y[idx], 1)
        boot_lines[i] = np.polyval(c, x_line)
        boot_intercepts[i] = c[1]

    ci_lo = np.percentile(boot_lines, 2.5, axis=0)
    ci_hi = np.percentile(boot_lines, 97.5, axis=0)
    ic_lo = float(np.percentile(boot_intercepts, 2.5))
    ic_hi = float(np.percentile(boot_intercepts, 97.5))

    return x_line, y_line, ci_lo, ci_hi, float(coeffs[1]), ic_lo, ic_hi


def _binned_means(x: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(float(x.min()), float(x.max()), n_bins + 1)
    centers, means, sems = [], [], []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        ys = y[mask]
        if len(ys) < 3:
            continue
        centers.append((edges[i] + edges[i + 1]) / 2)
        means.append(float(np.mean(ys)))
        sems.append(float(np.std(ys, ddof=1) / np.sqrt(len(ys))))
    return np.array(centers), np.array(means), np.array(sems)


def _draw_panel(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    models: list[str],
    channel: str,
    y_label: str,
) -> tuple[float, float, float]:
    """Draw one scatter panel. Returns (intercept, ci_lo, ci_hi)."""
    ax.patch.set_alpha(0)

    # Shaded matched-clarity band
    ax.axvspan(
        -CLARITY_MATCH_TOL,
        CLARITY_MATCH_TOL,
        alpha=0.10,
        color="#2196F3",
        zorder=0,
        label=r"$|\Delta\mathrm{clarity}|\leq0.25$",
    )

    # Pooled gap within band + horizontal reference
    band_mask = np.abs(xs) <= CLARITY_MATCH_TOL
    if band_mask.sum() >= 3:
        band_gap = float(np.mean(ys[band_mask]))
        ax.axhline(band_gap, color="#2196F3", lw=1.2, ls=":", alpha=0.8, zorder=2)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.35, zorder=1)

    # Raw scatter (small, semi-transparent, colored by family)
    for m, xi, yi in zip(models, xs, ys):
        fam = _DIR_FAMILY.get(m.replace("/", "_"), "Other")
        color = FAMILY_COLORS.get(fam, "#888888")
        ax.scatter(xi, yi, s=7, color=color, alpha=0.30, linewidths=0, zorder=2)

    # Regression line + CI band
    if len(xs) >= 10:
        x_line, y_line, ci_lo_arr, ci_hi_arr, intercept, ic_lo, ic_hi = _bootstrap_regression(xs, ys)
        ax.fill_between(x_line, ci_lo_arr, ci_hi_arr, alpha=0.15, color="black", zorder=3)
        ax.plot(x_line, y_line, color="black", lw=1.6, zorder=5)
    else:
        intercept, ic_lo, ic_hi = float("nan"), float("nan"), float("nan")

    # Binned means overlay
    bx, bm, bs = _binned_means(xs, ys)
    if len(bx):
        ax.errorbar(
            bx,
            bm,
            yerr=bs,
            marker="o",
            markersize=6,
            linestyle="",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            elinewidth=1.2,
            capsize=3,
            zorder=6,
        )

    ax.set_xlabel(r"$\Delta$ Model-rated clarity (explicit - implicit)", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title("User-message" if channel == "user" else "Tool-return", fontsize=12)
    ax.grid(linestyle=":", alpha=0.3)
    ax.tick_params(axis="both", length=0)

    if not math.isnan(intercept):
        ax.text(
            0.03,
            0.97,
            r"$\hat{a}_0 = $" + f"{intercept:.3f} [{ic_lo:.3f}, {ic_hi:.3f}]",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    return intercept, ic_lo, ic_hi


def _draw_panel_raw(
    ax,
    xs_exp: np.ndarray,
    ys_exp: np.ndarray,
    xs_imp: np.ndarray,
    ys_imp: np.ndarray,
    channel: str,
    y_label: str,
) -> None:
    """Raw scatter: explicit and implicit points on the same axes with two regression lines.

    A persistent vertical offset between the lines at all clarity levels rules
    out the pure cue-strength account (which predicts convergence).
    """
    ax.patch.set_alpha(0)

    # Scatter: explicit (filled circles) and implicit (open circles)
    ax.scatter(xs_exp, ys_exp, s=7, color=_EXP_COLOR, alpha=0.30, linewidths=0, zorder=2)
    ax.scatter(xs_imp, ys_imp, s=7, color=_IMP_COLOR, alpha=0.30, linewidths=0, zorder=2)

    # Regression lines with CI bands for each condition
    x_global = np.linspace(
        float(min(xs_exp.min(), xs_imp.min())),
        float(max(xs_exp.max(), xs_imp.max())),
        200,
    )
    rng = np.random.default_rng(42)

    for xs, ys, color, ls in [
        (xs_exp, ys_exp, _EXP_COLOR, "-"),
        (xs_imp, ys_imp, _IMP_COLOR, "--"),
    ]:
        if len(xs) < 10:
            continue
        a, b = _fit_logistic(xs, ys)
        y_line = _expit(a + b * x_global)

        boot_lines = np.empty((N_BOOT, len(x_global)))
        n = len(xs)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            a_b, b_b = _fit_logistic(xs[idx], ys[idx])
            boot_lines[i] = _expit(a_b + b_b * x_global)

        ci_lo = np.percentile(boot_lines, 2.5, axis=0)
        ci_hi = np.percentile(boot_lines, 97.5, axis=0)
        ax.fill_between(x_global, ci_lo, ci_hi, alpha=0.12, color=color, zorder=3)
        ax.plot(x_global, y_line, color=color, lw=1.6, ls=ls, zorder=5)

    # Binned means for each condition
    for xs, ys, color, marker in [
        (xs_exp, ys_exp, _EXP_COLOR, "o"),
        (xs_imp, ys_imp, _IMP_COLOR, "s"),
    ]:
        bx, bm, bs = _binned_means(xs, ys)
        if len(bx):
            ax.errorbar(
                bx,
                bm,
                yerr=bs,
                marker=marker,
                markersize=6,
                linestyle="",
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                elinewidth=1.2,
                capsize=3,
                zorder=6,
            )

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.25, zorder=1)
    ax.set_xlabel("Model-rated clarity", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_title("User-message" if channel == "user" else "Tool-return", fontsize=9)
    ax.grid(linestyle=":", alpha=0.3)
    ax.tick_params(axis="both", length=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="H6 clarity scatter plot")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--metric",
        choices=["vcr", "uar"],
        default="vcr",
        help="y-axis metric: vcr (verbalized commitment) or uar (unverbalized adoption)",
    )
    parser.add_argument(
        "--style",
        choices=["gap", "raw"],
        default="gap",
        help="gap: within-pair difference; raw: absolute clarity vs metric, two lines",
    )
    args = parser.parse_args()

    csv_path = args.figures_dir / "h6_per_scenario.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run analyze_h6_clarity_matched.py first: {csv_path}")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style(wide=True)
    plt.rcParams["axes.facecolor"] = "none"

    rows = load_csv(csv_path)
    metric_suffix = f"_{args.metric}" if args.metric != "vcr" else ""
    style_suffix = "_raw" if args.style == "raw" else ""
    print(f"Loaded {len(rows)} per-scenario points  (metric={args.metric}, style={args.style})")

    channels = ["user", "tool"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    if args.style == "gap":
        gap_col = METRIC_GAP_COL[args.metric]
        y_label = METRIC_YLABEL[args.metric]

        for col_idx, ch in enumerate(channels):
            ch_rows = [r for r in rows if r["channel"] == ch and gap_col in r and not math.isnan(r[gap_col])]
            print(f"  {ch}: {len(ch_rows)} points")
            if not ch_rows:
                continue
            xs = np.array([r["clarity_diff"] for r in ch_rows])
            ys = np.array([r[gap_col] for r in ch_rows])
            model_list = [r["model"] for r in ch_rows]

            intercept, ic_lo, ic_hi = _draw_panel(axes[col_idx], xs, ys, model_list, ch, y_label)
            excl_zero = ic_lo > 0 or ic_hi < 0
            print(
                f"    intercept={intercept:.3f} [{ic_lo:.3f},{ic_hi:.3f}] "
                f"({'excludes 0' if excl_zero else 'includes 0'})"
            )

        legend_handles = [
            mlines.Line2D(
                [],
                [],
                marker="o",
                color="black",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.5,
                linestyle="",
                markersize=6,
                label="Binned mean",
            ),
            mlines.Line2D([], [], color="black", lw=1.6, label="OLS regression"),
            mpatches.Patch(color="#2196F3", alpha=0.25, label=r"$|\Delta\mathrm{clarity}|\leq0.25$"),
        ]

    else:  # raw
        exp_col, imp_col = METRIC_RAW_COLS[args.metric]
        y_label = METRIC_RAW_YLABEL[args.metric]

        for col_idx, ch in enumerate(channels):
            ch_rows = [
                r
                for r in rows
                if r["channel"] == ch
                and exp_col in r
                and imp_col in r
                and not math.isnan(r[exp_col])
                and not math.isnan(r[imp_col])
            ]
            print(f"  {ch}: {len(ch_rows)} pairs → {2 * len(ch_rows)} points")
            if not ch_rows:
                continue
            xs_exp = np.array([r["clarity_exp"] for r in ch_rows])
            ys_exp = np.array([r[exp_col] for r in ch_rows])
            xs_imp = np.array([r["clarity_imp"] for r in ch_rows])
            ys_imp = np.array([r[imp_col] for r in ch_rows])

            _draw_panel_raw(axes[col_idx], xs_exp, ys_exp, xs_imp, ys_imp, ch, y_label)

        legend_handles = [
            mlines.Line2D(
                [],
                [],
                color=_EXP_COLOR,
                lw=1.6,
                ls="-",
                marker="o",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=_EXP_COLOR,
                markeredgewidth=1.5,
                label="Explicit",
            ),
            mlines.Line2D(
                [],
                [],
                color=_IMP_COLOR,
                lw=1.6,
                ls="--",
                marker="s",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=_IMP_COLOR,
                markeredgewidth=1.5,
                label="Implicit",
            ),
        ]

    fig.tight_layout()
    out = args.figures_dir / f"h6_clarity_scatter{metric_suffix}{style_suffix}.svg"
    save_figure(fig, out)
    print(f"Saved {out}")

    save_legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        args.figures_dir / f"h6_clarity_scatter{metric_suffix}{style_suffix}_legend.svg",
        ncol=len(legend_handles),
    )


if __name__ == "__main__":
    main()
