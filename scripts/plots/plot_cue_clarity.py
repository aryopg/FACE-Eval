"""Model-rated cue clarity and side-ID accuracy, explicit vs. implicit.

Two panels sharing the model y-axis (models on rows):

  (left)  Side-identification accuracy — % of items where the model correctly identified which
          side the artifact points to. Explicit (^) vs. implicit (v) per model.
          Reference line at x=0.70 (§3 H_comprehend pass threshold).

  (right) Model-rated clarity (1-5 scale). Same dumbbell structure.
          Reference line at x=3.0 (minimum detectable signal).

Explicit pool: context_type ∈ {user_turn, user_turn_structured, explicit}.
Implicit pool: context_type ∈ {user_turn_implicit, implicit}.

Outputs:
  figures/cue_clarity.svg
  figures/cue_clarity.csv
  figures/cue_clarity_legend.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting import (
    CLARITY_LABEL,
    DIR_FAMILY,
    ERRORBAR_KWARGS,
)
from src.utils.plotting import MODEL_LABEL_INLINE as MODEL_LABEL
from src.utils.plotting import (
    SIDE_ID_LABEL,
    assign_model_colors,
    save_figure,
    save_legend,
    save_table,
    select_models,
    setup_plot_style,
)

FIGURES_DIR = Path("figures")


EXPLICIT_CTYPES = ("user_turn", "user_turn_structured", "explicit")
IMPLICIT_CTYPES = ("user_turn_implicit", "implicit")


def _pool_conditions(
    marginals: dict,
    model: str,
    ctypes: tuple[str, ...],
    metric_key: str,
    ci_lo_key: str,
    ci_hi_key: str,
) -> tuple[float, float, float, int] | None:
    """Weighted mean of metric across conditions, CI as mean of CI bounds.

    Returns (mean, ci_lo, ci_hi, total_n) or None if no data for any condition.
    """
    vals, los, his, ns = [], [], [], []
    model_data = marginals.get(model, {})
    for ct in ctypes:
        cell = model_data.get(ct)
        if cell is None:
            continue
        v = cell.get(metric_key)
        lo = cell.get(ci_lo_key)
        hi = cell.get(ci_hi_key)
        n = cell.get("n_items", 0)
        if v is None or lo is None or hi is None or n == 0:
            continue
        vals.append(v)
        los.append(lo)
        his.append(hi)
        ns.append(n)
    if not vals:
        return None
    total_n = sum(ns)
    weights = np.array(ns, dtype=float) / total_n
    mean = float(np.dot(weights, vals))
    ci_lo = float(np.mean(los))
    ci_hi = float(np.mean(his))
    return mean, ci_lo, ci_hi, total_n


NO_THINK_OFFSET = 0.3  # y-offset for no-think dumbbells within each model row


def _draw_dumbbell(
    ax,
    explicit_tri: tuple[float, float, float],
    implicit_tri: tuple[float, float, float],
    y: float,
    color: tuple,
    linestyle: str = "-",
) -> None:
    x_explicit, x_implicit = explicit_tri[0], implicit_tri[0]
    ax.plot(
        [x_explicit, x_implicit],
        [y, y],
        color=color,
        linewidth=1.8,
        alpha=0.7,
        linestyle=linestyle,
        zorder=2,
    )
    for (x, lo, hi), marker in [(explicit_tri, ">"), (implicit_tri, "<")]:
        xerr = [[max(0.0, x - lo)], [max(0.0, hi - x)]]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            marker=marker,
            markersize=8,
            linestyle="",
            color="black",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=2,
            zorder=3,
            **ERRORBAR_KWARGS,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model-rated cue clarity, explicit vs. implicit.")
    parser.add_argument("--input", default="outputs/artifact_rating_marginals.json")
    parser.add_argument(
        "--input-no-think",
        default=None,
        help="Marginals JSON for no-think runs. When provided, plots dotted dumbbells below each model row.",
    )
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    FIGURES_DIR_OUT = Path(args.figures_dir)
    FIGURES_DIR_OUT.mkdir(parents=True, exist_ok=True)

    setup_plot_style(wide=True)
    plt.rcParams["axes.facecolor"] = "none"

    with open(args.input) as fh:
        marginals: dict = json.load(fh)

    no_think_marginals: dict | None = None
    if args.input_no_think:
        with open(args.input_no_think) as fh:
            no_think_marginals = json.load(fh)

    present = set(marginals.keys())
    models = select_models(present)
    if not models:
        raise ValueError(f"No known models found in {args.input}. Keys: {sorted(present)}")
    missing = [m for m in present if m not in models]
    if missing:
        print(f"Warning: {len(missing)} model(s) not found in input, skipping: {missing}")

    model_colors = assign_model_colors(models)
    n = len(models)
    ys = np.arange(n)

    # Pre-compute pooled stats for all models.
    rows: list[dict] = []
    for model in models:
        sid_e = _pool_conditions(marginals, model, EXPLICIT_CTYPES, "side_id_rate", "side_id_ci_lo", "side_id_ci_hi")
        sid_i = _pool_conditions(marginals, model, IMPLICIT_CTYPES, "side_id_rate", "side_id_ci_lo", "side_id_ci_hi")
        clar_e = _pool_conditions(marginals, model, EXPLICIT_CTYPES, "clarity_mean", "clarity_ci_lo", "clarity_ci_hi")
        clar_i = _pool_conditions(marginals, model, IMPLICIT_CTYPES, "clarity_mean", "clarity_ci_lo", "clarity_ci_hi")
        nt = no_think_marginals or {}
        sid_e_nt = _pool_conditions(nt, model, EXPLICIT_CTYPES, "side_id_rate", "side_id_ci_lo", "side_id_ci_hi")
        sid_i_nt = _pool_conditions(nt, model, IMPLICIT_CTYPES, "side_id_rate", "side_id_ci_lo", "side_id_ci_hi")
        clar_e_nt = _pool_conditions(nt, model, EXPLICIT_CTYPES, "clarity_mean", "clarity_ci_lo", "clarity_ci_hi")
        clar_i_nt = _pool_conditions(nt, model, IMPLICIT_CTYPES, "clarity_mean", "clarity_ci_lo", "clarity_ci_hi")
        rows.append(
            {
                "model": model,
                "label": MODEL_LABEL.get(model, model),
                "color": model_colors[model],
                "family": DIR_FAMILY.get(model, "Other"),
                "sid_e": sid_e,
                "sid_i": sid_i,
                "clar_e": clar_e,
                "clar_i": clar_i,
                "sid_e_nt": sid_e_nt,
                "sid_i_nt": sid_i_nt,
                "clar_e_nt": clar_e_nt,
                "clar_i_nt": clar_i_nt,
            }
        )

    row_height = 0.45 if no_think_marginals else 0.3
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6, row_height * n), sharey=True)
    ax_left.patch.set_alpha(0)
    ax_right.patch.set_alpha(0)

    # Reference lines
    ax_left.axvline(0.70, color="black", lw=1.0, ls="--", alpha=0.5, zorder=1)
    ax_right.axvline(3.0, color="black", lw=1.0, ls="--", alpha=0.5, zorder=1)

    prev_fam = None
    for y, row in zip(ys, rows):
        if prev_fam is not None and row["family"] != prev_fam:
            for ax in (ax_left, ax_right):
                ax.axhline(y - 0.5, color="grey", lw=0.8, ls=":", alpha=0.6, zorder=1)
        prev_fam = row["family"]

        color = row["color"]
        if row["sid_e"] is not None and row["sid_i"] is not None:
            _draw_dumbbell(ax_left, row["sid_e"][:3], row["sid_i"][:3], y, color, linestyle="-")
        if row["clar_e"] is not None and row["clar_i"] is not None:
            _draw_dumbbell(ax_right, row["clar_e"][:3], row["clar_i"][:3], y, color, linestyle="-")
        if row["sid_e_nt"] is not None and row["sid_i_nt"] is not None:
            _draw_dumbbell(ax_left, row["sid_e_nt"][:3], row["sid_i_nt"][:3], y + NO_THINK_OFFSET, color, linestyle=":")
        if row["clar_e_nt"] is not None and row["clar_i_nt"] is not None:
            _draw_dumbbell(
                ax_right, row["clar_e_nt"][:3], row["clar_i_nt"][:3], y + NO_THINK_OFFSET, color, linestyle=":"
            )

    # Compute axis limits from CI extents.
    def _xlim(col_e: str, col_i: str, pad: float, lo_bound: float, hi_bound: float) -> tuple[float, float]:
        xs: list[float] = []
        for row in rows:
            for col in (col_e, col_i, col_e + "_nt", col_i + "_nt"):
                tri = row.get(col)
                if tri is not None:
                    xs.extend([tri[1], tri[2]])
        if not xs:
            return lo_bound, hi_bound
        return max(lo_bound, min(xs) - pad), min(hi_bound, max(xs) + pad)

    # Stretch axes to data range so clustering near the right is visually clear.
    sid_xlo, sid_xhi = _xlim("sid_e", "sid_i", 0.03, 0.0, 1.0)
    clar_xlo, clar_xhi = _xlim("clar_e", "clar_i", 0.1, 1.0, 5.0)

    ax_left.set_xlim(sid_xlo, sid_xhi)
    ax_left.set_xlabel(SIDE_ID_LABEL)
    ax_left.grid(axis="x", linestyle=":", alpha=0.4)
    ax_left.tick_params(axis="both", length=0)

    ax_right.set_xlim(clar_xlo, clar_xhi)
    ax_right.set_xlabel(CLARITY_LABEL)
    ax_right.grid(axis="x", linestyle=":", alpha=0.4)
    ax_right.tick_params(axis="both", length=0)

    ax_left.set_yticks(ys)
    ax_left.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax_left.invert_yaxis()

    fig.tight_layout()

    out_svg = FIGURES_DIR_OUT / "cue_clarity.svg"
    save_figure(fig, out_svg)
    print(f"Saved {out_svg}")

    # CSV table — one row per (model, pool).
    table_rows: list[dict] = []
    for row in rows:
        for pool, sid, clar in [("explicit", row["sid_e"], row["clar_e"]), ("implicit", row["sid_i"], row["clar_i"])]:
            table_rows.append(
                {
                    "model": row["model"],
                    "model_label": row["label"],
                    "pool": pool,
                    "side_id_rate": sid[0] if sid is not None else "",
                    "side_id_ci_lo": sid[1] if sid is not None else "",
                    "side_id_ci_hi": sid[2] if sid is not None else "",
                    "n_items": sid[3] if sid is not None else "",
                    "clarity_mean": clar[0] if clar is not None else "",
                    "clarity_ci_lo": clar[1] if clar is not None else "",
                    "clarity_ci_hi": clar[2] if clar is not None else "",
                }
            )
    save_table(FIGURES_DIR_OUT / "cue_clarity.csv", table_rows)

    # Legend: explicit/implicit markers.
    handles = [
        mlines.Line2D(
            [],
            [],
            marker="<",
            color="black",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markeredgewidth=2,
            linestyle="",
            markersize=8,
            label="Implicit",
        ),
        mlines.Line2D(
            [],
            [],
            marker=">",
            color="black",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markeredgewidth=2,
            linestyle="",
            markersize=8,
            label="Explicit",
        ),
    ]
    save_legend(
        handles,
        [h.get_label() for h in handles],
        FIGURES_DIR_OUT / "cue_clarity_legend.svg",
        ncol=len(handles),
    )


if __name__ == "__main__":
    main()
