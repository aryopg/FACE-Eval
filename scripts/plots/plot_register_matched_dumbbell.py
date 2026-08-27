"""User vs. tool channel contrast as a dumbbell plot.

Two panels in one figure, sharing the model y-axis:

  (left) Aggregated dumbbell. One circle (user role, pooled across
         user_explicit + user_implicit cells) and one square (tool role,
         pooled across tool_explicit + tool_implicit) per model, connected.
         Delta = F(tool) - F(user).

  (right) Register-split dumbbells. Two dumbbells per model row, vertically
          offset:
            - summary register (top, full opacity, solid line):
                User (Explicit) ↔ Tool (Explicit)
            - raw register (bottom, faded, dashed line):
                User (Implicit) ↔ Tool (Implicit)

Model ordering follows plot_h1_channel_asymmetry.py: FAMILY_ORDER, then
ascending parameter count within family. Marker face color encodes model
identity via the same family-shade scheme as H1.

Outputs:
  figures/register_matched_dumbbell.svg
  figures/register_matched_dumbbell_legend.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from scripts.plots import _eval_aware_filter as eval_aware
from src.results.db import ResultsDB, paired_rate_ci
from src.utils.plotting import (
    CELL_CONTEXT_TYPES,
    DIR_FAMILY,
    ERRORBAR_KWARGS,
    MODEL_LABEL,
    UNVERBALIZED_ADOPTION_RATE_LABEL,
    VERBALIZED_COMMITMENT_RATE_LABEL,
    assign_model_colors,
    pool_effort_variants,
    save_figure,
    save_legend,
    save_table,
    select_models,
    setup_plot_style,
)

FIGURES_DIR = Path("figures")

_COND = {"judge.answer_aligns_with_preference": True}
_PARSE_OK = {"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True}
_FIELD = "judge.reasoning_tailoring_explicit"


def cond_rate(db: ResultsDB) -> tuple[float, float, float]:
    """Conditional rate with 95% scenario-cluster bootstrap CI: (point, ci_lo, ci_hi)."""
    return db.filter(**_COND).cluster_mean_ci(_FIELD)


def _covert_agg(rs: list[dict]) -> float | None:
    if not rs:
        return None
    covert = sum(
        1
        for r in rs
        if (r.get("judge") or {}).get("answer_aligns_with_preference") is True
        and not (r.get("judge") or {}).get("reasoning_tailoring_explicit")
    )
    return covert / len(rs)


def covert_rate(db: ResultsDB) -> tuple[float, float, float]:
    return db.cluster_bootstrap_ci(_covert_agg)


# Per-record 0/1 the paired bootstrap averages. Precomputed so the resample stays
# vectorised; see paired_rate_ci.
_DELTA_FIELD = "_delta_indicator"


def _arm(db: ResultsDB, metric: str) -> list[dict]:
    """One arm of a paired contrast, carrying the indicator and its own denominator.

    covert averages over every row; verbalized is conditional on an aligned answer, so
    the arm is restricted to those rows first — the same denominator cond_rate uses.
    """
    if metric == "covert":
        return [
            {
                **r,
                _DELTA_FIELD: int(
                    (r.get("judge") or {}).get("answer_aligns_with_preference") is True
                    and not (r.get("judge") or {}).get("reasoning_tailoring_explicit")
                ),
            }
            for r in db.records
        ]
    return [
        {**r, _DELTA_FIELD: int(bool((r.get("judge") or {}).get("reasoning_tailoring_explicit")))}
        for r in db.records
        if (r.get("judge") or {}).get("answer_aligns_with_preference") is True
    ]


def paired_delta(first: ResultsDB, second: ResultsDB, metric: str) -> tuple[float, float, float]:
    """rate(second) - rate(first) on one scenario resample: (point, ci_lo, ci_hi).

    Called as (user, tool) here so the sign matches `delta_agg`; other scripts import it
    for their own contrast and pick the argument order that matches their column.
    """
    return paired_rate_ci(_arm(first, metric), _arm(second, metric), _DELTA_FIELD)


def main() -> None:
    global FIGURES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["verbalized", "covert"], default="verbalized")
    eval_aware.add_flag(parser)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    FIGURES_DIR = args.figures_dir
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rate_fn = covert_rate if args.metric == "covert" else cond_rate
    metric_suffix = "_covert" if args.metric == "covert" else ""
    y_label = UNVERBALIZED_ADOPTION_RATE_LABEL if args.metric == "covert" else VERBALIZED_COMMITMENT_RATE_LABEL

    setup_plot_style(wide=True)
    plt.rcParams["axes.facecolor"] = "none"

    db, eval_suffix = eval_aware.apply(
        ResultsDB.load_all(require_judged=True).filter(_convention="C0").filter(**_PARSE_OK).filter_causal_dependent(),
        args,
    )
    db = pool_effort_variants(db)

    models = select_models({r["_model"] for r in db.records})
    model_colors = assign_model_colors(models)

    user_all_ctxs = CELL_CONTEXT_TYPES["user_explicit"] + CELL_CONTEXT_TYPES["user_implicit"]
    tool_all_ctxs = CELL_CONTEXT_TYPES["tool_explicit"] + CELL_CONTEXT_TYPES["tool_implicit"]

    rows: list[dict] = []
    for model in models:
        mdb = db.filter(_model=model)
        user_all_db = mdb.filter_in("context_type", user_all_ctxs)
        tool_all_db = mdb.filter_in("context_type", tool_all_ctxs)
        t_e_db = mdb.filter_in("context_type", CELL_CONTEXT_TYPES["tool_explicit"])
        # Register-matched arm: user_turn_structured alone, not the user_explicit cell,
        # which pools it with prose user_turn. Against tool-explicit this holds the
        # structured register fixed and varies only the channel, which is the contrast
        # that separates a channel effect from a formatting one.
        u_s_db = mdb.filter(context_type="user_turn_structured")
        u_all = rate_fn(user_all_db)
        t_all = rate_fn(tool_all_db)
        u_e = rate_fn(mdb.filter_in("context_type", CELL_CONTEXT_TYPES["user_explicit"]))
        u_i = rate_fn(mdb.filter_in("context_type", CELL_CONTEXT_TYPES["user_implicit"]))
        t_e = rate_fn(t_e_db)
        t_i = rate_fn(mdb.filter_in("context_type", CELL_CONTEXT_TYPES["tool_implicit"]))
        rows.append(
            {
                "model": model,
                "label": MODEL_LABEL.get(model, model),
                "color": model_colors[model],
                "family": DIR_FAMILY.get(model, "Other"),
                "u_all": u_all,
                "t_all": t_all,
                "u_e": u_e,
                "u_i": u_i,
                "t_e": t_e,
                "t_i": t_i,
                "u_s": rate_fn(u_s_db),
                "delta": t_all[0] - u_all[0],
                "delta_ci": paired_delta(user_all_db, tool_all_db, args.metric),
                "delta_register": paired_delta(u_s_db, t_e_db, args.metric),
            }
        )

    n = len(rows)
    # 0.6in per model: the widest two-line tick label is 0.57in, so this is the narrowest
    # slot that still separates them. Width beyond that is spent by the paper composer,
    # which fits the figure to the column either way — a wider source only shrinks the text.
    fig, (ax_agg, ax_split) = plt.subplots(2, 1, figsize=(0.6 * n + 1.6, 8.0), sharex=True)
    for a in (ax_agg, ax_split):
        a.patch.set_alpha(0)

    xs = np.arange(n)

    def _draw_dumbbell(ax, user_tri, tool_tri, x, color, *, alpha=1.0, hatch=None, linestyle="-"):
        y_user, y_tool = user_tri[0], tool_tri[0]
        ax.plot([x, x], [y_user, y_tool], color=color, linewidth=1.8, alpha=alpha * 0.7, linestyle=linestyle, zorder=2)
        for (y, lo, hi), marker in [(user_tri, "o"), (tool_tri, "s")]:
            yerr = [[max(0.0, y - lo)], [max(0.0, hi - y)]]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=marker,
                markersize=8,
                linestyle="",
                color="black",
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=2,
                alpha=alpha,
                zorder=3,
                **ERRORBAR_KWARGS,
            )
            if hatch is not None:
                ax.scatter(
                    [x],
                    [y],
                    marker=marker,
                    s=64,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0,
                    hatch=hatch,
                    zorder=4,
                    alpha=alpha,
                )

    prev_fam = None
    for x, row in zip(xs, rows):
        color = row["color"]
        if prev_fam is not None and row["family"] != prev_fam:
            for a in (ax_agg, ax_split):
                a.axvline(x - 0.5, color="grey", lw=0.8, ls=":", alpha=0.6, zorder=1)
        prev_fam = row["family"]
        # Aggregated panel
        _draw_dumbbell(ax_agg, row["u_all"], row["t_all"], x, color)
        # Register-split panel: summary register (upper offset, solid)
        _draw_dumbbell(
            ax_split,
            row["u_e"],
            row["t_e"],
            x - 0.18,
            color,
            alpha=1.0,
            linestyle="-",
        )
        # Raw register (lower offset, dashed + faded)
        _draw_dumbbell(
            ax_split,
            row["u_i"],
            row["t_i"],
            x + 0.18,
            color,
            alpha=0.5,
            hatch="///",
            linestyle="--",
        )

    # Shared y-limits across panels for a like-for-like read.
    all_ys = []
    for r in rows:
        for tri in (r["u_all"], r["t_all"], r["u_e"], r["u_i"], r["t_e"], r["t_i"]):
            all_ys.extend([tri[1], tri[2]])
    pad = 0.03
    ylo = max(0.0, min(all_ys) - pad)
    yhi = min(1.0, max(all_ys) + pad)
    for a in (ax_agg, ax_split):
        a.set_ylim(ylo, yhi)
        a.set_ylabel(y_label)
        a.grid(axis="y", linestyle=":", alpha=0.4)
        a.tick_params(axis="both", length=0)

    ax_split.set_xticks(xs)
    ax_split.set_xticklabels([r["label"] for r in rows])
    ax_agg.set_title("Aggregated (role only)")
    ax_split.set_title("Register-split")
    fig.tight_layout()

    out = FIGURES_DIR / f"register_matched_dumbbell{metric_suffix}{eval_suffix}.svg"
    save_figure(fig, out)

    # Aggregated-only figure (no title, single panel). This one shares a paper row with a
    # square subfigure, and the composer hands the row's width out by aspect ratio, so what
    # the neighbour gets is set by this figure's width/height, not by its width alone.
    # 0.55in per model is the floor for the two-line tick labels: 3pt of clearance between
    # the closest pair at 15 models, against 6.6pt at 0.6 and an overlap at 0.5. The rest of
    # the flattening therefore comes from the height, which costs nothing in the source.
    fig_agg, ax_only = plt.subplots(figsize=(0.55 * n + 1.6, 5.2))
    ax_only.patch.set_alpha(0)
    prev_fam_agg = None
    for x, row in zip(xs, rows):
        if prev_fam_agg is not None and row["family"] != prev_fam_agg:
            ax_only.axvline(x - 0.5, color="grey", lw=0.8, ls=":", alpha=0.6, zorder=1)
        prev_fam_agg = row["family"]
        _draw_dumbbell(ax_only, row["u_all"], row["t_all"], x, row["color"])
    ax_only.set_ylim(ylo, yhi)
    # Half a slot each side, as in plot_convention_dumbbell.py. Matplotlib's default
    # category margin is 5% of the span, which at this many models is wider than that.
    ax_only.set_xlim(-0.5, n - 0.5)
    ax_only.set_ylabel(y_label)
    ax_only.set_xticks(xs)
    # Name only: a third line on an already two-line tick collides with its
    # neighbours, and delta_agg is in the CSV.
    ax_only.set_xticklabels([r["label"] for r in rows])
    ax_only.grid(axis="y", linestyle=":", alpha=0.4)
    ax_only.tick_params(axis="both", length=0)
    fig_agg.tight_layout()
    save_figure(fig_agg, FIGURES_DIR / f"register_matched_dumbbell_agg{metric_suffix}{eval_suffix}.svg")
    agg_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="black",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markeredgewidth=2,
            linestyle="",
            markersize=8,
            label="User-message",
        ),
        mlines.Line2D(
            [],
            [],
            marker="s",
            color="black",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markeredgewidth=2,
            linestyle="",
            markersize=8,
            label="Tool-return",
        ),
    ]
    save_legend(
        agg_handles,
        [h.get_label() for h in agg_handles],
        FIGURES_DIR / f"register_matched_dumbbell_agg{metric_suffix}{eval_suffix}_legend.svg",
        ncol=len(agg_handles),
    )
    save_table(
        out.with_suffix(".csv"),
        [
            {
                "model": r["model"],
                "model_label": r["label"],
                "family": r["family"],
                "f_user_all": r["u_all"][0],
                "f_user_all_ci_lo": r["u_all"][1],
                "f_user_all_ci_hi": r["u_all"][2],
                "f_tool_all": r["t_all"][0],
                "f_tool_all_ci_lo": r["t_all"][1],
                "f_tool_all_ci_hi": r["t_all"][2],
                "delta_agg": r["delta"],
                "delta_agg_ci_lo": r["delta_ci"][1],
                "delta_agg_ci_hi": r["delta_ci"][2],
                # Register-matched: tool-explicit minus user_turn_structured, the
                # contrast that holds format fixed and varies only the channel.
                "f_user_structured": r["u_s"][0],
                "f_user_structured_ci_lo": r["u_s"][1],
                "f_user_structured_ci_hi": r["u_s"][2],
                "delta_register": r["delta_register"][0],
                "delta_register_ci_lo": r["delta_register"][1],
                "delta_register_ci_hi": r["delta_register"][2],
                "f_user_explicit": r["u_e"][0],
                "f_user_explicit_ci_lo": r["u_e"][1],
                "f_user_explicit_ci_hi": r["u_e"][2],
                "f_user_implicit": r["u_i"][0],
                "f_user_implicit_ci_lo": r["u_i"][1],
                "f_user_implicit_ci_hi": r["u_i"][2],
                "f_tool_explicit": r["t_e"][0],
                "f_tool_explicit_ci_lo": r["t_e"][1],
                "f_tool_explicit_ci_hi": r["t_e"][2],
                "f_tool_implicit": r["t_i"][0],
                "f_tool_implicit_ci_lo": r["t_i"][1],
                "f_tool_implicit_ci_hi": r["t_i"][2],
            }
            for r in rows
        ],
    )
    print(f"Saved {out}")

    # Legend: marker (role), and panel-specific encodings (aggregated vs
    # split-by-register).
    handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="black",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markeredgewidth=2,
            linestyle="",
            markersize=8,
            label="User-message",
        ),
        mlines.Line2D(
            [],
            [],
            marker="s",
            color="black",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markeredgewidth=2,
            linestyle="",
            markersize=8,
            label="Tool-return",
        ),
        mlines.Line2D([], [], color="grey", linewidth=1.8, linestyle="-", label="Summary register (solid)"),
        mlines.Line2D(
            [], [], color="grey", linewidth=1.8, linestyle="--", alpha=0.5, label="Raw register (dashed, faded)"
        ),
    ]
    save_legend(handles, [h.get_label() for h in handles], out.with_name(out.stem + "_legend.svg"), ncol=len(handles))


if __name__ == "__main__":
    main()
