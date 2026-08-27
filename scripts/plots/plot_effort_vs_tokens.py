"""Test-time compute: VCR, UAR and alignment vs. mean CoT tokens, per effort-swept model.

Recasts every effort sweep so the x-axis is a compute proxy (mean CoT token
count per cell, log scale) rather than the categorical effort label. One panel
per effort-swept model, because the effort levels are not shared: gpt-oss runs
low/medium/high, DeepSeek and GLM run high/max, and Inkling takes a float. Each
panel therefore keeps its own x-range, and points are annotated with the effort
value itself rather than an acronym.

Every model plotted here is effort-swept, so ResultsDB counts its reasoning with
that model's own tokenizer (see db.py::_counter_for) and x is a native token
count. The vocabularies differ between panels, so read across panels as a proxy;
within a panel the comparison is exact. A model whose tokenizer fails to load
falls back to `o200k_harmony` and says so at load time.
X-axis SE is cluster-bootstrap over `scenario_id` to match the y-axis.

Emits two figures with identical layout: conditional commitment-faithfulness
(the primary story) and answer-alignment rate (the secondary check showing
the alignment rate is roughly flat across compute).

Each figure is emitted twice: the full grid over every effort-swept model, and a
`_main` variant over MAIN_BASES for the main text. Colors are assigned from the
full model list, so a model keeps its shade between the two.

Outputs:
  figures/effort_vs_tokens.svg                 (F)
  figures/effort_vs_tokens_main.svg
  figures/effort_vs_tokens_alignment.svg       (Align_ans)
  figures/effort_vs_tokens_alignment_main.svg
  figures/effort_vs_tokens_legend.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from scripts.plots import _eval_aware_filter as eval_aware
from scripts.plots.plot_register_matched_dumbbell import paired_delta
from src.results.db import ResultsDB
from src.utils.plotting import (
    CELL_CONTEXT_TYPES,
    CELL_DISPLAY,
    CELLS_4,
    CUE_FOLLOWING_RATE_LABEL,
    EFFORT_VARIANTS,
    ERRORBAR_KWARGS,
    FONT_SIZE_AXIS_LABEL_WIDE,
    MODEL_LABEL_INLINE,
    UNVERBALIZED_ADOPTION_RATE_LABEL,
    VERBALIZED_COMMITMENT_RATE_LABEL,
    assign_model_colors,
    save_figure,
    save_legend,
    save_table,
    setup_plot_style,
    sort_models,
    sorted_effort_variants,
)

FIGURES_DIR = Path("figures")

CELL_STYLES: dict[str, tuple[str, str]] = {
    "user_explicit": ("-", "o"),
    "user_implicit": ("--", "o"),
    "tool_explicit": ("-", "D"),
    "tool_implicit": ("--", "D"),
}

# Every effort-swept model in the registry, in the usual family order.
BASES: list[str] = sort_models(EFFORT_VARIANTS)
MODEL_COLORS: dict[str, tuple] = assign_model_colors(BASES)
NCOLS = 3

# The main-text grid; the full six-panel version goes to the appendix.
MAIN_BASES: list[str] = [
    "openai_gpt-oss-120b",
    "deepseek-ai_DeepSeek-V4-Pro",
    "thinkingmachines_Inkling-NVFP4",
]

# One effort can land at two very different CoT lengths: DeepSeek and Inkling run
# their tool cells far shorter than their user cells, so a single label sits
# between the two clusters and names neither. GPT-OSS holds its four cells within
# a few percent of each other and still gets one label.
_LABEL_SPLIT_FACTOR = 1.5

_COND = {"judge.answer_aligns_with_preference": True}
_PARSE_OK = {"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True}


def cond_rate(db: ResultsDB, field: str = "judge.reasoning_tailoring_explicit") -> tuple[float, float, float]:
    """Conditional rate with 95% cluster bootstrap CI: (point, ci_lo, ci_hi)."""
    return db.filter(**_COND).cluster_mean_ci(field)


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


def align_rate(db: ResultsDB) -> tuple[float, float, float]:
    """Unconditional answer-alignment rate with 95% cluster bootstrap CI."""
    return db.cluster_mean_ci("judge.answer_aligns_with_preference")


def _mean_tokens_agg(records: list[dict]) -> float | None:
    if not records:
        return None
    return float(np.mean([int(r.get("reasoning_tokens") or 0) for r in records]))


def mean_cot_tokens(db: ResultsDB) -> tuple[float, float, float]:
    """Cluster-bootstrap 95% CI on mean CoT-token count, clustered on scenario_id."""
    if db.count() == 0:
        return float("nan"), float("nan"), float("nan")
    return db.cluster_bootstrap_ci(_mean_tokens_agg)


def _label_anchors(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Where to write one effort's label: one anchor per cluster of its cell positions.

    Takes this effort's (x, top-of-error-bar) per cell and returns (geometric-mean
    x, highest y) per cluster, ascending. Clusters split on a gap of
    `_LABEL_SPLIT_FACTOR` or more; the geometric mean is the midpoint on a log axis.
    """
    groups: list[list[tuple[float, float]]] = [[]]
    for x, y in sorted(points):
        if groups[-1] and x / groups[-1][-1][0] > _LABEL_SPLIT_FACTOR:
            groups.append([])
        groups[-1].append((x, y))
    return [(float(np.exp(np.mean(np.log([p[0] for p in g])))), max(p[1] for p in g)) for g in groups]


def _report_effort_separation(cells: dict) -> None:
    """Print how far consecutive efforts move the x-axis, per model.

    An effort knob that leaves the CoT length alone makes its panel a vertical
    smear rather than a compute sweep, which is grounds for dropping the panel.
    The same numbers are in the CSV, one row per (model, effort, cell).
    """
    print("Effort separation in mean CoT tokens:")
    for base, by_effort in cells.items():
        present = [e for e in (v.rsplit("_", 1)[-1] for v in sorted_effort_variants(base)) if e in by_effort]
        for lo, hi in zip(present, present[1:]):
            ratios, overlaps = [], 0
            for cell in sorted(set(by_effort[lo]) & set(by_effort[hi])):
                a, a_lo, a_hi = by_effort[lo][cell][0]
                b, b_lo, b_hi = by_effort[hi][cell][0]
                ratios.append(b / a)
                overlaps += not (a_hi < b_lo or b_hi < a_lo)
            print(
                f"  {MODEL_LABEL_INLINE.get(base, base):<20} {lo:>6} -> {hi:<6} "
                f"x{min(ratios):.2f}-{max(ratios):.2f}, CIs overlap in {overlaps}/{len(ratios)} cells"
            )


def _save_effort_deltas(db: ResultsDB, metric: str, out: Path) -> None:
    """Step-to-step effort deltas per (base model, cell), with a paired interval.

    The per-cell CIs in the main table are marginal, so "the rate falls as effort rises"
    rests on disjoint marginals rather than a tested difference. Effort variants run the
    same scenarios, which is what makes one scenario resample drive both arms here.
    """
    rows: list[dict] = []
    for base in BASES:
        variants = [v for v in sorted_effort_variants(base) if db.filter(_model=v).count()]
        for lower, higher in zip(variants, variants[1:]):
            for cell in CELLS_4:
                a = db.filter(_model=lower).filter_in("context_type", CELL_CONTEXT_TYPES[cell])
                b = db.filter(_model=higher).filter_in("context_type", CELL_CONTEXT_TYPES[cell])
                if not a.count() or not b.count():
                    continue
                point, lo, hi = paired_delta(a, b, metric)
                rows.append(
                    {
                        "base_model": base,
                        "effort_from": lower.rsplit("_", 1)[-1],
                        "effort_to": higher.rsplit("_", 1)[-1],
                        "cell": cell,
                        "cell_label": CELL_DISPLAY[cell],
                        "delta": point,
                        "delta_ci_lo": lo,
                        "delta_ci_hi": hi,
                        "excludes_zero": not (lo <= 0 <= hi),
                    }
                )
    save_table(out, rows)
    print(f"Saved {out}")


def main() -> None:
    global FIGURES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["verbalized", "covert"], default="verbalized")
    eval_aware.add_flag(parser)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    FIGURES_DIR = args.figures_dir
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_suffix = "_covert" if args.metric == "covert" else ""
    f_rate = covert_rate if args.metric == "covert" else cond_rate

    setup_plot_style(wide=True)
    plt.rcParams["axes.facecolor"] = "none"

    db, eval_suffix = eval_aware.apply(
        ResultsDB.load_all(require_judged=True).filter(_convention="C0").filter(**_PARSE_OK).filter_causal_dependent(),
        args,
    )

    # cells[base][effort][cell] = (t, A, F) where each is (mean, lo, hi).
    # A = unconditional answer-alignment rate; F = VCR or unverbalized adoption rate.
    cells: dict[
        str,
        dict[str, dict[str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]],
    ] = {}
    for base in BASES:
        for variant in sorted_effort_variants(base):
            effort = variant.rsplit("_", 1)[-1]
            mdb = db.filter(_model=variant)
            if mdb.count() == 0:
                continue
            for cell in CELLS_4:
                cdb = mdb.filter_in("context_type", CELL_CONTEXT_TYPES[cell])
                if cdb.count() == 0:
                    continue
                cells.setdefault(base, {}).setdefault(effort, {})[cell] = (
                    mean_cot_tokens(cdb),
                    align_rate(cdb),
                    f_rate(cdb),
                )

    _report_effort_separation(cells)

    def _draw(
        metric_idx: int, ylabel: str, out_path: Path, ylim: tuple | None = (0, 1.02), bases: list[str] = BASES
    ) -> None:
        nrows = -(-len(bases) // NCOLS)
        # sharey only: the point of one panel per model is that the token ranges differ.
        # Constrained layout, not tight_layout: it is the one that places supxlabel
        # against the tick labels instead of parking it at the figure edge, and it
        # gets the gap right for both the one-row and the two-row grid.
        fig, axes = plt.subplots(nrows, NCOLS, figsize=(4.2 * NCOLS, 3.5 * nrows), sharey=True, layout="constrained")
        flat = list(np.atleast_1d(axes).flat)
        for ax in flat[len(bases) :]:
            ax.set_visible(False)
        for ax, base in zip(flat, bases):
            label = MODEL_LABEL_INLINE.get(base, base)
            if base not in cells:
                ax.set_visible(False)
                continue
            color = MODEL_COLORS[base]
            # One annotation per effort, not per (effort, cell): the four cells sit
            # at nearly the same x for a given effort, so per-cell labels overprint.
            by_effort: dict[str, list[tuple[float, float]]] = {}
            for cell in CELLS_4:
                xs, x_los, x_his = [], [], []
                ys, y_los, y_his = [], [], []
                efforts: list[str] = []
                for variant in sorted_effort_variants(base):
                    effort = variant.rsplit("_", 1)[-1]
                    if effort not in cells[base] or cell not in cells[base][effort]:
                        continue
                    (t_mean, t_lo, t_hi), *_ = cells[base][effort][cell]
                    y_mean, y_lo, y_hi = cells[base][effort][cell][metric_idx]
                    xs.append(t_mean)
                    x_los.append(t_lo)
                    x_his.append(t_hi)
                    ys.append(y_mean)
                    y_los.append(y_lo)
                    y_his.append(y_hi)
                    efforts.append(effort)
                if not xs:
                    continue
                order = np.argsort(xs)
                xs = np.array(xs)[order]
                x_los = np.array(x_los)[order]
                x_his = np.array(x_his)[order]
                ys = np.array(ys)[order]
                y_los = np.array(y_los)[order]
                y_his = np.array(y_his)[order]
                efforts = [efforts[i] for i in order]
                xerr = [np.maximum(0.0, xs - x_los), np.maximum(0.0, x_his - xs)]
                yerr = [np.maximum(0.0, ys - y_los), np.maximum(0.0, y_his - ys)]
                linestyle, marker = CELL_STYLES[cell]
                ax.errorbar(
                    xs,
                    ys,
                    xerr=xerr,
                    yerr=yerr,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=7,
                    linewidth=1.6,
                    markeredgecolor="black",
                    markeredgewidth=2.0,
                    zorder=3,
                    **ERRORBAR_KWARGS,
                )
                for x, y_hi, e in zip(xs, y_his, efforts):
                    by_effort.setdefault(e, []).append((x, y_hi))
            for e, pts in by_effort.items():
                for anchor in _label_anchors(pts):
                    ax.annotate(
                        e.capitalize(),
                        anchor,
                        textcoords="offset points",
                        xytext=(0, 9),
                        ha="center",
                        fontsize=8,
                        color="#333333",
                    )
            ax.set_xscale("log")
            # Room for the effort annotations above the top points.
            ax.margins(x=0.20, y=0.18)
            ax.set_title(label)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, which="both", linestyle=":", alpha=0.4)
            ax.tick_params(axis="both", which="both", length=0, pad=5)
            # Pin the log ticks. Panels span anywhere from 0.1 to 2.3 decades, and
            # matplotlib picks its minor labels off that span: the wide panels get a
            # clean decade or two, the narrow ones a wall of overlapping 3 x 10^2.
            # Plain integers at 1/2/5 per decade stay legible at every span here.
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        # Figure-level labels: every panel measures the same two quantities, and a
        # per-row/column rule would drop the label whenever a panel has no data.
        # supxlabel/supylabel default to figure.labelsize, which setup_plot_style
        # leaves at matplotlib's oversized 'large'; the panels' own labels are 11.
        fig.supxlabel("Mean CoT length (tokens, log scale)", fontsize=FONT_SIZE_AXIS_LABEL_WIDE)
        fig.supylabel(ylabel, fontsize=FONT_SIZE_AXIS_LABEL_WIDE)
        save_figure(fig, out_path)
        print(f"Saved {out_path}")

    f_ylabel = UNVERBALIZED_ADOPTION_RATE_LABEL if args.metric == "covert" else VERBALIZED_COMMITMENT_RATE_LABEL
    f_ylim = None if args.metric == "covert" else (0, 1.02)
    out = FIGURES_DIR / f"effort_vs_tokens{metric_suffix}{eval_suffix}.svg"
    # `_main` is the three-model grid for the main text; the unsuffixed six-model
    # grid is the appendix version. Colors come from the full BASES either way, so
    # a model keeps its shade between the two.
    for bases, variant in ((BASES, ""), (MAIN_BASES, "_main")):
        _draw(
            metric_idx=2, ylabel=f_ylabel, out_path=out.with_name(f"{out.stem}{variant}.svg"), ylim=f_ylim, bases=bases
        )
        # No metric suffix: the alignment panel does not read --metric, so both runs
        # would write the same figure under two names.
        _draw(
            metric_idx=1,
            ylabel=CUE_FOLLOWING_RATE_LABEL,
            out_path=FIGURES_DIR / f"effort_vs_tokens_alignment{eval_suffix}{variant}.svg",
            bases=bases,
        )
    table_rows = [
        {
            "base_model": base,
            "effort": effort,
            "cell": cell,
            "cell_label": CELL_DISPLAY[cell],
            "mean_tokens": t[0],
            "mean_tokens_ci_lo": t[1],
            "mean_tokens_ci_hi": t[2],
            "A": a[0],
            "A_ci_lo": a[1],
            "A_ci_hi": a[2],
            "F": f[0],
            "F_ci_lo": f[1],
            "F_ci_hi": f[2],
        }
        for base, by_effort in cells.items()
        for effort, by_cell in by_effort.items()
        for cell, (t, a, f) in by_cell.items()
    ]
    save_table(out.with_suffix(".csv"), table_rows)
    _save_effort_deltas(db, args.metric, FIGURES_DIR / f"effort_deltas{metric_suffix}{eval_suffix}.csv")

    # Legend: channels (linestyle + marker, grey) and model shade (one color per
    # base model). Channel encoding shared across panels; color encodes model.
    handles = [
        mlines.Line2D(
            [],
            [],
            color="grey",
            linestyle=CELL_STYLES[cell][0],
            marker=CELL_STYLES[cell][1],
            markersize=7,
            linewidth=1.6,
            label=CELL_DISPLAY[cell],
        )
        for cell in CELLS_4
    ]
    # No effort key: each panel is one model, so points carry the effort value itself.
    save_legend(handles, [h.get_label() for h in handles], out.with_name(out.stem + "_legend.svg"), ncol=len(handles))


if __name__ == "__main__":
    main()
