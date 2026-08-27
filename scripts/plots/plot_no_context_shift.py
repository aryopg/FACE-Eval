"""Cue-following rate (CFR) per cell: fraction of cued samples whose answer lands on
the user's preferred side.

Since no_context always produces aligns_with_preference=null (no preference to
align with), this equals P(aligns | cued) − P(aligns | no_context) = P(aligns | cued)
trivially. A row is counted iff judge.answer_aligns_with_preference is True.

Population is the one every primary figure uses (plot_h1_phase_diagram.load_clean):
C0, both parse_ok flags, causal-dependent, effort variants pooled, registry models.
This script read a wider population until 2026-08-12 — parse-ok rows only, no matched
baseline — which put a different row set behind the same CFR symbol as the phase
diagram and reversed the tool-explicit vs. user-implicit ordering.

Error bars = 95% scenario-cluster bootstrap CI.

Output: figures/no_context_shift.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots import _eval_aware_filter as eval_aware
from src.results.db import ResultsDB
from src.utils.plotting import (
    CELL_CONTEXT_TYPES,
    CELL_DISPLAY,
    CELLS_4,
    CUE_FOLLOWING_RATE_LABEL,
    pool_effort_variants,
    save_figure,
    save_table,
    select_models,
    setup_plot_style,
)

FIGURES_DIR = Path("figures")

CELL_COLORS: dict[str, str] = {
    "user_explicit": "#66c2a5",
    "user_implicit": "#a6d854",
    "tool_explicit": "#fc8d62",
    "tool_implicit": "#ffd92f",
}

_PARSE_OK = {"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True}


def _shifted(r: dict) -> bool:
    return (r.get("judge") or {}).get("answer_aligns_with_preference") is True


def load_clean() -> ResultsDB:
    return (
        ResultsDB.load_all(require_judged=True).filter(_convention="C0").filter(**_PARSE_OK).filter_causal_dependent()
    )


def compute_shift_rates(db: ResultsDB) -> dict[str, tuple[float, float, float]]:
    """Per-cell shift rate with 95% scenario-cluster bootstrap CI: (point, lo, hi)."""

    def agg(rs: list[dict]) -> float | None:
        return (sum(_shifted(r) for r in rs) / len(rs)) if rs else None

    results: dict[str, tuple[float, float, float]] = {}
    for cell in CELLS_4:
        sub = db.filter_in("context_type", CELL_CONTEXT_TYPES[cell])
        n_total = sub.count()
        if n_total == 0:
            results[cell] = (0.0, 0.0, 0.0)
            continue
        point, lo, hi = sub.cluster_bootstrap_ci(agg)
        results[cell] = (point, lo, hi)
        print(f"  {cell}: {point:.1%} aligned (95% CI [{lo:.3f}, {hi:.3f}], n={n_total})")
    return results


def main() -> None:
    global FIGURES_DIR
    parser = argparse.ArgumentParser()
    eval_aware.add_flag(parser)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    FIGURES_DIR = args.figures_dir
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    db, eval_suffix = eval_aware.apply(load_clean(), args)
    db = pool_effort_variants(db)
    db = db.filter_in("_model", sorted(select_models({r["_model"] for r in db.records})))
    print(f"Loaded {db.count()} rows over {len({r['_model'] for r in db.records})} checkpoints")

    rates = compute_shift_rates(db)

    means = [rates[c][0] for c in CELLS_4]
    los = [rates[c][1] for c in CELLS_4]
    his = [rates[c][2] for c in CELLS_4]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.patch.set_alpha(0)
    x = np.arange(len(CELLS_4))

    yerr = [[max(0.0, m - lo) for m, lo in zip(means, los)], [max(0.0, hi - m) for m, hi in zip(means, his)]]
    ax.bar(x, means, color=[CELL_COLORS[c] for c in CELLS_4], width=0.55, alpha=0.85, zorder=2)
    ax.errorbar(x, means, yerr=yerr, fmt="none", color="black", capsize=4, linewidth=1.5, zorder=3)

    y_top = max(his) * 1.15
    ax.set_ylim(0, y_top)
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_DISPLAY[c].replace(" ", "\n") for c in CELLS_4])
    ax.set_ylabel(CUE_FOLLOWING_RATE_LABEL)
    # ax.set_xlabel("Cell")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.tick_params(axis="both", length=0)
    fig.tight_layout()

    out = FIGURES_DIR / f"no_context_shift{eval_suffix}.svg"
    save_figure(fig, out)
    save_table(
        out.with_suffix(".csv"),
        [
            {
                "cell": c,
                "cell_label": CELL_DISPLAY[c],
                "align_rate": rates[c][0],
                "ci_lo": rates[c][1],
                "ci_hi": rates[c][2],
            }
            for c in CELLS_4
        ],
        columns=["cell", "cell_label", "align_rate", "ci_lo", "ci_hi"],
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
