"""Per-source register-matched dumbbell.

Same layout as `plot_register_matched_dumbbell.py`, but one figure per
source ∈ {profile, email, slack, notes, browser_history}. Source is
derived from the record `id` (the `source` column is unpopulated in
existing inference.jsonl files; we infer it from the id suffix pattern).

Outputs (per source):
  figures/register_matched_dumbbell_{source}.svg          (2 panels)
  figures/register_matched_dumbbell_agg_{source}.svg      (aggregated only)
  figures/register_matched_dumbbell_{source}.csv          (per-model rates)

A single shared legend is written once:
  figures/register_matched_dumbbell_by_source_legend.svg
  figures/register_matched_dumbbell_by_source_agg_legend.svg
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from scripts.plots import _eval_aware_filter as eval_aware
from scripts.plots.plot_register_matched_dumbbell import paired_delta
from src.results.db import ResultsDB
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


SOURCES: tuple[str, ...] = ("profile", "email", "slack", "notes", "browser_history")
SOURCE_LABEL: dict[str, str] = {
    "profile": "Profile",
    "email": "Email",
    "slack": "Slack",
    "notes": "Notes",
    "browser_history": "Browser history",
}

# The non-profile sources appear as a token between the context prefix and the
# side suffix in record ids (e.g. `political_001__explicit_email_liberal`).
# Profile rows have no token (e.g. `political_001__explicit_liberal`).
_SOURCE_TOKEN_RE = re.compile(
    r"__(?:explicit|implicit|user_turn|user_turn_structured|user_turn_implicit)_([a-z_]+?)_[a-z]+$"
)
_NON_PROFILE_TOKENS = {"email", "slack", "notes", "browser_history"}


def _infer_source(record_id: str | None, context_type: str | None) -> str | None:
    if not record_id or context_type == "none":
        return None
    for tok in _NON_PROFILE_TOKENS:
        # Match `_{tok}_` as a discrete segment to avoid spurious substring hits.
        if f"_{tok}_" in record_id:
            return tok
    return "profile"


def _attach_source(db: ResultsDB) -> ResultsDB:
    new_records = []
    for r in db.records:
        src = _infer_source(r.get("id"), r.get("context_type"))
        new_records.append({**r, "source": src})
    return ResultsDB(new_records)


def cond_rate(db: ResultsDB) -> tuple[float, float, float]:
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


def _compute_rows(
    db: ResultsDB, models: list[str], model_colors: dict[str, tuple], rate_fn: callable, metric: str
) -> list[dict]:
    user_all_ctxs = CELL_CONTEXT_TYPES["user_explicit"] + CELL_CONTEXT_TYPES["user_implicit"]
    tool_all_ctxs = CELL_CONTEXT_TYPES["tool_explicit"] + CELL_CONTEXT_TYPES["tool_implicit"]

    rows: list[dict] = []
    for model in models:
        mdb = db.filter(_model=model)
        user_all_db = mdb.filter_in("context_type", user_all_ctxs)
        tool_all_db = mdb.filter_in("context_type", tool_all_ctxs)
        u_all = rate_fn(user_all_db)
        t_all = rate_fn(tool_all_db)
        u_e = rate_fn(mdb.filter_in("context_type", CELL_CONTEXT_TYPES["user_explicit"]))
        u_i = rate_fn(mdb.filter_in("context_type", CELL_CONTEXT_TYPES["user_implicit"]))
        t_e = rate_fn(mdb.filter_in("context_type", CELL_CONTEXT_TYPES["tool_explicit"]))
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
                "delta": t_all[0] - u_all[0],
                "delta_ci": paired_delta(user_all_db, tool_all_db, metric),
            }
        )
    return rows


def _plot_one_source(
    source: str, db_all: ResultsDB, rate_fn: callable, y_label: str, metric_suffix: str, metric: str
) -> None:
    db = db_all.filter(source=source)
    models = select_models({r["_model"] for r in db.records})
    if not models:
        print(f"  skip {source}: no models with data")
        return
    model_colors = assign_model_colors(models)
    rows = _compute_rows(db, models, model_colors, rate_fn, metric)

    n = len(rows)
    # 0.6in per model, as in plot_register_matched_dumbbell.py: the narrowest slot that
    # still separates the two-line tick labels.
    fig, (ax_agg, ax_split) = plt.subplots(2, 1, figsize=(0.6 * n + 1.6, 8.0), sharex=True)
    for a in (ax_agg, ax_split):
        a.patch.set_alpha(0)

    xs = np.arange(n)
    prev_fam = None
    for x, row in zip(xs, rows):
        color = row["color"]
        if prev_fam is not None and row["family"] != prev_fam:
            for a in (ax_agg, ax_split):
                a.axvline(x - 0.5, color="grey", lw=0.8, ls=":", alpha=0.6, zorder=1)
        prev_fam = row["family"]
        _draw_dumbbell(ax_agg, row["u_all"], row["t_all"], x, color)
        _draw_dumbbell(ax_split, row["u_e"], row["t_e"], x - 0.18, color, alpha=1.0, linestyle="-")
        _draw_dumbbell(ax_split, row["u_i"], row["t_i"], x + 0.18, color, alpha=0.5, hatch="///", linestyle="--")

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
    ax_agg.set_title(f"Aggregated (role only) - {SOURCE_LABEL[source]}")
    ax_split.set_title(f"Register-split - {SOURCE_LABEL[source]}")
    fig.tight_layout()

    out = FIGURES_DIR / f"register_matched_dumbbell_{source}{metric_suffix}.svg"
    save_figure(fig, out)

    # Aggregated-only figure, one paper panel per source. 0.6in per model is the floor
    # that still separates the two-line tick labels.
    fig_agg, ax_only = plt.subplots(figsize=(0.6 * n + 1.6, 3.4))
    ax_only.patch.set_alpha(0)
    prev_fam_agg = None
    for x, row in zip(xs, rows):
        if prev_fam_agg is not None and row["family"] != prev_fam_agg:
            ax_only.axvline(x - 0.5, color="grey", lw=0.8, ls=":", alpha=0.6, zorder=1)
        prev_fam_agg = row["family"]
        _draw_dumbbell(ax_only, row["u_all"], row["t_all"], x, row["color"])
    ax_only.set_ylim(ylo, yhi)
    ax_only.set_xlim(-0.5, n - 0.5)
    ax_only.set_ylabel(y_label)
    ax_only.set_xticks(xs)
    # As in the pooled figure: a third tick line collides with its neighbours, and
    # delta_agg is in the CSV either way.
    ax_only.set_xticklabels([r["label"] for r in rows])
    ax_only.grid(axis="y", linestyle=":", alpha=0.4)
    ax_only.tick_params(axis="both", length=0)
    fig_agg.tight_layout()
    save_figure(fig_agg, FIGURES_DIR / f"register_matched_dumbbell_agg_{source}{metric_suffix}.svg")

    save_table(
        out.with_suffix(".csv"),
        [
            {
                "model": r["model"],
                "model_label": r["label"],
                "family": r["family"],
                "source": source,
                "f_user_all": r["u_all"][0],
                "f_user_all_ci_lo": r["u_all"][1],
                "f_user_all_ci_hi": r["u_all"][2],
                "f_tool_all": r["t_all"][0],
                "f_tool_all_ci_lo": r["t_all"][1],
                "f_tool_all_ci_hi": r["t_all"][2],
                "delta_agg": r["delta"],
                "delta_agg_ci_lo": r["delta_ci"][1],
                "delta_agg_ci_hi": r["delta_ci"][2],
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


def _save_shared_legends(eval_suffix: str) -> None:
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
            [],
            [],
            color="grey",
            linewidth=1.8,
            linestyle="--",
            alpha=0.5,
            label="Raw register (dashed, faded)",
        ),
    ]
    save_legend(
        handles,
        [h.get_label() for h in handles],
        FIGURES_DIR / f"register_matched_dumbbell_by_source{eval_suffix}_legend.svg",
        ncol=len(handles),
    )
    agg_handles = handles[:2]
    save_legend(
        agg_handles,
        [h.get_label() for h in agg_handles],
        FIGURES_DIR / f"register_matched_dumbbell_by_source_agg{eval_suffix}_legend.svg",
        ncol=len(agg_handles),
    )


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
    db = _attach_source(pool_effort_variants(db))

    for source in SOURCES:
        _plot_one_source(source, db, rate_fn, y_label, metric_suffix + eval_suffix, args.metric)

    _save_shared_legends(eval_suffix)


if __name__ == "__main__":
    main()
