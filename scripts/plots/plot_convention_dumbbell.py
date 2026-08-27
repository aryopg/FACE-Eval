"""Convention contrast dumbbell plots (3-panel figure).

Three contrasts, each as one panel:
  A — C0 vs C3  : Default vs. Directive
  B — C0 vs MC0 : Default vs. Monitor-Aware
  C — C3 vs MC3 : Directive vs. Monitor-Aware + Directive

Mode v1 (default): dumbbell-of-deltas.
  x-axis = Δ = F(tool) - F(user). One segment per model from Δ_conv_A (circle)
  to Δ_conv_B (square). Errorbars = SEM(Δ) = sqrt(SEM_tool² + SEM_user²) on
  each endpoint. Vertical reference line at Δ = 0.

Mode v2: raw-F dumbbell pairs.
  x-axis = F(faithfulness). Two dumbbells per row offset vertically: conv_A
  (full opacity, upper) and conv_B (faded, lower). Circle = user channel,
  square = tool channel.

All panels use a global model y-index so rows align across panels even when
some models lack data for a particular convention pair.

Usage:
  python scripts/plots/plot_convention_dumbbell.py
  python scripts/plots/plot_convention_dumbbell.py --mode v2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from scripts.plots import _eval_aware_filter as eval_aware
from src.results.db import ResultsDB
from src.utils.plotting import (
    DIR_FAMILY,
    ERRORBAR_KWARGS,
    MODEL_LABEL,
    UNVERBALIZED_ADOPTION_RATE_LABEL,
    VERBALIZED_COMMITMENT_RATE_LABEL,
    assign_model_colors,
    pool_effort_variants,
    save_figure,
    save_legend,
    select_models,
    setup_plot_style,
)

FIGURES_DIR = Path("figures")

_COND = {"judge.answer_aligns_with_preference": True}
_PARSE_OK = {"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True}
_FIELD = "judge.reasoning_tailoring_explicit"

CONVENTION_LABEL: dict[str, str] = {
    "C0": "Default",
    "C3": "Directive",
    "MC0": "Monitor-Aware",
    "MC3": "Monitor-Aware + Directive",
}

# (conv_a, conv_b, panel title)
PANELS: list[tuple[str, str, str]] = [
    ("C0", "C3", "Default vs. Directive"),
    ("C0", "MC0", "Default vs. Monitor-Aware"),
    # ("C3", "MC3", "Directive vs. Monitor-Aware + Directive"),
]


_USER_CTXS = {"user_turn", "user_turn_structured"}
_TOOL_CTXS = {"explicit", "implicit"}


def fast_bootstrap(
    user_recs: list[dict], tool_recs: list[dict], field: str, n_boot: int = 2000, seed: int = 42
) -> dict:

    user_clusters = {}
    for r in user_recs:
        k = r.get("scenario_id")
        user_clusters.setdefault(k, []).append(r[field])

    tool_clusters = {}
    for r in tool_recs:
        k = r.get("scenario_id")
        tool_clusters.setdefault(k, []).append(r[field])

    all_keys = list(set(user_clusters.keys()) | set(tool_clusters.keys()))

    user_stats = np.array(
        [(sum(user_clusters.get(k, [])), len(user_clusters.get(k, []))) for k in all_keys], dtype=float
    )
    tool_stats = np.array(
        [(sum(tool_clusters.get(k, [])), len(tool_clusters.get(k, []))) for k in all_keys], dtype=float
    )

    u_sum = user_stats[:, 0].sum()
    u_count = user_stats[:, 1].sum()
    t_sum = tool_stats[:, 0].sum()
    t_count = tool_stats[:, 1].sum()

    point_u = u_sum / u_count if u_count > 0 else 0.0
    point_t = t_sum / t_count if t_count > 0 else 0.0
    point_d = point_u - point_t

    rng = np.random.default_rng(seed)
    n = len(all_keys)
    if n < 2 or n_boot < 2:
        return {
            "f_user": point_u,
            "lo_user": point_u,
            "hi_user": point_u,
            "f_tool": point_t,
            "lo_tool": point_t,
            "hi_tool": point_t,
            "delta": point_d,
            "lo_delta": point_d,
            "hi_delta": point_d,
        }

    idx = rng.integers(0, n, size=(n_boot, n))

    b_u_sum = user_stats[idx, 0].sum(axis=1)
    b_u_count = user_stats[idx, 1].sum(axis=1)
    b_t_sum = tool_stats[idx, 0].sum(axis=1)
    b_t_count = tool_stats[idx, 1].sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        b_u = np.where(b_u_count > 0, b_u_sum / b_u_count, 0.0)
        b_t = np.where(b_t_count > 0, b_t_sum / b_t_count, 0.0)

    b_d = b_u - b_t

    alpha = 0.05
    return {
        "f_user": point_u,
        "lo_user": float(np.quantile(b_u, alpha / 2)),
        "hi_user": float(np.quantile(b_u, 1 - alpha / 2)),
        "f_tool": point_t,
        "lo_tool": float(np.quantile(b_t, alpha / 2)),
        "hi_tool": float(np.quantile(b_t, 1 - alpha / 2)),
        "delta": point_d,
        "lo_delta": float(np.quantile(b_d, alpha / 2)),
        "hi_delta": float(np.quantile(b_d, 1 - alpha / 2)),
    }


def _channel_records(mdb: ResultsDB, convention: str, metric: str) -> tuple[list[dict], list[dict], str]:
    """(user records, tool records, scored field) for one convention under one metric."""
    if metric == "covert":
        cdb = mdb.filter(_convention=convention)
        field = "_is_covert"
    else:
        cdb = mdb.filter(_convention=convention).filter(**_COND)
        field = "_is_binary"

    user_recs = [
        *cdb.filter(context_type="user_turn").records,
        *cdb.filter(context_type="user_turn_structured").records,
    ]
    tool_recs = [
        *cdb.filter(context_type="explicit").records,
        *cdb.filter(context_type="implicit").records,
    ]
    return user_recs, tool_recs, field


def _channels(mdb: ResultsDB, convention: str, metric: str = "verbalized") -> dict | None:
    user_recs, tool_recs, field = _channel_records(mdb, convention, metric)
    if not user_recs or not tool_recs:
        return None

    return fast_bootstrap(user_recs, tool_recs, field)


def _cluster_stats(records: list[dict], key_index: dict[str, int], field: str) -> np.ndarray:
    """(sum, count) of `field` per scenario, on the shared scenario index."""
    stats = np.zeros((len(key_index), 2), dtype=float)
    for r in records:
        i = key_index.get(r.get("scenario_id"))
        if i is None:
            continue
        stats[i, 0] += r[field]
        stats[i, 1] += 1
    return stats


def _did_bootstrap(
    mdb: ResultsDB, conv_a: str, conv_b: str, metric: str, n_boot: int = 2000, seed: int = 42
) -> dict | None:
    """Joint cluster bootstrap on the convention contrast, everything as B - A.

    One scenario resample drives all four rates, so the A/B correlation survives and
    the CI is not the inflated one that differencing two independent bootstraps gives.
    Restricted to scenarios seen under both conventions: a scenario present in only one
    contributes to that arm's rate and an empty count to the other's, which would push
    the difference toward whichever convention has the fuller sweep.

      did    = delta_b - delta_a, delta = F(user) - F(tool)
      d_user = F_user_b - F_user_a
      d_tool = F_tool_b - F_tool_a

    The channel gap runs negative on UAR (tool above user), so a positive `did` is a gap
    that narrowed. Read `d_user` and `d_tool` to see which side moved.
    """
    user_a, tool_a, field = _channel_records(mdb, conv_a, metric)
    user_b, tool_b, _ = _channel_records(mdb, conv_b, metric)
    if not all((user_a, tool_a, user_b, tool_b)):
        return None

    keys_a = {r.get("scenario_id") for r in user_a + tool_a}
    keys_b = {r.get("scenario_id") for r in user_b + tool_b}
    keys = sorted(keys_a & keys_b)
    if len(keys) < 2:
        return None
    key_index = {k: i for i, k in enumerate(keys)}
    stats = [_cluster_stats(recs, key_index, field) for recs in (user_a, tool_a, user_b, tool_b)]

    totals = [(s[:, 0].sum(), s[:, 1].sum()) for s in stats]
    if any(c == 0 for _s, c in totals):
        return None
    p_ua, p_ta, p_ub, p_tb = (s / c for s, c in totals)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    sums = [s[idx, 0].sum(axis=1) for s in stats]
    counts = [s[idx, 1].sum(axis=1) for s in stats]
    # An empty resample of any of the four groups makes the difference undefined, so
    # drop those replicates rather than reading them as a rate of zero.
    keep = np.logical_and.reduce([c > 0 for c in counts])
    if keep.sum() < 2:
        return None
    b_ua, b_ta, b_ub, b_tb = (s[keep] / c[keep] for s, c in zip(sums, counts))

    alpha = 0.05

    def _ci(point_value: float, boot: np.ndarray) -> tuple[float, float, float]:
        return point_value, float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))

    did = _ci((p_ub - p_tb) - (p_ua - p_ta), (b_ub - b_tb) - (b_ua - b_ta))
    d_user = _ci(p_ub - p_ua, b_ub - b_ua)
    d_tool = _ci(p_tb - p_ta, b_tb - b_ta)
    return {
        "did": did[0],
        "did_lo": did[1],
        "did_hi": did[2],
        "d_user": d_user[0],
        "d_user_lo": d_user[1],
        "d_user_hi": d_user[2],
        "d_tool": d_tool[0],
        "d_tool_lo": d_tool[1],
        "d_tool_hi": d_tool[2],
        "n_scenarios": len(keys),
        "n_unpaired": len(keys_a ^ keys_b),
    }


def _collect_rows(
    db: ResultsDB,
    models: list[str],
    x_of: dict[str, int],
    conv_a: str,
    conv_b: str,
    metric: str = "verbalized",
) -> list[dict]:
    rows = []
    for model in models:
        mdb = db.filter(_model=model)
        ch_a = _channels(mdb, conv_a, metric=metric)
        ch_b = _channels(mdb, conv_b, metric=metric)
        if ch_a is None or ch_b is None:
            continue
        rows.append(
            {
                "model": model,
                "x": x_of[model],
                "label": MODEL_LABEL.get(model, model),
                "family": DIR_FAMILY.get(model, "Other"),
                "ch_a": ch_a,
                "ch_b": ch_b,
                "did": _did_bootstrap(mdb, conv_a, conv_b, metric),
            }
        )
    return rows


def _family_separators(models: list[str], x_of: dict[str, int]) -> list[float]:
    """X-positions of vertical separators between family groups (global model list)."""
    seps = []
    prev_fam = None
    for m in models:
        fam = DIR_FAMILY.get(m, "Other")
        if prev_fam is not None and fam != prev_fam:
            seps.append(x_of[m] - 0.5)
        prev_fam = fam
    return seps


def _label_model_axis(ax, global_models: list[str], is_bottom: bool) -> None:
    """Models sit on x; only the bottom panel of a stacked pair names them."""
    n = len(global_models)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(list(range(n)))
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in global_models] if is_bottom else [""] * n)


def _draw_panel_v1(
    ax,
    rows: list[dict],
    model_colors: dict,
    title: str,
    is_bottom: bool,
    global_models: list[str],
    x_of: dict[str, int],
    sep_xs: list[float],
) -> None:
    for sep_x in sep_xs:
        ax.axvline(sep_x, color="grey", lw=0.8, ls=":", alpha=0.6, zorder=1)

    all_los, all_his = [], []
    for row in rows:
        color = model_colors[row["model"]]
        a, b = row["ch_a"], row["ch_b"]
        x = row["x"]
        ax.plot([x, x], [a["delta"], b["delta"]], color=color, linewidth=1.8, alpha=0.7, zorder=2)
        for ch, marker in [(a, "o"), (b, "s")]:
            yerr = [[max(0.0, ch["delta"] - ch["lo_delta"])], [max(0.0, ch["hi_delta"] - ch["delta"])]]
            ax.errorbar(
                x,
                ch["delta"],
                yerr=yerr,
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
            all_los.append(ch["lo_delta"])
            all_his.append(ch["hi_delta"])

    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5, zorder=1)

    pad = 0.04
    lo = (min(all_los) - pad) if all_los else -0.5
    hi = (max(all_his) + pad) if all_his else 0.5
    ax.set_ylim(lo, hi)

    _label_model_axis(ax, global_models, is_bottom)
    ax.set_title(title, pad=4)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.tick_params(axis="both", length=0)


def _draw_panel_v2(
    ax,
    rows: list[dict],
    model_colors: dict,
    title: str,
    is_bottom: bool,
    global_models: list[str],
    x_of: dict[str, int],
    sep_xs: list[float],
) -> None:
    offset = 0.18
    for sep_x in sep_xs:
        ax.axvline(sep_x, color="grey", lw=0.8, ls=":", alpha=0.6, zorder=1)

    all_los, all_his = [], []
    for row in rows:
        color = model_colors[row["model"]]
        x = row["x"]
        for ch, xo, alpha in [(row["ch_a"], -offset, 1.0), (row["ch_b"], +offset, 0.45)]:
            xr = x + xo
            ax.plot([xr, xr], [ch["f_user"], ch["f_tool"]], color=color, linewidth=1.8, alpha=alpha * 0.7, zorder=2)
            for val, lo, hi, marker in [
                (ch["f_user"], ch["lo_user"], ch["hi_user"], "o"),
                (ch["f_tool"], ch["lo_tool"], ch["hi_tool"], "s"),
            ]:
                fc = (*mcolors.to_rgb(color), alpha)
                yerr = [[max(0.0, val - lo)], [max(0.0, hi - val)]]
                ax.errorbar(
                    xr,
                    val,
                    yerr=yerr,
                    marker=marker,
                    markersize=8,
                    linestyle="",
                    color="black",
                    markerfacecolor=fc,
                    markeredgecolor="black",
                    markeredgewidth=2,
                    zorder=3,
                    **ERRORBAR_KWARGS,
                )
                all_los.append(lo)
                all_his.append(hi)

    pad = 0.02
    ylo = (min(all_los) - pad) if all_los else 0.0
    yhi = (max(all_his) + pad) if all_his else 1.0
    ax.set_ylim(ylo, yhi)

    _label_model_axis(ax, global_models, is_bottom)
    ax.set_title(title, pad=4)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.tick_params(axis="both", length=0)


def main() -> None:
    global FIGURES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["v1", "v2"], default="v2")
    parser.add_argument("--metric", choices=["verbalized", "covert"], default="verbalized")
    eval_aware.add_flag(parser)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    FIGURES_DIR = args.figures_dir
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    setup_plot_style(wide=True)
    plt.rcParams["axes.facecolor"] = "none"

    db, eval_suffix = eval_aware.apply(
        ResultsDB.load_all(require_judged=True).filter(**_PARSE_OK).filter_causal_dependent(),
        args,
    )
    db = pool_effort_variants(db)

    for r in db.records:
        r["_is_user"] = r.get("context_type") in _USER_CTXS
        r["_is_tool"] = r.get("context_type") in _TOOL_CTXS
        j = r.get("judge") or {}
        r["_is_covert"] = j.get("answer_aligns_with_preference") is True and not j.get("reasoning_tailoring_explicit")
        r["_is_binary"] = 1 if j.get("reasoning_tailoring_explicit") else 0

    all_models = {r["_model"] for r in db.records}
    models = select_models(all_models)
    x_of = {m: i for i, m in enumerate(models)}
    model_colors = assign_model_colors(models)
    sep_xs = _family_separators(models, x_of)

    metric_suffix = "_covert" if args.metric == "covert" else ""

    panel_rows = []
    for conv_a, conv_b, title in PANELS:
        rows = _collect_rows(db, models, x_of, conv_a, conv_b, metric=args.metric)
        print(f"Panel '{title}': {len(rows)} / {len(models)} models have data")
        panel_rows.append(rows)

    csv_path = FIGURES_DIR / f"convention_dumbbell_{args.mode}{metric_suffix}{eval_suffix}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "panel",
                "conv_a",
                "conv_b",
                "model",
                "label",
                "family",
                "f_user_a",
                "f_user_a_ci_lo",
                "f_user_a_ci_hi",
                "f_tool_a",
                "f_tool_a_ci_lo",
                "f_tool_a_ci_hi",
                "delta_a",
                "delta_a_ci_lo",
                "delta_a_ci_hi",
                "f_user_b",
                "f_user_b_ci_lo",
                "f_user_b_ci_hi",
                "f_tool_b",
                "f_tool_b_ci_lo",
                "f_tool_b_ci_hi",
                "delta_b",
                "delta_b_ci_lo",
                "delta_b_ci_hi",
                # Joint bootstrap over the paired scenarios, all as B - A.
                "did",
                "did_ci_lo",
                "did_ci_hi",
                "d_user",
                "d_user_ci_lo",
                "d_user_ci_hi",
                "d_tool",
                "d_tool_ci_lo",
                "d_tool_ci_hi",
                "n_scenarios_paired",
                "n_scenarios_unpaired",
            ]
        )
        for (conv_a, conv_b, title), rows in zip(PANELS, panel_rows):
            for row in rows:
                a, b = row["ch_a"], row["ch_b"]
                d = row["did"] or {}
                writer.writerow(
                    [
                        title,
                        conv_a,
                        conv_b,
                        row["model"],
                        row["label"],
                        row["family"],
                        a["f_user"],
                        a["lo_user"],
                        a["hi_user"],
                        a["f_tool"],
                        a["lo_tool"],
                        a["hi_tool"],
                        a["delta"],
                        a["lo_delta"],
                        a["hi_delta"],
                        b["f_user"],
                        b["lo_user"],
                        b["hi_user"],
                        b["f_tool"],
                        b["lo_tool"],
                        b["hi_tool"],
                        b["delta"],
                        b["lo_delta"],
                        b["hi_delta"],
                        d.get("did"),
                        d.get("did_lo"),
                        d.get("did_hi"),
                        d.get("d_user"),
                        d.get("d_user_lo"),
                        d.get("d_user_hi"),
                        d.get("d_tool"),
                        d.get("d_tool_lo"),
                        d.get("d_tool_hi"),
                        d.get("n_scenarios"),
                        d.get("n_unpaired"),
                    ]
                )
    print(f"Saved {csv_path}")

    global_n = len(models)
    # 0.6in per model, as in plot_register_matched_dumbbell.py: the narrowest slot that
    # still separates the two-line tick labels.
    fig, axes = plt.subplots(2, 1, figsize=(0.6 * global_n + 1.6, 8.0), sharex=True)

    for i, ((conv_a, conv_b, title), rows) in enumerate(zip(PANELS, panel_rows)):
        ax = axes[i]
        ax.patch.set_alpha(0)
        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, pad=4)
            continue

        kwargs = dict(
            rows=rows,
            model_colors=model_colors,
            title=title,
            is_bottom=(i == len(PANELS) - 1),
            global_models=models,
            x_of=x_of,
            sep_xs=sep_xs,
        )
        if args.mode == "v1":
            _draw_panel_v1(ax, **kwargs)
            if args.metric == "covert":
                ax.set_ylabel(r"$\Delta$ Unverbalized Adoption Rate (user $-$ tool)")
            else:
                ax.set_ylabel(r"$\Delta\,P(\mathrm{Comm}_{\mathrm{CoT}} \mid \cdot)$ (user $-$ tool)")
        else:
            _draw_panel_v2(ax, **kwargs)
            ax.set_ylabel(
                UNVERBALIZED_ADOPTION_RATE_LABEL if args.metric == "covert" else VERBALIZED_COMMITMENT_RATE_LABEL
            )

    fig.tight_layout(h_pad=0.3)

    out = FIGURES_DIR / f"convention_dumbbell_{args.mode}{metric_suffix}{eval_suffix}.svg"
    save_figure(fig, out)
    print(f"Saved {out}")

    # One shared legend: marker shape encodes convention position (first / second named
    # in each panel title); family colour encodes model.
    if args.mode == "v1":
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
                label="First system prompt",
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
                label="Second system prompt",
            ),
        ]
    else:
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
            mlines.Line2D([], [], color="#888888", linewidth=2.0, alpha=1.0, label="First system prompt"),
            mlines.Line2D([], [], color="#888888", linewidth=2.0, alpha=0.45, label="Second system prompt"),
        ]

    save_legend(
        handles,
        [h.get_label() for h in handles],
        out.with_name(out.stem + "_legend.svg"),
        ncol=4,
    )
    print("Saved legend")


if __name__ == "__main__":
    main()
