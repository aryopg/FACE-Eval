"""H1 — role × register interaction (gap-of-gaps scatter).

For each model, compute the within-role register gap of conditional
faithfulness, oriented so positive = predicted H3 direction:

    Δ_tool = F(Tool Explicit) − F(Tool Implicit)
    Δ_user = F(User Explicit) − F(User Implicit)

where F is the conditional rate P(L_signal | aligned) on the
causally-dependent slice (stance ≠ 'none'). User (Explicit) pools
`user_turn` + `user_turn_structured` per the 2026-05-17 cell-pooling
decision; the two share register and only differ in template.

Headline figure: Δ_tool on x, Δ_user on y, one dot per model with
scenario-cluster bootstrap CIs. Reference lines x=0, y=0, y=x answer:

    - upper-right quadrant : register effect runs in predicted direction
                             on both roles
    - above y=x            : user-side register effect larger than
                             tool-side (i.e. role × register interaction)
    - on y=x               : no interaction

Pooled mean shown as a black star with crosshair CI (mean of per-model
gaps; CI = bootstrap over models).

L3 (headline): tailoring_explicit

Outputs:
  figures/h1_register_l3.svg + _legend.svg + .csv
  figures/h1_register_forest_l3.svg + .csv   (appendix)
  figures/h1_register_forest_l1.svg + .csv   (appendix)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from scripts.plots import _eval_aware_filter as eval_aware
from src.results.db import ResultsDB
from src.utils.plotting import (
    CELL_CONTEXT_TYPES,
)
from src.utils.plotting import DIR_FAMILY as _DIR_FAMILY
from src.utils.plotting import (
    FAMILY_COLORS,
    MODEL_PARAMS,
    UNVERBALIZED_ADOPTION_RATE_LABEL,
    VERBALIZED_COMMITMENT_RATE_LABEL,
    conditional_faithfulness_label,
    pool_effort_variants,
    save_figure,
    save_legend,
    save_table,
    select_models,
    setup_plot_style,
)

FIGURES_DIR = Path("figures")

MODEL_LABEL: dict[str, str] = {
    "Qwen_Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen_Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen_Qwen3.5-27B": "Qwen3.5-27B",
    "google_gemma-4-E4B-it": "Gemma4-E4B",
    "google_gemma-4-26B-A4B-it": "Gemma4-26B",
    "google_gemma-4-31B-it": "Gemma4-31B",
    "allenai_Olmo-3-7B-Think": "OLMo3-7B",
    "allenai_Olmo-3.1-32B-Think": "OLMo3.1-32B",
    "openai_gpt-oss-20b": "GPT-OSS-20B",
    "openai_gpt-oss-120b": "GPT-OSS-120B",
}

N_BOOT = 2000
BOOT_SEED = 42


def load_clean(convention: str) -> ResultsDB:
    return pool_effort_variants(
        ResultsDB.load_all(require_judged=True)
        .filter(_convention=convention)
        .filter(**{"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True})
        .filter_causal_dependent()
    )


def _family_shade(family_color: str, rank: int, n: int) -> tuple:
    c = np.array(mcolors.to_rgb(family_color))
    if n == 1:
        return tuple(c.tolist())
    light = 0.4 * (1 - rank / (n - 1))
    return tuple(np.clip(c + (1 - c) * light, 0, 1).tolist())


def _assign_model_colors(models: list[str]) -> dict[str, tuple]:
    by_family: dict[str, list[tuple[int, str]]] = {}
    for m in models:
        fam = _DIR_FAMILY.get(m, "Other")
        by_family.setdefault(fam, []).append((MODEL_PARAMS.get(m, 0), m))
    colors: dict[str, tuple] = {}
    for fam, entries in by_family.items():
        entries.sort()
        fam_color = FAMILY_COLORS.get(fam, "#888")
        for rank, (_, m) in enumerate(entries):
            colors[m] = _family_shade(fam_color, rank, len(entries))
    return colors


def _scenario_tally(
    records: list[dict], explicit_cts: tuple[str, ...], implicit_cts: tuple[str, ...], cot_field: str
) -> np.ndarray:
    """Per-scenario (n_total_e, n_ac_e, n_signal_e, n_total_i, n_ac_i, n_signal_i).

    n_total_* : all records in that register bucket (cued conditions)
    n_ac_*    : subset where answer is aligned with the user's preferred side
    n_signal_*: subset of n_ac where the CoT field is True
    """
    e_set, i_set = set(explicit_cts), set(implicit_cts)
    by_scenario: dict[str, list[int]] = {}
    for r in records:
        rj = r["judge"]
        ct = r["context_type"]
        if ct in e_set:
            base = 0
        elif ct in i_set:
            base = 3
        else:
            continue
        aligned = rj.get("answer_aligns_with_preference") is True
        signal = rj.get(cot_field) is True
        sid = r["scenario_id"]
        if sid not in by_scenario:
            by_scenario[sid] = [0, 0, 0, 0, 0, 0]
        by_scenario[sid][base] += 1  # n_total
        if aligned:
            by_scenario[sid][base + 1] += 1  # n_ac
            if signal:
                by_scenario[sid][base + 2] += 1  # n_signal
    if not by_scenario:
        return np.zeros((0, 6), dtype=float)
    return np.array(list(by_scenario.values()), dtype=float)


def _cell_stats(
    stats: np.ndarray, metric: str = "verbalized", n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> dict[str, float]:
    """Per-(role) explicit/implicit rates, gap, and scenario-cluster bootstrap 95% CIs.

    Column layout: (n_total_e, n_ac_e, n_signal_e, n_total_i, n_ac_i, n_signal_i)

    verbalized: F = P(signal | aligned) = n_signal / n_ac  (denominator = n_ac)
    covert:     F = P(align ∧ ¬signal | cued) = (n_ac - n_signal) / n_total

    The gap CI is computed within the bootstrap (paired across scenarios), so it
    correctly captures within-scenario covariance between the two arms.
    """
    out = {
        "F_e": float("nan"),
        "F_e_lo": float("nan"),
        "F_e_hi": float("nan"),
        "F_i": float("nan"),
        "F_i_lo": float("nan"),
        "F_i_hi": float("nan"),
        "gap": float("nan"),
        "gap_lo": float("nan"),
        "gap_hi": float("nan"),
    }
    if len(stats) == 0:
        return out
    totals = stats.sum(axis=0)
    # columns: 0=n_total_e, 1=n_ac_e, 2=n_signal_e, 3=n_total_i, 4=n_ac_i, 5=n_signal_i
    if metric == "covert":
        denom_e, num_e = totals[0], totals[1] - totals[2]
        denom_i, num_i = totals[3], totals[4] - totals[5]
    else:  # verbalized
        denom_e, num_e = totals[1], totals[2]
        denom_i, num_i = totals[4], totals[5]
    if denom_e == 0 or denom_i == 0:
        return out
    out["F_e"] = float(num_e / denom_e)
    out["F_i"] = float(num_i / denom_i)
    out["gap"] = out["F_i"] - out["F_e"] if metric == "covert" else out["F_e"] - out["F_i"]
    n_s = len(stats)
    if n_boot < 2 or n_s < 2:
        out["F_e_lo"] = out["F_e_hi"] = out["F_e"]
        out["F_i_lo"] = out["F_i_hi"] = out["F_i"]
        out["gap_lo"] = out["gap_hi"] = out["gap"]
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_s, size=(n_boot, n_s))
    b = stats[idx].sum(axis=1)  # shape (n_boot, 6)
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "covert":
            f_e = np.where(b[:, 0] > 0, (b[:, 1] - b[:, 2]) / b[:, 0], np.nan)
            f_i = np.where(b[:, 3] > 0, (b[:, 4] - b[:, 5]) / b[:, 3], np.nan)
        else:
            f_e = np.where(b[:, 1] > 0, b[:, 2] / b[:, 1], np.nan)
            f_i = np.where(b[:, 4] > 0, b[:, 5] / b[:, 4], np.nan)
    f_e_ok = f_e[np.isfinite(f_e)]
    f_i_ok = f_i[np.isfinite(f_i)]
    if len(f_e_ok) >= 2:
        out["F_e_lo"], out["F_e_hi"] = float(np.quantile(f_e_ok, 0.025)), float(np.quantile(f_e_ok, 0.975))
    if len(f_i_ok) >= 2:
        out["F_i_lo"], out["F_i_hi"] = float(np.quantile(f_i_ok, 0.025)), float(np.quantile(f_i_ok, 0.975))
    raw_gaps = (f_i - f_e) if metric == "covert" else (f_e - f_i)
    gaps = raw_gaps[np.isfinite(raw_gaps)]
    if len(gaps) >= 2:
        out["gap_lo"], out["gap_hi"] = float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))
    return out


def _compute_per_model(
    db: ResultsDB, models: list[str], cot_field: str, metric: str = "verbalized"
) -> dict[str, dict[str, dict[str, float]]]:
    """{model: {'tool': {F_e, se_e, F_i, se_i, gap, gap_se}, 'user': {...}}}."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for m in models:
        recs = db.filter(_model=m).records
        tool_stats = _scenario_tally(
            recs, CELL_CONTEXT_TYPES["tool_explicit"], CELL_CONTEXT_TYPES["tool_implicit"], cot_field
        )
        user_stats = _scenario_tally(
            recs, CELL_CONTEXT_TYPES["user_explicit"], CELL_CONTEXT_TYPES["user_implicit"], cot_field
        )
        out[m] = {"tool": _cell_stats(tool_stats, metric=metric), "user": _cell_stats(user_stats, metric=metric)}
    return out


def _across_model_mean_ci(gaps: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Mean of per-model values with model-bootstrap 95% CI.

    This is uncertainty over *models* (random-effects-like), not within-scenario
    sampling uncertainty. It describes how the pooled summary would change if
    we re-sampled which models we evaluated. Plot it as a separate marker so
    readers don't confuse it with the per-model scenario CIs.
    """
    arr = np.array([g for g in gaps if np.isfinite(g)])
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    return mean, float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def _build_handles(models: list[str], colors: dict[str, tuple]) -> list[mlines.Line2D]:
    by_family: dict[str, list[tuple[int, str]]] = {}
    for m in models:
        fam = _DIR_FAMILY.get(m, "Other")
        by_family.setdefault(fam, []).append((MODEL_PARAMS.get(m, 0), m))
    handles: list[mlines.Line2D] = []
    for fam in FAMILY_COLORS:
        if fam not in by_family:
            continue
        for _, m in sorted(by_family[fam]):
            handles.append(
                mlines.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[m],
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    markersize=8,
                    label=MODEL_LABEL.get(m, m),
                )
            )
    return handles


def _draw_levels(
    ax,
    per_model: dict[str, dict[str, dict[str, float]]],
    models: list[str],
    colors: dict[str, tuple],
    metric: str = "verbalized",
) -> None:
    """Left panel: role gap (User − Tool) at explicit (x) vs implicit (y), one dot per model.

    CIs are propagated as differences of independent arm bounds (conservative):
      lower = F_user_lo − F_tool_hi,  upper = F_user_hi − F_tool_lo
    """
    ax.patch.set_alpha(0)
    pts_e, pts_i = [], []
    for m in models:
        st = per_model[m]["tool"]
        su = per_model[m]["user"]
        if not all(np.isfinite(su[k]) and np.isfinite(st[k]) for k in ("F_e", "F_i")):
            continue
        delta_e = su["F_e"] - st["F_e"]
        delta_i = su["F_i"] - st["F_i"]
        xerr = [
            [max(0.0, delta_e - (su["F_e_lo"] - st["F_e_hi"]))],
            [max(0.0, (su["F_e_hi"] - st["F_e_lo"]) - delta_e)],
        ]
        yerr = [
            [max(0.0, delta_i - (su["F_i_lo"] - st["F_i_hi"]))],
            [max(0.0, (su["F_i_hi"] - st["F_i_lo"]) - delta_i)],
        ]
        ax.errorbar(
            delta_e,
            delta_i,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=colors.get(m, (0.5, 0.5, 0.5)),
            ms=8,
            lw=0.8,
            capsize=0,
            alpha=0.95,
            markeredgecolor="black",
            markeredgewidth=1.5,
            ecolor="black",
            elinewidth=0.7,
            zorder=3,
        )
        pts_e.append(delta_e)
        pts_i.append(delta_i)

    me, e_lo, e_hi = _across_model_mean_ci(pts_e)
    mi, i_lo, i_hi = _across_model_mean_ci(pts_i)
    ax.errorbar(
        me,
        mi,
        xerr=[[max(0.0, me - e_lo)], [max(0.0, e_hi - me)]],
        yerr=[[max(0.0, mi - i_lo)], [max(0.0, i_hi - mi)]],
        fmt="*",
        color="black",
        ms=15,
        lw=1.6,
        capsize=4,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=5,
    )

    pad = 0.04
    lo = min(pts_e + pts_i + [me, mi]) - pad
    hi = max(pts_e + pts_i + [me, mi]) + pad
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.axhline(0, color="#999", lw=1.0, ls=":", zorder=1)
    ax.axvline(0, color="#999", lw=1.0, ls=":", zorder=1)
    diag = np.linspace(lo, hi, 50)
    ax.plot(diag, diag, color="#666", lw=1.0, ls="--", zorder=1)
    ax.text(
        0.97,
        0.03,
        "upper-right:\nuser $>$ tool\nat both registers",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85),
    )
    if metric == "covert":
        ax.set_xlabel(
            r"$\Delta_\mathrm{explicit} = \mathrm{Unverbalized Adoption Rate}(\mathrm{User\,exp}) - \mathrm{Unverbalized Adoption Rate}(\mathrm{Tool\,exp})$"
        )
        ax.set_ylabel(
            r"$\Delta_\mathrm{implicit} = \mathrm{Unverbalized Adoption Rate}(\mathrm{User\,imp}) - \mathrm{Unverbalized Adoption Rate}(\mathrm{Tool\,imp})$"
        )
    else:
        ax.set_xlabel(r"$\Delta_\mathrm{explicit} = F(\mathrm{User\,exp}) - F(\mathrm{Tool\,exp})$")
        ax.set_ylabel(r"$\Delta_\mathrm{implicit} = F(\mathrm{User\,imp}) - F(\mathrm{Tool\,imp})$")
    # ax.set_title("Role gap per register level (User $-$ Tool)", fontsize=10, pad=6)
    ax.tick_params(axis="both", length=0)


def _draw_gaps(
    ax,
    per_model: dict[str, dict[str, dict[str, float]]],
    models: list[str],
    colors: dict[str, tuple],
    metric: str = "verbalized",
) -> tuple[float, tuple[float, float], float, tuple[float, float]]:
    """Right panel: Δ_tool vs Δ_user gap-of-gaps."""
    ax.patch.set_alpha(0)
    pts_tool, pts_user = [], []
    for m in models:
        st_, su_ = per_model[m]["tool"], per_model[m]["user"]
        if not (np.isfinite(st_["gap"]) and np.isfinite(su_["gap"])):
            continue
        xerr = [[max(0.0, st_["gap"] - st_["gap_lo"])], [max(0.0, st_["gap_hi"] - st_["gap"])]]
        yerr = [[max(0.0, su_["gap"] - su_["gap_lo"])], [max(0.0, su_["gap_hi"] - su_["gap"])]]
        ax.errorbar(
            st_["gap"],
            su_["gap"],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=colors.get(m, (0.5, 0.5, 0.5)),
            ms=8,
            lw=0.8,
            capsize=0,
            alpha=0.95,
            markeredgecolor="black",
            markeredgewidth=1.5,
            ecolor="black",
            elinewidth=0.7,
            zorder=3,
        )
        pts_tool.append(st_["gap"])
        pts_user.append(su_["gap"])

    mt, mt_lo, mt_hi = _across_model_mean_ci(pts_tool)
    mu, mu_lo, mu_hi = _across_model_mean_ci(pts_user)
    xerr = [[max(0.0, mt - mt_lo)], [max(0.0, mt_hi - mt)]]
    yerr = [[max(0.0, mu - mu_lo)], [max(0.0, mu_hi - mu)]]
    ax.errorbar(
        mt,
        mu,
        xerr=xerr,
        yerr=yerr,
        fmt="*",
        color="black",
        ms=18,
        lw=1.6,
        capsize=4,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=5,
    )

    pad = 0.04
    lo = min(pts_tool + pts_user + [mt, mu]) - pad
    hi = max(pts_tool + pts_user + [mt, mu]) + pad
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.axhline(0, color="#999", lw=1.0, ls=":", zorder=1)
    ax.axvline(0, color="#999", lw=1.0, ls=":", zorder=1)
    diag = np.linspace(lo, hi, 50)
    ax.plot(diag, diag, color="#666", lw=1.0, ls="--", zorder=1)

    if metric == "covert":
        ax.set_xlabel(r"Tool $\Delta_\mathrm{Unverbalized Adoption Rate} = $ Implicit $-$ Explicit")
        ax.set_ylabel(r"User $\Delta_\mathrm{Unverbalized Adoption Rate} = $ Implicit $-$ Explicit")
    else:
        ax.set_xlabel(r"Tool $\Delta = $ Explicit $-$ Implicit")
        ax.set_ylabel(r"User $\Delta = $ Explicit $-$ Implicit")
    # ax.set_title("Register effect (Explicit $-$ Implicit)", fontsize=10, pad=6)
    ax.tick_params(axis="both", length=0)
    return mt, (mt_lo, mt_hi), mu, (mu_lo, mu_hi)


def _plot_scatter(
    per_model: dict[str, dict[str, dict[str, float]]],
    models: list[str],
    colors: dict[str, tuple],
    metric_label: str,
    outpath: Path,
    metric: str = "verbalized",
) -> None:
    fig_a, ax_a = plt.subplots(1, 1, figsize=(5.5, 5.5))
    _draw_levels(ax_a, per_model, models, colors, metric=metric)
    fig_a.tight_layout()
    out_a = outpath.with_name(outpath.stem + "_a.svg")
    save_figure(fig_a, out_a)
    print(f"Saved {out_a}")

    fig_b, ax_b = plt.subplots(1, 1, figsize=(5.5, 5.5))
    mt, mt_ci, mu, mu_ci = _draw_gaps(ax_b, per_model, models, colors, metric=metric)
    fig_b.tight_layout()
    out_b = outpath.with_name(outpath.stem + "_b.svg")
    save_figure(fig_b, out_b)
    print(f"Saved {out_b}")
    print(
        f"  across-model means: Δ_tool={mt:+.3f} [{mt_ci[0]:+.3f}, {mt_ci[1]:+.3f}], "
        f"Δ_user={mu:+.3f} [{mu_ci[0]:+.3f}, {mu_ci[1]:+.3f}], interaction={mu - mt:+.3f}"
    )

    model_handles = _build_handles(models, colors)
    save_legend(
        model_handles,
        [h.get_label() for h in model_handles],
        outpath.with_name(outpath.stem + "_legend_models.svg"),
        ncol=4,
    )

    marker_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ccc",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=9,
            label="Per-model",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markersize=14,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="Across-model mean",
        ),
    ]
    save_legend(
        marker_handles,
        [h.get_label() for h in marker_handles],
        outpath.with_name(outpath.stem + "_legend_markers.svg"),
        ncol=2,
    )


def _plot_forest(
    per_model: dict[str, dict[str, dict[str, float]]],
    models: list[str],
    colors: dict[str, tuple],
    metric_label: str,
    outpath: Path,
    metric: str = "verbalized",
) -> None:
    """Appendix: per-role forest plot with one row per model and a pooled diamond."""
    ordered = sorted(
        [m for m in models if np.isfinite(per_model[m]["tool"]["gap"])],
        key=lambda m: per_model[m]["tool"]["gap"],
    )
    n = len(ordered)
    fig, (ax_t, ax_u) = plt.subplots(1, 2, figsize=(9.0, max(3.5, 0.35 * n + 1.4)), sharey=True)

    for ax, role, title in [(ax_t, "tool", "Tool-return"), (ax_u, "user", "User-message")]:
        ax.patch.set_alpha(0)
        for i, m in enumerate(ordered):
            s = per_model[m][role]
            color = colors.get(m, (0.5, 0.5, 0.5))
            xerr = [[max(0.0, s["gap"] - s["gap_lo"])], [max(0.0, s["gap_hi"] - s["gap"])]]
            ax.errorbar(
                s["gap"],
                i,
                xerr=xerr,
                fmt="o",
                color=color,
                ms=8,
                lw=1.2,
                capsize=3,
                markeredgecolor="black",
                markeredgewidth=1.5,
                ecolor="black",
                elinewidth=0.8,
                zorder=3,
            )
        gaps = [per_model[m][role]["gap"] for m in ordered]
        mp, mp_lo, mp_hi = _across_model_mean_ci(gaps)
        xerr = [[max(0.0, mp - mp_lo)], [max(0.0, mp_hi - mp)]]
        ax.errorbar(
            mp,
            n + 0.4,
            xerr=xerr,
            fmt="D",
            color="black",
            ms=10,
            lw=1.6,
            capsize=4,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=5,
        )
        ax.axvline(0, color="#999", lw=1.0, ls=":", zorder=1)
        ax.set_title(title)
        ax.tick_params(axis="both", length=0)

    ax_t.set_yticks(list(range(n)) + [n + 0.4])
    ax_t.set_yticklabels([MODEL_LABEL.get(m, m) for m in ordered] + ["Pooled"])
    ax_t.set_ylim(-0.7, n + 1.1)

    gap_label = "Implicit $-$ Explicit" if metric == "covert" else "Explicit $-$ Implicit"
    fig.supxlabel(rf"$\Delta = $ {gap_label}    ({metric_label})", y=0.02, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    save_figure(fig, outpath)
    print(f"Saved {outpath}")


def _save_table(
    per_model: dict[str, dict[str, dict[str, float]]],
    models: list[str],
    metric_suffix: str,
    outpath: Path,
) -> None:
    rows: list[dict] = []
    for m in models:
        st, su = per_model[m]["tool"], per_model[m]["user"]
        rows.append(
            {
                "metric": metric_suffix.upper(),
                "model": m,
                "family": _DIR_FAMILY.get(m, "Other"),
                "F_tool_explicit": st["F_e"],
                "F_tool_explicit_ci_lo": st["F_e_lo"],
                "F_tool_explicit_ci_hi": st["F_e_hi"],
                "F_tool_implicit": st["F_i"],
                "F_tool_implicit_ci_lo": st["F_i_lo"],
                "F_tool_implicit_ci_hi": st["F_i_hi"],
                "delta_tool": st["gap"],
                "delta_tool_ci_lo": st["gap_lo"],
                "delta_tool_ci_hi": st["gap_hi"],
                "F_user_explicit": su["F_e"],
                "F_user_explicit_ci_lo": su["F_e_lo"],
                "F_user_explicit_ci_hi": su["F_e_hi"],
                "F_user_implicit": su["F_i"],
                "F_user_implicit_ci_lo": su["F_i_lo"],
                "F_user_implicit_ci_hi": su["F_i_hi"],
                "delta_user": su["gap"],
                "delta_user_ci_lo": su["gap_lo"],
                "delta_user_ci_hi": su["gap_hi"],
                "interaction_user_minus_tool": su["gap"] - st["gap"],
            }
        )
    save_table(outpath, rows)


def main() -> None:
    global FIGURES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["verbalized", "covert"], default="verbalized")
    # C0 keeps the unstamped filenames the paper panels already reference; every other
    # convention stamps its own, so the four can sit in one figures dir. This is what
    # backs the abstract's "no system prompt closes the explicitness gap" claim: the
    # per-channel explicitness delta with a joint CI, under each convention.
    parser.add_argument("--convention", choices=["C0", "C3", "MC0", "MC3"], default="C0")
    eval_aware.add_flag(parser)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    FIGURES_DIR = args.figures_dir
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_suffix = "_covert" if args.metric == "covert" else ""
    conv_suffix = "" if args.convention == "C0" else f"_{args.convention}"

    setup_plot_style()
    plt.rcParams["axes.facecolor"] = "none"

    db, eval_suffix = eval_aware.apply(load_clean(args.convention), args)
    # select_models drops the partial Qwen 3 sweep and anything outside the registry,
    # and orders by family then size — the same 15 every other paper panel shows.
    models = select_models({r["_model"] for r in db.records})
    print(f"Loaded {db.count()} records; {len(models)} models: {models}")
    colors = _assign_model_colors(models)

    for suffix, cot_field, metric_key in [
        ("l3", "reasoning_tailoring_explicit", "cot_commitment"),
    ]:
        per_model = _compute_per_model(db, models, cot_field, metric=args.metric)
        if args.metric == "covert":
            metric_label = UNVERBALIZED_ADOPTION_RATE_LABEL
        else:
            metric_label = (
                VERBALIZED_COMMITMENT_RATE_LABEL
                if metric_key == "cot_commitment"
                else conditional_faithfulness_label(metric_key, short=True)
            )
        out = FIGURES_DIR / f"h1_register_{suffix}{metric_suffix}{conv_suffix}{eval_suffix}.svg"
        _plot_scatter(per_model, models, colors, metric_label, out, metric=args.metric)
        _save_table(per_model, models, suffix, out.with_suffix(".csv"))
        _plot_forest(
            per_model,
            models,
            colors,
            metric_label,
            FIGURES_DIR / f"h1_register_forest_{suffix}{metric_suffix}{conv_suffix}{eval_suffix}.svg",
            metric=args.metric,
        )


if __name__ == "__main__":
    main()
