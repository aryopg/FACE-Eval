"""Quantitative tests for the Qwen3.5 convention-backfire pattern.

H-A: Anti-sycophancy suppression — does L1 (verbalization) drop alongside L3
     (commitment) under C3 in the user channel? If yes, suppression is active
     at the mention level; if L1 holds and only L3 drops, it's H-C instead.

H-C: Compliance theater — does C3 produce shallower CoTs (fewer Qwen-tokenized
     tokens) in unfaithful (L3=False) user-channel instances relative to C0?

Denominator: tracks X = P(aligned ∧ committed | cued) for all models, C0 vs C3,
     user-channel, without the causal filter, to expose denominator shifts that
     confound any conditional Y = P(L3=False | aligned ∧ committed) comparison.

Outputs (all written to figures/):
  convention_backfire_ha.svg/.csv   — ΔL1 and ΔL3 (C3 − C0, Qwen3.5 user-channel)
  convention_backfire_denom.svg/.csv — X and Y for all models (C0 vs C3, user-channel)
  convention_backfire_hc.svg/.csv   — CoT token-length by (convention × L3 label)
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.results.db import ResultsDB
from src.results.storage import discover_runs, load_merged_results
from src.utils.plotting import (
    DIR_FAMILY,
    ERRORBAR_KWARGS,
    FAMILY_COLORS,
)
from src.utils.plotting import MODEL_LABEL_INLINE as MODEL_LABEL
from src.utils.plotting import (
    save_figure,
    select_models,
    setup_plot_style,
)

FIGURES_DIR = Path("figures")
RESULTS_DIR = "results/agentic"

QWEN35_MODELS = frozenset({"Qwen_Qwen3.5-4B", "Qwen_Qwen3.5-9B", "Qwen_Qwen3.5-27B"})
USER_CTXS = ("user_turn", "user_turn_structured")
_PARSE_OK = {"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True}
_ALIGNED_COMMITTED = {
    "judge.answer_aligns_with_preference": True,
    "judge.answer_committed": True,
}
_QWEN_COLOR = FAMILY_COLORS["Qwen 3.5"]

_QWEN_TOKENIZER = None


def _qwen_token_count(text: str | None) -> int | None:
    """Tokenize text with the Qwen3.5 tokenizer; returns None if unavailable."""
    global _QWEN_TOKENIZER
    if not text:
        return 0
    if _QWEN_TOKENIZER is None:
        try:
            from transformers import AutoTokenizer

            _QWEN_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
        except Exception as exc:
            warnings.warn(f"Qwen tokenizer unavailable ({exc}); H-C skipped.")
            return None
    return len(_QWEN_TOKENIZER.encode(text, add_special_tokens=False))


def _load_raw_qwen(results_dir: str = RESULTS_DIR) -> list[dict]:
    """Load Qwen3.5 C0/C3 records with full reasoning text (judge-merged)."""
    records: list[dict] = []
    for run in discover_runs(results_dir):
        if run["model"] not in QWEN35_MODELS:
            continue
        if run.get("convention", "C0") not in ("C0", "C3"):
            continue
        if not (run["has_inference"] and run["has_judged"]):
            continue
        for r in load_merged_results(Path(run["path"])):
            r["_model"] = run["model"]
            r["_seed"] = run["seed"]
            r["_convention"] = run.get("convention", "C0")
            records.append(r)
    return records


# ---------------------------------------------------------------------------
# Shared bootstrap helpers (cluster = scenario_id)
# ---------------------------------------------------------------------------


def _fraction(records: list[dict], field: str) -> float:
    vals = []
    for r in records:
        parts = field.split(".")
        v = r
        for p in parts:
            v = v.get(p) if isinstance(v, dict) else None
        if v is not None:
            vals.append(1 if v else 0)
    return float(np.mean(vals)) if vals else float("nan")


def _bootstrap_ci(
    records: list[dict],
    field: str,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Scenario-cluster bootstrap CI. Returns (point, lo, hi)."""
    by_cluster: dict = {}
    for r in records:
        by_cluster.setdefault(r.get("scenario_id"), []).append(r)
    cluster_lists = list(by_cluster.values())
    point = _fraction(records, field)
    if np.isnan(point) or len(cluster_lists) < 2 or n_boot < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    n = len(cluster_lists)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [r for i in idx for r in cluster_lists[i]]
        boot.append(_fraction(sample, field))
    arr = np.asarray(boot)
    return point, float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def _bootstrap_mean_ci(
    values: list[float],
    cluster_ids: list,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI on mean of a numeric variable, clustered by scenario."""
    by_cluster: dict = {}
    for cid, v in zip(cluster_ids, values):
        by_cluster.setdefault(cid, []).append(v)
    cluster_lists = list(by_cluster.values())
    point = float(np.mean(values)) if values else float("nan")
    if np.isnan(point) or len(cluster_lists) < 2 or n_boot < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    n = len(cluster_lists)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [v for i in idx for v in cluster_lists[i]]
        boot.append(float(np.mean(sample)))
    arr = np.asarray(boot)
    return point, float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# H-A: ΔL1 and ΔL3 (C3 − C0) for Qwen3.5 user-channel
# ---------------------------------------------------------------------------


def _ha_rates(db: ResultsDB, model: str, convention: str) -> dict | None:
    """Compute L1 and L3 rates (conditional on aligned ∧ committed) with 95% CIs.

    `db` must already have parse_ok + causal_dependent applied.
    """
    sub = (
        db.filter(_model=model, _convention=convention)
        .filter_in("context_type", USER_CTXS)
        .filter(**_ALIGNED_COMMITTED)
    )
    if sub.count() < 5:
        return None
    l1 = _bootstrap_ci(sub.records, "judge.reasoning_acknowledges_preference")
    l3 = _bootstrap_ci(sub.records, "judge.reasoning_tailoring_explicit")
    return {"l1": l1, "l3": l3, "n": sub.count()}


def plot_ha(db: ResultsDB) -> None:
    models_ordered = [
        m
        for m in ("Qwen_Qwen3.5-4B", "Qwen_Qwen3.5-9B", "Qwen_Qwen3.5-27B")
        if any(r["_model"] == m for r in db.records)
    ]

    rows = []
    for model in models_ordered:
        r0 = _ha_rates(db, model, "C0")
        r3 = _ha_rates(db, model, "C3")
        if r0 is None or r3 is None:
            continue
        # Δ = C3 rate − C0 rate (positive = C3 more faithful)
        dl1 = r3["l1"][0] - r0["l1"][0]
        dl3 = r3["l3"][0] - r0["l3"][0]
        # Conservative CI on Δ: propagate via bootstrap difference
        rows.append(
            {
                "model": model,
                "label": MODEL_LABEL.get(model, model),
                "dl1": dl1,
                "dl3": dl3,
                "l1_c0": r0["l1"],
                "l1_c3": r3["l1"],
                "l3_c0": r0["l3"],
                "l3_c3": r3["l3"],
                "n_c0": r0["n"],
                "n_c3": r3["n"],
            }
        )

    if not rows:
        print("H-A: no data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 0.9 * len(rows) + 1.2))

    ys = list(range(len(rows)))
    for y, row in enumerate(rows):
        # ΔL3 (square, primary)
        ax.scatter(
            row["dl3"],
            y - 0.12,
            marker="s",
            s=60,
            color=_QWEN_COLOR,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            label="$\\Delta L3$ (Commit$_\\mathrm{CoT}$)" if y == 0 else "",
        )
        # ΔL1 (circle, secondary)
        ax.scatter(
            row["dl1"],
            y + 0.12,
            marker="o",
            s=60,
            color=_QWEN_COLOR,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            alpha=0.55,
            label="$\\Delta L1$ (Verb$_\\mathrm{CoT}$)" if y == 0 else "",
        )

    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.5, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel(r"$\Delta$ Rate (C3 $-$ C0),  positive $=$ more faithful under C3")
    ax.set_xlim(-0.55, 0.55)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.tick_params(length=0)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(r"H-A: Convention effect on L1 and L3 (user channel, aligned $\wedge$ committed)", pad=6)

    fig.tight_layout()
    out = FIGURES_DIR / "convention_backfire_ha.svg"
    save_figure(fig, out)
    print(f"Saved {out}")

    csv_path = FIGURES_DIR / "convention_backfire_ha.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "model",
                "label",
                "n_c0",
                "n_c3",
                "l1_c0",
                "l1_c0_lo",
                "l1_c0_hi",
                "l1_c3",
                "l1_c3_lo",
                "l1_c3_hi",
                "delta_l1",
                "l3_c0",
                "l3_c0_lo",
                "l3_c0_hi",
                "l3_c3",
                "l3_c3_lo",
                "l3_c3_hi",
                "delta_l3",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["label"],
                    row["n_c0"],
                    row["n_c3"],
                    *row["l1_c0"],
                    *row["l1_c3"],
                    row["dl1"],
                    *row["l3_c0"],
                    *row["l3_c3"],
                    row["dl3"],
                ]
            )
    print(f"Saved {csv_path}")


# ---------------------------------------------------------------------------
# Denominator: X = P(aligned ∧ committed | cued) alongside Y, C0 vs C3
# ---------------------------------------------------------------------------


def _denom_cell(db: ResultsDB, model: str, convention: str) -> dict | None:
    """
    X = P(aligned ∧ committed | causal-dependent cued) — denominator rate
    Y = P(L3=False | aligned ∧ committed)              — conditional unfaithfulness

    `db` must already have parse_ok + causal_dependent applied (no_context rows
    already removed by filter_causal_dependent).
    """
    cued = db.filter(_model=model, _convention=convention).filter_in("context_type", USER_CTXS)
    if cued.count() < 5:
        return None

    # Tag each record with the joint aligned ∧ committed condition so we can
    # bootstrap over it as a single binary field without mutating the DB.
    tagged = []
    for r in cued.records:
        j = r.get("judge") or {}
        rc = {**r, "_ac": bool(j.get("answer_aligns_with_preference") and j.get("answer_committed"))}
        tagged.append(rc)
    x_pt, x_lo, x_hi = _bootstrap_ci(tagged, "_ac")

    # For Y we need the aligned+committed subset
    ac = cued.filter(**_ALIGNED_COMMITTED)
    if ac.count() < 3:
        y_pt, y_lo, y_hi = float("nan"), float("nan"), float("nan")
    else:
        # L3=False rate = 1 − L3=True rate
        l3_pt, l3_lo, l3_hi = _bootstrap_ci(ac.records, "judge.reasoning_tailoring_explicit")
        y_pt, y_lo, y_hi = 1 - l3_pt, 1 - l3_hi, 1 - l3_lo  # invert CI bounds

    return {
        "x": (x_pt, x_lo, x_hi),
        "y": (y_pt, y_lo, y_hi),
        "n_cued": cued.count(),
        "n_ac": ac.count(),
    }


def plot_denom(db_raw: ResultsDB) -> None:
    """Two-panel figure: X (left) and Y (right) for all models, C0 vs C3."""
    all_models = select_models({r["_model"] for r in db_raw.records})

    rows = []
    for model in all_models:
        c0 = _denom_cell(db_raw, model, "C0")
        c3 = _denom_cell(db_raw, model, "C3")
        if c0 is None and c3 is None:
            continue
        rows.append(
            {
                "model": model,
                "label": MODEL_LABEL.get(model, model),
                "family": DIR_FAMILY.get(model, "Other"),
                "c0": c0,
                "c3": c3,
            }
        )

    if not rows:
        print("Denominator: no data, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(rows) + 1.5))
    ys = list(range(len(rows)))

    conv_style = {
        "C0": {"marker": "o", "alpha": 1.0, "label": "C0 (Default)"},
        "C3": {"marker": "s", "alpha": 0.75, "label": "C3 (Directive)"},
    }

    csv_rows = []
    for panel_idx, (ax, metric, title, xlabel) in enumerate(
        [
            (
                axes[0],
                "x",
                r"Denominator: $X = P(\mathrm{Align}_\mathrm{ans} \wedge \mathrm{Commit}_\mathrm{ans} \mid \mathrm{cued})$",
                "Denominator rate (aligned $\\wedge$ committed)",
            ),
            (
                axes[1],
                "y",
                r"Conditional unfaithfulness: $Y = P(\neg L3 \mid \mathrm{Align} \wedge \mathrm{Commit})$",
                "Unverbalized adoption rate",
            ),
        ]
    ):
        all_pts = []
        for y, row in enumerate(rows):
            fam = row["family"]
            color = FAMILY_COLORS.get(fam, "#888888")
            for conv_key in ("C0", "C3"):
                cell = row.get(conv_key) or row.get("c0" if conv_key == "C0" else "c3")
                cell = row["c0"] if conv_key == "C0" else row["c3"]
                if cell is None:
                    continue
                pt, lo, hi = cell[metric]
                if np.isnan(pt):
                    continue
                sty = conv_style[conv_key]
                yo = -0.15 if conv_key == "C0" else 0.15
                xerr = [[max(0.0, pt - lo)], [max(0.0, hi - pt)]]
                ax.errorbar(
                    pt,
                    y + yo,
                    xerr=xerr,
                    marker=sty["marker"],
                    markersize=7,
                    linestyle="",
                    color="black",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=1.2,
                    alpha=sty["alpha"],
                    zorder=3,
                    label=sty["label"] if y == 0 else "",
                    **ERRORBAR_KWARGS,
                )
                all_pts.extend([lo, hi])
                if panel_idx == 0:
                    csv_rows.append(
                        {
                            "model": row["model"],
                            "label": row["label"],
                            "family": fam,
                            "convention": conv_key,
                            "x_pt": cell["x"][0],
                            "x_lo": cell["x"][1],
                            "x_hi": cell["x"][2],
                            "y_pt": cell["y"][0],
                            "y_lo": cell["y"][1],
                            "y_hi": cell["y"][2],
                            "n_cued": cell["n_cued"],
                            "n_ac": cell["n_ac"],
                        }
                    )
            # connector line C0→C3
            c0_pt = (row["c0"] or {}).get(metric, (float("nan"),))[0] if row["c0"] else float("nan")
            c3_pt = (row["c3"] or {}).get(metric, (float("nan"),))[0] if row["c3"] else float("nan")
            if not np.isnan(c0_pt) and not np.isnan(c3_pt):
                fam_color = FAMILY_COLORS.get(row["family"], "#888888")
                ax.plot([c0_pt, c3_pt], [y - 0.15, y + 0.15], color=fam_color, lw=1.0, alpha=0.4, zorder=2)

        pad = 0.03
        lo_all = min(all_pts) - pad if all_pts else 0.0
        hi_all = max(all_pts) + pad if all_pts else 1.0
        ax.set_xlim(max(0.0, lo_all), min(1.0, hi_all))
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(ys)
        if panel_idx == 0:
            ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, pad=4, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", ls=":", alpha=0.4)
        ax.tick_params(length=0)
        if panel_idx == 0:
            ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout(w_pad=0.5)
    out = FIGURES_DIR / "convention_backfire_denom.svg"
    save_figure(fig, out)
    print(f"Saved {out}")

    csv_path = FIGURES_DIR / "convention_backfire_denom.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "label",
                "family",
                "convention",
                "x_pt",
                "x_lo",
                "x_hi",
                "y_pt",
                "y_lo",
                "y_hi",
                "n_cued",
                "n_ac",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {csv_path}")


# ---------------------------------------------------------------------------
# H-C: CoT token length by (convention × L3 label) — Qwen tokenizer
# ---------------------------------------------------------------------------


def plot_hc(raw_records: list[dict]) -> None:
    """Violin plot of Qwen-tokenized CoT lengths, C0 vs C3, L3=True vs False.

    Applies the same parse_ok + causal_dependent filter as plot_convention_dumbbell.py
    before restricting to user-channel and aligned ∧ committed.
    """
    causal_db = ResultsDB(raw_records).filter(**_PARSE_OK).filter_causal_dependent()
    filtered = [
        r
        for r in causal_db.records
        if r.get("context_type") in USER_CTXS
        and (r.get("judge") or {}).get("answer_aligns_with_preference")
        and (r.get("judge") or {}).get("answer_committed")
    ]
    if not filtered:
        print("H-C: no records after filtering, skipping.")
        return

    # Tokenize (returns None if tokenizer unavailable — abort early)
    probe = _qwen_token_count(filtered[0].get("reasoning") or "test")
    if probe is None:
        print("H-C: Qwen tokenizer unavailable, skipping.")
        return

    groups: dict[tuple[str, bool], list[int]] = {}
    clusters: dict[tuple[str, bool], list] = {}
    csv_rows = []
    for r in filtered:
        conv = r.get("_convention", "C0")
        j = r.get("judge") or {}
        l3 = bool(j.get("reasoning_tailoring_explicit"))
        n_tok = _qwen_token_count(r.get("reasoning"))
        if n_tok is None:
            continue
        key = (conv, l3)
        groups.setdefault(key, []).append(n_tok)
        clusters.setdefault(key, []).append(r.get("scenario_id"))
        csv_rows.append(
            {
                "model": r.get("_model"),
                "convention": conv,
                "l3_faithful": l3,
                "scenario_id": r.get("scenario_id"),
                "tokens": n_tok,
            }
        )

    conditions = [("C0", True), ("C0", False), ("C3", True), ("C3", False)]
    labels = ["C0\nL3=True", "C0\nL3=False", "C3\nL3=True", "C3\nL3=False"]
    colors = ["#4878CF", "#E8A020", "#4878CF", "#E8A020"]
    alphas = [1.0, 1.0, 0.55, 0.55]

    fig, ax = plt.subplots(figsize=(7, 4))

    plot_data = []
    for cond in conditions:
        vals = groups.get(cond, [])
        plot_data.append(vals if vals else [0])

    parts = ax.violinplot(plot_data, positions=list(range(4)), widths=0.65, showmedians=True, showextrema=False)

    for i, (pc, color, alpha) in enumerate(zip(parts["bodies"], colors, alphas)):
        pc.set_facecolor(color)
        pc.set_alpha(alpha * 0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.8)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    # Overlay individual points (jittered)
    rng = np.random.default_rng(0)
    for i, cond in enumerate(conditions):
        vals = groups.get(cond, [])
        if not vals:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(i + jitter, vals, s=8, alpha=0.3, color=colors[i], edgecolors="none", zorder=4)

    # Bootstrap mean + CI
    for i, cond in enumerate(conditions):
        vals = groups.get(cond, [])
        cids = clusters.get(cond, [])
        if len(vals) < 3:
            continue
        mn, lo, hi = _bootstrap_mean_ci(vals, cids)
        ax.errorbar(
            i,
            mn,
            yerr=[[mn - lo], [hi - mn]],
            marker="D",
            markersize=6,
            color="black",
            markerfacecolor="white",
            markeredgewidth=1.5,
            linestyle="",
            zorder=5,
            **ERRORBAR_KWARGS,
        )

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CoT length (Qwen tokens)")
    ax.set_title("H-C: CoT length by convention $\\times$ L3 faithfulness (Qwen3.5 user-channel)", pad=6)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.tick_params(length=0)

    # Annotate n per condition
    for i, cond in enumerate(conditions):
        n = len(groups.get(cond, []))
        ax.text(i, ax.get_ylim()[1] * 0.98, f"n={n}", ha="center", va="top", fontsize=7, color="#555555")

    fig.tight_layout()
    out = FIGURES_DIR / "convention_backfire_hc.svg"
    save_figure(fig, out)
    print(f"Saved {out}")

    csv_path = FIGURES_DIR / "convention_backfire_hc.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "convention", "l3_faithful", "scenario_id", "tokens"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global FIGURES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    FIGURES_DIR = args.figures_dir
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    # Load slim cache and apply the same filter chain as plot_convention_dumbbell.py:
    # parse_ok → causal_dependent (removes no_context + uncaused rows).
    print("Loading slim cache…")
    db = ResultsDB.load_all(results_dir=RESULTS_DIR, require_judged=True).filter(**_PARSE_OK).filter_causal_dependent()

    # H-A: conditional on aligned ∧ committed (causal filter already applied above)
    print("\n--- H-A ---")
    plot_ha(db)

    # Denominator: X = P(aligned ∧ committed | causal-dependent cued)
    print("\n--- Denominator ---")
    plot_denom(db)

    # H-C: needs full reasoning text — load Qwen3.5 C0/C3 only (slim=False)
    print("\n--- H-C (loading raw reasoning, may be slow) ---")
    raw_records = _load_raw_qwen(RESULTS_DIR)
    print(f"  Loaded {len(raw_records)} raw records")
    plot_hc(raw_records)


if __name__ == "__main__":
    main()
