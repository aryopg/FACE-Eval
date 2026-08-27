"""Shared plotting configuration, model metadata, and figure/CSV helpers.

The paper scripts use this module as the single source of truth for display
labels, model ordering, colours, and output conventions.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.results.db import ResultsDB

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIGURE_SIZE = (5.5, 3.5)
FONT_SIZE_AXIS_LABEL = 11
FONT_SIZE_AXIS_TICKS = 9
FONT_SIZE_LEGEND = 9

# Wide figures: designed at 10 in, scaled to ~5.5 in in LaTeX (factor ≈ 1.82).
# Font sizes are pre-compensated so they render correctly after scaling.
FIGURE_SIZE_WIDE = (10.0, 3.5)
FONT_SIZE_AXIS_LABEL_WIDE = 11
FONT_SIZE_AXIS_TICKS_WIDE = 9
FONT_SIZE_LEGEND_WIDE = 9
FONT_SIZE_TITLE = 12
FONT_SIZE_TITLE_WIDE = 12

# Standard error bar styling — thin black caps, consistent across all figures.
ERRORBAR_KWARGS: dict = {"ecolor": "black", "elinewidth": 0.8, "capsize": 2, "capthick": 0.8}
# For ax.bar(yerr=..., error_kw=ERROR_KW_BAR, capsize=2)
ERROR_KW_BAR: dict = {"ecolor": "black", "lw": 0.8}

# ---------------------------------------------------------------------------
# Model metadata for scaling plots
# ---------------------------------------------------------------------------

# Channel names follow the manuscript: the cue arrives in a user-message or in a
# tool-return.
CHANNEL_DISPLAY: dict[str, str] = {
    "user_turn": "User-message",
    "user_turn_structured": "User-message structured",
    "user_turn_implicit": "User-message implicit",
    "explicit": "Explicit (tool-return)",
    "implicit": "Implicit (tool-return)",
}

# Headline channel x salience taxonomy. user_explicit pools the prose and XML-tagged
# variants, which differ in template but not in register. Tool cells are source-agnostic:
# `source` is its own field, so filtering on `context_type` already pools all five.
CELL_DISPLAY: dict[str, str] = {
    "user_explicit": "User-message (Explicit)",
    "user_implicit": "User-message (Implicit)",
    "tool_explicit": "Tool-return (Explicit)",
    "tool_implicit": "Tool-return (Implicit)",
}
CELL_CONTEXT_TYPES: dict[str, tuple[str, ...]] = {
    "user_explicit": ("user_turn", "user_turn_structured"),
    "user_implicit": ("user_turn_implicit",),
    "tool_explicit": ("explicit",),
    "tool_implicit": ("implicit",),
}
CELLS_4: tuple[str, ...] = ("user_explicit", "user_implicit", "tool_explicit", "tool_implicit")
# Marker per cell, for plots that encode the cell by shape (model by colour).
CELL_MARKERS: dict[str, str] = {
    "user_explicit": "o",
    "user_implicit": "s",
    "tool_explicit": "^",
    "tool_implicit": "D",
}

# ---------------------------------------------------------------------------
# Cue-salience annotation experiment: 4-way condition grouping
# ---------------------------------------------------------------------------
# user_turn and user_turn_structured are always pooled as "User-message (explicit)".
# Context-type members per group come from CELL_CONTEXT_TYPES above.
CUE_CONDITIONS: tuple[str, ...] = ("user_explicit", "user_implicit", "tool_explicit", "tool_implicit")
CUE_CONDITION_LABEL: dict[str, str] = {
    "user_explicit": "User-message\n(explicit)",
    "user_implicit": "User-message\n(implicit)",
    "tool_explicit": "Tool-return\n(explicit)",
    "tool_implicit": "Tool-return\n(implicit)",
}
SIDE_ID_LABEL: str = "Side-identification accuracy"
CLARITY_LABEL: str = "Model-rated clarity (1-5)"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# The dicts below list every model the figures report. Every dict must hold the
# same key set: a model in one but not the others gets a figure with a missing
# label, colour or parameter count.
MODEL_PRETTY_NAMES: dict[str, str] = {
    "Qwen/Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen/Qwen3.5-27B": "Qwen3.5-27B",
    "google/gemma-4-E4B-it": "Gemma4-E4B",
    "google/gemma-4-26B-A4B-it": "Gemma4-26B",
    "google/gemma-4-31B-it": "Gemma4-31B",
    "allenai/Olmo-3-7B-Think": "OLMo3-7B",
    "allenai/Olmo-3.1-32B-Think": "OLMo3.1-32B",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
    "openai/gpt-oss-120b": "GPT-OSS-120B",
    "deepseek-ai/DeepSeek-V4-Flash": "DeepSeek V4 Flash",
    "deepseek-ai/DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "zai-org/GLM-5.2-FP8": "GLM-5.2",
    "moonshotai/Kimi-K2.6": "Kimi-K2.6",
    "thinkingmachines/Inkling-NVFP4": "Inkling",
}

MODEL_FAMILY: dict[str, str] = {
    "Qwen/Qwen3.5-4B": "Qwen 3.5",
    "Qwen/Qwen3.5-9B": "Qwen 3.5",
    "Qwen/Qwen3.5-27B": "Qwen 3.5",
    "google/gemma-4-E4B-it": "Gemma 4",
    "google/gemma-4-26B-A4B-it": "Gemma 4",
    "google/gemma-4-31B-it": "Gemma 4",
    "allenai/Olmo-3-7B-Think": "OLMo 3",
    "allenai/Olmo-3.1-32B-Think": "OLMo 3",
    "openai/gpt-oss-20b": "GPT-OSS",
    "openai/gpt-oss-120b": "GPT-OSS",
    "deepseek-ai/DeepSeek-V4-Flash": "DeepSeek V4",
    "deepseek-ai/DeepSeek-V4-Pro": "DeepSeek V4",
    "zai-org/GLM-5.2-FP8": "GLM 5.2",
    "moonshotai/Kimi-K2.6": "Kimi",
    "thinkingmachines/Inkling-NVFP4": "Inkling",
}

# Colorblind-safe palette (seaborn 'colorblind')
FAMILY_COLORS: dict[str, str] = {
    "Qwen 3.5": "#21409A",
    "Gemma 4": "#BE1E2D",
    "OLMo 3": "#E8972E",
    "GPT-OSS": "#41A9AC",
    # Frontier families. Inkling is the achromatic one and stays light (L* 68) so the
    # black hatch on its bars reads. Closest pair under normal, deutan or protan vision
    # is dE 5.4; keep any new colour above that.
    "DeepSeek V4": "#2E6E4E",
    "GLM 5.2": "#C3557F",
    "Kimi": "#8E5FC0",
    "Inkling": "#A6A6A6",
}

# ---------------------------------------------------------------------------
# Core model registry (H3–H6 hypothesis scripts)
#
# Adding a model requires editing only this file.
# ---------------------------------------------------------------------------

# Total parameter counts (in billions), keyed by dir-style name (/ → _).
# Total, not active: gemma-4-26B-A4B and gpt-oss-120b are stored at 26 and 120.
# Used for sort order and within-family colour shading, not for a compute claim.
MODEL_PARAMS: dict[str, int] = {
    "Qwen_Qwen3.5-4B": 4,
    "Qwen_Qwen3.5-9B": 9,
    "Qwen_Qwen3.5-27B": 27,
    "google_gemma-4-E4B-it": 4,
    "google_gemma-4-26B-A4B-it": 26,
    "google_gemma-4-31B-it": 31,
    "allenai_Olmo-3-7B-Think": 7,
    "allenai_Olmo-3.1-32B-Think": 32,
    "openai_gpt-oss-20b": 20,
    "openai_gpt-oss-120b": 120,
    "deepseek-ai_DeepSeek-V4-Flash": 284,
    "zai-org_GLM-5.2-FP8": 744,
    "thinkingmachines_Inkling-NVFP4": 975,
    "moonshotai_Kimi-K2.6": 1040,
    "deepseek-ai_DeepSeek-V4-Pro": 1600,
}

# Multi-line labels — for x-axis ticks in bar/grouped-bar figures.
MODEL_LABEL: dict[str, str] = {
    "Qwen_Qwen3.5-4B": "Qwen3.5\n4B",
    "Qwen_Qwen3.5-9B": "Qwen3.5\n9B",
    "Qwen_Qwen3.5-27B": "Qwen3.5\n27B",
    "google_gemma-4-E4B-it": "Gemma4\nE4B",
    "google_gemma-4-26B-A4B-it": "Gemma4\n26B",
    "google_gemma-4-31B-it": "Gemma4\n31B",
    "allenai_Olmo-3-7B-Think": "OLMo3\n7B",
    "allenai_Olmo-3.1-32B-Think": "OLMo3.1\n32B",
    "openai_gpt-oss-20b": "GPT-OSS\n20B",
    "openai_gpt-oss-120b": "GPT-OSS\n120B",
    # Frontier models carry no parameter count: the sizes are approximate, and
    # they are not part of a scaling sweep the way the open-weight families are.
    "deepseek-ai_DeepSeek-V4-Flash": "DeepSeek\nV4 Flash",
    "zai-org_GLM-5.2-FP8": "GLM-5.2",
    "thinkingmachines_Inkling-NVFP4": "Inkling",
    "moonshotai_Kimi-K2.6": "Kimi-K2.6",
    "deepseek-ai_DeepSeek-V4-Pro": "DeepSeek\nV4 Pro",
}

# Single-line labels — for y-axis rows in dumbbell / forest plots.
MODEL_LABEL_INLINE: dict[str, str] = {
    "Qwen_Qwen3.5-4B": "Qwen3.5 4B",
    "Qwen_Qwen3.5-9B": "Qwen3.5 9B",
    "Qwen_Qwen3.5-27B": "Qwen3.5 27B",
    "google_gemma-4-E4B-it": "Gemma4 E4B",
    "google_gemma-4-26B-A4B-it": "Gemma4 26B",
    "google_gemma-4-31B-it": "Gemma4 31B",
    "allenai_Olmo-3-7B-Think": "OLMo3 7B",
    "allenai_Olmo-3.1-32B-Think": "OLMo3.1 32B",
    "openai_gpt-oss-20b": "GPT-OSS 20B",
    "openai_gpt-oss-120b": "GPT-OSS 120B",
    "deepseek-ai_DeepSeek-V4-Flash": "DeepSeek V4 Flash",
    "zai-org_GLM-5.2-FP8": "GLM-5.2",
    "thinkingmachines_Inkling-NVFP4": "Inkling",
    "moonshotai_Kimi-K2.6": "Kimi-K2.6",
    "deepseek-ai_DeepSeek-V4-Pro": "DeepSeek V4 Pro",
}

# Display order for family separators and color assignment.
FAMILY_ORDER: list[str] = [
    "Qwen 3.5",
    "Gemma 4",
    "OLMo 3",
    "GPT-OSS",
    "DeepSeek V4",
    "GLM 5.2",
    "Kimi",
    "Inkling",
]

# Dir-style model name → family, covering effort-level suffixes automatically.
# Inkling's efforts are continuous floats rather than names, so its run dirs carry
# the float itself (thinkingmachines_Inkling-NVFP4_0.7). Its 0.2 sweep is left out
# on purpose, here and in EFFORT_VARIANTS: an effort registered here but not there
# has no base to pool into, so it reaches the axis unpooled under its raw dir name.
DIR_FAMILY: dict[str, str] = {k.replace("/", "_"): v for k, v in MODEL_FAMILY.items()}
for _base_k, _fam_v in list(DIR_FAMILY.items()):
    for _effort in ("low", "medium", "high", "max", "0.7", "0.99"):
        DIR_FAMILY[f"{_base_k}_{_effort}"] = _fam_v

# Effort variants pooled into canonical base entries. Kimi-K2.6 has no effort axis,
# so it is absent here and passes through unpooled.
EFFORT_VARIANTS: dict[str, set[str]] = {
    **{
        f"openai_gpt-oss-{s}": {f"openai_gpt-oss-{s}_{e}" for e in ("low", "medium", "high", "max")}
        for s in ("20b", "120b")
    },
    **{
        m: {f"{m}_high", f"{m}_max"}
        for m in ("zai-org_GLM-5.2-FP8", "deepseek-ai_DeepSeek-V4-Flash", "deepseek-ai_DeepSeek-V4-Pro")
    },
    **{m: {f"{m}_{e}" for e in ("0.7", "0.99")} for m in ("thinkingmachines_Inkling-NVFP4",)},
}
VARIANT_TO_BASE: dict[str, str] = {v: k for k, vs in EFFORT_VARIANTS.items() for v in vs}

# Named effort levels, lowest first. Inkling's efforts are floats and compare
# numerically; no model mixes named and float efforts, so the two never meet.
_EFFORT_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "max": 3}


def _effort_rank(variant: str) -> float:
    effort = variant.rsplit("_", 1)[-1]
    return float(_EFFORT_RANK[effort]) if effort in _EFFORT_RANK else float(effort)


def sorted_effort_variants(base: str) -> list[str]:
    """The registered effort variants of one model, lowest effort first.

    Returns dir-style names. Registration is not presence: a variant that was
    never run still appears, so callers skip the ones with no records.
    """
    return sorted(EFFORT_VARIANTS[base], key=_effort_rank)


def highest_effort_variants(models: Iterable[str]) -> list[str]:
    """Keep one variant per effort-swept model: the highest effort present.

    Models with no effort axis (Kimi, the open-weight families) pass through.
    """
    best: dict[str, str] = {}
    for m in models:
        base = VARIANT_TO_BASE.get(m)
        if base is None:
            best[m] = m
        elif base not in best or _effort_rank(m) > _effort_rank(best[base]):
            best[base] = m
    return sort_models(best.values())


def pool_effort_variants(db: ResultsDB) -> ResultsDB:
    """Relabel effort-variant records with the base model name."""
    from src.results.db import ResultsDB as _DB

    new_records = [
        {**r, "_model": VARIANT_TO_BASE[r["_model"]]} if r.get("_model") in VARIANT_TO_BASE else r for r in db.records
    ]
    return _DB(new_records)


def sort_models(models: Iterable[str], family_order: list[str] | None = None) -> list[str]:
    """Sort models by family order then ascending parameter count."""
    _fo = family_order if family_order is not None else FAMILY_ORDER

    def _key(m: str) -> tuple:
        fam = DIR_FAMILY.get(m, "Other")
        fam_rank = _fo.index(fam) if fam in _fo else 99
        return (fam_rank, MODEL_PARAMS.get(m, 0), m)

    return sorted(models, key=_key)


def select_models(models: Iterable[str], family_order: list[str] | None = None) -> list[str]:
    """Keep the models whose family a figure shows, in display order.

    The family list is what a figure varies: pass a subset to leave a family out
    (e.g. Qwen 3, whose sweep is partial). Models outside it — and models absent
    from the registry entirely — are dropped.
    """
    _fo = family_order if family_order is not None else FAMILY_ORDER
    return sort_models([m for m in models if DIR_FAMILY.get(m) in _fo], _fo)


def family_shade(family_color: str, rank: int, n: int) -> tuple:
    """Lighten a family color by rank: rank=0 is 40% white blend, rank=n-1 is full color."""
    c = np.array(mcolors.to_rgb(family_color))
    if n == 1:
        return tuple(c.tolist())
    light = 0.4 * (1 - rank / (n - 1))
    return tuple(np.clip(c + (1 - c) * light, 0, 1).tolist())


def assign_model_colors(
    models: list[str],
    family_colors: dict[str, str] | None = None,
) -> dict[str, tuple]:
    """Assign family-shaded colors to models (lightest = smallest, darkest = largest)."""
    _fc = family_colors if family_colors is not None else FAMILY_COLORS
    by_family: dict[str, list[tuple[int, str]]] = {}
    for m in models:
        fam = DIR_FAMILY.get(m, "Other")
        by_family.setdefault(fam, []).append((MODEL_PARAMS.get(m, 0), m))
    colors: dict[str, tuple] = {}
    for fam, entries in by_family.items():
        entries.sort()
        fam_color = _fc.get(fam, "#888888")
        for rank, (_, m) in enumerate(entries):
            colors[m] = family_shade(fam_color, rank, len(entries))
    return colors


# ---------------------------------------------------------------------------
# Metric naming convention
# ---------------------------------------------------------------------------
# Answer-side and CoT-side metrics live in distinct namespaces. Subscripts
# (_ans / _CoT) disambiguate words that collide between stages — notably
# "Commitment", which appears on both sides.
#
# Stage  | Key                 | Display           | Judge field
# -------|---------------------|-------------------|-------------------------------------
# answer | answer_alignment    | Alignment_ans     | judge.answer_aligns_with_preference
# answer | answer_commitment   | Commitment_ans    | judge.answer_committed
# CoT    | cot_verbalization   | Verbalization_CoT | judge.reasoning_acknowledges_preference
# CoT    | cot_attribution     | Attribution_CoT   | judge.reasoning_cites_{user_statement,tool_return}
# CoT    | cot_commitment      | Commitment_CoT    | judge.reasoning_tailoring_explicit
METRIC_LABEL: dict[str, str] = {
    "answer_alignment": r"$\mathrm{Align}_{\mathrm{ans}}$",
    "answer_commitment": r"$\mathrm{Commit}_{\mathrm{ans}}$",
    "cot_verbalization": r"$\mathrm{Verb}_{\mathrm{CoT}}$",
    "cot_attribution": r"$\mathrm{Attrib}_{\mathrm{CoT}}$",
    "cot_commitment": r"$\mathrm{Commit}_{\mathrm{CoT}}$",
}
# Short forms for tight axis labels.
METRIC_LABEL_SHORT: dict[str, str] = {
    "answer_alignment": r"$\mathrm{Align}_{\mathrm{ans}}$",
    "answer_commitment": r"$\mathrm{Comm}_{\mathrm{ans}}$",
    "cot_verbalization": r"$\mathrm{Verb}_{\mathrm{CoT}}$",
    "cot_attribution": r"$\mathrm{Attr}_{\mathrm{CoT}}$",
    "cot_commitment": r"$\mathrm{Comm}_{\mathrm{CoT}}$",
}

# Verbal names for the two headline rates. Used as axis labels in place of
# the raw math forms (e.g. "Verbalized Commitment Rate" instead of
# r"$P(Commit_CoT | Align_ans)$").
VERBALIZED_COMMITMENT_RATE_LABEL: str = "Verbalized Commitment Rate"
CUE_FOLLOWING_RATE_LABEL: str = "Cue-Following Rate"
UNVERBALIZED_ADOPTION_RATE_LABEL: str = "Unverbalized Adoption Rate"


def conditional_faithfulness_label(metric_key: str, short: bool = False) -> str:
    """Build the axis label for a CoT metric, conditioned on the answer aligning.

    Returns e.g. r"$P(\\mathrm{Commit}_\\mathrm{CoT} \\mid \\mathrm{Align}_\\mathrm{ans})$".

    Args:
        metric_key: one of the cot_* keys in METRIC_LABEL.
        short: use the short forms (Verb / Attr / Comm) instead of full words.
    """
    labels = METRIC_LABEL_SHORT if short else METRIC_LABEL

    def _bare(s: str) -> str:
        return s.strip("$")

    cond = _bare(labels["answer_alignment"])
    return rf"$P({_bare(labels[metric_key])} \mid {cond})$"


def setup_plot_style(wide: bool = False):
    """Set up consistent plotting style for all visualizations in the project.

    Args:
        wide: if True, uses FIGURE_SIZE_WIDE and pre-compensated font sizes for
              figures designed at 10 in that will be scaled to ~5.5 in in LaTeX.
    """
    rc_params = plt.rcParams

    sns.set_theme(style="whitegrid")
    rc_params["text.usetex"] = False
    rc_params["font.size"] = "12.5"
    rc_params["figure.dpi"] = 190
    rc_params["axes.unicode_minus"] = False
    rc_params["font.family"] = "cmr10"
    rc_params["mathtext.fontset"] = "cm"
    rc_params["axes.formatter.use_mathtext"] = True

    # Set rc_params for the border color and ticks
    rc_params["axes.edgecolor"] = "black"  # Set border color
    rc_params["axes.linewidth"] = 1.5  # Set border width

    # Black border on all bars (patches)
    rc_params["patch.edgecolor"] = "black"
    rc_params["patch.linewidth"] = 2.0
    rc_params["patch.force_edgecolor"] = True
    rc_params["xtick.color"] = "black"  # Set xtick color
    rc_params["ytick.color"] = "black"  # Set ytick color

    # set background color
    rc_params["axes.facecolor"] = "#EFEFEAFF"

    # set grid color
    rc_params["grid.color"] = "white"
    rc_params["grid.alpha"] = 0.7
    rc_params["grid.linewidth"] = 1.5
    rc_params["grid.linestyle"] = "--"

    # make ticks show
    rc_params["xtick.bottom"] = True  # Ensure xticks are shown at the bottom
    rc_params["ytick.left"] = True  # Ensure yticks are shown on the left

    sns.set_context(context="talk", font_scale=0.9)

    # Standardized font sizes
    rc_params["axes.titlesize"] = FONT_SIZE_TITLE_WIDE if wide else FONT_SIZE_TITLE
    rc_params["axes.labelsize"] = FONT_SIZE_AXIS_LABEL_WIDE if wide else FONT_SIZE_AXIS_LABEL
    rc_params["axes.labelpad"] = 0
    rc_params["xtick.labelsize"] = FONT_SIZE_AXIS_TICKS_WIDE if wide else FONT_SIZE_AXIS_TICKS
    rc_params["ytick.labelsize"] = FONT_SIZE_AXIS_TICKS_WIDE if wide else FONT_SIZE_AXIS_TICKS
    # Matplotlib's own default — anything tighter puts the labels against the axes.
    rc_params["xtick.major.pad"] = 3.5
    rc_params["ytick.major.pad"] = 3.5
    rc_params["legend.fontsize"] = FONT_SIZE_LEGEND_WIDE if wide else FONT_SIZE_LEGEND
    rc_params["figure.figsize"] = FIGURE_SIZE_WIDE if wide else FIGURE_SIZE


def facecolor_alpha(color, alpha: float):
    """Return an RGBA tuple with alpha applied only to the facecolor.

    Use instead of the `alpha` kwarg when you want a transparent fill but a
    fully opaque (black) border on the same patch.
    """
    import matplotlib.colors as mcolors

    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, alpha)


def yerr_from_ci(point: float, ci_lo: float, ci_hi: float) -> list[list[float]]:
    """Asymmetric matplotlib yerr from (point, ci_lo, ci_hi). Clamps to >=0.

    Use:  ax.errorbar(x, point, yerr=yerr_from_ci(p, lo, hi), ...)
       or ax.bar(x, point, yerr=yerr_from_ci(p, lo, hi), ...)
    """
    return [[max(0.0, point - ci_lo)], [max(0.0, ci_hi - point)]]


def yerrs_from_cis(points: list[float], ci_los: list[float], ci_his: list[float]) -> list[list[float]]:
    """Batch version of `yerr_from_ci` for vector inputs. Returns a (2, N) list."""
    return [
        [max(0.0, p - lo) for p, lo in zip(points, ci_los)],
        [max(0.0, hi - p) for p, hi in zip(points, ci_his)],
    ]


def tight_ylim(
    values: list[float],
    sems: list[float] | None = None,
    zero_floor: bool = True,
    margin: float = 0.12,
) -> tuple[float, float]:
    """Compute y-axis limits zoomed to the data range with margin.

    Args:
        values: data values to span.
        sems: optional error bar sizes; limits include ±sem.
        zero_floor: clamp ymin to 0 (use for bar charts / rates).
        margin: fractional padding beyond the data range on each side.
    """
    if not values:
        return 0.0, 1.05
    _s = sems if sems is not None else [0.0] * len(values)
    lo = min(v - s for v, s in zip(values, _s))
    hi = max(v + s for v, s in zip(values, _s))
    span = max(hi - lo, 0.01)
    pad = span * margin
    ymin = max(0.0, lo - pad) if zero_floor else lo - pad
    return ymin, hi + pad


def save_figure(fig, path, **kwargs):
    """Save figure as SVG without white padding, no title."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_alpha(0)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0, **kwargs)
    plt.close(fig)


def save_table(path, rows: list[dict], columns: list[str] | None = None) -> None:
    """Save a list of dicts as CSV next to a figure for eyeballing / paper use.

    `columns` pins the column order; defaults to the union of keys across rows.
    Missing values are written as empty cells. Numeric values are written in
    their native repr — Excel-friendly formatting is the caller's job.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        seen: set[str] = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    columns.append(k)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


def figure_suffix(model: str | None) -> str:
    """Filename suffix for a model/monitor-specific figure: ``__{sanitized}`` or ``""``.

    Slashes become underscores so the result is a valid filename. Use this everywhere a
    figure is parameterized by a model/monitor so every script names files identically
    (a figure parameterized by N monitors is N single-monitor files, never one clobbered
    file).
    """
    return f"__{model.replace('/', '_')}" if model else ""


def short_model_name(model: str) -> str:
    """Run-model id → compact pretty label.

    'Qwen_Qwen3.5-27B' → 'Qwen3.5-27B'; 'openai_gpt-oss-120b_medium' → 'GPT-OSS-120B'.
    """
    base = VARIANT_TO_BASE.get(model, model)
    slash_name = base.replace("_", "/", 1) if "_" in base else base
    if slash_name in MODEL_PRETTY_NAMES:
        return MODEL_PRETTY_NAMES[slash_name]
    if base in MODEL_LABEL_INLINE:
        return MODEL_LABEL_INLINE[base]
    return (base.split("_", 1)[-1] if "_" in base else base).replace("_", "-")


def short_monitor_name(monitor: str) -> str:
    """Monitor model id → display label for panel titles."""
    if "4o-mini" in monitor:
        return "GPT-4o mini"
    if "luna" in monitor:
        return "GPT-5.6-Luna"
    return monitor


def to_float(v) -> float:
    """Parse a CSV cell to float; nan on empty/unparseable."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_metric_rows(path, index_by: tuple[str, ...] | None = None, str_cols: tuple[str, ...] = ()):
    """Load an analysis CSV, coercing numeric cells to float.

    Returns a list of row dicts. If ``index_by`` (a tuple of column names) is given,
    returns a dict keyed by those columns' values (a bare value for one key column, a
    tuple for several) instead. Columns named in ``index_by`` or ``str_cols`` are kept
    as strings.
    """
    import csv

    keep_str = set(str_cols) | set(index_by or ())
    with open(path) as f:
        rows = [{k: (v if k in keep_str else to_float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    if index_by is None:
        return rows
    if len(index_by) == 1:
        col = index_by[0]
        return {r[col]: r for r in rows}
    return {tuple(r[k] for k in index_by): r for r in rows}


def save_legend(handles, labels, path, ncol=1, **kwargs):
    """Save legend as a separate SVG file.

    The frame is fixed here rather than per caller: these files are composed into paper
    panels side by side, where a legend with matplotlib's grey default edge reads as a
    different kind of object from its neighbour. The face is left clear for the same
    reason — some callers set `axes.facecolor` to none and some do not, and an inherited
    face turns up as a grey box on the panel's cream background.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis("off")
    legend = ax.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        edgecolor="black",
        facecolor="none",
        framealpha=0.9,
        **kwargs,
    )
    fig.savefig(
        path,
        format="svg",
        bbox_inches=legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted()),
        transparent=True,
        pad_inches=0.0,
    )
    plt.close(fig)
