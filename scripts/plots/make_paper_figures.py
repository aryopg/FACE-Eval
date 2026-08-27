#!/usr/bin/env python3
"""Compose paper-ready figure panels from the standalone SVGs in ``figures/``.

Each panel is a single self-contained SVG one TMLR column wide (6.5in = 468pt): a
cream background box, a left-aligned title, the figure (or a row of subfigures
with subtitles), and the legend(s). A panel marked ``tight`` is narrower — only as
wide as its own content — so the LaTeX can include it at a smaller fraction of the
line without shrinking the art.

The source SVGs are matplotlib output, so 1 user unit = 1pt and the ids are
uuid-salted per figure — they nest 1:1 with no rewriting. Titles are rendered
through matplotlib so they come out in Computer Modern like the panel text.

Usage:
    python scripts/plots/make_paper_figures.py
    python scripts/plots/make_paper_figures.py --only 5,7,13 --png
    python scripts/plots/make_paper_figures.py --figures-dir figures_remote figures
"""

from __future__ import annotations

import argparse
import io
import shutil
import string
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from scripts.plots._eval_aware_filter import SUFFIX as EVAL_UNAWARE_SUFFIX
from src.utils.plotting import setup_plot_style

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

# Geometry, all in points (1pt = 1/72in).
PAPER_WIDTH_PT = 468.0  # 6.5in — TMLR single-column text width
PAD_X = 12.0
PAD_Y = 6.0
CONTENT_WIDTH = PAPER_WIDTH_PT - 2 * PAD_X
BG_COLOR = "#FBF4E4"

# The vertical gaps are deliberately tight: matplotlib already leaves whitespace
# inside each source SVG, so the gap here is added to that, not to the ink.
GAP_TITLE = 6.0  # title -> first row
GAP_SUBTITLE = 3.0  # subtitle -> what follows it
GAP_LEGEND = 4.0  # subfigure -> its own legend
GAP_ROW = 7.0  # row -> row
GAP_SHARED_LEGEND = 6.0  # last row -> shared legend
GAP_COL = 10.0  # column -> column within a row (horizontal, so no page cost)

FONT_SIZE_PANEL_TITLE = 14.0
FONT_SIZE_SUBTITLE = 10.0

# CMU Serif Bold is Computer Modern Bold *Extended*, so it registers as
# stretch="expanded"; without asking for that, matplotlib picks the regular Roman.
FONT = "CMU Serif"
FONT_WEIGHT = "bold"
FONT_STRETCH = "expanded"

# The H2 figure names carry the monitor id via plotting.figure_suffix(). Pin it
# so the panel does not silently follow whichever monitor happens to sort last.
H2_MONITOR = "gpt-5.6-luna"
# The weak monitor's appendix twin of figure 7. It saw one checkpoint the strong one
# did not (Inkling-NVFP4 at 0.99), so its panel carries 9 checkpoints against 7's 8.
H2_MONITOR_WEAK = "gpt-4o-mini-2024-07-18"

# Same for the inter-judge scatters, whose filenames carry the second judge and
# the convention it was measured under (Makefile: SECOND_JUDGE, JUDGE_CONVENTION).
INTER_JUDGE = "__gpt-5.6-luna__ALL"


@dataclass
class Sub:
    """One subfigure: an SVG stem plus optional subtitle and legend."""

    svg: str
    subtitle: str | None = None
    legend: str | None = None  # placed below the subfigure
    legend_above: str | None = None  # placed above it instead (Figure 5)
    # Keep the unfiltered art in an otherwise eval-unaware panel. The clarity rating
    # is a separate pass that the paper reports before the filter, so Figure 3 mixes
    # an unfiltered (a) with a filtered (b).
    eval_unaware_exempt: bool = False


@dataclass
class Panel:
    number: int
    title: str  # "" for a panel whose LaTeX caption already names it
    rows: list[list[Sub]]
    stem: str
    legend: str | None = None  # shared legend, centered under the last row
    # Shrink the box to its content instead of padding it out to the full column.
    # For art that is narrower than the column at 1:1, the cream margin is dead
    # width, and the panel is then included at a smaller \linewidth fraction with
    # the art no smaller than before.
    tight: bool = False
    subs: list[Sub] = field(init=False)

    def __post_init__(self) -> None:
        self.subs = [s for row in self.rows for s in row]


PANELS: list[Panel] = [
    Panel(
        number=2,
        title="VCR in Agentic Scenarios",
        stem="channel_rates",
        rows=[[Sub("h1_channel_rates_l3_4bar")]],
        legend="h1_channel_rates_l3_4bar_legend",
    ),
    Panel(
        number=3,
        title="Model-Rated Cue Clarity and Cue-Following Rate",
        stem="cue_saliency",
        rows=[
            [
                Sub(
                    "cue_clarity",
                    subtitle="Model-Rated Cue Clarity",
                    legend="cue_clarity_legend",
                    eval_unaware_exempt=True,
                ),
                Sub("no_context_shift", subtitle="CFR"),
            ]
        ],
    ),
    Panel(
        number=4,
        title="UAR Across Channels and Explicitness",
        stem="phase_diagram",
        rows=[[Sub("h1_phase_diagram_l3")]],
        legend="h1_phase_diagram_legend",
    ),
    Panel(
        number=5,
        title=r"$\Delta$UAR Between Channels and Explicitness",
        stem="delta_channel_explicitness",
        rows=[
            [
                Sub(
                    "register_matched_dumbbell_agg_covert",
                    subtitle="User-Message vs Tool-Return",
                    legend_above="register_matched_dumbbell_agg_covert_legend",
                ),
                Sub(
                    "h1_register_l3_covert_b",
                    subtitle="Explicit vs Implicit",
                    legend_above="h1_register_l3_covert_legend_markers",
                ),
            ]
        ],
        legend="h1_phase_diagram_legend",
    ),
    Panel(
        number=6,
        title="Cue Explicitness Gap vs Model-Rated Clarity Gap",
        stem="clarity_scatter",
        rows=[[Sub("h6_clarity_scatter")]],
        legend="h6_clarity_scatter_legend",
    ),
    Panel(
        number=7,
        title="Monitor AUROC Across Cue Conditions vs UAR",
        stem="monitor_auroc",
        rows=[
            [
                Sub(
                    f"h2_monitor_capability_bprime__{H2_MONITOR}",
                    subtitle="GPT-5.6-Luna Monitor AUROC",
                    legend=f"h2_monitor_capability_bprime__{H2_MONITOR}_legend",
                ),
                # No checkpoint legend box: the scatter's own in-axes legend already
                # names every checkpoint against its line colour, so a second one only
                # repeats it at panel width.
                Sub(
                    f"h2_thesis_scatter_auroc_bymodel_lines_bprime__{H2_MONITOR}",
                    subtitle="UAR vs Action + CoT Monitor AUROC",
                    legend=f"h2_thesis_scatter_bprime__{H2_MONITOR}_legend_channels",
                ),
            ]
        ],
    ),
    Panel(
        number=8,
        title=r"$\Delta$UAR Between Channels and System Prompts",
        stem="delta_channel_convention",
        rows=[[Sub("convention_dumbbell_v2_covert")]],
        legend="convention_dumbbell_v2_covert_legend",
    ),
    Panel(
        number=9,
        title="UAR vs Generated Reasoning Tokens",
        stem="effort_vs_tokens_covert",
        rows=[[Sub("effort_vs_tokens_covert")]],
        legend="effort_vs_tokens_legend",
    ),
    Panel(
        number=10,
        title="Position Bias in Side-Identification",
        stem="position_bias",
        rows=[[Sub("position_bias")]],
        legend="position_bias_legend",
    ),
    Panel(
        number=11,
        title="VCR vs Generated Reasoning Tokens",
        stem="effort_vs_tokens_verbalized",
        rows=[[Sub("effort_vs_tokens")]],
        legend="effort_vs_tokens_legend",
    ),
    Panel(
        number=12,
        title="CFR vs Generated Reasoning Tokens",
        stem="effort_vs_tokens_alignment",
        rows=[[Sub("effort_vs_tokens_alignment")]],
        legend="effort_vs_tokens_legend",
    ),
    # One panel per source, numbered 18-22 as a block. Stacking all five in one panel
    # composed to 783pt, and TMLR leaves 557pt for a graphic once the caption is set;
    # the height could only be bought back by scaling the text down. Split, each one
    # composes to ~207pt at full text size. 13 is left unused: the numbers are filenames,
    # and the paper order is set by the LaTeX.
    Panel(
        number=18,
        title=r"$\Delta$UAR Between Channels - User Profile",
        stem="delta_channel_profile",
        rows=[[Sub("register_matched_dumbbell_agg_profile_covert")]],
        legend="register_matched_dumbbell_agg_covert_legend",
    ),
    Panel(
        number=19,
        title=r"$\Delta$UAR Between Channels - Email",
        stem="delta_channel_email",
        rows=[[Sub("register_matched_dumbbell_agg_email_covert")]],
        legend="register_matched_dumbbell_agg_covert_legend",
    ),
    Panel(
        number=20,
        title=r"$\Delta$UAR Between Channels - Slack",
        stem="delta_channel_slack",
        rows=[[Sub("register_matched_dumbbell_agg_slack_covert")]],
        legend="register_matched_dumbbell_agg_covert_legend",
    ),
    Panel(
        number=21,
        title=r"$\Delta$UAR Between Channels - Notes",
        stem="delta_channel_notes",
        rows=[[Sub("register_matched_dumbbell_agg_notes_covert")]],
        legend="register_matched_dumbbell_agg_covert_legend",
    ),
    Panel(
        number=22,
        title=r"$\Delta$UAR Between Channels - Browser History",
        stem="delta_channel_browser_history",
        rows=[[Sub("register_matched_dumbbell_agg_browser_history_covert")]],
        legend="register_matched_dumbbell_agg_covert_legend",
    ),
    Panel(
        number=14,
        title="FACE-Eval - Evaluation Awareness",
        stem="eval_awareness",
        rows=[[Sub("h4_eval_awareness")]],
        legend="h4_eval_awareness_legend",
    ),
    Panel(
        number=15,
        title="Inter-Judge Agreement on VCR",
        stem="inter_judge_agreement_vcr",
        rows=[[Sub(f"inter_judge_agreement_vcr{INTER_JUDGE}")]],
        legend=f"inter_judge_agreement_vcr{INTER_JUDGE}_legend",
        tight=True,
    ),
    Panel(
        number=16,
        title="Inter-Judge Agreement on UAR",
        stem="inter_judge_agreement_uar",
        rows=[[Sub(f"inter_judge_agreement_uar{INTER_JUDGE}")]],
        legend=f"inter_judge_agreement_uar{INTER_JUDGE}_legend",
        tight=True,
    ),
    Panel(
        number=23,
        title="Monitor AUROC Across Cue Conditions vs UAR",
        stem="monitor_auroc_weak",
        rows=[
            [
                Sub(
                    f"h2_monitor_capability_bprime__{H2_MONITOR_WEAK}",
                    subtitle="GPT-4o mini Monitor AUROC",
                    legend=f"h2_monitor_capability_bprime__{H2_MONITOR_WEAK}_legend",
                ),
                Sub(
                    f"h2_thesis_scatter_auroc_bymodel_lines_bprime__{H2_MONITOR_WEAK}",
                    subtitle="UAR vs Action + CoT Monitor AUROC",
                    legend=f"h2_thesis_scatter_bprime__{H2_MONITOR_WEAK}_legend_channels",
                ),
            ]
        ],
    ),
    Panel(
        # Main text; figure 9 is the same plot over all six effort-swept models and
        # stays in the appendix. Numbered last to leave every other panel's filename
        # alone — the paper order is set by the LaTeX, not by this list.
        number=17,
        title="UAR vs Generated Reasoning Tokens",
        stem="effort_vs_tokens_covert_main",
        rows=[[Sub("effort_vs_tokens_covert_main")]],
        legend="effort_vs_tokens_legend",
    ),
]


# ---------------------------------------------------------------------------
# SVG pieces
# ---------------------------------------------------------------------------


@dataclass
class Piece:
    """A parsed SVG ready to be dropped into a parent at its natural size."""

    root: ET.Element
    width: float
    height: float


# Panels drawn from the C0 conditioning population, so a --exclude-eval-aware rerun
# of their source scripts moves the numbers. Panel 14 measures eval-awareness itself
# and is deliberately absent; the rest read artifact-rating annotations or analyze_* CSVs that
# carry no eval-awareness flag of their own.
EVAL_UNAWARE_PANELS = {2, 3, 4, 5, 8, 9, 11, 12, 17, 18, 19, 20, 21, 22}

# The suffix lands at the end of the base stem, ahead of whatever structural tail the
# plotting script appends to it (the _a/_b halves, the _main grid, the legends).
_STRUCTURAL_TAILS = ("_legend_models", "_legend_markers", "_legend_channels", "_legend", "_main", "_a", "_b")


def _eval_unaware_stem(stem: str) -> str:
    for tail in _STRUCTURAL_TAILS:
        if stem.endswith(tail):
            return f"{stem[: -len(tail)]}{EVAL_UNAWARE_SUFFIX}{tail}"
    return f"{stem}{EVAL_UNAWARE_SUFFIX}"


def _eval_unaware_panel(panel: Panel) -> Panel:
    """Same panel, pointed at the `_evalunaware` art and composed under its own name."""
    rows = [
        [
            (
                s
                if s.eval_unaware_exempt
                else Sub(
                    _eval_unaware_stem(s.svg),
                    subtitle=s.subtitle,
                    legend=_eval_unaware_stem(s.legend) if s.legend else None,
                    legend_above=_eval_unaware_stem(s.legend_above) if s.legend_above else None,
                )
            )
            for s in row
        ]
        for row in panel.rows
    ]
    return replace(
        panel,
        rows=rows,
        stem=f"{panel.stem}{EVAL_UNAWARE_SUFFIX}",
        legend=_eval_unaware_stem(panel.legend) if panel.legend else None,
    )


def _pt(value: str) -> float:
    return float(value.removesuffix("pt"))


def find_svg(stem: str, source_dirs: list[Path]) -> Path | None:
    """First `<stem>.svg` across the source directories, searched in order."""
    return next((d / f"{stem}.svg" for d in source_dirs if (d / f"{stem}.svg").exists()), None)


def read_svg(stem: str, source_dirs: list[Path]) -> Piece:
    root = ET.parse(find_svg(stem, source_dirs)).getroot()
    return Piece(root, _pt(root.get("width", "0")), _pt(root.get("height", "0")))


def register_chrome_font() -> None:
    """Make CMU Serif available to matplotlib, whose cache does not index it.

    The source figures need no font beyond cmr10, so only the composition step
    depends on this: regenerate figures wherever you like, compose on a box that
    has CMU Serif installed.
    """
    _INSTALL_HINT = (
        "Install it with `apt-get install fonts-cmu` (Debian/Ubuntu), or drop the cmun* font files into "
        "~/.local/share/fonts. Nothing else in the pipeline needs it — only this script, so the figures "
        "themselves can be generated on a box without it."
    )
    for path in fm.findSystemFonts():
        if Path(path).name.startswith("cmun"):
            fm.fontManager.addfont(path)
    try:
        found = fm.findfont(
            fm.FontProperties(family=FONT, weight=FONT_WEIGHT, stretch=FONT_STRETCH), fallback_to_default=False
        )
    except ValueError as exc:  # matplotlib raises before the check below can run
        raise SystemExit(
            f"{FONT} {FONT_WEIGHT} is not installed, so the panel chrome has no font.\n{_INSTALL_HINT}"
        ) from exc
    # Stem, not filename: Debian's fonts-cmu ships cmunbx.otf where macOS has
    # cmunbx.ttf, and matplotlib scans both.
    if Path(found).stem != "cmunbx":
        raise SystemExit(f"Expected CMU Serif Bold (cmunbx), matplotlib resolved {found}.\n{_INSTALL_HINT}")


def text_width(text: str, fontsize: float) -> float:
    """Rendered width of a single line, in points.

    The extent comes back in device pixels, and some backends (macosx) override
    the requested dpi with a retina one — so divide by the figure's actual dpi.
    """
    fig = plt.figure(figsize=(20.0, 1.0), dpi=72)
    artist = fig.text(0.0, 0.0, text, fontsize=fontsize, family=FONT, weight=FONT_WEIGHT, stretch=FONT_STRETCH)
    pixels = artist.get_window_extent(fig.canvas.get_renderer()).width
    dpi = fig.dpi
    plt.close(fig)
    return pixels * 72.0 / dpi


def wrap_text(text: str, fontsize: float, max_width: float) -> str:
    """Greedily break `text` into lines that each fit `max_width`."""
    if text_width(text, fontsize) <= max_width:
        return text
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and text_width(trial, fontsize) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return "\n".join(lines)


def render_text(text: str, fontsize: float, max_width: float | None = None, align: str = "center") -> Piece:
    """Render text (mathtext allowed) to an SVG piece, wrapped to `max_width`.

    The canvas is tiny on purpose: bbox_inches="tight" unions it with the text,
    so a larger one would show up as padding. The text overflows it and tight
    still finds it.

    `align` only matters once the text wraps: it aligns the lines against each other.
    """
    if max_width is not None:
        text = wrap_text(text, fontsize, max_width)
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.patch.set_alpha(0)
    fig.text(
        0.0,
        0.0,
        text,
        fontsize=fontsize,
        family=FONT,
        weight=FONT_WEIGHT,
        stretch=FONT_STRETCH,
        ha="left",
        va="baseline",
        ma=align,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    root = ET.parse(buf).getroot()
    return Piece(root, _pt(root.get("width", "0")), _pt(root.get("height", "0")))


def place(parent: ET.Element, piece: Piece, x: float, y: float, scale: float = 1.0) -> None:
    """Nest a piece into the parent at (x, y), scaled about its top-left corner."""
    g = ET.SubElement(parent, f"{{{SVG_NS}}}g")
    g.set("transform", f"translate({x:.4f} {y:.4f}) scale({scale:.6f})")
    for child in piece.root:
        if child.tag != f"{{{SVG_NS}}}metadata":
            g.append(child)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass
class Column:
    """One subfigure's art, measured at scale 1. The subtitle is filled in later,
    once the panel scale is known and the wrap width can be computed."""

    figure: Piece
    legend: Piece | None
    legend_above: Piece | None
    subtitle: Piece | None = None

    def width(self) -> float:
        parts = [self.figure.width] + [p.width for p in (self.legend, self.legend_above) if p is not None]
        return max(parts)


def load_columns(panel: Panel, source_dirs: list[Path]) -> list[list[Column]]:
    return [
        [
            Column(
                figure=read_svg(sub.svg, source_dirs),
                legend=read_svg(sub.legend, source_dirs) if sub.legend else None,
                legend_above=read_svg(sub.legend_above, source_dirs) if sub.legend_above else None,
            )
            for sub in row
        ]
        for row in panel.rows
    ]


def row_width(cols: list[Column]) -> float:
    """Width the row occupies at scale 1, gaps included."""
    return sum(c.width() for c in cols) + GAP_COL * (len(cols) - 1)


def row_scales(cols: list[Column], content_width: float) -> list[float]:
    """Per-column scales that give every subfigure in the row the same height.

    Solve for the common height H that makes the row exactly fill the content
    width: sum(width_i * H / height_i) + gaps = content_width. If that would
    enlarge any subfigure past 1:1, shrink the whole row so the largest sits at
    1:1 — equal heights are preserved either way.
    """
    budget = content_width - GAP_COL * (len(cols) - 1)
    height = budget / sum(c.width() / c.figure.height for c in cols)
    scales = [height / c.figure.height for c in cols]
    excess = max(scales)
    return [s / excess for s in scales] if excess > 1.0 else scales


def compose(panel: Panel, source_dirs: list[Path]) -> tuple[ET.Element, float, float]:
    rows = load_columns(panel, source_dirs)
    shared_legend = read_svg(panel.legend, source_dirs) if panel.legend else None

    # The shared legend counts as content: leaving it out would put it over the
    # box edge, and capping it to the art would scale its text below the figure's.
    natural = max([row_width(cols) for cols in rows] + [shared_legend.width if shared_legend else 0.0])
    content_width = min(CONTENT_WIDTH, natural) if panel.tight else CONTENT_WIDTH
    panel_width = content_width + 2 * PAD_X
    # An empty title still renders as the tiny blank canvas render_text draws on, so
    # skip it outright rather than paying for a phantom line plus GAP_TITLE.
    title = render_text(panel.title, FONT_SIZE_PANEL_TITLE, content_width, align="left") if panel.title else None

    scales = [row_scales(cols, content_width) for cols in rows]
    # The shared legend is sized off the largest subfigure so its text tracks the
    # panel's, then capped to the art it annotates. A multi-column row is justified
    # edge to edge, so it spans the content width; a lone column keeps its own
    # width and centers, and a legend wider than that reads as a separate object.
    content_span = max(
        content_width if len(cols) > 1 else cols[0].width() * col_scales[0] for cols, col_scales in zip(rows, scales)
    )
    legend_scale = (
        min(max(s for row in scales for s in row), content_span / shared_legend.width) if shared_legend else 0.0
    )

    # Subtitles are chrome (never scaled), so they must be wrapped to the width
    # the column ends up occupying, not to the source figure's natural width.
    # Lettering runs a) b) c) … in reading order and restarts every panel.
    letters = iter(string.ascii_lowercase)
    for spec_row, cols, col_scales in zip(panel.rows, rows, scales):
        for sub, col, s in zip(spec_row, cols, col_scales):
            if sub.subtitle:
                text = f"{next(letters)}) {sub.subtitle}"
                col.subtitle = render_text(text, FONT_SIZE_SUBTITLE, col.width() * s, align="left")

    root = ET.Element(f"{{{SVG_NS}}}svg", {"version": "1.1"})
    body = ET.SubElement(root, f"{{{SVG_NS}}}g")  # everything above the box goes here

    y = PAD_Y
    if title is not None:
        place(body, title, PAD_X, y)
        y += title.height + GAP_TITLE

    for cols, col_scales in zip(rows, scales):
        widths = [c.width() * s for c, s in zip(cols, col_scales)]
        # Justify: equalizing heights leaves the columns at whatever widths that
        # implies, so any slack goes into the gaps between them. The row then
        # spans edge to edge and the left and right margins are both PAD_X. A
        # lone column has nowhere to put slack, so it gets centered instead.
        slack = content_width - sum(widths)
        gap = slack / (len(cols) - 1) if len(cols) > 1 else 0.0
        x = PAD_X if len(cols) > 1 else PAD_X + slack / 2
        # Bands are row-wide so that every figure in the row starts at the same
        # y even when one subtitle wrapped to two lines and another did not.
        subtitle_band = max((c.subtitle.height + GAP_SUBTITLE for c in cols if c.subtitle), default=0.0)
        above_band = max(
            (c.legend_above.height * s + GAP_SUBTITLE for c, s in zip(cols, col_scales) if c.legend_above),
            default=0.0,
        )
        figure_top = y + subtitle_band + above_band
        row_bottom = figure_top
        for col, scale, col_w in zip(cols, col_scales, widths):
            if col.subtitle is not None:  # left-aligned on the column, bottom of its band
                place(body, col.subtitle, x, y + subtitle_band - GAP_SUBTITLE - col.subtitle.height)
            if col.legend_above is not None:
                legend_h = col.legend_above.height * scale
                place(
                    body,
                    col.legend_above,
                    x + (col_w - col.legend_above.width * scale) / 2,
                    figure_top - GAP_SUBTITLE - legend_h,
                    scale,
                )
            place(body, col.figure, x + (col_w - col.figure.width * scale) / 2, figure_top, scale)
            bottom = figure_top + col.figure.height * scale
            if col.legend is not None:
                bottom += GAP_LEGEND
                place(body, col.legend, x + (col_w - col.legend.width * scale) / 2, bottom, scale)
                bottom += col.legend.height * scale
            row_bottom = max(row_bottom, bottom)
            x += col_w + gap
        y = row_bottom + GAP_ROW

    y -= GAP_ROW
    if shared_legend is not None:
        y += GAP_SHARED_LEGEND
        place(body, shared_legend, PAD_X + (content_width - shared_legend.width * legend_scale) / 2, y, legend_scale)
        y += shared_legend.height * legend_scale

    height = y + PAD_Y
    root.set("width", f"{panel_width:.4f}pt")
    root.set("height", f"{height:.4f}pt")
    root.set("viewBox", f"0 0 {panel_width:.4f} {height:.4f}")

    # Inline style, not a fill attribute: the nested panels each carry a global
    # `*{...}` CSS rule, and a `*` selector outranks presentation attributes.
    box = ET.Element(f"{{{SVG_NS}}}rect")
    box.set("x", "0")
    box.set("y", "0")
    box.set("width", f"{panel_width:.4f}")
    box.set("height", f"{height:.4f}")
    box.set("style", f"fill: {BG_COLOR}")
    root.insert(0, box)
    return root, panel_width, height


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def check_sources(panels: list[Panel], source_dirs: list[Path]) -> None:
    missing = []
    for panel in panels:
        stems = [s.svg for s in panel.subs]
        stems += [s.legend for s in panel.subs if s.legend]
        stems += [s.legend_above for s in panel.subs if s.legend_above]
        if panel.legend:
            stems.append(panel.legend)
        missing += [f"fig{panel.number}: {s}.svg" for s in stems if find_svg(s, source_dirs) is None]
    if missing:
        raise SystemExit("Missing source SVGs:\n  " + "\n  ".join(missing))


def convert(svg: Path, fmt: str, extra: list[str]) -> None:
    subprocess.run(
        ["rsvg-convert", "-f", fmt, *extra, "-o", str(svg.with_suffix(f".{fmt}")), str(svg)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir",
        nargs="+",
        default=["figures"],
        help="one or more source directories, searched in order — the first one holding a given SVG wins "
        "(e.g. --figures-dir figures_remote figures to prefer pod-generated figures)",
    )
    parser.add_argument("--out-dir", default=None, help="default: <first figures-dir>/paper")
    parser.add_argument("--only", default=None, help="comma-separated figure numbers, e.g. 5,7,13")
    parser.add_argument("--no-pdf", action="store_true", help="skip the PDF render")
    parser.add_argument("--png", action="store_true", help="also write a 300dpi PNG for eyeballing")
    parser.add_argument(
        "--eval-unaware",
        action="store_true",
        help="compose from the '_evalunaware' art written by the plot scripts' --exclude-eval-aware runs; "
        "only the panels that filter can move are composed",
    )
    args = parser.parse_args()

    source_dirs = [Path(d) for d in args.figures_dir]
    missing_dirs = [d for d in source_dirs if not d.is_dir()]
    if missing_dirs:
        raise SystemExit(f"Not a directory: {', '.join(str(d) for d in missing_dirs)}")
    out_dir = Path(args.out_dir) if args.out_dir else source_dirs[0] / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = PANELS
    if args.only:
        wanted = {int(n) for n in args.only.split(",")}
        panels = [p for p in PANELS if p.number in wanted]
        unknown = wanted - {p.number for p in PANELS}
        if unknown:
            raise SystemExit(f"Unknown figure numbers: {sorted(unknown)}")

    if args.eval_unaware:
        skipped = sorted(p.number for p in panels if p.number not in EVAL_UNAWARE_PANELS)
        if skipped:
            print(f"eval-unaware: skipping figures {skipped} — the filter cannot move them")
        panels = [_eval_unaware_panel(p) for p in panels if p.number in EVAL_UNAWARE_PANELS]
        if not panels:
            raise SystemExit("No eval-unaware panels selected.")

    check_sources(panels, source_dirs)
    setup_plot_style()
    register_chrome_font()
    print(f"sources: {' -> '.join(str(d) for d in source_dirs)}")

    have_rsvg = shutil.which("rsvg-convert") is not None
    if not have_rsvg and (not args.no_pdf or args.png):
        print("warning: rsvg-convert not found — writing SVG only")

    for panel in panels:
        root, width, height = compose(panel, source_dirs)
        out = out_dir / f"fig{panel.number:02d}_{panel.stem}.svg"
        ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
        print(f"fig{panel.number:02d}  {width:5.1f} x {height:7.1f} pt  {out}")
        if have_rsvg:
            if not args.no_pdf:
                convert(out, "pdf", [])
            if args.png:
                convert(out, "png", ["-d", "300", "-p", "300"])


if __name__ == "__main__":
    main()
