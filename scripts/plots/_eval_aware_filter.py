"""Shared ``--exclude-eval-aware`` wiring for the paper-figure scripts.

H4 closed on "eval-awareness is below 2% in every (model, cell)", which held for the
open-weight sweep. It does not hold for the frontier additions (Inkling-Small reaches
15% on the user channel), so the headline figures need a variant measured on the
eval-unaware subset. The flag is off by default: with no arguments every script writes
the same filenames with the same numbers as before.

Filtered runs stamp ``_evalunaware`` on every output stem so both variants coexist in
one figures dir and a missing filtered file fails loudly instead of resolving to stale
unfiltered art.
"""

from __future__ import annotations

import argparse

from src.results.db import ResultsDB
from src.utils.plotting import CELL_CONTEXT_TYPES

SUFFIX = "_evalunaware"

_CELLS = ("user_explicit", "user_implicit", "tool_explicit", "tool_implicit")
_CELL_OF = {ctx: cell for cell in _CELLS for ctx in CELL_CONTEXT_TYPES[cell]}


def add_flag(parser: argparse.ArgumentParser) -> None:
    """Register --exclude-eval-aware on a script's parser."""
    parser.add_argument(
        "--exclude-eval-aware",
        action="store_true",
        help="Drop rows the reasoning judge flagged eval-aware; stamps '_evalunaware' on every output stem.",
    )


def suffix(args: argparse.Namespace) -> str:
    """Output-stem suffix for this run, for scripts that need it before loading a db."""
    return SUFFIX if getattr(args, "exclude_eval_aware", False) else ""


def apply(db: ResultsDB, args: argparse.Namespace) -> tuple[ResultsDB, str]:
    """Return (possibly filtered db, output-stem suffix).

    Pass the causal-dependent conditioning population before effort variants are
    pooled: filtering is row-wise either way, but the report then names the variant
    that lost the rows, and awareness splits hard across efforts.
    """
    if not suffix(args):
        return db, ""

    filtered = db.filter_eval_unaware()
    _report(db, filtered)
    return filtered, SUFFIX


def _tally(db: ResultsDB) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for r in db.records:
        cell = _CELL_OF.get(r.get("context_type"))
        if cell is not None:
            counts[(r["_model"], cell)] = counts.get((r["_model"], cell), 0) + 1
    return counts


def _report(before: ResultsDB, after: ResultsDB) -> None:
    """Print kept/total per (model, cell), the denominators the figures divide by.

    Awareness concentrates on the user channel, so a modest per-model drop can be a
    large one in a single cell. hypotheses.md sets the usable floor at N ≈ 50 per
    (model, cell); printing per cell is what makes a shifted gap distinguishable
    from a cell that simply ran out of rows.
    """
    kept, total = _tally(after), _tally(before)
    print("Eval-aware filter, conditioning population kept/total per (model, cell):")
    for model in sorted({m for m, _ in total}):
        cells = "  ".join(f"{cell} {kept.get((model, cell), 0)}/{total.get((model, cell), 0)}" for cell in _CELLS)
        print(f"  {model:44s} {cells}")
