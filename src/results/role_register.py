"""Shared helpers for the role x register scaffolds.

Centralises:
  - the 2x2 cell mapping (role x register -> context_type)
  - the metric field names used everywhere
  - the no_context-normalised, parse-ok-filtered ResultsDB loader
  - scenario-cluster bootstrap on per-cell binary means
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.results.db import ResultsDB
from src.utils.plotting import pool_effort_variants

CONVENTION_DEFAULT = "C0"
CONVENTION_CHOICES = ("ALL", "C0", "C3", "MC0", "MC3")

METRICS: dict[str, str] = {
    "L3": "judge.reasoning_tailoring_explicit",
}

# 2x2 design: (role, register) -> tuple of context_types pooled into that cell.
# The (user, summary) cell pools user_turn (prose) and user_turn_structured
# (XML-tagged) — they differ only in template; both are summary-register in
# the user channel. Tool-side cells are source-agnostic by construction
# because `source` lives in its own record field.
CELLS_2X2: dict[tuple[str, str], tuple[str, ...]] = {
    ("tool", "summary"): ("explicit",),
    ("tool", "raw"): ("implicit",),
    ("user", "summary"): ("user_turn", "user_turn_structured"),
    ("user", "raw"): ("user_turn_implicit",),
}

# Canonical cell names used as keys throughout bootstrap / contrast machinery.
_CANONICAL_2X2: frozenset[str] = frozenset({"tool_summary", "tool_raw", "user_summary", "user_raw"})

# Q4 within-user-turn gradient (increasing realism)
USER_GRADIENT: list[str] = ["user_turn", "user_turn_structured", "user_turn_implicit"]

PARSE_OK = {"judge.reasoning_parse_ok": True, "judge.answer_parse_ok": True}


def load_clean(
    convention: str = CONVENTION_DEFAULT,
    results_dir: str = "results/agentic",
) -> ResultsDB:
    """no_context-normalised, parse-ok-filtered, convention-fixed result set.

    .filter_causal_dependent() keeps only cued rows where (a) stance_label != 'none'
    (cue shifted the answer) AND (b) the no_context baseline shows committed==False
    (model had no prior commitment). No_context rows are excluded from the output.

    convention="ALL" pools across all conventions (C0, C3, MC0, MC3).
    """
    db = ResultsDB.load_all(results_dir=results_dir, require_judged=True)
    if convention != "ALL":
        db = db.filter(_convention=convention)
    return pool_effort_variants(db.filter(**PARSE_OK).filter_causal_dependent())


# ---------------------------------------------------------------------------
# Bootstrap on contrasts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSummary:
    rate: float
    se: float
    n: int


def cell_rate(db: ResultsDB, field: str) -> CellSummary:
    rate, se = db.cluster_mean_sem(field)
    return CellSummary(rate=rate, se=se, n=db.count())


def _records_by_cluster(records: list[dict], cluster_key: str = "scenario_id") -> dict:
    out: dict = {}
    for r in records:
        out.setdefault(r.get(cluster_key), []).append(r)
    return out


def _binary(field: str, r: dict) -> int | None:
    parts = field.split(".")
    v = r
    for p in parts:
        if not isinstance(v, dict):
            return None
        v = v.get(p)
    if v is None:
        return None
    return 1 if v else 0


def bootstrap_cells_and_contrasts(
    db: ResultsDB,
    cell_to_context: dict[str, str | tuple[str, ...]],
    field: str,
    n_boot: int = 2000,
    seed: int = 42,
    cluster_key: str = "scenario_id",
) -> dict:
    """Cluster-bootstrap per-cell means and linear contrasts.

    `cell_to_context` maps each cell name to the set of context_types pooled
    into that cell. A string value is accepted as shorthand for a one-element
    tuple, so gradient-style callers (one context_type per cell) keep working.

    Returns a dict shaped like:
      {
        "cells": {cell_name: {"point": float, "ci": (lo, hi), "n": int}},
        "contrasts": {name: {"point": float, "ci": (lo, hi)}},  # 2-sided 95%
        "missing_cells": [cell_name, ...],
      }
    Contrasts assume the four 2x2 cell names ("tool_summary", "tool_raw",
    "user_summary", "user_raw"). If any are absent, contrasts are omitted.
    """
    rng = np.random.default_rng(seed)

    # Normalise: every cell maps to a tuple of context_types.
    cell_ctxs: dict[str, tuple[str, ...]] = {
        cell: ((ctx,) if isinstance(ctx, str) else tuple(ctx)) for cell, ctx in cell_to_context.items()
    }
    # Reverse index: which cell does a given context_type belong to?
    ctx_to_cell: dict[str, str] = {ctx: cell for cell, ctxs in cell_ctxs.items() for ctx in ctxs}

    cell_records: dict[str, list[dict]] = {}
    for cell, ctxs in cell_ctxs.items():
        cell_records[cell] = db.filter_in("context_type", ctxs).records

    missing = [c for c, recs in cell_records.items() if not recs]

    def cell_mean(records: list[dict]) -> float:
        vals = [_binary(field, r) for r in records]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    point_cells = {c: cell_mean(recs) for c, recs in cell_records.items()}
    n_cells = {c: len(recs) for c, recs in cell_records.items()}

    # Bootstrap by scenario cluster across the union of all cells.
    all_records = [r for recs in cell_records.values() for r in recs]
    by_cluster = _records_by_cluster(all_records, cluster_key)
    cluster_keys = list(by_cluster.keys())

    boot_cells: dict[str, list[float]] = {c: [] for c in cell_records}
    boot_contrasts: dict[str, list[float]] = {"role_main": [], "register_main": [], "interaction": []}

    can_contrast = _CANONICAL_2X2.issubset(cell_ctxs.keys()) and all(
        c in point_cells and not np.isnan(point_cells[c]) for c in _CANONICAL_2X2
    )

    if cluster_keys and n_boot >= 2:
        for _ in range(n_boot):
            idx = rng.integers(0, len(cluster_keys), size=len(cluster_keys))
            resample = [by_cluster[cluster_keys[i]] for i in idx]
            resample_records = [r for cluster in resample for r in cluster]
            # Re-bucket into cells.
            buckets: dict[str, list[dict]] = {c: [] for c in cell_records}
            for r in resample_records:
                cell = ctx_to_cell.get(r.get("context_type"))
                if cell is not None:
                    buckets[cell].append(r)
            means = {c: cell_mean(recs) for c, recs in buckets.items()}
            for c, m in means.items():
                if not np.isnan(m):
                    boot_cells[c].append(m)
            if can_contrast and all(not np.isnan(means[c]) for c in _CANONICAL_2X2):
                ts = means["tool_summary"]
                tr = means["tool_raw"]
                us = means["user_summary"]
                ur = means["user_raw"]
                boot_contrasts["role_main"].append(0.5 * ((ts + tr) - (us + ur)))
                boot_contrasts["register_main"].append(0.5 * ((tr + ur) - (ts + us)))
                boot_contrasts["interaction"].append((ts - us) - (tr - ur))

    def ci(samples: list[float]) -> tuple[float, float]:
        if len(samples) < 2:
            return (float("nan"), float("nan"))
        return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))

    cells_out = {c: {"point": point_cells[c], "ci": ci(boot_cells[c]), "n": n_cells[c]} for c in cell_records}

    if can_contrast:
        ts = point_cells["tool_summary"]
        tr = point_cells["tool_raw"]
        us = point_cells["user_summary"]
        ur = point_cells["user_raw"]
        contrasts_out = {
            "role_main": {"point": 0.5 * ((ts + tr) - (us + ur)), "ci": ci(boot_contrasts["role_main"])},
            "register_main": {"point": 0.5 * ((tr + ur) - (ts + us)), "ci": ci(boot_contrasts["register_main"])},
            "interaction": {"point": (ts - us) - (tr - ur), "ci": ci(boot_contrasts["interaction"])},
        }
    else:
        contrasts_out = {}

    return {"cells": cells_out, "contrasts": contrasts_out, "missing_cells": missing}
