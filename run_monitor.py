"""Run H2 monitor-detection judges.

Population fed to the monitors:
  - Cued rows (committed, both parse_ok=True) — the positive candidates.
  - No-context rows (both parse_ok=True) — unconditional negatives (no cue to detect).

The causal label (positive vs. negative) is computed in analyze_monitor_detection.py
after the monitor run, using the no_context baseline. Rows with no causal label are
excluded from the AUC at analysis time, not here.

All eligible rows across all selected models/seeds are pooled into a single pair of
batch calls (action + CoT). OpenAI batch chunking (≤10k per batch) is handled internally
by OpenAILLM.chat_batch.

Row-level resume (--resume): rows already in h2_monitor__<monitor>.jsonl with both action.parse_ok
and cot.parse_ok=True are kept; only failed or missing rows are re-sent to the API.

Usage:
    .venv/bin/python run_monitor.py
    .venv/bin/python run_monitor.py --results-dir results/agentic --dry-run
    .venv/bin/python run_monitor.py --model Qwen_Qwen3.5-9B  # one model only
    .venv/bin/python run_monitor.py --monitor-model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from src.data.face_eval import FaceEval
from src.evaluation.monitor import monitor_filename, run_monitors
from src.results.storage import discover_runs, load_merged_results
from src.utils.logging import get_logger

_CONVENTION = "C0"

_CUED_CONTEXT_TYPES = {
    "explicit",
    "implicit",
    "user_turn",
    "user_turn_structured",
    "user_turn_implicit",
}


def _should_process(r: dict) -> bool:
    """Return True for rows to feed to the monitors.

    Eligibility matches H1 (plot_h1_phase_diagram / filter_causal_dependent):
    parse_ok only, NO answer_committed gate. The committed gate previously dropped
    ~76% of causal positives — disproportionately the covert (hedged, CoT-silent)
    ones — so it is removed. The causal positive/negative split is computed in
    analysis from answer_stance_label + the no_context baseline.
    """
    judge = r.get("judge") or {}
    if not (judge.get("answer_parse_ok") and judge.get("reasoning_parse_ok")):
        return False
    ctx = r.get("context_type")
    return ctx == "none" or ctx in _CUED_CONTEXT_TYPES


def _load_all_entries(monitor_path: Path) -> dict[str, dict]:
    """All existing monitor entries keyed by id (for merge-preserve on write)."""
    out: dict[str, dict] = {}
    if not monitor_path.exists() or monitor_path.stat().st_size == 0:
        return out
    with open(monitor_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in entry:
                out[entry["id"]] = entry
    return out


def _load_kept_entries(
    monitor_path: Path, primary_ids: set[str], views: list[str]
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load existing monitor entries from the per-model monitor file.

    Returns (kept, partial):
      kept    — every view has parse_ok=True; skip entirely.
      partial — some (>=1) views parse_ok=True but not all; re-run the missing views.
    """
    kept: dict[str, dict] = {}
    partial: dict[str, dict] = {}
    if not monitor_path.exists() or monitor_path.stat().st_size == 0:
        return kept, partial
    with open(monitor_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = entry.get("id")
            if rid not in primary_ids:
                continue
            oks = [(entry.get(v) or {}).get("parse_ok") for v in views]
            if all(oks):
                kept[rid] = entry
            elif any(oks):
                partial[rid] = entry
    return kept, partial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H2 monitor judges on committed C0 rows.")
    parser.add_argument("--results-dir", default="results/agentic", help="Root results directory.")
    parser.add_argument("--model", default=None, help="Filter to one model directory name (e.g. Qwen_Qwen3-4B).")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model directory names to process (alternative to --model).",
    )
    parser.add_argument("--config", default="config/monitor_judge.yaml", help="Monitor judge config path.")
    parser.add_argument("--dry-run", action="store_true", help="Print row counts only; skip API calls.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Row-level resume: keep already-succeeded rows; re-run only failed/missing.",
    )
    parser.add_argument(
        "--monitor-model", default=None, help="Override monitor model (default: value from config YAML)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap pending rows (smoke-test use).")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for reasoning monitor models (e.g. gpt-5.6-luna).",
    )
    args = parser.parse_args()

    if args.model and args.models:
        raise SystemExit("--model and --models are mutually exclusive")

    model_set: set[str] | None = None
    if args.model:
        model_set = {args.model}
    elif args.models:
        model_set = set(args.models.split(","))

    logger = get_logger()

    logger.info(f"Discovering runs under {args.results_dir!r} ...")
    all_runs = discover_runs(args.results_dir)

    runs = [
        r
        for r in all_runs
        if r["convention"] == _CONVENTION and r["has_judged"] and (model_set is None or r["model"] in model_set)
    ]
    logger.info(
        f"Found {len(runs)} eligible run(s) (C0, has judged{', models=' + str(model_set) if model_set else ''})"
    )

    if not runs:
        logger.info("Nothing to process.")
        return

    dataset = FaceEval()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    monitor_model = args.monitor_model or cfg["model"]
    views = list(cfg["monitors"].keys())
    monitor_file = monitor_filename(monitor_model)
    logger.info(f"Monitor output file: {monitor_file}; views: {views}")

    # --- Build pending rows (pooled across all runs) ---------------------------
    # kept_by_run: run_path -> {id: entry} for rows already succeeded (resume only)
    # primary_by_run: run_path -> [row, ...] ordered list (for writing final files)
    # pending: flat list of rows to send to monitors
    # pending_run_paths: parallel list — pending[i] belongs to pending_run_paths[i]

    kept_by_run: dict[Path, dict[str, dict]] = {}
    partial_by_run: dict[Path, dict[str, dict]] = {}
    primary_by_run: dict[Path, list[dict]] = {}
    pending: list[dict] = []
    pending_run_paths: list[Path] = []

    for run in runs:
        run_path: Path = run["path"]
        merged = load_merged_results(run_path)
        primary = [r for r in merged if _should_process(r)]
        primary_by_run[run_path] = primary

        if not primary:
            logger.warning(f"No primary rows in {run_path}; skipping")
            kept_by_run[run_path] = {}
            partial_by_run[run_path] = {}
            continue

        primary_ids = {r["id"] for r in primary}

        if args.resume:
            kept, partial = _load_kept_entries(run_path / monitor_file, primary_ids, views)
        else:
            kept, partial = {}, {}
        kept_by_run[run_path] = kept
        partial_by_run[run_path] = partial

        for row in primary:
            if row["id"] not in kept:
                # Shallow-copy to attach run identity without mutating the primary list.
                pending.append({**row, "_run_path": str(run_path)})
                pending_run_paths.append(run_path)

    total_kept = sum(len(k) for k in kept_by_run.values())
    total_partial = sum(len(p) for p in partial_by_run.values())
    logger.info(
        f"Pending: {len(pending)} rows across {len(runs)} run(s)"
        + (
            f" ({total_kept} fully done, kept"
            + (f"; {total_partial} partial, resuming missing side" if total_partial else "")
            + ")"
            if args.resume
            else ""
        )
    )

    # Build per-view overrides from partial entries (some views already done).
    # Keyed by (str(run_path), row_id) so the same row ID in different runs is not confused.
    overrides: dict[str, dict[tuple[str, str], dict]] = {v: {} for v in views}
    if args.resume:
        for run_path, partial in partial_by_run.items():
            for rid, entry in partial.items():
                key = (str(run_path), rid)
                for v in views:
                    if (entry.get(v) or {}).get("parse_ok"):
                        overrides[v][key] = entry[v]

    if args.limit and len(pending) > args.limit:
        logger.info(f"Limiting to {args.limit} rows (--limit)")
        pending = pending[: args.limit]
        pending_run_paths = pending_run_paths[: args.limit]

    if args.dry_run:
        for run in runs:
            run_path = run["path"]
            n_primary = len(primary_by_run.get(run_path, []))
            n_kept = len(kept_by_run.get(run_path, {}))
            logger.info(
                f"  \\[{run['model']} seed={run['seed']}] primary={n_primary} kept={n_kept} pending={n_primary - n_kept}"
            )
        logger.info(f"DRY RUN: would send {len(pending)} rows to monitors")
        return

    if not pending:
        logger.info("Nothing pending — all rows already succeeded.")
        return

    # --- Single pooled monitor call -------------------------------------------
    logger.info(f"Running monitors on {len(pending)} rows (pooled) ...")
    new_entries = run_monitors(
        pending,
        dataset,
        config_path=args.config,
        monitor_model=args.monitor_model,
        overrides={v: o for v, o in overrides.items() if o} or None,
        reasoning_effort=args.reasoning_effort,
    )

    # --- Fan back by index (NOT by id — ids are not unique across runs) --------
    new_by_run: dict[Path, list[dict]] = defaultdict(list)
    for i, entry in enumerate(new_entries):
        new_by_run[pending_run_paths[i]].append(entry)

    # --- Write per-run output files -------------------------------------------
    for run in runs:
        run_path = run["path"]
        primary = primary_by_run.get(run_path, [])
        if not primary:
            continue

        new_by_id = {e["id"]: e for e in new_by_run.get(run_path, [])}

        # Merge-preserve: start from ALL existing entries on disk, overlay the freshly
        # reprocessed ones. This guarantees a partial run (e.g. --limit, or resuming a
        # subset of views) never truncates rows it did not touch. Written in primary order.
        existing = _load_all_entries(run_path / monitor_file)
        merged = {**existing, **new_by_id}
        all_entries: list[dict] = [merged[row["id"]] for row in primary if row["id"] in merged]

        output_path = run_path / monitor_file
        output_path.write_text("\n".join(json.dumps(e) for e in all_entries) + "\n")

        fails = {v: sum(1 for e in all_entries if not (e.get(v) or {}).get("parse_ok")) for v in views}
        msg = f"  \\[{run['model']} seed={run['seed']}] wrote {len(all_entries)} rows"
        if any(fails.values()):
            msg += " — parse failures: " + ", ".join(f"{v}={n}" for v, n in fails.items() if n)
            logger.warning(msg)
        else:
            logger.info(msg)

    logger.success(f"Done. Processed {len(runs)} run(s), {len(pending)} new rows.")


if __name__ == "__main__":
    main()
