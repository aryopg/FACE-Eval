#!/usr/bin/env python3
"""Check inference outputs for judge-readiness.

Usage:
    python check_inference.py              # check all runs
    python check_inference.py --expected 100  # custom expected count
    python check_inference.py --results results/agentic  # custom dir

Exits non-zero if any run fails, so it can gate judge execution:
    python check_inference.py || exit 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.results.check import MIN_LEGIBLE_RATE, ModelSummary, RunCheckResult, check_run, summarize_by_model
from src.results.storage import discover_runs, load_results

console = Console()

# 4 conventions (C0/C3/MC0/MC3) x 3 seeds. A model with fewer runs is missing inference.
MATRIX_RUNS = 12
# A model is flagged once missing content passes the complement of the legibility threshold.
SUBSTANTIAL_RATE = 1 - MIN_LEGIBLE_RATE


def _row_style(result: RunCheckResult) -> str:
    return "" if result.ok else "bold red"


def _rate_str(count: int, rate: float) -> str:
    if not count:
        return "[green]0[/green]"
    color = "red" if rate >= SUBSTANTIAL_RATE else "yellow"
    return f"[{color}]{count} ({rate:.1%})[/{color}]"


def _print_summary(summaries: list[ModelSummary]) -> None:
    """Per-model rollup, worst missing-content rate first."""
    # Rich falls back to 80 columns when output is piped (as it is on the judge pod),
    # which squeezes the headers into ellipses. Model dir names alone need more.
    out = Console(width=max(console.width, 130))
    table = Table(title=f"Per-model summary — flagged at ≥{SUBSTANTIAL_RATE:.0%} missing (rates over actual rows)")
    table.add_column("Model", no_wrap=True)
    table.add_column("Runs", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Short runs", justify="right")
    table.add_column("Missing reasoning", justify="right")
    table.add_column("Missing answer", justify="right")
    table.add_column("Worst run reasoning", justify="right")

    for s in summaries:
        runs_str = f"{s.runs}/{MATRIX_RUNS}"
        table.add_row(
            s.model,
            runs_str if s.runs >= MATRIX_RUNS else f"[yellow]{runs_str}[/yellow]",
            f"{s.rows}/{s.expected}" if s.rows != s.expected else str(s.rows),
            f"[yellow]{s.short_runs}[/yellow]" if s.short_runs else "[green]0[/green]",
            _rate_str(s.empty_reasoning, s.missing_reasoning_rate),
            _rate_str(s.empty_raw_answer, s.missing_answer_rate),
            f"{s.worst_reasoning_rate:.1%}",
        )
    out.print(table)
    out.print("[dim]Models with no inference on disk do not appear at all.[/dim]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check inference outputs before judging.")
    parser.add_argument("--results", default="results/agentic", help="Results directory")
    parser.add_argument("--expected", type=int, default=500, help="Expected records per run")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a per-model rollup instead of the per-run table (per-run detail still goes to _check_report.json)",
    )
    args = parser.parse_args()

    runs = discover_runs(args.results)
    if not runs:
        console.print(f"[yellow]No runs found in {args.results}[/yellow]")
        return 1

    inference_runs = [r for r in runs if r["has_inference"]]
    if not inference_runs:
        console.print(f"[yellow]No inference.jsonl files found under {args.results}[/yellow]")
        return 1

    table = Table(
        title=f"Inference check — expected {args.expected} records/run (legible threshold: {MIN_LEGIBLE_RATE:.0%})"
    )
    table.add_column("Model", no_wrap=True)
    table.add_column("Convention")
    table.add_column("Seed", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Legible", justify="right")
    table.add_column("Missing reasoning", justify="right")
    table.add_column("Empty raw_answer", justify="right")
    table.add_column("Missing fields", justify="right")
    table.add_column("Status")

    any_failed = False
    failing_runs: list[tuple[str, str, int, RunCheckResult]] = []
    checked_runs: list[tuple[str, RunCheckResult]] = []
    report_runs: list[dict] = []
    for run in inference_runs:
        convention = run.get("convention", "C0")
        try:
            records = load_results(Path(run["path"]), "inference")
        except Exception as e:
            console.print(f"[red]Failed to load {run['path']}/inference.jsonl: {e}[/red]")
            any_failed = True
            report_runs.append(
                {
                    "model": run["model"],
                    "convention": convention,
                    "seed": run["seed"],
                    "ok": False,
                    "load_error": str(e),
                }
            )
            continue

        result = check_run(records, expected_count=args.expected)
        checked_runs.append((run["model"], result))
        if not result.ok:
            any_failed = True
            failing_runs.append((run["model"], convention, run["seed"], result))
        report_runs.append(
            {
                "model": run["model"],
                "convention": convention,
                "seed": run["seed"],
                "ok": result.ok,
                "count": result.count,
                "expected": result.expected,
                "legible_count": result.legible_count,
                "legible_rate": result.legible_rate,
                "empty_reasoning": result.empty_reasoning,
                "reasoning_missing_rate": result.reasoning_missing_rate,
                "reasoning_missing_by_condition": result.reasoning_missing_by_condition,
                "empty_raw_answer": result.empty_raw_answer,
                "missing_fields": result.missing_fields,
            }
        )

        if args.summary:
            continue

        style = _row_style(result)
        count_str = f"[yellow]{result.count}[/yellow]" if result.count != result.expected else str(result.count)
        legible_color = "green" if result.legible_rate >= MIN_LEGIBLE_RATE else "red"
        legible_str = f"[{legible_color}]{result.legible_count} ({result.legible_rate:.1%})[/{legible_color}]"
        if result.empty_reasoning:
            reasoning_str = f"[yellow]{result.empty_reasoning} ({result.reasoning_missing_rate:.1%})[/yellow]"
        else:
            reasoning_str = "[green]0[/green]"
        table.add_row(
            run["model"],
            convention,
            str(run["seed"]),
            count_str,
            legible_str,
            reasoning_str,
            str(result.empty_raw_answer) if result.empty_raw_answer else "[green]0[/green]",
            str(result.missing_fields) if result.missing_fields else "[green]0[/green]",
            "[green]OK[/green]" if result.ok else "[red]FAIL[/red]",
            style=style,
        )

    if args.summary:
        _print_summary(summarize_by_model(checked_runs))
    else:
        console.print(table)

        for model, convention, seed, result in failing_runs:
            if result.reasoning_missing_by_condition:
                detail = Table(
                    title=f"Missing reasoning by condition — {model} [{convention}] seed {seed}", show_header=True
                )
                detail.add_column("Condition")
                detail.add_column("Missing", justify="right")
                for cond, cnt in sorted(result.reasoning_missing_by_condition.items(), key=lambda x: -x[1]):
                    detail.add_row(cond, str(cnt))
                console.print(detail)

    report_path = Path(args.results) / "_check_report.json"
    report_path.write_text(
        json.dumps(
            {
                "expected": args.expected,
                "legible_threshold": MIN_LEGIBLE_RATE,
                "runs": report_runs,
            },
            indent=2,
        )
    )
    console.print(f"[dim]Wrote report to {report_path}[/dim]")

    if any_failed:
        console.print("[bold red]Some runs failed the check. Fix before judging.[/bold red]")
        return 1

    console.print(f"[bold green]All {len(inference_runs)} run(s) ready to judge.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
