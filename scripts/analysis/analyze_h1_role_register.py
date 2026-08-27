"""2x2 role x register analysis on L1/L3 metrics.

For each (metric in {L1, L3}, source, model) and a model-pooled aggregate,
estimates:
  - per-cell rate (point + scenario-cluster 95% bootstrap CI)
  - role main effect:     (tool - user)   averaged across registers
  - register main effect: (raw - summary) averaged across roles
  - role x register interaction: (tool_summary - user_summary) - (tool_raw - user_raw)

All rates are no_context-normalized (filter_causal_dependent: only rows whose
answer stance shifted off the no_context baseline).

Writes analysis/h1_role_register.json and prints a compact summary table.

This is the scaffold: it runs against whatever cells are present today, and
will start reporting interaction terms once user_turn_implicit_* exists.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.results.db import ResultsDB
from src.results.role_register import (
    CELLS_2X2,
    CONVENTION_CHOICES,
    METRICS,
    bootstrap_cells_and_contrasts,
    load_clean,
)
from src.utils.plotting import sort_models

ANALYSIS_DIR = Path("analysis")


def _format(p: float | None, ci: tuple[float, float] | None) -> str:
    if p is None or (isinstance(p, float) and (p != p)):
        return "  n/a "
    if ci is None or any((x != x) for x in ci):
        return f"{p:+.3f}"
    return f"{p:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _cells_by_role_register(db: ResultsDB) -> dict[str, str]:
    return {f"{role}_{reg}": ctx for (role, reg), ctx in CELLS_2X2.items()}


def _present_sources(db: ResultsDB) -> list[str]:
    seen = {r.get("source") for r in db.records}
    seen = {s for s in seen if s}
    return sorted(seen) if seen else ["profile"]


def _analyze_block(db: ResultsDB, metric_field: str, n_boot: int, seed: int) -> dict:
    return bootstrap_cells_and_contrasts(
        db=db,
        cell_to_context=_cells_by_role_register(db),
        field=metric_field,
        n_boot=n_boot,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convention", choices=CONVENTION_CHOICES, default="C0")
    parser.add_argument("--results-dir", default="results/agentic")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path; defaults to analysis/h1_role_register_{convention}.json",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = str(ANALYSIS_DIR / f"h1_role_register_{args.convention}.json")

    db = load_clean(convention=args.convention, results_dir=args.results_dir)
    if db.count() == 0:
        raise SystemExit("No records loaded — check --results-dir and --convention.")

    models = sort_models({r["_model"] for r in db.records})
    sources = _present_sources(db)

    print(f"Loaded {db.count()} records.")
    print(f"  convention: {args.convention}")
    print(f"  models: {len(models)} ({', '.join(models)})")
    print(f"  sources: {sources}")
    ctx_counts = Counter(r.get("context_type") for r in db.records)
    print(f"  context_types: {dict(ctx_counts)}")
    print()

    results: dict = {
        "config": {"convention": args.convention, "n_boot": args.n_boot, "seed": args.seed},
        "blocks": [],
    }

    header = f"{'metric':<3} {'source':<16} {'model':<28} {'role_main':<28} {'register_main':<28} {'interaction':<28}"
    print(header)
    print("-" * len(header))

    for metric_name, field in METRICS.items():
        for source in sources:
            src_db = db.filter(source=source) if any(r.get("source") for r in db.records) else db

            # Pooled across models.
            pooled = _analyze_block(src_db, field, args.n_boot, args.seed)
            results["blocks"].append({"metric": metric_name, "source": source, "model": "_pooled", **pooled})
            ct = pooled["contrasts"]
            print(
                f"{metric_name:<3} {source:<16} {'_pooled':<28} "
                f"{_format(ct.get('role_main', {}).get('point'), ct.get('role_main', {}).get('ci')):<28} "
                f"{_format(ct.get('register_main', {}).get('point'), ct.get('register_main', {}).get('ci')):<28} "
                f"{_format(ct.get('interaction', {}).get('point'), ct.get('interaction', {}).get('ci')):<28}"
            )

            # Per-model.
            for m in models:
                mdb = src_db.filter(_model=m)
                if mdb.count() == 0:
                    continue
                block = _analyze_block(mdb, field, args.n_boot, args.seed)
                results["blocks"].append({"metric": metric_name, "source": source, "model": m, **block})
                ct = block["contrasts"]
                print(
                    f"{metric_name:<3} {source:<16} {m:<28} "
                    f"{_format(ct.get('role_main', {}).get('point'), ct.get('role_main', {}).get('ci')):<28} "
                    f"{_format(ct.get('register_main', {}).get('point'), ct.get('register_main', {}).get('ci')):<28} "
                    f"{_format(ct.get('interaction', {}).get('point'), ct.get('interaction', {}).get('ci')):<28}"
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Flat CSV companion: one row per (metric, source, model, cell|contrast).
    from src.utils.plotting import save_table

    csv_rows: list[dict] = []
    for block in results["blocks"]:
        for cell_name, cell_data in block.get("cells", {}).items():
            csv_rows.append(
                {
                    "convention": args.convention,
                    "metric": block["metric"],
                    "source": block["source"],
                    "model": block["model"],
                    "cell": cell_name,
                    "point": cell_data["point"],
                    "ci_lo": cell_data["ci"][0],
                    "ci_hi": cell_data["ci"][1],
                    "n": cell_data["n"],
                }
            )
        for con_name, con_data in block.get("contrasts", {}).items():
            csv_rows.append(
                {
                    "convention": args.convention,
                    "metric": block["metric"],
                    "source": block["source"],
                    "model": block["model"],
                    "cell": f"contrast:{con_name}",
                    "point": con_data["point"],
                    "ci_lo": con_data["ci"][0],
                    "ci_hi": con_data["ci"][1],
                    "n": "",
                }
            )
    save_table(out_path.with_suffix(".csv"), csv_rows)
    print(f"Saved {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
