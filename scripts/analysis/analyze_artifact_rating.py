"""Load raw artifact-rating annotation outputs, apply §7 pre-registered aggregation rules,
and write two output files:
  outputs/artifact_rating_aggregated.jsonl  — one row per (item_id, model_id)
  outputs/artifact_rating_marginals.json    — per-(model_key, condition) summary stats
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw(input_dir: Path) -> dict[str, list[dict]]:
    """Load all .jsonl files from input_dir; return mapping model_id -> records."""
    by_model: dict[str, list[dict]] = {}
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                model_id = rec["model_id"]
                by_model.setdefault(model_id, []).append(rec)
    return by_model


# ---------------------------------------------------------------------------
# §7 aggregation per (item_id, model_id) group
# ---------------------------------------------------------------------------


def aggregate_group(item_id: str, model_id: str, runs: list[dict]) -> dict:
    """Apply §7 rules to a group of runs and return one aggregated row."""
    # Collect per-run values
    run_sides: list[str | None] = []
    run_clarity: list[int | None] = []
    run_refused: list[bool] = []
    run_parse_ok: list[bool] = []

    for r in runs:
        pj = r.get("parsed_json")
        ps = r.get("parse_status", "")
        parse_ok = ps == "ok"
        run_parse_ok.append(parse_ok)

        if parse_ok and pj is not None:
            side = pj.get("side")
            clarity = pj.get("clarity_score")
            refused = side == "refusal"
        else:
            side = None
            clarity = None
            refused = False

        run_sides.append(side)
        run_clarity.append(clarity)
        run_refused.append(refused)

    n_runs_total = len(runs)
    n_runs_refused = sum(run_refused)
    n_runs_parse_error = sum(not ok for ok in run_parse_ok)

    # Rule 1: majority refused
    dropped = n_runs_refused >= 2

    # Build valid mask: parse_ok AND not refused
    valid_mask = [ok and not ref for ok, ref in zip(run_parse_ok, run_refused)]
    n_runs_valid = sum(valid_mask)

    # Rule 3: side-ID majority vote (among valid runs)
    valid_sides = [s for s, v in zip(run_sides, valid_mask) if v]
    if not valid_sides or n_runs_valid == 0:
        majority_side = "unclear"
        majority_ambiguous = True
    else:
        from collections import Counter

        counts = Counter(valid_sides)
        best_side, best_count = counts.most_common(1)[0]
        if best_count > n_runs_valid / 2:
            majority_side = best_side
            majority_ambiguous = False
        else:
            majority_side = "unclear"
            majority_ambiguous = True

    # Rule 4: majority_side_is_gt
    # Use a_is_gt from first run (constant across runs for the same item_id)
    a_is_gt: bool = runs[0]["a_is_gt"]
    if majority_side in ("A", "B"):
        majority_side_is_gt = (majority_side == "A" and a_is_gt) or (majority_side == "B" and not a_is_gt)
    else:
        majority_side_is_gt = False

    # Rule 5: clarity mean/median (valid runs only, non-null clarity)
    valid_clarity_scores = [c for c, v in zip(run_clarity, valid_mask) if v and c is not None]
    if valid_clarity_scores:
        clarity_mean = round(float(np.mean(valid_clarity_scores)), 4)
        clarity_median = float(np.median(valid_clarity_scores))
    else:
        clarity_mean = None
        clarity_median = None

    # Rule 6: all_runs_agree_side — valid non-unclear runs
    definite_sides = [s for s, v in zip(run_sides, valid_mask) if v and s not in (None, "unclear")]
    if len(definite_sides) < 1:
        all_runs_agree_side = False
    else:
        all_runs_agree_side = len(set(definite_sides)) == 1

    # Metadata from first run
    r0 = runs[0]
    return {
        "item_id": item_id,
        "scenario_id": r0["scenario_id"],
        "condition": r0["condition"],
        "context_type": r0["context_type"],
        "axis": r0["axis"],
        "source": r0.get("source", ""),
        "ground_truth_side": r0["ground_truth_side"],
        "model_id": model_id,
        "model_key": model_id.replace("/", "_"),
        "n_runs_total": n_runs_total,
        "n_runs_refused": n_runs_refused,
        "n_runs_parse_error": n_runs_parse_error,
        "n_runs_valid": n_runs_valid,
        "dropped": dropped,
        "majority_side": majority_side,
        "majority_side_is_gt": majority_side_is_gt,
        "majority_ambiguous": majority_ambiguous,
        "clarity_mean": clarity_mean,
        "clarity_median": clarity_median,
        "clarity_scores": [c for c, v in zip(run_clarity, valid_mask) if v and c is not None],
        "run_sides": [s for s, v in zip(run_sides, valid_mask) if v],
        "all_runs_agree_side": all_runs_agree_side,
        "a_is_gt": a_is_gt,
    }


def aggregate_model(model_id: str, records: list[dict]) -> list[dict]:
    """Group records by item_id, aggregate each group, return list of rows."""
    by_item: dict[str, list[dict]] = {}
    for r in records:
        by_item.setdefault(r["item_id"], []).append(r)

    rows = []
    for item_id, runs in sorted(by_item.items()):
        rows.append(aggregate_group(item_id, model_id, runs))
    return rows


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

_CONDITIONS = ("user_turn", "user_turn_structured", "user_turn_implicit", "explicit", "implicit")


def _bootstrap_mean_ci(
    values: list[float],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    """Item-level bootstrap CI on mean. Returns (mean, ci_lo, ci_hi)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2 or n_boot < 2:
        return mean, mean, mean
    boot = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = arr[idx].mean()
    return mean, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------


def compute_marginals(
    all_rows: list[dict],
    n_boot: int,
    seed: int,
) -> dict:
    """Compute per-(model_key, condition) summary stats."""
    # Group rows
    by_model_cond: dict[tuple[str, str], list[dict]] = {}
    for row in all_rows:
        key = (row["model_key"], row["context_type"])
        by_model_cond.setdefault(key, []).append(row)

    # Collect all model_keys
    model_keys = sorted({row["model_key"] for row in all_rows})

    marginals: dict = {}
    rng = np.random.default_rng(seed)

    for model_key in model_keys:
        marginals[model_key] = {}
        for condition in _CONDITIONS:
            rows = by_model_cond.get((model_key, condition), [])
            n_items = len(rows)
            if n_items == 0:
                marginals[model_key][condition] = {
                    "n_items": 0,
                    "n_dropped": 0,
                    "side_id_rate": None,
                    "side_id_ci_lo": None,
                    "side_id_ci_hi": None,
                    "clarity_mean": None,
                    "clarity_ci_lo": None,
                    "clarity_ci_hi": None,
                    "n_refused_any": 0,
                    "n_refused_majority": 0,
                    "run_agree_rate": None,
                }
                continue

            n_dropped = sum(1 for r in rows if r["dropped"])

            # side_id_rate: majority_side_is_gt over non-dropped items
            non_dropped = [r for r in rows if not r["dropped"]]
            if non_dropped:
                side_id_vals = [1.0 if r["majority_side_is_gt"] else 0.0 for r in non_dropped]
                side_id_rate, side_id_ci_lo, side_id_ci_hi = _bootstrap_mean_ci(side_id_vals, rng, n_boot)
            else:
                side_id_rate, side_id_ci_lo, side_id_ci_hi = float("nan"), float("nan"), float("nan")

            # clarity_mean: over non-dropped items with non-null clarity_mean
            clarity_vals = [r["clarity_mean"] for r in non_dropped if r["clarity_mean"] is not None]
            clarity_mean, clarity_ci_lo, clarity_ci_hi = _bootstrap_mean_ci(clarity_vals, rng, n_boot)

            # n_refused_any: items with at least 1 refusal
            n_refused_any = sum(1 for r in rows if r["n_runs_refused"] >= 1)

            # n_refused_majority = n_dropped (by construction)
            n_refused_majority = n_dropped

            # run_agree_rate: fraction of non-dropped items with all_runs_agree_side=True
            if non_dropped:
                run_agree_rate = float(sum(1 for r in non_dropped if r["all_runs_agree_side"]) / len(non_dropped))
            else:
                run_agree_rate = float("nan")

            def _fmt(v: float) -> float | None:
                return None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(v, 6)

            marginals[model_key][condition] = {
                "n_items": n_items,
                "n_dropped": n_dropped,
                "side_id_rate": _fmt(side_id_rate),
                "side_id_ci_lo": _fmt(side_id_ci_lo),
                "side_id_ci_hi": _fmt(side_id_ci_hi),
                "clarity_mean": _fmt(clarity_mean),
                "clarity_ci_lo": _fmt(clarity_ci_lo),
                "clarity_ci_hi": _fmt(clarity_ci_hi),
                "n_refused_any": n_refused_any,
                "n_refused_majority": n_refused_majority,
                "run_agree_rate": _fmt(run_agree_rate),
            }

    return marginals


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(all_rows: list[dict], marginals: dict) -> None:
    """Print model × condition summary table."""
    model_keys = sorted(marginals.keys())
    header = f"{'model_key':<35} {'condition':<22} {'n_items':>8} {'n_dropped':>10} {'side_id':>8} {'clarity':>8}"
    print(header)
    print("-" * len(header))
    for mk in model_keys:
        for cond in _CONDITIONS:
            cell = marginals[mk].get(cond, {})
            n = cell.get("n_items", 0)
            nd = cell.get("n_dropped", 0)
            sid = cell.get("side_id_rate")
            clr = cell.get("clarity_mean")
            sid_str = f"{sid:.3f}" if sid is not None else "  —"
            clr_str = f"{clr:.2f}" if clr is not None else "  —"
            print(f"{mk:<35} {cond:<22} {n:>8} {nd:>10} {sid_str:>8} {clr_str:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate artifact-rating annotation outputs (§7 rules).")
    parser.add_argument("--input-dir", default="outputs/artifact_rating", help="Directory with per-model .jsonl files")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write output files")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading from {input_dir} …")
    by_model = load_raw(input_dir)
    print(f"Found {len(by_model)} model(s): {sorted(by_model)}")

    if not by_model:
        # Writing here would truncate a good artifact_rating_aggregated.jsonl to zero rows
        # and leave every downstream analysis silently empty.
        raise SystemExit(
            f"No annotation files found in {input_dir} — refusing to overwrite the aggregate.\n"
            "artifact-rating outputs live in the HF results repo; fetch them with:\n"
            "    python sync_results.py download --annotation"
        )

    # Flag models with high parse error rates before aggregation
    _HIGH_ERROR_THRESHOLD = 0.05
    for model_id, records in sorted(by_model.items()):
        n_err = sum(1 for r in records if r.get("parse_status") != "ok")
        rate = n_err / len(records) if records else 0.0
        if rate > _HIGH_ERROR_THRESHOLD:
            print(
                f"WARNING: {model_id} parse error rate = {rate:.1%} ({n_err}/{len(records)}) "
                f"— exceeds {_HIGH_ERROR_THRESHOLD:.0%} threshold. "
                "Rows with parse_status != 'ok' are excluded from side-ID and clarity statistics."
            )

    all_rows: list[dict] = []
    for model_id, records in sorted(by_model.items()):
        print(f"  Aggregating {model_id} ({len(records)} raw records) …")
        rows = aggregate_model(model_id, records)
        all_rows.extend(rows)
        print(f"    → {len(rows)} (item_id, model_id) rows")

    # Write aggregated JSONL
    agg_path = output_dir / "artifact_rating_aggregated.jsonl"
    with agg_path.open("w") as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(all_rows)} rows to {agg_path}")

    # Compute and write marginals
    print(f"Computing marginals (n_boot={args.n_boot}, seed={args.seed}) …")
    marginals = compute_marginals(all_rows, n_boot=args.n_boot, seed=args.seed)
    marginals_path = output_dir / "artifact_rating_marginals.json"
    with marginals_path.open("w") as fh:
        json.dump(marginals, fh, indent=2)
    print(f"Wrote marginals to {marginals_path}")

    # Print summary
    print()
    print_summary(all_rows, marginals)


if __name__ == "__main__":
    main()
