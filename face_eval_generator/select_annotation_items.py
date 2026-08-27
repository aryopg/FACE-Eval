"""Select the 100-item human-annotation subset (50 matched explicit/implicit pairs) for human annotation."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.data.face_eval import FaceEval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select human-annotation items for human annotation.")
    parser.add_argument("--dataset-path", default=None, help="Local HF dataset path.")
    parser.add_argument("--dataset-name", default="edinburgh-dawg/face-eval")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    ds = FaceEval(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        use_auth_token=os.getenv("HF_TOKEN"),
    )

    # Build lookup tables: key = (scenario_id, source, side)
    explicit_index: dict[tuple[str, str, str], str] = {}  # key → id
    # NOTE: stratifying by (source, axis) rather than source alone (as §7 pre-registers).
    # §7 specifies explicitness × source = 10 strata (~10 items each). This uses
    # source × axis = up to 25 strata (~2 pairs each), giving stronger per-axis balance.
    # §7 permits "opportunistic" axis balancing within strata — this is a stricter form.
    # Document as a minor protocol deviation in the paper appendix.
    implicit_by_stratum: dict[tuple[str, str], list[tuple[str, str, str, str]]] = defaultdict(list)
    # implicit stratum key = (source, axis); value = list of (id, scenario_id, source, side)

    for row in ds.dataset:
        context_type = row.get("context_type", "")
        if context_type not in ("explicit", "implicit"):
            continue

        condition: str = row["condition"]
        side: str = condition.split("_")[-1]
        scenario_id: str = row["scenario_id"]
        source: str = row["source"]
        axis: str = row["axis"]
        row_id: str = row["id"]

        key = (scenario_id, source, side)
        if context_type == "explicit":
            explicit_index[key] = row_id
        else:
            implicit_by_stratum[(source, axis)].append((row_id, scenario_id, source, side))

    rng = random.Random(args.seed)

    strata = sorted(implicit_by_stratum.keys())
    n_strata = len(strata)
    target_total = 50
    base_per_stratum = target_total // n_strata  # items per stratum (floor)
    remainder = target_total % n_strata  # distribute extras to first N strata

    # Strata may have fewer items than quota (missing source × axis combinations in data).
    # Per-stratum counts are printed below — review before sending to Prolific.
    pairs: list[dict] = []
    for i, stratum_key in enumerate(strata):
        items = list(implicit_by_stratum[stratum_key])
        rng.shuffle(items)

        quota = base_per_stratum + (1 if i < remainder else 0)
        selected: list[dict] = []
        for row_id, scenario_id, source, side in items:
            if len(selected) >= quota:
                break
            explicit_key = (scenario_id, source, side)
            if explicit_key not in explicit_index:
                continue  # no matching explicit — skip
            selected.append(
                {
                    "explicit_id": explicit_index[explicit_key],
                    "implicit_id": row_id,
                    "axis": stratum_key[1],
                    "source": source,
                    "scenario_id": scenario_id,
                    "side": side,
                }
            )
        pairs.extend(selected)
        source_label, axis_label = stratum_key
        print(f"  stratum ({source_label}, {axis_label}): quota={quota}, selected={len(selected)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "arm_a_items.json"
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nTotal pairs written: {len(pairs)} → {output_path}")


if __name__ == "__main__":
    main()
