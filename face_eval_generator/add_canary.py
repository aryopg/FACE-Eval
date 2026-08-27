"""Add a contamination-detection canary field to the private HF dataset.

Usage:
    python -m face_eval_generator.add_canary --uuid <your-private-uuid4>

The UUID is not generated here — pass it explicitly so re-runs with the same
UUID are idempotent. Store the UUID privately (password manager or HF org secret).

Canary field value per row:
    "7b7b0000-0000-0000-0000-000000000000 CANARY-AGENTIC-SYCOPHANCY-<uuid>"

The BIG-bench GUID is the de-facto standard for automated decontamination
pipelines. The project-specific UUID is the probe string for manual checks:
    prompt: "Complete this string: CANARY-AGENTIC-SYCOPHANCY-"
    if the model returns the UUID verbatim, it has seen this dataset.
"""

from __future__ import annotations

import argparse
import os

import datasets

DATASET_NAME = "edinburgh-dawg/face-eval"
BIGBENCH_GUID = "7b7b0000-0000-0000-0000-000000000000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Add canary field to the private HF dataset.")
    parser.add_argument("--uuid", required=True, help="Project-specific UUID4 for the canary string.")
    parser.add_argument("--split", default="train", help="Dataset split (default: train).")
    parser.add_argument("--dry-run", action="store_true", help="Print canary value and row count; do not push.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        raise EnvironmentError("HF_TOKEN environment variable is required to push to HuggingFace.")

    canary_value = f"{BIGBENCH_GUID} CANARY-AGENTIC-SYCOPHANCY-{args.uuid}"
    print(f"Canary value: {canary_value!r}")

    print(f"Loading {DATASET_NAME} ({args.split})...")
    ds = datasets.load_dataset(DATASET_NAME, split=args.split, token=token)
    print(f"Loaded {len(ds)} rows.")

    if "canary" in ds.column_names:
        already_set = sum(1 for v in ds["canary"] if v)
        print(
            f"canary column already exists ({already_set}/{len(ds)} rows have a value). Skipping rows with existing values."
        )

        def add_canary(row: dict) -> dict:
            if not row.get("canary"):
                row["canary"] = canary_value
            return row

        ds = ds.map(add_canary)
    else:
        ds = ds.map(lambda row: {**row, "canary": canary_value})

    n_set = sum(1 for v in ds["canary"] if v)
    print(f"Rows with canary set: {n_set}/{len(ds)}")

    if args.dry_run:
        print("Dry run — not pushing to HuggingFace.")
        return

    print(f"Pushing to {DATASET_NAME}...")
    ds.push_to_hub(DATASET_NAME, split=args.split, token=token)
    print("Done.")


if __name__ == "__main__":
    main()
