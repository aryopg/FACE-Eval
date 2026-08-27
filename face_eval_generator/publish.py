"""Push the assembled agentic-sycophancy dataset to HuggingFace.

Reads `data/finalized_combined.jsonl` (run
`python -m face_eval_generator.export_full_combined` first if it is not up
to date), builds a HuggingFace `Dataset`, and pushes to the configured repo
along with a fresh dataset card.

Defaults:
  --repo-id  edinburgh-dawg/face-eval
  --private  True (pass --public to override)

Requires `HF_TOKEN` in the environment (or `huggingface-cli login`).

Examples:
  # Dry-run: build the dataset locally without pushing
  python -m face_eval_generator.publish --dry-run

  # Actual push (private)
  python -m face_eval_generator.publish

  # Push as public dataset
  python -m face_eval_generator.publish --public

  # Push to a different repo
  python -m face_eval_generator.publish --repo-id myorg/agentic-sycophancy-v2
"""

from __future__ import annotations

import argparse
from collections import Counter

from face_eval_generator.generate import (
    DEFAULT_DATA_DIR,
    _generate_dataset_readme,
    load_jsonl,
    push_to_huggingface,
)

COMBINED_PATH = DEFAULT_DATA_DIR / "finalized_combined.jsonl"

DEFAULT_REPO = "edinburgh-dawg/face-eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--public", action="store_true", help="Push as public (default: private)")
    parser.add_argument("--dry-run", action="store_true", help="Build dataset locally without pushing")
    parser.add_argument(
        "--write-readme-only",
        action="store_true",
        help="Print the README that would be uploaded; do not build or push",
    )
    args = parser.parse_args()

    if not COMBINED_PATH.exists():
        raise SystemExit(f"missing {COMBINED_PATH}; run python -m face_eval_generator.export_full_combined first")

    rows = load_jsonl(COMBINED_PATH)
    print(f"loaded {len(rows)} rows from {COMBINED_PATH}")
    breakdown = Counter((r.get("context_type"), r.get("source") or "—") for r in rows)
    print("breakdown:")
    for (ct, src), n in sorted(breakdown.items()):
        print(f"  {ct:25s} {src:20s} {n}")

    if args.write_readme_only:
        print()
        print("=" * 60)
        print(_generate_dataset_readme(rows, args.repo_id))
        return

    if args.dry_run:
        from datasets import Dataset

        ds = Dataset.from_list(rows)
        print()
        print(f"would push to: {args.repo_id} (private={not args.public})")
        print(f"dataset rows: {len(ds)}")
        print(f"columns: {ds.column_names}")
        print()
        print("schema:")
        for k, v in ds.features.items():
            print(f"  {k}: {v}")
        print()
        print("(dry-run — pass without --dry-run to actually push)")
        return

    print()
    print(f"pushing to {args.repo_id} (private={not args.public})...")
    data_dir = COMBINED_PATH.parent
    url = push_to_huggingface(data_dir, args.repo_id, private=not args.public)
    print(f"done: {url}")


if __name__ == "__main__":
    main()
