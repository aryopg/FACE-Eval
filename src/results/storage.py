"""Structured result storage and loading for faithfulness evaluation runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# The pre-registered judge for every result on disk. Its verdicts stay in the
# unsuffixed judged.jsonl; any other judge is keyed by model so a second judge
# run can never overwrite the first.
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_JUDGED_FILE = "judged.jsonl"


def judged_filename(judge_model: str) -> str:
    """Per-run judge output filename for a given judge model."""
    if judge_model == DEFAULT_JUDGE_MODEL:
        return DEFAULT_JUDGED_FILE
    return f"judged__{judge_model.replace('/', '_')}.jsonl"


def get_run_dir(
    output_dir: str,
    model: str,
    seed: int,
    reasoning_effort: str | None = None,
    convention: str = "C0",
) -> Path:
    """Get (and create) the run directory for a (model, seed, convention) triple.

    C0 keeps the legacy `seed_{N}/` path. C1-C3 append `_{convention}`.
    """
    model_name = model.replace("/", "_")
    if reasoning_effort is not None:
        model_name = f"{model_name}_{reasoning_effort}"
    seed_name = f"seed_{seed}" if convention == "C0" else f"seed_{seed}_{convention}"
    run_dir = Path(output_dir) / model_name / seed_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_results(
    run_dir: Path,
    results: list[dict[str, Any]],
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save results to a JSONL file and update metadata.json."""
    jsonl_path = run_dir / f"{stage}.jsonl"
    with open(jsonl_path, "w") as f:
        for result in results:
            f.write(json.dumps(result, default=str) + "\n")

    meta_path = run_dir / "metadata.json"
    existing_meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Use stage-specific key so inference and judge counts don't overwrite each other.
    existing_meta[f"{stage}_completed_at"] = datetime.now().isoformat()
    existing_meta[f"{stage}_total_samples"] = len(results)
    if metadata:
        existing_meta.update(metadata)

    meta_path.write_text(json.dumps(existing_meta, indent=2))


def load_results(run_dir: Path, stage: str) -> list[dict[str, Any]]:
    """Load results from a JSONL file."""
    jsonl_path = run_dir / f"{stage}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Results file not found: {jsonl_path}")

    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def discover_runs(
    output_dir: str = "results/agentic",
    judged_file: str = DEFAULT_JUDGED_FILE,
) -> list[dict[str, Any]]:
    """Discover all runs under output_dir.

    Expects layout: {output_dir}/{model}/seed_{N}[_C{1,2,3}]/
    Default matches the path written by run.py (results/agentic).
    `judged_file` selects which judge's verdicts `has_judged` reports on.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    runs = []
    for model_dir in sorted(output_path.iterdir()):
        if not model_dir.is_dir():
            continue
        for seed_dir in sorted(model_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            name = seed_dir.name
            if not name.startswith("seed_"):
                continue
            parts = name.removeprefix("seed_").split("_", 1)
            try:
                seed = int(parts[0])
            except ValueError:
                continue
            convention = parts[1] if len(parts) > 1 else "C0"
            meta_path = seed_dir / "metadata.json"
            # Guard against partially-written or corrupted metadata files.
            meta: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except json.JSONDecodeError:
                    meta = {"_metadata_parse_error": True}

            runs.append(
                {
                    "model": model_dir.name,
                    "seed": seed,
                    "convention": convention,
                    "path": seed_dir,
                    "has_inference": (seed_dir / "inference.jsonl").exists(),
                    "has_judged": (seed_dir / judged_file).exists(),
                    "metadata": meta,
                }
            )

    runs.sort(key=lambda r: (r["model"], r["seed"], r["convention"]))
    return runs


def load_merged_results(run_dir: Path, judged_file: str = DEFAULT_JUDGED_FILE) -> list[dict[str, Any]]:
    """Load inference results merged with judge annotations.

    Joins inference.jsonl and `judged_file` on "id" field.
    If the judge file doesn't exist, returns inference results without judge fields.
    """
    results = load_results(run_dir, "inference")
    judged_path = run_dir / judged_file
    if not judged_path.exists():
        return results

    judge_by_id: dict[str, Any] = {}
    with open(judged_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                judge_by_id[entry["id"]] = entry.get("judge")

    for r in results:
        r["judge"] = judge_by_id.get(r["id"])

    return results
