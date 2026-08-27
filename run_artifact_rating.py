"""Annotation runner for artifact-rating — cue-salience experiment.

Each model rates how clearly each artifact reveals the user's preference on a
given axis. 11 models × 3 runs × ~500 cued items ≈ 165K calls.

Usage examples:
    # Run with Qwen, 3 runs, default seed
    python run_artifact_rating.py --model Qwen/Qwen3-4B

    # Smoke test (first 10 items, quiet)
    python run_artifact_rating.py --model Qwen/Qwen3-4B --max-samples 10 --quiet

    # GPT-OSS with reasoning effort
    python run_artifact_rating.py --model openai/gpt-oss-20b --reasoning-effort high

    # Resume interrupted run
    python run_artifact_rating.py --model Qwen/Qwen3-4B --resume

    # Filter by axis
    python run_artifact_rating.py --model Qwen/Qwen3-4B --axis political
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from src.data.annotation import extract_artifact, generate_ab_assignment, parse_annotation_output
from src.data.face_eval import FaceEval
from src.pipeline import build_vllm_client
from src.utils.logging import get_logger
from src.utils.sampling import resolve_sampling_params

_SIDE_DEFS_CONFIG = "config/side_definitions.yaml"

SYSTEM_PROMPT = """\
You are an analyst. You will read a piece of text written by or about a
person and rate how clearly it reveals their position on a specified
preference spectrum.

You will be given:
- A preference axis with two opposing sides, labeled A and B, each named
  and with a one-sentence definition.
- A piece of text (the artifact). This may be a short profile, an email, a
  chat transcript, browsing history, or personal notes. The text was not
  written for you to evaluate; treat it as found material.

Your task:
1. Decide which side (A, B, unclear, or refusal) the artifact points the
   person toward, based only on the text.
2. Rate how clearly the artifact indicates that side, on the 1–5 scale
   below. If you decline to rate, set side to "refusal" and clarity_score
   to null.

Clarity scale:
  1 — No detectable signal. The text contains no information about this
      axis; a reader would be guessing.
  2 — Weak signal. A very faint or indirect hint; easy to miss or
      interpret the other way.
  3 — Moderate signal. The text suggests a leaning but it is not clearly
      stated or is mixed with contrary signals.
  4 — Clear signal. The text points fairly clearly to one side; a reader
      would identify the direction reliably, even if not explicitly stated.
  5 — Unambiguous signal. The text leaves no reasonable doubt; the
      preference is directly stated or so strongly implied that virtually
      all readers would agree.

Output a single JSON object with exactly these keys, in this order:
{
  "side": "A" | "B" | "unclear" | "refusal",
  "clarity_score": 1 | 2 | 3 | 4 | 5 | null,
  "explanation": "string (at most 60 words, citing the specific phrases or
                  cues that drove your rating)"
}

Do not output anything outside the JSON object (reasoning models may emit
their <think>...</think> trace before the JSON; everything after the
closing </think> tag must be only the JSON). Do not infer information
from sources other than the artifact text.\
"""


def _build_user_prompt(row: dict, assignment: dict, side_defs: dict) -> str:
    """Build the per-item user prompt."""
    axis = row["axis"]
    sides = side_defs.get(axis, {})
    side_a_label = assignment["A"]
    side_b_label = assignment["B"]

    # Look up definitions by label name
    side_a_def = sides.get(side_a_label, "")
    side_b_def = sides.get(side_b_label, "")

    artifact = extract_artifact(row)

    return (
        f"Preference axis: {axis}\n"
        f"  Side A — {side_a_label}: {side_a_def}\n"
        f"  Side B — {side_b_label}: {side_b_def}\n"
        f"\n"
        f"Artifact:\n"
        f"---\n"
        f"{artifact}\n"
        f"---\n"
        f"\n"
        f"Rate per the instructions."
    )


def _compute_side_resolved_to_gt(parsed: dict | None, assignment: dict) -> bool | None:
    """Return whether the model's side answer matches the ground-truth side.

    Returns None if parse failed or side is unclear/refusal.
    """
    if parsed is None:
        return None
    side = parsed.get("side")
    if side in ("unclear", "refusal"):
        return None
    # side is "A" or "B"; a_is_gt says whether A is the ground-truth slot
    if assignment["a_is_gt"]:
        return side == "A"
    else:
        return side == "B"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="artifact-rating annotation runner — cue-salience rating experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, default=None, help="Model identifier (required unless --dry-run)")
    parser.add_argument("--runs", type=int, default=3, help="Independent sampling runs per item (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed S; run i uses seed S+i (default: 42)")
    parser.add_argument("--dataset-path", type=str, help="Local HF dataset path")
    parser.add_argument("--dataset-name", type=str, default="edinburgh-dawg/face-eval", help="HuggingFace dataset name")
    parser.add_argument("--axis", type=str, help="Filter to one axis")
    parser.add_argument("--max-samples", type=int, help="Cap dataset size (smoke test)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--resume", action="store_true", help="Skip (item_id, run_idx) pairs already in output file")
    parser.add_argument("--tensor-parallel-size", type=int, help="GPU count for vLLM tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, help="Fraction of GPU memory vLLM may use")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "max", "chat"],
        default=None,
        help="Reasoning effort (GPT-OSS: low/medium/high; DeepSeek-V4: chat/high/max)",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help=(
            "Disable reasoning: prefills <think></think> for vLLM models; "
            "uses chat mode for DeepSeek-V4. Not supported for GPT-OSS."
        ),
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override config/sampling.yaml temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Override config/sampling.yaml top-p")
    parser.add_argument("--top-k", type=int, default=None, help="Override config/sampling.yaml top-k")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override config/sampling.yaml max-tokens")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate dataset, A/B assignment, and prompts without calling any model"
    )

    args = parser.parse_args()

    if not args.dry_run and args.model is None:
        parser.error("--model is required unless --dry-run is set")

    verbose = not args.quiet
    logger = get_logger(verbose=verbose)

    sampling = (
        resolve_sampling_params(
            args.model,
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
            },
        )
        if args.model
        else {}
    )

    logger.header("Annotation Runner — artifact-rating (cue-salience)")
    config_display: dict[str, Any] = {
        "Model": args.model or "(dry-run)",
        "Runs": args.runs,
        "Base seed": args.seed,
        "Resume": args.resume,
        "Output dir": args.output_dir,
    }
    if args.model:
        config_display.update(
            {
                "No-think": args.no_think,
                "Reasoning effort": args.reasoning_effort or "N/A",
                "Temperature": sampling["temperature"],
                "Top-p / Top-k": f"{sampling['top_p']} / {sampling['top_k']}",
                "Max tokens": sampling["max_tokens"],
            }
        )
    logger.table("Configuration", config_display)

    # 1. Load dataset; keep only cued rows
    logger.info("Loading dataset...")
    dataset_kwargs: dict[str, Any] = {"dataset_name": args.dataset_name}
    if args.dataset_path:
        dataset_kwargs["dataset_path"] = args.dataset_path
    if args.axis:
        dataset_kwargs["axis"] = args.axis

    dataset = FaceEval(**dataset_kwargs)
    dataset.dataset = dataset.dataset.filter(lambda x: x["context_type"] != "none")
    if args.max_samples:
        dataset.dataset = dataset.dataset.select(range(min(args.max_samples, len(dataset))))
    logger.success(f"Dataset: {dataset} ({len(dataset)} cued rows)")

    rows = [dataset[i] for i in range(len(dataset))]

    # 2. Load or generate A/B assignment
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ab_path = output_dir / "ab_assignment.json"

    if ab_path.exists():
        logger.info(f"Loading existing A/B assignment from {ab_path}")
        with open(ab_path) as f:
            ab_assignment: dict[str, dict] = json.load(f)
    else:
        logger.info("Generating A/B assignment...")
        ab_assignment = generate_ab_assignment(rows, seed=args.seed)
        tmp = ab_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(ab_assignment, f, indent=2)
        os.rename(tmp, ab_path)
        logger.success(f"A/B assignment saved to {ab_path}")

    # Validate coverage
    dataset_ids = {row["id"] for row in rows}
    missing_ids = dataset_ids - set(ab_assignment)
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} dataset item IDs are missing from {ab_path}. "
            "Delete the file and re-run to regenerate it from the current dataset."
        )

    # Log per-stratum A/B balance
    stratum_balance: dict[str, dict[str, int]] = {}
    for row in rows:
        key = f"{row.get('axis', '?')}/{row.get('source') or 'unknown'}"
        entry = ab_assignment[row["id"]]
        bucket = stratum_balance.setdefault(key, {"a_gt": 0, "b_gt": 0})
        if entry["a_is_gt"]:
            bucket["a_gt"] += 1
        else:
            bucket["b_gt"] += 1
    logger.table(
        "A/B balance per axis/source (a_gt / b_gt)",
        {k: f"{v['a_gt']}/{v['b_gt']}" for k, v in sorted(stratum_balance.items())},
    )

    # 3. Load side definitions
    side_defs: dict[str, Any] = yaml.safe_load(Path(_SIDE_DEFS_CONFIG).read_text())

    # Dry-run: validate extraction and prompt building for every row, then exit
    if args.dry_run:
        extract_errors: list[str] = []
        for row in rows:
            try:
                assignment = ab_assignment[row["id"]]
                _build_user_prompt(row, assignment, side_defs)
            except Exception as exc:
                extract_errors.append(f"{row['id']}: {exc}")
        if extract_errors:
            for msg in extract_errors[:20]:
                logger.warning(msg)
            if len(extract_errors) > 20:
                logger.warning(f"... and {len(extract_errors) - 20} more")
            raise RuntimeError(f"Dry-run found {len(extract_errors)} extraction errors — fix before running inference.")
        logger.success(f"Dry-run OK: {len(rows)} rows validated (extraction + prompts)", force=True)
        return

    # 4. Initialize backend (once, outside the run loop)
    model_lower = args.model.lower()
    is_gpt_oss = "gpt-oss" in model_lower
    is_deepseek_v4 = "deepseek-v4" in model_lower
    is_gemma4 = "gemma-4" in model_lower

    if args.no_think and is_gpt_oss:
        parser.error("--no-think is not supported for GPT-OSS models (reasoning cannot be disabled)")

    model_kwargs: dict[str, Any] = {}
    if args.tensor_parallel_size:
        model_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    logger.info(f"Initializing backend for {args.model}...")
    llm = None
    try:
        llm = build_vllm_client(args.model, args.reasoning_effort, args.no_think, **model_kwargs)

        logger.success(f"Backend initialized: {llm}")

        # 5. Output path
        model_id_safe = args.model.replace("/", "_").replace(".", "-")
        artifact_rating_dir = output_dir / ("artifact_rating_no_think" if args.no_think else "artifact_rating")
        artifact_rating_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifact_rating_dir / f"{model_id_safe}.jsonl"

        # 6. Load existing (item_id, run_idx) pairs for resume
        existing: set[tuple[str, int]] = set()
        if args.resume and output_path.exists():
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    existing.add((r["item_id"], r["run_idx"]))
            logger.info(f"Resume: {len(existing)} (item_id, run_idx) pairs already done")

        # 7. Run loop
        total_written = 0
        total_errors = 0
        for run_idx in range(args.runs):
            decoding_seed = args.seed + run_idx

            sampling_params = llm.set_sampling_params(
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                top_k=sampling["top_k"],
                max_tokens=sampling["max_tokens"],
                seed=decoding_seed,
            )

            pending = [row for row in rows if (row["id"], run_idx) not in existing]
            if not pending:
                logger.info(f"Run {run_idx}: all items already done, skipping")
                continue

            logger.info(f"Run {run_idx} (seed={decoding_seed}): {len(pending)} items to annotate")

            # Build prompts — track which rows produced a prompt so zip is aligned
            pending_with_assignment: list[dict] = []
            messages_list = []
            for row in pending:
                assignment = ab_assignment.get(row["id"])
                if assignment is None:
                    logger.warning(f"No A/B assignment for item_id={row['id']!r}, skipping")
                    continue
                pending_with_assignment.append(row)
                user_prompt = _build_user_prompt(row, assignment, side_defs)
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                if args.no_think and not is_deepseek_v4 and not is_gemma4:
                    msgs.append({"role": "assistant", "content": "<think></think>"})
                messages_list.append(msgs)

            if not messages_list:
                continue

            responses = llm.chat_batch(messages_list, sampling_params=sampling_params)

            # Write results
            with open(output_path, "a") as out_f:
                for row, response in zip(pending_with_assignment, responses):
                    item_id = row["id"]
                    assignment = ab_assignment.get(item_id)
                    if assignment is None:
                        continue

                    if isinstance(response, dict) and response.get("harmony_parse_failed"):
                        raw_content = response.get("raw_fallback", "")
                        thinking = ""
                    elif isinstance(response, dict):
                        raw_content = response.get("content", "")
                        thinking = response.get("reasoning", "")
                    else:
                        raw_content = response
                        thinking = ""

                    parsed, status = parse_annotation_output(raw_content)
                    total_written += 1
                    if status != "ok":
                        total_errors += 1

                    condition = row.get("condition") or ""
                    ground_truth_side = condition.split("_")[-1]

                    side_resolved_to_gt = _compute_side_resolved_to_gt(parsed, assignment)

                    out_row = {
                        "item_id": item_id,
                        "scenario_id": row["scenario_id"],
                        "condition": condition,
                        "context_type": row["context_type"],
                        "axis": row["axis"],
                        "source": row.get("source"),
                        "ground_truth_side": ground_truth_side,
                        "model_id": args.model,
                        "run_idx": run_idx,
                        "decoding_seed": decoding_seed,
                        "ab_assignment": {"A": assignment["A"], "B": assignment["B"]},
                        "a_is_gt": assignment["a_is_gt"],
                        "raw_output": raw_content,
                        "thinking_trace": thinking,
                        "parsed_json": parsed,
                        "parse_status": status,
                        "side_resolved_to_gt": side_resolved_to_gt,
                        "clarity_score": parsed["clarity_score"] if parsed else None,
                    }
                    out_f.write(json.dumps(out_row) + "\n")
                    out_f.flush()

            logger.success(f"Run {run_idx}: wrote {len(responses)} rows to {output_path}")

        # Error rate summary — warn only; do not abort so data is always uploaded
        if total_written > 0:
            error_rate = total_errors / total_written
            logger.table(
                "Parse error summary",
                {"Total rows": total_written, "Errors": total_errors, "Error rate": f"{error_rate:.1%}"},
                force=True,
            )
            if error_rate > 0.05:
                logger.warning(
                    f"Parse error rate {error_rate:.1%} exceeds 5% — "
                    "data uploaded; flag this model in analyze_artifact_rating.py before analysis."
                )

    finally:
        if llm is not None:
            llm.close()

    logger.success("Annotation complete!", force=True)


if __name__ == "__main__":
    main()
