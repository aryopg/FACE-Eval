"""Convert an inspect `.eval` log into the repo's results layout.

`inspect eval` writes its own log; every analysis and plotting script reads
`results/agentic/{model}/seed_{N}/{inference,judged}.jsonl`. This bridges the two, so a
run produced through `eval/inspect/task.py` can go through `make analyze` and the figure
scripts unchanged.

Unlike `task.py` and `judge.py`, this module is repo-coupled on purpose: it imports
`src.results.storage` so the run-directory convention keeps one definition instead of a
second copy that can drift.

Two things decide whether the converted run is analyzable at all:

  * The causal-dependence filter matches each cued row to its `no_context` baseline, so a
    run filtered with `-T axis=` or `-T condition=` has no baseline and every row drops
    out. Convert unfiltered runs.
  * One log is one seed. Anything that pools over seeds needs several runs, each generated
    under a different `run_eval.py --seed`.

The seed is read out of the log's own `task_args`, not passed here — it is a property of
how the run was generated, and relabelling it at conversion time would only produce a
directory name the run does not answer to. An unseeded log is therefore not convertible.

Usage:
    python eval/inspect/to_results.py logs/agentic/Qwen_Qwen3.5-9B/seed_42/<run>.eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inspect_ai.log import read_eval_log

from eval.inspect.judge import _reasoning_and_answer
from src.results.storage import DEFAULT_JUDGE_MODEL, get_run_dir, judged_filename, save_results
from src.utils.plotting import DIR_FAMILY

SCORER = "face_eval_scorer"

# Written by run.py's judge stage; the analysis layer reads these names.
_JUDGE_FIELDS = (
    "reasoning_acknowledges_preference",
    "reasoning_tailoring_explicit",
    "reasoning_eval_awareness",
    "reasoning_explanation",
    "reasoning_parse_ok",
    "has_reasoning",
    "answer_aligns_with_preference",
    "answer_committed",
    "answer_stance_label",
    "answer_explanation",
    "answer_parse_ok",
    "answer_tailored",
)


def strip_provider(model: str) -> str:
    """Drop the inspect provider prefix: `anthropic/claude-haiku-4-5` → `claude-haiku-4-5`.

    Judge model names are recorded and filed under the bare name run.py uses, so the
    pre-registered judge lands in judged.jsonl and any other judge gets its own file.
    """
    parts = model.split("/")
    return parts[-1] if len(parts) > 1 else model


def model_dir_name(model: str) -> str:
    """Inspect model string → the directory name run.py would have written.

    `openai-api/vllm/Qwen/Qwen3.5-9B` and `vllm/Qwen/Qwen3.5-9B` both give
    `Qwen_Qwen3.5-9B`. Anything else, pass --model-name.
    """
    parts = model.split("/")
    if parts[0] == "openai-api":
        parts = parts[2:]  # drop provider and the service prefix it requires
    elif len(parts) > 1:
        parts = parts[1:]
    return "_".join(parts)


def convert(
    log_path: Path,
    output_dir: str,
    model_name: str | None,
    convention: str,
    force: bool = False,
) -> Path:
    """Write inference.jsonl / judged.jsonl / metadata.json for one inspect log."""
    log = read_eval_log(str(log_path))
    if log.status != "success":
        raise SystemExit(f"{log_path}: eval status is {log.status!r}, not 'success' — nothing converted")
    if not log.samples:
        raise SystemExit(f"{log_path}: no samples in the log")

    # `inspect eval` records every task arg including defaults, so an unseeded run comes
    # back as an explicit None rather than a missing key.
    seed = log.eval.task_args.get("seed")
    if seed is None:
        raise SystemExit(
            f"{log_path}: this run was generated unseeded, so there is no seed to file it under. "
            f"run.py seeds every run; regenerate with `run_eval.py --seed N` (or `inspect eval "
            f"... -T seed=N`)."
        )

    name = model_name or model_dir_name(log.eval.model)

    # get_run_dir only creates the directory, so checking for the payload afterwards is
    # safe and keeps one definition of the path convention. The guard matters because
    # results/agentic is where run.py's sweeps live: a converted trial run takes the same
    # (model, seed, convention) path and would otherwise overwrite GPU-hours of paper data.
    run_dir = get_run_dir(output_dir, name, seed, convention=convention)
    if (run_dir / "inference.jsonl").exists() and not force:
        raise SystemExit(
            f"{run_dir} already holds a run. Converting would overwrite it — if that came from "
            f"run.py you would lose it. Regenerate the log under an unused `run_eval.py --seed`, "
            f"write elsewhere with --output-dir, or pass --force to overwrite deliberately."
        )

    inference, judged = [], []
    unscored = 0
    for sample in log.samples:
        meta = sample.metadata or {}
        reasoning, raw_answer = _reasoning_and_answer(sample.output)
        inference.append(
            {
                "id": sample.id,
                "axis": meta.get("axis", ""),
                "condition": meta.get("condition", ""),
                "context_type": meta.get("context_type", ""),
                "scenario_id": meta.get("scenario_id", ""),
                "question": meta.get("question", ""),
                "reasoning": reasoning,
                "has_reasoning": bool(reasoning),
                "no_think": False,
                "raw_answer": raw_answer,
            }
        )

        score = (sample.scores or {}).get(SCORER)
        if score is None:
            unscored += 1
            continue
        judged.append({"id": sample.id, "judge": {k: (score.metadata or {}).get(k) for k in _JUDGE_FIELDS}})

    # A non-default judge must not be filed as judged.jsonl: that name is reserved for the
    # pre-registered judge, and the analysis layer selects between them by filename.
    # `inspect eval` records every task arg including defaults, so the key is missing only
    # for a log from a hand-built Task.
    recorded_judge = log.eval.task_args.get("judge_model")
    if not recorded_judge:
        print(f"WARNING: log records no judge model; filing under the default ({DEFAULT_JUDGE_MODEL})")
    judge_model = strip_provider(recorded_judge) if recorded_judge else DEFAULT_JUDGE_MODEL
    judged_stage = judged_filename(judge_model).removesuffix(".jsonl")

    save_results(run_dir, inference, "inference", metadata={"source_eval_log": str(log_path), "model": name})
    save_results(run_dir, judged, judged_stage, metadata={"judge_model": judge_model})

    print(f"{len(inference)} rows → {run_dir} ({judged_stage}.jsonl)")
    if unscored:
        print(f"WARNING: {unscored} samples had no {SCORER} score and are in inference.jsonl only")
    if not any(r["context_type"] == "none" for r in inference):
        print(
            "WARNING: no no_context rows in this log. The causal-dependence filter needs them, "
            "so the primary figures will come out empty. Convert an unfiltered run."
        )
    if name not in DIR_FAMILY:
        print(
            f"WARNING: {name!r} is not in the plotting registry (src/utils/plotting.py). "
            "select_models() drops unregistered models silently, so analyze_* will read this run "
            "but the figures will not show it. Add it to MODEL_FAMILY / MODEL_PARAMS / MODEL_LABEL."
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path, help="Path to the .eval log written by `inspect eval`")
    parser.add_argument("--output-dir", default="results/agentic", help="Results root (default: results/agentic)")
    parser.add_argument("--model-name", default=None, help="Override the derived model directory name")
    parser.add_argument("--convention", default="C0", help="System-prompt convention label (default: C0)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run directory")
    args = parser.parse_args()

    convert(args.log, args.output_dir, args.model_name, args.convention, args.force)


if __name__ == "__main__":
    main()
