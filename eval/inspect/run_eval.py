"""Run the inspect-eval task into a run.py-shaped log directory.

`inspect eval` names its log `{timestamp}_{task}_{id}.eval` and drops every run in a flat
`logs/`, so which model or seed a log belongs to is only visible by opening it. Inspect
resolves that path before the task runs and the filename pattern takes only `{task}`,
`{model}` and `{id}` — so the structure has to go in the directory, and something outside
inspect has to compute it. This wrapper does, mirroring `run.py`'s results layout:

    logs/agentic[_no_think]/{model}/seed_{N}/{timestamp}_face-eval_{id}.eval

The model segment is the same string `to_results.py` derives, so a log directory and the
`results/agentic/` directory it converts into carry the same name.

Usage:
    python eval/inspect/run_eval.py --model anthropic/claude-sonnet-4-6 --seed 9001
    python eval/inspect/run_eval.py --model openai-api/vllm/Qwen/Qwen3.5-9B --seed 9001 --axis political

`--seed` is required, and 42/43/44 are the paper's sweep seeds — reusing one on a model
already in the sweep produces a log `to_results.py` will refuse to convert over.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inspect_ai import eval as inspect_eval

from eval.inspect.to_results import model_dir_name

TASK = "eval/inspect/task.py"


def log_dir_for(model: str, seed: int, no_think: bool, log_root: str, model_name: str | None = None) -> Path:
    """Log directory for one run, shaped like `run.py`'s results layout."""
    family = "agentic_no_think" if no_think else "agentic"
    return Path(log_root) / family / (model_name or model_dir_name(model)) / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Inspect model string for the subject model")
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Sampling seed. Required: it names both the log directory and, after conversion, the "
            "results directory, and to_results.py reads it back out of the log rather than relabelling."
        ),
    )
    parser.add_argument("--axis", default=None, help="Filter dataset by axis (e.g. political)")
    parser.add_argument("--condition", default=None, help="Filter dataset by condition")
    parser.add_argument(
        "--judge-model",
        default="anthropic/claude-haiku-4-5-20251001",
        help="Inspect model string for both judges (default: the pre-registered judge)",
    )
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode (mirrors run.py --no-think)")
    parser.add_argument("--model-name", default=None, help="Override the derived model directory name")
    parser.add_argument("--log-root", default="logs", help="Log root (default: logs)")
    args = parser.parse_args()

    log_dir = log_dir_for(args.model, args.seed, args.no_think, args.log_root, args.model_name)
    task_args = {
        "seed": args.seed,
        "think": not args.no_think,
        "judge_model": args.judge_model,
    }
    if args.axis:
        task_args["axis"] = args.axis
    if args.condition:
        task_args["condition"] = args.condition

    print(f"Logging to {log_dir}/")
    inspect_eval(TASK, model=args.model, task_args=task_args, log_dir=str(log_dir))


if __name__ == "__main__":
    main()
