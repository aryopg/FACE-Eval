"""Multi-stage faithfulness evaluation runner with multi-seed and resume support.

Agentic sycophancy substrate. DAFT-Math runner archived in legacy/.

Usage examples:
    # Full pipeline (inference + judge) with vLLM backend
    python run.py --model Qwen/Qwen3-4B --seeds 42,43,44

    # Inference only with Anthropic backend
    python run.py --model claude-sonnet-4-20250514 --backend anthropic --stage inference

    # Filter by axis
    python run.py --model Qwen/Qwen3-4B --axis political

    # GPT-OSS with reasoning effort control
    python run.py --model openai/gpt-oss-20b --reasoning-effort high --seeds 42

    # Resume (skip seeds that already have inference.jsonl)
    python run.py --model Qwen/Qwen3-4B --seeds 42,43 --resume

    # Second judge (writes judged__gpt-5.6-luna.jsonl, leaves judged.jsonl alone)
    python run.py --model Qwen/Qwen3-4B --stage judge --batch \
        --judge-config config/judge_gpt.yaml

    # Quiet mode
    python run.py --model Qwen/Qwen3-4B --quiet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.face_eval import FaceEval
from src.pipeline import build_vllm_client, run_inference, run_judge_stage, run_needs_engine
from src.utils.logging import get_logger
from src.utils.sampling import resolve_sampling_params

_BACKENDS_CONFIG = "config/backends.yaml"


def _resolve_backend(model: str, cli_backend: str | None) -> str:
    """Return backend from CLI override, or infer from model name via config."""
    if cli_backend is not None:
        return cli_backend
    config = yaml.safe_load(Path(_BACKENDS_CONFIG).read_text())
    model_lower = model.lower()
    for entry in config.get("patterns", []):
        if entry["pattern"].lower() in model_lower:
            return entry["backend"]
    return config.get("default", "vllm")


def main():
    parser = argparse.ArgumentParser(
        description="Run faithfulness evaluation pipeline (agentic sycophancy substrate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--stage",
        choices=["all", "inference", "judge"],
        default="all",
        help="Which stage(s) to run (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed samples and only run inference on new ones",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated list of random seeds (default: 42)",
    )

    parser.add_argument("--dataset-path", type=str, help="Local path to HF dataset directory")
    parser.add_argument("--dataset-name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--axis", type=str, help="Filter dataset by axis (e.g., political)")
    parser.add_argument("--condition", type=str, help="Filter dataset by condition")
    parser.add_argument("--max-samples", type=int, help="Cap dataset size (for smoke testing)")

    parser.add_argument("--model", type=str, required=True, help="Model identifier")
    parser.add_argument(
        "--backend",
        choices=["vllm", "vllm_server", "anthropic", "openai", "gemini", "openrouter"],
        default=None,
        help="Inference backend. If omitted, inferred from model name via config/backends.yaml.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for the vllm_server backend (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--no-batch",
        action="store_false",
        dest="use_batch",
        help="Disable batch API (Anthropic/OpenAI); use sequential/concurrent calls instead",
    )
    parser.set_defaults(use_batch=True)
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=10,
        help="Max concurrent calls for async backends (Gemini, OpenRouter, OpenAI non-batch; default: 10)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Number of GPUs for vLLM tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="Fraction of GPU memory vLLM may use (e.g., 0.4)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help=(
            "Max context length (prompt + output) for the vLLM engine, to cap KV-cache "
            "memory on large models. Distinct from --max-tokens (output only)."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow models that ship custom code (e.g. Kimi-K2.6, DeepSeek-V4) to run it.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Reasoning effort(s), comma-separated (validated per backend). GPT-OSS: low/medium/high "
            "(default high). DeepSeek-V4: chat (Non-think) / high (Think High) / max (Think Max). "
            "Inkling: a float in [0.0, 1.0) (e.g. 0.5) or a named preset. Multiple values sweep efforts "
            "on one loaded engine (e.g. 0.1,0.5,0.99). Ignored for other backends."
        ),
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help=(
            "Run the no-think arm for reasoning-capable models. Writes to "
            "results/agentic_no_think by default and treats empty reasoning as intentional."
        ),
    )

    parser.add_argument(
        "--convention",
        default="C0",
        help=(
            "Named system-prompt arm(s), comma-separated. C0–C3 = monitor-oblivious "
            "attribution ladder (C0 default, C3 directive). MC0/MC3 = monitor-aware "
            "variants of C0/C3. Multiple conventions reuse one loaded engine "
            "(e.g. --convention C0,C3,MC0,MC3). See src/data/conventions.py."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config/sampling.yaml)",
    )
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (overrides config/sampling.yaml)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (overrides config/sampling.yaml)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens to generate (overrides config/sampling.yaml)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--judge-config",
        type=str,
        default="config/judge.yaml",
        help=(
            "Judge config path. The judge model named in it decides the output file: the "
            "pre-registered judge writes judged.jsonl, any other writes judged__{model}.jsonl."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent judge API calls (default: 10)",
    )
    parser.add_argument("--batch", action="store_true", help="Use the provider's Batch API for the judge")

    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging")

    args = parser.parse_args()

    effective_output_dir = f"{args.output_dir}/agentic_no_think" if args.no_think else f"{args.output_dir}/agentic"

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    conventions = [c.strip() for c in args.convention.split(",")]
    allowed_conventions = {"C0", "C1", "C2", "C3", "MC0", "MC3"}
    if invalid := set(conventions) - allowed_conventions:
        parser.error(
            f"Invalid convention(s): {', '.join(sorted(invalid))}. Allowed: {', '.join(sorted(allowed_conventions))}"
        )

    # A comma list of efforts sweeps them on one loaded engine (per-request setting, not engine-level).
    efforts: list[str | None] = (
        [e.strip() for e in args.reasoning_effort.split(",")] if args.reasoning_effort else [None]
    )
    if args.no_think and len(efforts) > 1:
        parser.error("--no-think cannot be combined with multiple --reasoning-effort values")

    verbose = not args.quiet
    logger = get_logger(verbose=verbose)

    sampling = resolve_sampling_params(
        args.model,
        {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
        },
    )

    backend = _resolve_backend(args.model, args.backend)
    if args.no_think and backend != "vllm":
        parser.error("--no-think is currently supported only for vLLM-backed local models")
    if args.no_think and "gpt-oss" in args.model.lower():
        parser.error("--no-think is not supported for GPT-OSS models")

    logger.header("Faithfulness Evaluation Pipeline")
    logger.table(
        "Configuration",
        {
            "Model": args.model,
            "Backend": backend,
            "Substrate": "agentic_sycophancy",
            "Stage": args.stage,
            "Seeds": ", ".join(str(s) for s in seeds),
            "Convention": ", ".join(conventions),
            "No-think": args.no_think,
            "Reasoning Effort": ", ".join(str(e) for e in efforts),
            "Resume": args.resume,
            "Temperature": sampling["temperature"],
            "Top-p / Top-k": f"{sampling['top_p']} / {sampling['top_k']}",
            "Max tokens": sampling["max_tokens"],
        },
    )

    model_kwargs = {}
    if args.tensor_parallel_size:
        model_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        model_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        model_kwargs["max_model_len"] = args.max_model_len
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    # Load the dataset once (identical across seeds) and, for the offline vLLM
    # backend, build the engine once so its weights aren't reloaded per seed.
    dataset = None
    preloaded_llm = None
    if args.stage in ("all", "inference"):
        logger.info("Loading dataset...")
        dataset_kwargs = {}
        if args.dataset_path:
            dataset_kwargs["dataset_path"] = args.dataset_path
        if args.dataset_name:
            dataset_kwargs["dataset_name"] = args.dataset_name
        if args.axis:
            dataset_kwargs["axis"] = args.axis
        if args.condition:
            dataset_kwargs["condition"] = args.condition
        dataset = FaceEval(**dataset_kwargs)
        if args.max_samples:
            dataset.dataset = dataset.dataset.select(range(min(args.max_samples, len(dataset))))
        logger.success(f"Dataset: {dataset}")

        if backend == "vllm" and any(
            run_needs_engine(
                effective_output_dir,
                args.model,
                seeds,
                dataset,
                args.resume,
                reasoning_effort=effort,
                convention=convention,
            )
            for effort in efforts
            for convention in conventions
        ):
            logger.info("Loading vLLM engine (once, reused across efforts, conventions, and seeds)...")
            preloaded_llm = build_vllm_client(args.model, efforts[0], args.no_think, **model_kwargs)
            # Validate every effort up front so a typo fails before the long run, not mid-sweep.
            if hasattr(preloaded_llm, "set_reasoning_effort"):
                for effort in efforts:
                    if effort is not None:
                        preloaded_llm.set_reasoning_effort(effort)

    try:
        for effort in efforts:
            # Retune the loaded engine's per-request effort; base/Gemma clients have no effort knob.
            if preloaded_llm is not None and effort is not None and hasattr(preloaded_llm, "set_reasoning_effort"):
                preloaded_llm.set_reasoning_effort(effort)
            for convention in conventions:
                for seed in seeds:
                    label = f"Convention {convention} · Seed {seed}"
                    if effort is not None:
                        label = f"Effort {effort} · {label}"
                    logger.header(label)

                    if args.stage in ("all", "inference"):
                        logger.rule("Inference")
                        run_inference(
                            model=args.model,
                            seed=seed,
                            dataset=dataset,
                            output_dir=effective_output_dir,
                            temperature=sampling["temperature"],
                            top_p=sampling["top_p"],
                            top_k=sampling["top_k"],
                            max_tokens=sampling["max_tokens"],
                            resume=args.resume,
                            backend=backend,
                            reasoning_effort=effort,
                            no_think=args.no_think,
                            convention=convention,
                            use_batch=args.use_batch,
                            api_concurrency=args.api_concurrency,
                            preloaded_llm=preloaded_llm,
                            base_url=args.base_url,
                            **model_kwargs,
                        )

                    if args.stage in ("all", "judge"):
                        logger.rule("Judge")
                        run_judge_stage(
                            model=args.model,
                            seed=seed,
                            output_dir=effective_output_dir,
                            judge_config_path=args.judge_config,
                            concurrency=args.concurrency,
                            use_batch=args.batch,
                            resume=args.resume,
                            reasoning_effort=effort,
                            no_think=args.no_think,
                            convention=convention,
                        )
    finally:
        if preloaded_llm is not None:
            preloaded_llm.close()

    logger.success("Pipeline complete!", force=True)


if __name__ == "__main__":
    main()
