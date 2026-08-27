"""Pipeline logic: inference execution and judge evaluation.

Agentic sycophancy substrate only. DAFT-Math code has been archived to legacy/.
"""

from __future__ import annotations

import json
from collections import defaultdict

from src.data.conventions import CONVENTIONS
from src.data.face_eval import FaceEval
from src.evaluation.parsing import parse_model_output
from src.results.storage import get_run_dir, judged_filename, load_results, save_results
from src.utils.logging import get_logger


def _append_convention(msgs: list[dict], addendum: str) -> list[dict]:
    """Append convention addendum to the dataset's system message.

    C0 has an empty addendum, so messages are returned unchanged. For other
    conventions the addendum is appended to the existing system-message content
    (or placed in a new system message if none is present).
    """
    if not addendum:
        return msgs
    if msgs and msgs[0].get("role") == "system":
        new_msg = {**msgs[0], "content": f"{msgs[0]['content']} {addendum}"}
        return [new_msg] + msgs[1:]
    return [{"role": "system", "content": addendum}] + list(msgs)


def _prefill_empty_think(msgs: list[dict]) -> list[dict]:
    """Append an assistant prefill that closes the reasoning channel.

    Qwen/OLMo no-think mode needs both template-level thinking disabled and an
    empty-think prefill so generation starts in the answer channel.
    """
    return list(msgs) + [{"role": "assistant", "content": "<think></think>"}]


def build_vllm_client(model: str, reasoning_effort: str | None, no_think: bool, **model_kwargs):
    """Construct the vLLM client for a model.

    Kept separate from run_inference so the engine (the expensive weight load) can
    be built once and reused across seeds, rather than reloaded per seed.
    """
    model_lower = model.lower()
    if "gpt-oss" in model_lower:
        from src.llm.gpt_oss import GPTOSSClient

        return GPTOSSClient(
            model=model,
            reasoning_effort=reasoning_effort or "high",
            include_reasoning=True,
            **model_kwargs,
        )
    if "deepseek-v4" in model_lower:
        from src.llm.deepseek_v4 import DeepSeekV4Client

        # GPT-OSS-style "low"/"medium" don't map to V4; coerce so the CLI default
        # ("high") still works and Think Max is reachable via --reasoning-effort max.
        v4_effort = "chat" if no_think else reasoning_effort or "high"
        if v4_effort in ("low", "medium"):
            raise ValueError(f"Reasoning effort {v4_effort} is not supported for DeepSeek V4")
        return DeepSeekV4Client(
            model=model,
            reasoning_effort=v4_effort,
            include_reasoning=True,
            **model_kwargs,
        )
    if "inkling" in model_lower:
        from src.llm.inkling import InklingClient

        return InklingClient(
            model=model,
            reasoning_effort=reasoning_effort,
            include_reasoning=True,
            **model_kwargs,
        )
    if "gemma-4" in model_lower:
        from src.llm.gemma4 import Gemma4Client

        return Gemma4Client(
            model=model,
            enable_thinking=not no_think,
            include_reasoning=True,
            **model_kwargs,
        )
    from src.llm.vllm import VLLMClient

    return VLLMClient(
        model=model,
        enable_thinking=not no_think,
        include_reasoning=True,
        **model_kwargs,
    )


def run_needs_engine(
    output_dir: str,
    model: str,
    seeds: list[int],
    dataset: FaceEval,
    resume: bool,
    reasoning_effort: str | None = None,
    convention: str = "C0",
) -> bool:
    """True if any seed still has inference work to do.

    Used to avoid loading the (expensive) vLLM engine up front only to have every
    seed early-return on a fully-completed --resume run.
    """
    if not resume:
        return True
    dataset_ids = {dataset[i]["id"] for i in range(len(dataset))}
    for seed in seeds:
        run_dir = get_run_dir(output_dir, model, seed, reasoning_effort=reasoning_effort, convention=convention)
        inference_path = run_dir / "inference.jsonl"
        if not inference_path.exists():
            return True
        existing = {r["id"] for r in load_results(run_dir, "inference")}
        if dataset_ids - existing:
            return True
    return False


def run_inference(
    model: str,
    seed: int,
    dataset: FaceEval,
    output_dir: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    resume: bool,
    backend: str,
    reasoning_effort: str | None = None,
    no_think: bool = False,
    convention: str = "C0",
    use_batch: bool = True,
    api_concurrency: int = 10,
    preloaded_llm=None,
    base_url: str | None = None,
    **model_kwargs,
) -> list[dict]:
    """Run inference stage on the agentic sycophancy dataset.

    Returns a list of result dicts with parsed outputs.
    """
    logger = get_logger()
    if no_think and backend != "vllm":
        raise ValueError("no_think is currently supported only for the vLLM backend")
    if no_think and "gpt-oss" in model.lower():
        raise ValueError("no_think is not supported for GPT-OSS models")

    run_dir = get_run_dir(output_dir, model, seed, reasoning_effort=reasoning_effort, convention=convention)

    existing_results: list[dict] = []
    existing_ids: set[str] = set()
    inference_path = run_dir / "inference.jsonl"
    if resume and inference_path.exists():
        existing_results = load_results(run_dir, "inference")
        existing_ids = {r["id"] for r in existing_results}
        logger.info(f"Resuming: found {len(existing_results)} existing results")

    # Recover results from a partial async run (gemini/openrouter) that was interrupted.
    # The partial file is written per-response during the run; it persists on crash.
    # We always check for it — not just on --resume — since interruptions can happen
    # even on a fresh run.
    partial_path = run_dir / "inference_partial.jsonl"
    if partial_path.exists():
        _partial_by_idx: dict[int, object] = {}
        with open(partial_path) as _pf:
            for _line in _pf:
                _line = _line.strip()
                if not _line:
                    continue
                _e = json.loads(_line)
                _partial_by_idx[_e["dataset_idx"]] = _e["response"]
        if _partial_by_idx:
            _recover_indices = [idx for idx in sorted(_partial_by_idx) if dataset[idx]["id"] not in existing_ids]
            if _recover_indices:
                _rec_results, _, _ = _build_agentic_results(
                    dataset,
                    [_partial_by_idx[idx] for idx in _recover_indices],
                    _recover_indices,
                    no_think=no_think,
                )
                existing_results.extend(_rec_results)
                existing_ids.update(r["id"] for r in _rec_results)
                logger.info(f"Recovered {len(_rec_results)} responses from partial run at {partial_path}")

    all_indices = list(range(len(dataset)))
    pending_indices = [i for i in all_indices if dataset[i]["id"] not in existing_ids]

    if not pending_indices:
        logger.info("All samples already processed, nothing to do")
        return existing_results

    skipped = len(all_indices) - len(pending_indices)
    if skipped > 0:
        logger.info(f"{len(pending_indices)} new samples to process (skipping {skipped} already done)")
    else:
        logger.info(f"{len(pending_indices)} samples to process")

    logger.info("Preparing prompts...")
    messages_list = []
    tools_list = []
    for i in pending_indices:
        msgs, tools = dataset.get_messages_and_tools(i)
        messages_list.append(msgs)
        tools_list.append(tools)

    logger.info(f"Running batch inference on {len(messages_list)} prompts...")
    # `seed` reaches vLLM only. Anthropic and Gemini have no seed parameter, OpenAI is
    # not wired to one here, and OpenRouter forwards it best-effort.

    injected = [_append_convention(msgs, CONVENTIONS[convention]) for msgs in messages_list]

    # For async backends: stream each result to a partial file as it arrives so a
    # crash mid-run doesn't discard work. The partial file is read back on the
    # next invocation (see recovery block above) before pending_indices is computed.
    _partial_fh = None
    _on_result = None
    if backend in ("gemini", "openrouter"):
        _partial_fh = open(partial_path, "a")

        def _on_result(list_idx: int, response: object) -> None:
            _partial_fh.write(json.dumps({"dataset_idx": pending_indices[list_idx], "response": response}) + "\n")
            _partial_fh.flush()

    llm = None
    try:
        if backend == "vllm":
            is_gemma4 = "gemma-4" in model.lower()
            llm = (
                preloaded_llm
                if preloaded_llm is not None
                else build_vllm_client(model, reasoning_effort, no_think, **model_kwargs)
            )

            sampling_params = llm.set_sampling_params(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                seed=seed,
            )
            vllm_messages = (
                [_prefill_empty_think(msgs) for msgs in injected] if no_think and not is_gemma4 else injected
            )
            responses = llm.chat_batch(
                vllm_messages,
                sampling_params=sampling_params,
                tools_list=tools_list,
            )

        elif backend == "anthropic":
            from src.llm.anthropic import AnthropicLLM

            llm = AnthropicLLM(
                model=model,
                use_batch=use_batch,
                include_reasoning=True,
                reasoning_effort=reasoning_effort,
            )
            responses = llm.chat_batch(
                injected,
                max_tokens=max_tokens,
                temperature=temperature,
                tools_list=tools_list,
            )

        elif backend == "openai":
            from src.llm.openai_llm import OpenAILLM

            llm = OpenAILLM(
                model=model,
                use_batch=use_batch,
                concurrency=api_concurrency,
                reasoning_effort=reasoning_effort,
            )
            responses = llm.chat_batch(
                injected,
                max_tokens=max_tokens,
                temperature=temperature,
                tools_list=tools_list,
            )

        elif backend == "gemini":
            from src.llm.gemini import GeminiLLM

            llm = GeminiLLM(
                model=model,
                concurrency=api_concurrency,
                include_reasoning=True,
                reasoning_effort=reasoning_effort,
            )
            responses = llm.chat_batch(
                injected,
                max_tokens=max_tokens,
                temperature=temperature,
                tools_list=tools_list,
                on_result=_on_result,
            )

        elif backend == "openrouter":
            from src.llm.openrouter import OpenRouterLLM

            llm = OpenRouterLLM(model=model, concurrency=api_concurrency)
            responses = llm.chat_batch(
                injected,
                max_tokens=max_tokens,
                temperature=temperature,
                tools_list=tools_list,
                on_result=_on_result,
            )

        elif backend == "vllm_server":
            from src.llm.vllm_server import VLLMServerClient

            llm = VLLMServerClient(
                model=model,
                base_url=base_url,
                concurrency=api_concurrency,
                seed=seed,
                enable_thinking=not no_think,
            )
            responses = llm.chat_batch(
                injected,
                max_tokens=max_tokens,
                temperature=temperature,
                tools_list=tools_list,
            )

        else:
            raise ValueError(f"Unknown backend: {backend}")
    finally:
        # Don't tear down a caller-owned engine (preloaded across seeds).
        if llm is not None and preloaded_llm is None:
            llm.close()
        if _partial_fh is not None:
            _partial_fh.close()

    logger.success(f"Generated {len(responses)} responses")

    logger.info("Parsing outputs...")
    skipped_ids_path = run_dir / "skipped_inference_ids.json"
    parse_failures_path = run_dir / "parse_failures_debug.jsonl"
    new_results, skipped_ids, parse_failures = _build_agentic_results(
        dataset,
        responses,
        pending_indices,
        no_think=no_think,
    )

    if parse_failures:
        with open(parse_failures_path, "w") as _f:
            for _entry in parse_failures:
                _f.write(json.dumps(_entry) + "\n")

    results = existing_results + new_results
    if skipped_ids:
        skipped_ids_path.write_text(json.dumps(skipped_ids, indent=2))
        logger.warning(f"Skipped {len(skipped_ids)} samples due to LLM output parse failures")
        logger.warning(f"Debug data (raw text + error) saved to {parse_failures_path}")
    else:
        if skipped_ids_path.exists():
            skipped_ids_path.unlink()
        if parse_failures_path.exists():
            parse_failures_path.unlink()

    metadata = {
        "model": model,
        "seed": seed,
        "backend": backend,
        "reasoning_effort": reasoning_effort,
        "no_think": no_think,
        "convention": convention,
        "dataset_type": "agentic",
        "skipped_inference_ids_count": len(skipped_ids),
        "sampling_params": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
        },
    }

    axis_counts: dict[str, int] = defaultdict(int)
    for r in results:
        axis_counts[r["axis"]] += 1
    logger.table("Rows per Axis", dict(axis_counts), force=True)

    save_results(run_dir, results, stage="inference", metadata=metadata)
    logger.success(f"Inference results saved to {run_dir / 'inference.jsonl'}")

    if partial_path.exists():
        partial_path.unlink()

    return results


def _build_agentic_results(
    dataset: FaceEval,
    responses: list,
    indices: list[int] | None = None,
    no_think: bool = False,
) -> tuple[list[dict], list[str], list[dict]]:
    """Build result dicts for agentic sycophancy dataset."""
    if indices is None:
        indices = list(range(len(dataset)))

    if len(responses) != len(indices):
        raise ValueError(
            f"responses/indices length mismatch: {len(responses)} responses for {len(indices)} indices. "
            "This indicates a silent batch failure in the LLM backend."
        )

    results = []
    skipped_ids = []
    parse_failures = []
    for idx, response in zip(indices, responses):
        item = dataset[idx]
        if isinstance(response, dict) and response.get("harmony_parse_failed"):
            skipped_ids.append(item["id"])
            parse_failures.append(
                {
                    "id": item["id"],
                    "raw_text": response.get("raw_fallback", ""),
                    "parse_error": response.get("parse_error", "unknown"),
                }
            )
            continue

        if isinstance(response, dict):
            reasoning_from_model = response.get("reasoning", "")
            content = response.get("content", "")
        else:
            reasoning_from_model = ""
            content = response

        parsed = parse_model_output(content, reasoning_from_model=reasoning_from_model)
        has_reasoning = bool((parsed.reasoning or "").strip())

        results.append(
            {
                "id": item["id"],
                "axis": item["axis"],
                "condition": item["condition"],
                "context_type": item["context_type"],
                "scenario_id": item["scenario_id"],
                "question": item["question"],
                "reasoning": parsed.reasoning,
                "has_reasoning": has_reasoning,
                "no_think": no_think,
                "raw_answer": parsed.raw_answer,
            }
        )
    return results, skipped_ids, parse_failures


def run_judge_stage(
    model: str,
    seed: int,
    output_dir: str,
    judge_config_path: str,
    concurrency: int = 10,
    use_batch: bool = False,
    resume: bool = False,
    reasoning_effort: str | None = None,
    no_think: bool = False,
    convention: str = "C0",
) -> list[dict]:
    """Run both judges (reasoning + answer) on existing inference results.

    Loads `inference.jsonl` from the run directory, runs both judges via
    `src.evaluation.judges.run_judges`, and writes one record per row
    to `judged.jsonl` keyed by `id`. Each record contains the 6 metrics plus the
    derived `answer_tailored` flag.

    The output file is named after the judge model: the pre-registered judge writes
    `judged.jsonl`, any other writes `judged__{model}.jsonl`, so a second judge never
    overwrites the first.

    If `resume=True`, rows whose id already appears in that file are skipped.
    """
    from src.evaluation.judges import load_judge_config, run_judges

    logger = get_logger()
    allow_empty_reasoning = no_think
    run_dir = get_run_dir(output_dir, model, seed, reasoning_effort=reasoning_effort, convention=convention)

    inference_path = run_dir / "inference.jsonl"
    if not inference_path.exists():
        raise FileNotFoundError(
            f"No inference results found at {inference_path}. Run inference first (--stage inference or --stage all)."
        )

    inference_results = load_results(run_dir, "inference")
    logger.info(f"Loaded {len(inference_results)} inference rows from {run_dir}")

    missing_by_condition: dict[str, int] = {}
    missing_count = 0
    for r in inference_results:
        if not (r.get("reasoning") or "").strip():
            missing_count += 1
            cond = r.get("condition", "unknown")
            missing_by_condition[cond] = missing_by_condition.get(cond, 0) + 1
    missing_rate = missing_count / len(inference_results) if inference_results else 0.0
    if missing_count and not allow_empty_reasoning:
        logger.warning(
            f"Skipping {missing_count} rows with empty reasoning ({missing_rate:.1%}) — "
            f"by condition: {missing_by_condition}"
        )
        inference_results = [r for r in inference_results if (r.get("reasoning") or "").strip()]
    elif missing_count:
        logger.warning(
            f"Keeping {missing_count} rows with intentionally empty reasoning ({missing_rate:.1%}) — "
            f"by condition: {missing_by_condition}"
        )

    judge_model = load_judge_config(judge_config_path)["model"]
    stage = judged_filename(judge_model).removesuffix(".jsonl")
    judged_path = run_dir / f"{stage}.jsonl"
    logger.info(f"Judge model: {judge_model} -> {judged_path.name}")

    existing_judged: list[dict] = []
    clean_ids: set[str] = set()  # rows where BOTH judges parsed cleanly — skip
    stale_ids: set[str] = set()  # rows where at least one judge failed — re-run
    if resume and judged_path.exists():
        with open(judged_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                judge = entry.get("judge") or {}
                # Rows written before parse_ok was recorded lack the fields;
                # treat them as clean rather than re-judging old results.
                r_ok = judge.get("reasoning_parse_ok", True)
                a_ok = judge.get("answer_parse_ok", True)
                if r_ok and a_ok:
                    existing_judged.append(entry)
                    clean_ids.add(entry["id"])
                else:
                    stale_ids.add(entry["id"])
        logger.info(f"Resuming: {len(clean_ids)} rows clean, {len(stale_ids)} rows flagged for re-judge")

    pending = [r for r in inference_results if r["id"] not in clean_ids]
    if not pending:
        logger.info("All rows already judged cleanly; nothing to do")
        return existing_judged

    if resume and clean_ids:
        logger.info(f"{len(pending)} rows to judge ({len(stale_ids)} re-runs, {len(pending) - len(stale_ids)} new)")

    # Stream completed rows to a partial file so an API failure mid-run does not
    # discard already-finished work. On success the partial file is promoted to
    # the final judged.jsonl.
    partial_path = judged_path.with_suffix(".partial.jsonl")
    new_entries = run_judges(
        pending,
        config_path=judge_config_path,
        concurrency=concurrency,
        use_batch=use_batch,
        output_path=partial_path,
        allow_empty_reasoning=allow_empty_reasoning,
    )
    partial_path.replace(judged_path)

    all_entries = existing_judged + new_entries
    # Nested under the stage key so a second judge records its own provenance
    # instead of overwriting the first judge's.
    metadata = {
        stage: {
            "judge_model": judge_model,
            "judge_config_path": judge_config_path,
            "judge_concurrency": concurrency,
            "judge_use_batch": use_batch,
            "total_judged": len(all_entries),
            "reasoning_missing_count": missing_count,
            "reasoning_missing_rate": missing_rate,
            "reasoning_missing_by_condition": missing_by_condition,
            "allow_empty_reasoning": allow_empty_reasoning,
            "no_think": no_think,
        }
    }
    save_results(run_dir, all_entries, stage=stage, metadata=metadata)
    logger.success(f"Judge results saved to {judged_path}")
    return all_entries
