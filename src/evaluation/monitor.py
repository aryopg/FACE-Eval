"""H2 monitor-detection judge.

Blind monitor "views" over existing inference+judge results, defined in
config['monitors'] and run together in one batch. Built-in views:
  - action   : system prompt + user message + tool calls/returns + final answer (no CoT).
  - cot       : all of the above, plus the assistant's reasoning trace.
  - cot_only  : same transcript as `action` but the reasoning trace replaces the final
                answer (answer<->reasoning swap; cue context held constant).

No view receives condition, preferred_side, or axis — each must discover any
preference signal from the observable record alone. Adding a view = adding a block
under config['monitors']; the runner, resume, and recovery handle it generically.

Results are stored per-run as h2_monitor__{monitor_model}.jsonl (one key per view).
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from src.data.face_eval import FaceEval
from src.evaluation.common import load_dataset_index, render_action_context
from src.llm.anthropic import AnthropicLLM
from src.utils.logging import get_logger
from src.utils.parsing import extract_json_object, substitute

# ---------------------------------------------------------------------------
# Output file naming
# ---------------------------------------------------------------------------


def monitor_filename(model: str) -> str:
    """Per-run monitor output filename, keyed by monitor model so different
    monitors do not overwrite each other.

    The h2_ prefix is the published name: results already on HuggingFace carry it,
    and sync_results.py globs for it. Renaming orphans them."""
    return f"h2_monitor__{model.replace('/', '_')}.jsonl"


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_monitor_response(response: str) -> dict:
    """Parse a monitor judge response.

    Returns {tailoring_score, tailoring_detected, evidence, direction, parse_ok}.
    On success: tailoring_score clamped to [0, 100]; parse_ok=True.
    On failure: tailoring_score=None, tailoring_detected=None so that any code
    that uses the score without checking parse_ok crashes loudly rather than
    silently computing wrong results.
    """
    try:
        data = extract_json_object(response)
        raw_score = data["tailoring_score"]
        score = int(round(float(raw_score)))
        score = max(0, min(100, score))
        direction = data.get("direction")
        return {
            "tailoring_score": score,
            "tailoring_detected": bool(data.get("tailoring_detected", False)),
            "evidence": str(data.get("evidence", "")),
            "direction": None if direction is None else str(direction),
            "parse_ok": True,
        }
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return {
            "tailoring_score": None,
            "tailoring_detected": None,
            "evidence": None,
            "direction": None,
            "parse_ok": False,
            "raw_response": response,
        }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_view_prompt(row: dict, action_context: str, template: str) -> str:
    """Format a monitor-view user prompt.

    All view variables are made available; each view's template references only the
    ones it needs (e.g. the action view omits ``{reasoning}``, the cot_only view omits
    ``{action_context}``). ``substitute`` ignores unused keys.
    """
    return substitute(
        template,
        {
            "action_context": action_context,
            "reasoning": str(row.get("reasoning", "")),
            "answer": str(row.get("raw_answer", "")),
        },
    )


# ---------------------------------------------------------------------------
# Backend dispatch helpers
# ---------------------------------------------------------------------------


def _resp_to_str(r: str | dict) -> str:
    """Normalise a batch response to a plain string."""
    if isinstance(r, dict):
        return r.get("content", "")
    return r or ""


def _call_anthropic(
    row_contexts: list[tuple[dict, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    views: list[tuple[str, dict]],
    max_batch_size: int,
    logger,
    skip: dict[str, frozenset[tuple[str, str]]],
) -> dict[str, list[str]]:
    """Run all monitor views via Anthropic Batch API. Returns {view_name: [str, ...]}."""
    llm = AnthropicLLM(model=model, use_batch=True)

    requests: list[dict] = []
    for name, cfg in views:
        skip_ids = skip.get(name, frozenset())
        for i, (row, action_context) in enumerate(row_contexts):
            if (row.get("_run_path", ""), row["id"]) in skip_ids:
                continue
            requests.append(
                {
                    "custom_id": f"{name}__{i}",
                    "params": {
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system": cfg["system_prompt"],
                        "messages": [
                            {
                                "role": "user",
                                "content": build_view_prompt(row, action_context, cfg["user_prompt_template"]),
                            }
                        ],
                    },
                }
            )

    n_chunks = max(1, math.ceil(len(requests) / max_batch_size))
    chunk_size = math.ceil(len(requests) / n_chunks)
    chunks = [requests[i : i + chunk_size] for i in range(0, len(requests), chunk_size)]
    logger.info(f"Anthropic: submitting {len(chunks)} sub-batch(es) ({len(requests)} requests total)")

    batch_ids = [llm.create_batch(chunk) for chunk in chunks]
    responses: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(20, len(batch_ids))) as executor:
        futures = {executor.submit(llm.poll_batch, bid): bid for bid in batch_ids}
        for future in as_completed(futures):
            responses.update(future.result())

    out: dict[str, list[str]] = {name: [""] * len(row_contexts) for name, _ in views}
    for name, _ in views:
        skip_ids = skip.get(name, frozenset())
        for i, (row, _ctx) in enumerate(row_contexts):
            key = f"{name}__{i}"
            if key not in responses and (row.get("_run_path", ""), row["id"]) not in skip_ids:
                logger.warning(f"Batch response missing key {key!r}; row id={row['id']!r} will have parse_ok=False")
            out[name][i] = _resp_to_str(responses.get(key, ""))
    return out


def _call_openai(
    row_contexts: list[tuple[dict, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    views: list[tuple[str, dict]],
    logger,
    skip: dict[str, frozenset[tuple[str, str]]],
    reasoning_effort: str | None = None,
) -> dict[str, list[str]]:
    """Run all monitor views via a single OpenAI Batch API call.

    Messages for every view are concatenated into one batch (view-major order) and
    split back by recorded (view, row) index. Rows whose (run_path, id) key appears
    in ``skip[view]`` are omitted; their slots stay empty strings.
    """
    from src.llm.openai_llm import OpenAILLM

    llm = OpenAILLM(model=model, use_batch=True, reasoning_effort=reasoning_effort)

    out: dict[str, list[str]] = {name: [""] * len(row_contexts) for name, _ in views}
    combined_msgs: list[list[dict]] = []
    index: list[tuple[str, int]] = []
    for name, cfg in views:
        skip_ids = skip.get(name, frozenset())
        for i, (row, ctx) in enumerate(row_contexts):
            if (row.get("_run_path", ""), row["id"]) in skip_ids:
                continue
            combined_msgs.append(
                [
                    {"role": "system", "content": cfg["system_prompt"]},
                    {"role": "user", "content": build_view_prompt(row, ctx, cfg["user_prompt_template"])},
                ]
            )
            index.append((name, i))

    if not combined_msgs:
        return out

    counts = {name: sum(1 for n, _ in index if n == name) for name, _ in views}
    logger.info(
        f"OpenAI: submitting combined batch ({', '.join(f'{n}={counts[n]}' for n, _ in views)} = {len(combined_msgs)} rows)"
    )
    combined_raw = llm.chat_batch(combined_msgs, max_tokens=max_tokens, temperature=temperature)

    for (name, i), r in zip(index, combined_raw):
        result = _resp_to_str(r)
        out[name][i] = result
        if not result:
            logger.warning(f"Empty {name} response for row id={row_contexts[i][0]['id']!r}; will have parse_ok=False")
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_views(config: dict) -> list[tuple[str, dict]]:
    """Ordered list of (view_name, view_cfg) from config['monitors']."""
    return list(config["monitors"].items())


def run_monitors(
    results: list[dict],
    dataset: FaceEval,
    config_path: str = "config/monitor_judge.yaml",
    max_batch_size: int = 7000,
    monitor_model: str | None = None,
    overrides: dict[str, dict] | None = None,
    reasoning_effort: str | None = None,
) -> list[dict]:
    """Run all blind monitor views (config['monitors']) over inference results.

    Returns list of {id, <view>: {tailoring_score, tailoring_detected, evidence,
    direction, parse_ok}, ...} — one key per view (e.g. action, cot, cot_only).

    ``monitor_model`` overrides the model in the config file when provided.
    ``overrides``: {view_name: {(run_path, row_id): result}} — rows present skip the
    corresponding API call and reuse the pre-filled result (partial resume).
    """
    logger = get_logger()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = monitor_model or config["model"]
    logger.info(f"Monitor model: {model}")
    max_tokens = int(config.get("max_tokens", 256))
    temperature = float(config.get("temperature", 0.0))
    views = load_views(config)
    logger.info(f"Monitor views: {[n for n, _ in views]}")

    backend = config.get("backend", "anthropic")
    effort = reasoning_effort or config.get("reasoning_effort")
    if effort:
        logger.info(f"Reasoning effort: {effort}")
    dataset_index = load_dataset_index(dataset)

    overrides = overrides or {}
    # Skip sets are (run_path_str, row_id) tuples to avoid cross-run ID collisions.
    skip: dict[str, frozenset[tuple[str, str]]] = {name: frozenset(overrides.get(name, {})) for name, _ in views}
    n_skip = sum(len(s) for s in skip.values())
    if n_skip:
        logger.info(f"Partial resume: skipping {', '.join(f'{n}={len(skip[n])}' for n, _ in views)} (already done)")

    # Build (row, action_context) pairs — shared across backends and views
    row_contexts: list[tuple[dict, str]] = []
    for row in results:
        row_id = row["id"]
        ds_row = dataset_index.get(row_id)
        if ds_row is None:
            logger.warning(f"Dataset row not found for id={row_id!r}; using empty action context")
            action_context = "(context unavailable)"
        else:
            try:
                action_context = render_action_context(ds_row)
            except Exception as exc:
                logger.warning(f"Failed to render action context for id={row_id!r}: {exc}; using empty context")
                action_context = "(action context rendering failed)"
        row_contexts.append((row, action_context))

    # Dispatch to backend
    if backend == "openai":
        view_strs = _call_openai(
            row_contexts, model, max_tokens, temperature, views, logger, skip, reasoning_effort=effort
        )
    else:
        view_strs = _call_anthropic(row_contexts, model, max_tokens, temperature, views, max_batch_size, logger, skip)

    entries: list[dict] = []
    for i, (row, _) in enumerate(row_contexts):
        rid = row["id"]
        key = (row.get("_run_path", ""), rid)
        entry: dict = {"id": rid}
        for name, _ in views:
            ov = overrides.get(name, {})
            entry[name] = ov[key] if key in ov else _parse_monitor_response(view_strs[name][i])
        entries.append(entry)

    logger.success(f"H2 monitor evaluation complete for {len(entries)} rows")
    return entries
