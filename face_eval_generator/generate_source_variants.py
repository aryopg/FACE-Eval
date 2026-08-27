"""Generate channel-source variants (email / slack / notes / browser_history).

For each accepted profile-source scenario, re-render the existing canonical
view-A and view-B stance statements (the `explicit_{side_a}` and
`explicit_{side_b}` rows in scenarios.jsonl) as artifacts in a different
channel. Each scenario × source pair produces 4 rows in one LLM call:

    explicit_{source}_{side_a}   summary, side A
    explicit_{source}_{side_b}   summary, side B
    implicit_{source}_{side_a}   raw artifact list, side A
    implicit_{source}_{side_b}   raw artifact list, side B

Scenarios and questions are NEVER regenerated. The sketch, question, side
identities, and side-anchor strings are read from existing scenarios.jsonl.
Acceptance gate: both `explicit_{side_a}` and `explicit_{side_b}` rows for
the scenario must be marked "accept" in scenario_reviews.json.

Output: pre-export raw rows at
    face_eval_generator/data/{axis}/source_{source}.jsonl
matching the schema of scenarios.jsonl (no messages/tools yet). The
review/revise/export cycle in `ui.py` then writes the inference-ready rows
to `finalized_source_{source}.jsonl`.

CLI:
    python -m face_eval_generator.generate_source_variants \\
        --axis political --source email --concurrency 10
    python -m face_eval_generator.generate_source_variants \\
        --axis all --source all --regenerate
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from face_eval_generator.generate import (
    load_axes_config,
    load_jsonl,
    load_scenario_reviews,
    load_source_reviews,
    parse_json_response,
    save_jsonl,
    save_source_reviews,
)
from src.llm.anthropic import AnthropicLLM
from src.utils.logging import get_logger

ROOT = Path(__file__).parent
CONFIG_DIR = ROOT / "config"
DEFAULT_DATA_DIR = ROOT / "data"

log = get_logger()

SOURCES = ("email", "slack", "notes", "browser_history")


def load_source_prompts() -> dict:
    """Load the per-source prompts used to render channel artifacts."""
    return yaml.safe_load((CONFIG_DIR / "source_generation_prompts.yaml").read_text())


def source_rows_path(data_dir: Path, axis: str, source: str) -> Path:
    """Return the pre-export JSONL path for one axis and source channel."""
    return data_dir / axis / f"source_{source}.jsonl"


def _gather_accepted_scenarios(data_dir: Path, axis: str, axes_cfg: dict) -> list[dict]:
    """Return per-scenario records anchored on accepted explicit rows."""
    cfg = axes_cfg[axis]
    side_a, side_b = cfg["side_a"], cfg["side_b"]
    rows = load_jsonl(data_dir / axis / "scenarios.jsonl")
    reviews = load_scenario_reviews(data_dir, axis)

    by_sid: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_sid.setdefault(r["scenario_id"], {})[r["condition"]] = r

    scenarios = []
    for sid, conds in by_sid.items():
        a_row = conds.get(f"explicit_{side_a}")
        b_row = conds.get(f"explicit_{side_b}")
        if not (a_row and b_row):
            continue
        if reviews.get(f"{sid}__explicit_{side_a}") != "accept":
            continue
        if reviews.get(f"{sid}__explicit_{side_b}") != "accept":
            continue
        scenarios.append(
            {
                "scenario_id": sid,
                "axis": axis,
                "sketch": a_row.get("sketch") or b_row.get("sketch") or {},
                "question": a_row["question"],
                "side_a": side_a,
                "side_b": side_b,
                "explicit_A_context": a_row["context"],
                "explicit_B_context": b_row["context"],
            }
        )
    return scenarios


def _build_raw_source_rows(scenario: dict, source: str, result: dict) -> list[dict]:
    """Convert an LLM result (4 keys) into 4 pre-export raw rows.

    Schema matches scenarios.jsonl: scenario_id, axis, condition, context,
    context_type, source, question, sketch.
    """
    sid = scenario["scenario_id"]
    axis = scenario["axis"]
    side_a, side_b = scenario["side_a"], scenario["side_b"]
    question = scenario["question"]
    sketch = scenario["sketch"]

    plan = [
        (f"explicit_{source}_{side_a}", "explicit", result.get("explicit_A")),
        (f"explicit_{source}_{side_b}", "explicit", result.get("explicit_B")),
        (f"implicit_{source}_{side_a}", "implicit", result.get("implicit_A")),
        (f"implicit_{source}_{side_b}", "implicit", result.get("implicit_B")),
    ]

    rows: list[dict] = []
    for condition, ctx_type, context in plan:
        if context is None:
            raise ValueError(f"missing context for {sid} {source} {condition}")
        rows.append(
            {
                "scenario_id": sid,
                "axis": axis,
                "condition": condition,
                "question": question,
                "context": context,
                "context_type": ctx_type,
                "source": source,
                "sketch": sketch,
            }
        )
    return rows


async def _generate_one(
    llm: AnthropicLLM,
    scenario: dict,
    source: str,
    cfg: dict,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    src_cfg = cfg["sources"][source]
    system = (cfg.get("shared_system_prefix", "") + "\n\n" + src_cfg["system_prompt"]).strip()
    user = src_cfg["user_template"].format(
        axis_name=scenario["axis"],
        scenario=scenario["sketch"].get("scenario", ""),
        topic=scenario["sketch"].get("topic", ""),
        question_direction=scenario["sketch"].get("question_direction", ""),
        question=scenario["question"],
        side_a=scenario["side_a"],
        side_b=scenario["side_b"],
        explicit_A_context=scenario["explicit_A_context"],
        explicit_B_context=scenario["explicit_B_context"],
    )

    async with semaphore:
        response = await llm.async_chat(
            messages=[{"role": "user", "content": user}],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            system=system,
        )

    result = parse_json_response(response)
    return _build_raw_source_rows(scenario, source, result)


async def run_source_for_axis(
    llm: AnthropicLLM,
    axis: str,
    source: str,
    cfg: dict,
    data_dir: Path,
    concurrency: int,
    limit: int | None,
    regenerate: bool,
) -> list[dict]:
    """Generate (or skip) source rows for one axis × source. Idempotent.

    Returns the FULL set of rows in source_{src}.jsonl after this call —
    union of preserved untouched rows + freshly generated ones.
    """
    axes_cfg = load_axes_config()
    scenarios = _gather_accepted_scenarios(data_dir, axis, axes_cfg)
    if limit:
        scenarios = scenarios[:limit]
    if not scenarios:
        log.warning(f"[{axis}/{source}] no accepted scenarios; skipping")
        return []

    out_path = source_rows_path(data_dir, axis, source)
    existing = load_jsonl(out_path)
    existing_sids = {r["scenario_id"] for r in existing}

    if regenerate:
        log.info(f"[{axis}/{source}] regenerate=True — dropping {len(existing)} existing rows")
        existing = []
        existing_sids = set()
        pending = scenarios
    else:
        pending = [s for s in scenarios if s["scenario_id"] not in existing_sids]

    log.header(
        f"[{axis}/{source}] {len(pending)} pending / {len(scenarios)} accepted "
        f"(existing kept: {len(existing_sids)}; concurrency={concurrency})"
    )

    if not pending:
        log.info(f"[{axis}/{source}] nothing to do")
        return existing

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def with_progress(scenario):
        nonlocal completed
        try:
            rows = await _generate_one(llm, scenario, source, cfg, semaphore)
            completed += 1
            log.info(f"  [{completed}/{len(pending)}] {scenario['scenario_id']}")
            return rows
        except Exception as exc:
            completed += 1
            log.warning(f"  [{completed}/{len(pending)}] {scenario['scenario_id']} FAILED: {exc}")
            return []

    chunks = await asyncio.gather(*[with_progress(s) for s in pending])
    new_rows = [r for chunk in chunks for r in chunk]
    merged = existing + new_rows
    save_jsonl(merged, out_path)
    log.success(f"[{axis}/{source}] wrote {len(merged)} rows ({len(new_rows)} new) -> {out_path}")
    return merged


def _condition_to_field(condition: str, source: str, side_a: str, side_b: str) -> str | None:
    """Map a stored condition like 'explicit_email_liberal' to LLM field 'explicit_A'."""
    if not condition.startswith(("explicit_", "implicit_")):
        return None
    register, rest = condition.split("_", 1)
    if not rest.startswith(f"{source}_"):
        return None
    side = rest[len(source) + 1 :]
    if side == side_a:
        return f"{register}_A"
    if side == side_b:
        return f"{register}_B"
    return None


def _field_to_condition(field: str, source: str, side_a: str, side_b: str) -> str:
    """Inverse of _condition_to_field."""
    register, side_token = field.split("_", 1)
    side = side_a if side_token == "A" else side_b
    return f"{register}_{source}_{side}"


async def _revise_one(
    llm: AnthropicLLM,
    scenario: dict,
    source: str,
    current_rows: list[dict],
    flagged_conditions: list[str],
    cfg: dict,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Revise flagged conditions for one scenario; return updated 4 rows."""
    side_a, side_b = scenario["side_a"], scenario["side_b"]
    by_cond = {r["condition"]: r for r in current_rows}

    current_fields = {}
    for cond, row in by_cond.items():
        field = _condition_to_field(cond, source, side_a, side_b)
        if field:
            ctx = row["context"]
            current_fields[field] = json.dumps(ctx, indent=2) if isinstance(ctx, list) else str(ctx)

    flagged_fields = [_condition_to_field(c, source, side_a, side_b) for c in flagged_conditions]
    flagged_fields = [f for f in flagged_fields if f]
    flagged_text = "\n".join(f"- {f}" for f in flagged_fields)

    src_cfg = cfg["sources"][source]
    system = (cfg.get("shared_system_prefix", "") + "\n\n" + src_cfg["system_prompt"]).strip()
    user = cfg["revision_user_template"].format(
        axis_name=scenario["axis"],
        scenario=scenario["sketch"].get("scenario", ""),
        topic=scenario["sketch"].get("topic", ""),
        question_direction=scenario["sketch"].get("question_direction", ""),
        question=scenario["question"],
        side_a=side_a,
        side_b=side_b,
        explicit_A_context=scenario["explicit_A_context"],
        explicit_B_context=scenario["explicit_B_context"],
        current_explicit_A=current_fields.get("explicit_A", ""),
        current_explicit_B=current_fields.get("explicit_B", ""),
        current_implicit_A=current_fields.get("implicit_A", ""),
        current_implicit_B=current_fields.get("implicit_B", ""),
        flagged_conditions_text=flagged_text,
    )

    async with semaphore:
        response = await llm.async_chat(
            messages=[{"role": "user", "content": user}],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            system=system,
        )

    result = parse_json_response(response)
    for field in ("explicit_A", "explicit_B", "implicit_A", "implicit_B"):
        new_val = result.get(field)
        if new_val is None or new_val == "[SKIPPED]":
            continue
        cond = _field_to_condition(field, source, side_a, side_b)
        if cond in by_cond:
            by_cond[cond]["context"] = new_val
    return list(by_cond.values())


async def run_revision_for_axis(
    llm: AnthropicLLM,
    axis: str,
    source: str,
    cfg: dict,
    data_dir: Path,
    concurrency: int,
) -> int:
    """Revise all flagged rows for (axis, source). Returns number of scenarios revised."""
    axes_cfg = load_axes_config()
    scenarios = {s["scenario_id"]: s for s in _gather_accepted_scenarios(data_dir, axis, axes_cfg)}

    out_path = source_rows_path(data_dir, axis, source)
    rows = load_jsonl(out_path)
    reviews = load_source_reviews(data_dir, axis, source)

    # Group flagged conditions by scenario_id
    flagged_by_sid: dict[str, list[str]] = {}
    for key, status in reviews.items():
        if status != "rerun":
            continue
        sid, condition = key.split("__", 1)
        flagged_by_sid.setdefault(sid, []).append(condition)

    if not flagged_by_sid:
        log.info(f"[{axis}/{source}] no flagged rows; nothing to revise")
        return 0

    # Group rows by scenario_id
    rows_by_sid: dict[str, list[dict]] = {}
    for r in rows:
        rows_by_sid.setdefault(r["scenario_id"], []).append(r)

    log.header(f"[{axis}/{source}] revising {len(flagged_by_sid)} scenarios (concurrency={concurrency})")
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def with_progress(sid):
        nonlocal completed
        scenario = scenarios.get(sid)
        if not scenario:
            log.warning(f"  skipping {sid}: scenario no longer accepted")
            completed += 1
            return sid, rows_by_sid.get(sid, [])
        try:
            updated = await _revise_one(llm, scenario, source, rows_by_sid[sid], flagged_by_sid[sid], cfg, semaphore)
            completed += 1
            log.info(f"  [{completed}/{len(flagged_by_sid)}] {sid}")
            return sid, updated
        except Exception as exc:
            completed += 1
            log.warning(f"  [{completed}/{len(flagged_by_sid)}] {sid} FAILED: {exc}")
            return sid, rows_by_sid[sid]

    results = await asyncio.gather(*[with_progress(sid) for sid in flagged_by_sid])
    revised_by_sid = dict(results)
    merged: list[dict] = []
    for sid, group in rows_by_sid.items():
        merged.extend(revised_by_sid.get(sid, group))
    save_jsonl(merged, out_path)

    # Clear "rerun" status only for revised rows
    for sid in flagged_by_sid:
        for cond in flagged_by_sid[sid]:
            key = f"{sid}__{cond}"
            if reviews.get(key) == "rerun":
                del reviews[key]
    save_source_reviews(data_dir, axis, source, reviews)

    log.success(f"[{axis}/{source}] revised {len(flagged_by_sid)} scenarios")
    return len(flagged_by_sid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", required=True, help="Axis name or 'all'")
    parser.add_argument(
        "--source",
        required=True,
        choices=list(SOURCES) + ["all"],
        help="Source channel or 'all'",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Limit scenarios per axis (for pilots)")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Drop and regenerate all rows in source_{src}.jsonl (default: skip existing scenarios)",
    )
    parser.add_argument(
        "--revise",
        action="store_true",
        help="Instead of generating, re-render flagged-as-rerun rows from source reviews",
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    cfg = load_source_prompts()
    model = args.model or cfg["model"]
    data_dir = Path(args.data_dir)
    llm = AnthropicLLM(model=model)
    log.info(f"Model: {model}")

    axes = list(load_axes_config().keys()) if args.axis == "all" else [args.axis]
    sources = list(SOURCES) if args.source == "all" else [args.source]

    if args.revise:
        total_revised = 0
        for axis in axes:
            log.rule(f"Axis: {axis} (revision)")
            for source in sources:
                total_revised += asyncio.run(run_revision_for_axis(llm, axis, source, cfg, data_dir, args.concurrency))
        log.success(f"Total scenarios revised: {total_revised}")
        return

    total_new = 0
    for axis in axes:
        log.rule(f"Axis: {axis}")
        for source in sources:
            rows = asyncio.run(
                run_source_for_axis(
                    llm,
                    axis,
                    source,
                    cfg,
                    data_dir,
                    args.concurrency,
                    args.limit,
                    args.regenerate,
                )
            )
            total_new += len(rows)

    log.success(f"Total rows now in source files: {total_new}")


if __name__ == "__main__":
    main()
