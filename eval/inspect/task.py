"""Inspect-eval task for the agentic sycophancy benchmark.

Self-contained: no dependency on src/. Loads the dataset directly from
HuggingFace and reads judge prompts from config/judge.yaml
(relative to the repo root — see README.md).

Usage:
    # Full eval
    inspect eval eval/inspect/task.py --model anthropic/claude-sonnet-4-6

    # Filter by axis
    inspect eval eval/inspect/task.py --model openai/gpt-4o -T axis=political

    # Swap judge
    inspect eval eval/inspect/task.py --model openai/gpt-4o -T judge_model=google/gemini-2.5-pro

    # Seeded, as run.py --seeds does (vLLM backends only)
    inspect eval eval/inspect/task.py --model vllm/Qwen/Qwen3.5-9B -T seed=42

    # Open-weight subject model on a vLLM server (VLLM_BASE_URL / VLLM_API_KEY)
    inspect eval eval/inspect/task.py --model openai-api/vllm/Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import datasets
import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolCall, ToolDef, ToolParams

from eval.inspect.judge import face_eval_scorer

# ---------------------------------------------------------------------------
# Flat-to-inspect message conversion (inlined — no src/ dependency)
# ---------------------------------------------------------------------------


def _convert_flat_to_inspect_messages(messages: list[dict]) -> list[ChatMessage]:
    """Convert the dataset's flat tool_call messages to inspect ChatMessage objects.

    The dataset stores a pre-baked tool exchange as four flat rows: system, user, an
    assistant row whose `tool_call` names the function, and a tool row carrying the
    return. inspect needs those as typed messages with a matching tool-call id, so the
    ids are synthesised here (`call_0`, `call_1`, …) and the tool row is paired with the
    assistant row that precedes it.
    """
    converted: list[ChatMessage] = []
    call_counter = 0

    for msg in messages:
        role = msg["role"]
        content = msg.get("content") or ""
        tool_call = msg.get("tool_call")

        if not tool_call:
            if role == "system":
                converted.append(ChatMessageSystem(content=content))
            elif role == "assistant":
                converted.append(ChatMessageAssistant(content=content))
            else:
                converted.append(ChatMessageUser(content=content))
            continue

        if role == "assistant":
            call_id = f"call_{call_counter}"
            call_counter += 1
            converted.append(
                ChatMessageAssistant(
                    content=content,
                    tool_calls=[ToolCall(id=call_id, function=tool_call, arguments={})],
                )
            )
        elif role == "tool":
            converted.append(
                ChatMessageTool(
                    content=content,
                    tool_call_id=f"call_{max(0, call_counter - 1)}",
                    function=tool_call,
                )
            )

    return converted


def _condition_to_preferred_side(condition: str) -> str:
    """Preference label carried by a condition string, or "none" for the control.

    A condition is `{context_type prefix}_{source channel}_{side}`, so the side is the
    last token. Kept inline because this task deliberately has no `src/` dependency;
    the canonical copy is `src/evaluation/common.py::preferred_side` and the two must
    agree.
    """
    if condition == "no_context":
        return "none"
    return condition.rsplit("_", 1)[-1]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_hf_dataset(
    axis: str | None,
    condition: str | None,
    token: str | None,
) -> list[Sample]:
    ds = datasets.load_dataset(
        "edinburgh-dawg/face-eval",
        split="train",
        token=token or os.getenv("HF_TOKEN"),
    )
    if axis:
        ds = ds.filter(lambda x: x["axis"] == axis)
    if condition:
        ds = ds.filter(lambda x: x["condition"] == condition)

    samples = []
    for row in ds:
        input_msgs = _convert_flat_to_inspect_messages(row["messages"])

        # Parse tools JSON string → list
        tools: list[dict] = []
        if row.get("tools"):
            try:
                tools = json.loads(row["tools"])
            except (json.JSONDecodeError, TypeError):
                pass

        # user_turn conditions embed preference in the user message, not via tool calls.
        context_type = row.get("context_type", "")
        if context_type in ("user_turn", "user_turn_structured", "user_turn_implicit"):
            tools = []

        preferred_side = row.get("preferred_side") or _condition_to_preferred_side(row.get("condition", ""))

        samples.append(
            Sample(
                id=row["id"],
                input=input_msgs,
                metadata={
                    "axis": row.get("axis", ""),
                    "condition": row.get("condition", ""),
                    "context_type": context_type,
                    "scenario_id": row.get("scenario_id", ""),
                    "question": row.get("question", ""),
                    "preferred_side": preferred_side,
                    "tools": tools,
                },
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------

# Path to the sampling config relative to the repo root (two levels up).
_SAMPLING_CONFIG = Path(__file__).parent.parent.parent / "config" / "sampling.yaml"


@lru_cache(maxsize=None)
def _sampling_for(model: str) -> tuple[float, float, int, int]:
    """Per-model sampling parameters, matching what run.py sends.

    Same resolution as `src.utils.sampling.resolve_sampling_params`: global defaults, then
    the first model pattern whose key appears in the model name. Inlined because this task
    deliberately has no `src/` dependency, and the two must agree — without it the eval
    would silently sample at the provider's defaults instead of the paper's.

    Inspect model strings carry a provider prefix (`openai-api/vllm/Qwen/Qwen3.5-9B`), but
    matching is on substrings, so the same pattern wins either way.
    """
    config = yaml.safe_load(_SAMPLING_CONFIG.read_text())
    params: dict[str, Any] = dict(config.get("defaults", {}))
    model_lower = model.lower()
    for pattern, overrides in config.get("models", {}).items():
        if pattern.lower() in model_lower:
            params.update(overrides)
            break
    return params["temperature"], params["top_p"], params["top_k"], params["max_tokens"]


# Providers that render a chat template locally and accept chat_template_kwargs. An
# `openai-api/<service>/<model>` string names the service in its second segment, which is
# how a vLLM or SGLang server reached over the OpenAI-compatible API is identified.
_TEMPLATE_BACKENDS = frozenset({"vllm", "vllm-completions", "sglang", "hf", "ollama", "llama-cpp-python"})


def _serves_chat_template(model: str) -> bool:
    parts = model.split("/")
    if parts[0] in _TEMPLATE_BACKENDS:
        return True
    return parts[0].startswith("openai-api") and len(parts) > 1 and parts[1] in _TEMPLATE_BACKENDS


def _sampling_kwargs(model: str, think: bool = True) -> dict[str, Any]:
    """Sampling parameters as inspect GenerateConfig kwargs."""
    temperature, top_p, top_k, max_tokens = _sampling_for(model)
    kwargs: dict[str, Any] = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}

    # Neither of these is an OpenAI request parameter, so inspect never puts them in the
    # body itself; a vLLM server reads both out of extra_body.
    extra: dict[str, Any] = {}
    # A negative top_k means "disabled", which is also what omitting it does — and omitting
    # keeps API providers from rejecting the field.
    if top_k > 0:
        extra["top_k"] = top_k
    if _serves_chat_template(model):
        # run.py always states this (src/llm/vllm.py::_chat_template_kwargs) rather than
        # relying on the template's default. The whole measurement is about the CoT, so
        # whether the model produces one cannot be left to a template we do not control.
        extra["chat_template_kwargs"] = {"enable_thinking": think}
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


# ---------------------------------------------------------------------------
# Solver — single-pass, no agent loop
# ---------------------------------------------------------------------------


async def _unused_tool() -> str:
    """Body for a declared-but-never-called tool. Generation runs with tool_calls="none"."""
    return ""


def _raw_schemas_to_tool_defs(schemas: list[dict]) -> list[ToolDef]:
    """Convert OpenAI-format tool schema dicts to inspect-ai ToolDef objects.

    `TaskState.tools` takes `Tool | ToolDef`, so a schema alone is not enough — each
    definition needs a callable. `_unused_tool` is that callable and is never invoked.
    """
    defs = []
    for s in schemas:
        fn = s.get("function", s)  # accept both {type, function: {...}} and bare {...}
        defs.append(
            ToolDef(
                tool=_unused_tool,
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                parameters=ToolParams(**fn.get("parameters", {})),
            )
        )
    return defs


@solver
def single_pass_solver(seed: int | None = None, think: bool = True) -> Solver:
    """Single generate call, with the tool schemas in the prompt.

    The pipeline is single-pass: the tool exchange is already baked into the messages.
    The schemas still have to be declared, because `run.py` renders them into the prompt
    via `llm.chat(..., tools=tools)` and the model's chat template lays them out in the
    preamble. Dropping them changes the prompt the model sees on exactly the tool-channel
    cells, which measurably shifts VCR — so tool_choice is left alone even though that
    leaves the model free to open a new call. `tool_calls='none'` still stops inspect from
    resolving one, and the scorer records `emitted_tool_call` so such a row is visible
    rather than silently unscored.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sampling = _sampling_kwargs(str(state.model), think=think)
        if seed is not None:
            sampling["seed"] = seed
        raw_schemas = state.metadata.get("tools", [])
        if raw_schemas:
            state.tools = _raw_schemas_to_tool_defs(raw_schemas)
            state = await generate(state, tool_calls="none", **sampling)
        else:
            state = await generate(state, **sampling)
        return state

    return solve


# ---------------------------------------------------------------------------
# Task entry point
# ---------------------------------------------------------------------------


@task
def face_eval(
    axis: str | None = None,
    condition: str | None = None,
    judge_model: str = "anthropic/claude-haiku-4-5-20251001",
    seed: int | None = None,
    think: bool = True,
    hf_token: str | None = None,
) -> Task:
    """Agentic sycophancy benchmark task.

    Args:
        axis: Filter to one axis (political, ethics, egalitarianism,
              epistemic-posture, domain-expertise). None = all axes.
        condition: Filter to one condition, e.g. explicit_liberal,
                   user_turn_structured_novice, no_context. None = all conditions.
        judge_model: Inspect model string for both judges. Defaults to the
                     pre-registered judge every result on disk was scored with.
        seed: Sampling seed, as run.py's --seeds takes. Reaches vLLM only —
              the API backends have no seed parameter. None leaves generation
              unseeded, which run.py never does.
        think: Thinking mode, the inverse of run.py's --no-think. Sent as
               chat_template_kwargs to chat-template backends. The eval measures
               the CoT, so a run with think=False scores no reasoning at all.
        hf_token: HuggingFace token. Falls back to HF_TOKEN env var.
    """
    samples = _load_hf_dataset(axis=axis, condition=condition, token=hf_token)
    dataset = MemoryDataset(samples=samples, name="agentic-sycophancy")

    return Task(
        dataset=dataset,
        solver=[single_pass_solver(seed=seed, think=think)],
        scorer=face_eval_scorer(judge_model=judge_model),
    )
