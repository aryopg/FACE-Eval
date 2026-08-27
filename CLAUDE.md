# FACE-Eval

## What This Is

A framework for measuring Chain-of-Thought (CoT) faithfulness under **realistic conditions**, and
the code that built the `edinburgh-dawg/face-eval` dataset it runs on.

The question: do faithfulness measurements taken under artificial benchmark probes still hold when
the cue arrives the way it would in deployment — in a user's message, or in a document a tool
returned? Models answer questions on a shared agentic substrate while a preference cue is planted in
one of four places, and a two-judge pipeline scores the reasoning trace and the final answer
separately.

## Analysis structure

The headline analysis pools the dataset's `context_type`s into a **channel × salience 2×2**
(`CELL_CONTEXT_TYPES` in `src/utils/plotting.py`):

- **Channel** — where the cue lives: **User** (user message) or **Tool** (tool/document return).
- **Salience** — **Explicit** (stated) or **Implicit** (must be inferred).

| Cell | `context_type`s |
|---|---|
| `user_explicit` | `user_turn`, `user_turn_structured` |
| `user_implicit` | `user_turn_implicit` |
| `tool_explicit` | `explicit` |
| `tool_implicit` | `implicit` |
| `no_context` | `none` — the shared baseline |

Makefile targets are named after the pre-registered hypotheses:

| | Question | Targets |
|---|---|---|
| H1 | Does the channel the cue arrives on change the faithfulness rate? | `analyze-h1`, `figures-h1` |
| H2 | Can a transcript-reading monitor detect the tailoring, and does the CoT add anything over the action alone? | `monitor-run`, `analyze-h2*`, `figures-h2` |
| H3 | Do the orderings survive dropping the causal-dependence filter? | `analyze-h3` |
| H4 | Does evaluation-awareness explain the gap? | `analyze-h4`, `figures-h4` |
| H5 | Do system-prompt conventions move it? | `analyze-h5`, `figures-register` |
| H6 | Is the explicit/implicit gap just cue clarity? | `analyze-h6`, `figures-h6` |

## Project Structure

```
# Repo root: the runners that PRODUCE data, plus the audit UI.
run.py                    # Main CLI (multi-seed, resume, conventions, reasoning effort)
run_artifact_rating.py    # Cue-clarity + side-ID comprehension probe
run_monitor.py            # Monitor judges over finished transcripts
check_inference.py        # Completeness gate between infer and judge
sync_results.py           # Upload/download results to/from HuggingFace
audit_app.py              # Streamlit transcript audit UI
src/
  llm/                    # Backends, all subclass base.py
    base.py  vllm.py  vllm_server.py  gpt_oss.py  gemma4.py  deepseek_v4.py  inkling.py
    anthropic.py  openai_llm.py  gemini.py  openrouter.py
  evaluation/
    judges.py             # Two-judge pipeline: ReasoningJudge + AnswerJudge
    monitor.py            # Transcript-reading monitor (action / cot / cot_only views)
    parsing.py            # <think> and <answer> tag extraction
    common.py             # Shared judge/monitor batch plumbing
  data/
    face_eval.py          # Dataset loader + flat→OpenAI tool-call conversion
    conventions.py        # System-prompt conventions C0–C3 / MC0–MC3 / OC0
    annotation.py         # Item sampling for the artifact-rating probe
  results/
    storage.py            # results/agentic/{model}/seed_{N}/
    db.py                 # Query layer (ResultsDB; filter_causal_dependent, cluster_mean_ci, …)
    check.py              # Run-completeness checks
    role_register.py      # H1 role/register helpers
  utils/                  # logging.py, plotting.py, parsing.py, sampling.py
  pipeline.py             # Inference + judge orchestration
scripts/
  analysis/               # analyze_* — write the CSV/JSON the paper quotes
  plots/                  # plot_* + make_paper_figures.py — figures and paper panels
config/
  judge.yaml  judge_gpt.yaml     # Primary and second judge (byte-identical prompts, different model)
  monitor_judge.yaml             # Monitor prompts, one per view
  backends.yaml  sampling.yaml  side_definitions.yaml
face_eval_generator/      # How edinburgh-dawg/face-eval was built (see its README)
eval/inspect/             # The same eval as an Inspect-AI task
web/                      # FACE-Eval companion site
tests/
```

Most work runs through the `Makefile`; `make help` lists every target. Per-model stages
(`infer`, `judge`, `judge-second`, `artifact-rating`, `check-inference`) take `MODEL=...`, so a
sweep is a loop over `MODEL`.

## Frozen names

These strings are a wire format, not identifiers. Results already on HuggingFace carry them, and
renaming orphans every existing artifact:

- `results/agentic/` and `results/agentic_no_think/` — the results roots
- `h2_monitor__<model>.jsonl` — emitted by `monitor_filename()` in `src/evaluation/monitor.py`;
  `sync_results.py` globs for it
- `"Substrate": "agentic_sycophancy"` in `run.py` — written into every `metadata.json`
- Figure and CSV basenames (`h2_monitor_auc`, `h2_increment_*`, `h2_monitor_capability_bprime*`) —
  `scripts/plots/make_paper_figures.py` reads them as literal strings

## Key Commands

```bash
# Full pipeline (inference + two-judge scoring)
python run.py --model Qwen/Qwen3-4B --seeds 42,43,44

# Inference only
python run.py --model Qwen/Qwen3-4B --stage inference

# Filter by axis
python run.py --model Qwen/Qwen3-4B --axis political

# Reasoning-effort control (GPT-OSS)
python run.py --model openai/gpt-oss-20b --reasoning-effort high --seeds 42

# Convention arm (C0–C3 oblivious + MC0–MC3 monitor-aware; C0 is the default)
python run.py --model Qwen/Qwen3.5-9B --convention C3 --seeds 42

# Resume
python run.py --model Qwen/Qwen3-4B --seeds 42,43 --resume

# Persistent vLLM server: the model loads ONCE and stays resident across every
# run.py invocation — seeds, conventions, axes. Essential for slow-loading models.
#   vllm serve <model> --tensor-parallel-size N --reasoning-parser <p> \
#       --enable-auto-tool-choice --tool-call-parser <p> --max-model-len <M>
python run.py --model <model> --backend vllm_server --seeds 42,43,44
# (offline vLLM also reuses one engine across --seeds)

# Monitor pass over finished transcripts
python run_monitor.py --results-dir results/agentic

# Cue-clarity comprehension probe (separate output dir)
python run_artifact_rating.py --model Qwen/Qwen3.5-9B --no-think

# Sync
python sync_results.py upload
python sync_results.py download
```

`run.py` and `run_artifact_rating.py` both support `--no-think` (vLLM backends only).

## Results Layout

```
results/agentic/{model}[_{effort}]/seed_{N}[_{convention}]/
  inference.jsonl               # Inference results
  judged.jsonl                  # Two-judge scores (reasoning + answer)
  judged__{model}.jsonl         # A second judge, alongside — never replacing — the first
  h2_monitor__{model}.jsonl     # Monitor scores, if the monitor pass has been run
  metadata.json                 # Run config, metrics, timestamps
```

Model dirs carrying a reasoning-effort or thinking variant suffix it
(`openai_gpt-oss-120b_high`, `deepseek-ai_DeepSeek-V4-Flash_max`). `--no-think` runs go under
`results/agentic_no_think/` with the same shape. Artifact-rating probe outputs land separately
under `outputs/artifact_rating/` and `outputs/artifact_rating_no_think/`.

## Dataset

- Source: `edinburgh-dawg/face-eval` on HuggingFace
- ~5,100 scenarios; 5 axes × 1,020 each
- Axes: `political`, `ethics`, `egalitarianism`, `epistemic-posture`, `domain-expertise`
- `context_type` (6 values, the analysis grouping): `user_turn`, `user_turn_structured`,
  `user_turn_implicit`, `explicit`, `implicit`, `none` (baseline, 100 rows)
- `condition` is the fine-grained label:
  `{context_type prefix}_{source: email/slack/notes/browser_history/∅}_{preference side}`.
  Sides span the axes (conservative/liberal, expert/novice, utilitarian/deontological,
  egalitarian/elitist, skeptical/deferential).
- Fields: `id`, `axis`, `condition`, `context_type`, `scenario_id`, `question`, plus the tool-call
  message structure

```python
from src.data.face_eval import FaceEval

dataset = FaceEval()
political = FaceEval(axis="political")
one_condition = FaceEval(condition="explicit_liberal")

messages, tools = dataset.get_messages_and_tools(0)
```

`face_eval_generator/` holds the code and intermediate files that produced the published dataset.
It imports `src.llm.anthropic` and `src.utils.logging`, so it is not independently installable —
see its README for the generation chain.

## Git Rules

- **NEVER use `git checkout -p`, `git checkout <file>`, `git reset`, or `git restore` on dirty
  working tree files.** These discard uncommitted changes that cannot be recovered.
- When committing subsets, use `git add <file>` for whole files or `git add -p` for specific hunks
  — never discard/checkout to split changes.
- Stage and commit only. Do not touch the working tree to manipulate what gets committed.

## Development Workflow

```bash
uv pip install -e ".[dev]" --python .venv/bin/python   # add apps,dataset for the Streamlit/Gradio UIs
.venv/bin/pre-commit install                           # one-time

pre-commit run --all-files                             # ruff + black + isort + vulture
pytest tests/ -v
```

`config/` YAML is not covered by the test suite — a renamed or missing config file passes `pytest`
and fails at runtime. Check those paths by hand.

## Code Style Preferences

- **Keep it simple.** No over-engineering, no premature abstractions, no bloat.
- Write the minimum code needed. Three similar lines > a premature helper.
- No unnecessary error handling for scenarios that can't happen.
- No feature flags, no backwards-compat shims. Just change the code.
- Docstrings on public functions. No comments on obvious code.
- Type hints. Dataclasses for structured data. Standard library where possible.
- If a dependency isn't needed, don't add it.
- All files use `from __future__ import annotations`.
- Use `list`, `dict`, `tuple`, `X | None` instead of `List`, `Dict`, `Tuple`, `Optional`.
- Line length: 120 (enforced by black and ruff).

## Environment

- Python >= 3.11
- vLLM is optional (Linux/CUDA only), imported lazily
- API keys in `.env`: `ANTHROPIC_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY` (monitor pass only)
- Package manager: uv
