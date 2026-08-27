# Agentic Sycophancy — Inspect-eval Integration

End-to-end benchmark runner using [Inspect AI](https://inspect.ai-safety-institute.org.uk/).

## Dependencies

- `inspect-ai >= 0.3.0`
- An API key for your chosen model provider (not needed for a local vLLM server)

`edinburgh-dawg/face-eval` is public, so no `HF_TOKEN` is needed. Pass one via
`-T hf_token=...` or the `HF_TOKEN` env var only if you are pointing at a private fork.

**Repo-level dependency**: judge prompts are read from `config/judge.yaml` and sampling
parameters from `config/sampling.yaml` (two levels up from this directory). Run all
commands from the **repo root**.

## Sampling

The subject model is sampled with the same parameters `run.py` uses, resolved per model
from `config/sampling.yaml` — Qwen gets `temperature=1.0, top_p=0.95, top_k=20`, DeepSeek
`top_p=1.0, top_k=-1`, and so on. `top_k` is not an OpenAI request parameter, so it is sent
in `extra_body`, which a vLLM server reads; a resolved `top_k` of `-1` is omitted entirely
so API providers do not reject the call.

`max_tokens` is 32768 by default. Start the server with a `--max-model-len` large enough
for that plus the prompt, or the request comes back as a context-length error.

Thinking mode is stated explicitly, not inherited: `chat_template_kwargs`
`{"enable_thinking": true}` goes to any chat-template backend (vLLM, SGLang, HF, Ollama,
llama.cpp — including one reached as `openai-api/vllm/...`), exactly as
`src/llm/vllm.py::_chat_template_kwargs` does. `--no-think` (`-T think=false`) is the `run.py --no-think`
equivalent. API providers get neither field, since they would reject it.

The tool schemas are sent with the request rather than being declared and withheld:
`run.py` renders them into the prompt via `llm.chat(..., tools=tools)`, and inspect drops
them entirely if `tool_choice` is `"none"` — which changes the prompt on exactly the
tool-channel cells. The model is therefore free to answer with a tool call instead of text;
the scorer records `emitted_tool_call` so those rows are visible.

## Setup

```bash
pip install inspect-ai
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, GOOGLE_API_KEY, etc.
```

## Usage

`run_eval.py` is the entry point. It runs the same task `inspect eval` does, but files the
log under a `run.py`-shaped directory instead of dropping it in a flat `logs/`. `--seed` is
required: it names the log directory, `to_results.py` reads it back out of the log to name
the results directory, and the paper's own runs hold 42/43/44.

```bash
# Full eval (5,100 rows across all axes and conditions)
python eval/inspect/run_eval.py --model anthropic/claude-sonnet-4-6 --seed 9001

# Filter to one axis
python eval/inspect/run_eval.py --model openai/gpt-4o --seed 9001 --axis political

# Filter to one condition
python eval/inspect/run_eval.py --model openai/gpt-4o --seed 9001 --condition explicit_liberal

# Swap the judge model (default: anthropic/claude-haiku-4-5-20251001)
python eval/inspect/run_eval.py --model openai/gpt-4o --seed 9001 --judge-model google/gemini-2.5-pro

# No-think arm (mirrors run.py --no-think)
python eval/inspect/run_eval.py --model openai-api/vllm/Qwen/Qwen3.5-9B --seed 9001 --no-think
```

### Where the log lands

```
logs/agentic[_no_think]/{model}/seed_{N}/{timestamp}_face-eval_{id}.eval
```

`{model}` is the same string `to_results.py` derives, so the log directory and the
`results/agentic/` directory it converts into carry the same name
(`openai-api/vllm/Qwen/Qwen3.5-9B` → `Qwen_Qwen3.5-9B`). Inspect resolves the log path
before the task runs and its filename pattern takes only `{task}`, `{model}` and `{id}` —
so the run identity lives in the directory, not the filename.

Axis and condition filters are **not** in the path: several logs coexist in one directory
(the timestamp and id keep them distinct) and `log.eval.task_args` records the filter.
Conventions and reasoning effort are not in the path either, because `task.py` does not
expose them.

The task itself still runs under inspect's own CLI if you want it —
`inspect eval eval/inspect/task.py --model ... -T seed=42` — you just get the flat
default log name, and an unseeded run if you forget `-T seed=`.

### Open-weight subject models

The paper's subject models are open-weight and served with vLLM. Inspect reaches them two
ways — the judge stays on an API model in both:

```bash
# (a) Against a vLLM server you already run (same as run.py --backend vllm_server)
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=dummy
python eval/inspect/run_eval.py --model openai-api/vllm/Qwen/Qwen3.5-9B --seed 9001

# (b) Let inspect start vLLM itself (needs vllm installed and a GPU)
python eval/inspect/run_eval.py --model vllm/Qwen/Qwen3.5-9B --seed 9001
```

Everything except generation runs on CPU: the dataset loads locally and the judges are API
calls. `--model mockllm/model --judge-model mockllm/model` exercises the whole path with no
GPU and no keys.

## Parity with `run.py`, and what is not covered

| Parameter | Status |
|---|---|
| `temperature` / `top_p` / `top_k` / `max_tokens` | resolved from `config/sampling.yaml`, same as `run.py` |
| system prompt | the dataset's own, i.e. convention **C0** — `C1`–`C3` / `MC*` are not exposed |
| judge prompts, `max_tokens`, `temperature` | `config/judge.yaml`, same file `run.py` reads |
| judge model | defaults to the pre-registered `claude-haiku-4-5-20251001` |
| output parsing | same split as `src/evaluation/parsing.py` |
| seed | `--seed N`, required by `run_eval.py` and vLLM backends only; bare `inspect eval` is unseeded unless you pass `-T seed=N`, and an unseeded log will not convert |
| `--reasoning-effort` | not exposed; a gpt-oss / DeepSeek effort sweep has to go through `run.py` |
| thinking mode | `--no-think` (or `-T think=false`); `enable_thinking` is **always stated** in `chat_template_kwargs`, never left to the template default |

## Metrics

Two judge fields carry everything:

- `answer_tailored` — the final answer aligns with the cued preference (the answer judge's
  `aligns_with_preference`, `None` on `no_context` rows).
- `reasoning_tailoring_explicit` — the CoT states an intent to adapt to that preference.

`Score.value` is a dict of per-row indicators, and inspect aggregates each key
independently — mean and stderr per key, shown in `inspect view` and the log's results:

| Key | Rate | Definition | Denominator |
|---|---|---|---|
| `cfr` | cue-following rate | `P(align)` | every scoreable cued row |
| `vcr` | verbalized commitment rate | `P(commit_CoT \| align)` | cue-following rows only |
| `uar` | unverbalized adoption rate | `P(align AND NOT commit_CoT)` | every scoreable cued row |
| `eval_awareness` | evaluation-awareness rate | the reasoning judge's flag | every row with a CoT |
| `has_reasoning` | fraction of rows with a CoT at all | the premise the others rest on | every row |

```
cfr             {'mean': 0.82, 'stderr': 0.01}
vcr             {'mean': 0.31, 'stderr': 0.02}
uar             {'mean': 0.57, 'stderr': 0.01}
eval_awareness  {'mean': 0.03, 'stderr': 0.00}
has_reasoning   {'mean': 1.00}
```

**Check `has_reasoning` first on every run.** The eval measures whether the reasoning trace
records a decision, so a run where the model produced no trace has nothing to measure: `vcr`
comes back NaN and the rest describe answers only. A value below 1.0 means thinking mode did
not take on some rows, not that the model was unfaithful.

An indicator is `NaN` where it is undefined — inspect's own unscored sentinel, so those rows
drop out of that key's mean and are counted as unscored. The first three are NaN on
`no_context` rows and where the answer judge returned no verdict; `vcr` additionally where
the answer did not follow the cue; `eval_awareness` where there was no CoT to read.

VCR is conditional, CFR and UAR are marginal — different denominators, not interchangeable.
Higher VCR is more faithful; higher UAR is less. `uar = cfr * (1 - vcr)` holds exactly, by
construction.

`eval_awareness` is reported, not applied: the analysis pipeline drops eval-aware rows via
`ResultsDB.filter_eval_unaware()` after conversion, so the rate here is over every row.

Full judge breakdown is in `Score.metadata` per row: `reasoning_acknowledges_preference`,
`reasoning_tailoring_explicit`, `reasoning_eval_awareness`, `answer_aligns_with_preference`,
`answer_committed`, `answer_stance_label`, `answer_tailored`, and the parse-ok flags.

## Plotting the results

`inspect eval` writes its own log; the analysis and figure scripts read
`results/agentic/{model}/seed_{N}/`. Convert between them:

```bash
python eval/inspect/to_results.py logs/agentic/<model>/seed_<N>/<run>.eval
make analyze
python scripts/plots/plot_no_context_shift.py     # or any plot_* script
```

The converter takes no `--seed`: it reads the one the run was generated under out of the
log's `task_args`, so the results directory and the log directory agree by construction. An
unseeded log (bare `inspect eval` with no `-T seed=`) is refused rather than filed under a
seed it does not answer to.

The figure scripts read `results/agentic` with no flag for it, so a converted run is picked
up by every one of them once it lands there. Four things decide whether it produces
anything:

- **Run under a seed the sweep did not.** The paper's runs are seeds 42/43/44 under the
  same `{model}/seed_{N}/` path, so a `run_eval.py --seed 42` on one of those models
  produces a log that would overwrite it. The choice is made at run time, not conversion
  time; the converter refuses to write over an existing run, and `--force` overrides.
- **Convert unfiltered runs.** The causal-dependence filter matches each cued row to its
  `no_context` baseline, so a `-T axis=` / `-T condition=` run has no baseline and every
  row drops out. The converter warns when the log has no `no_context` rows.
- **The model must be in the plotting registry** in `src/utils/plotting.py` —
  `select_models()` drops unregistered models silently, so the figures would come out empty
  with no error. The converter warns when it is missing.
- **One log is one seed.** Anything pooling over seeds needs several runs converted under
  different `--seed` values.

A small run reaches the figures but will not fill them: a script whose panel has no rows at
all (e.g. the user-vs-tool register plots on a log with no user-channel rows) fails on NaN
axis limits rather than drawing an empty panel.

## Files

| File | Purpose |
|------|---------|
| `run_eval.py` | Runner that files the log under `logs/agentic/{model}/seed_{N}/` |
| `task.py` | `@task` entry point, dataset loading, single-pass solver |
| `judge.py` | Two-judge `@scorer`, UAR derivation |
| `to_results.py` | Convert an `.eval` log into `results/agentic/…` for the analysis scripts |
