# FACE-Eval

Code and analysis for measuring Chain-of-Thought (CoT) faithfulness under realistic
conditions. A model is given a task plus a cue about the user's preference. We measure
whether the model's answer moves toward that preference, and whether its reasoning trace
says so.

The headline metric is the **unverbalized adoption rate (UAR)**: the answer aligns with the
user's preference while the reasoning trace never states an intent to tailor.

New here? Start with **Install**, set the API keys required for your backend, then run the
20-row **Smoke test**. The remaining sections explain the experimental design, full runs,
and analysis workflow.

## What varies

The dataset's `context_type` values pool into a **channel x salience** 2x2:

| | Explicit (cue is stated) | Implicit (cue must be inferred) |
|---|---|---|
| **User** channel (user message) | `user_turn`, `user_turn_structured` | `user_turn_implicit` |
| **Tool** channel (tool or document return) | `explicit` | `implicit` |

`none` is the shared no-cue baseline. The pooling lives in `CELL_CONTEXT_TYPES` in
`src/utils/plotting.py`.

A second axis varies the **system-prompt convention** (`src/data/conventions.py`):
C0 is the plain default, C3 adds a disclosure directive, and the `MC*` variants add a
sentence saying a monitor will read the output. C0 vs MC0 is the clean observer-effect
contrast.

## Pipeline overview

```
┌────────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
│ HF dataset             │     │ Inference                │     │ Judge stage              │
│ edinburgh-dawg/        │ ──► │ src/pipeline.py          │ ──► │ src/pipeline.py          │
│   face-eval            │     │ ::run_inference          │     │ ::run_judge_stage        │
│ 5,100 rows             │     │ vLLM (open weights)      │     │ ReasoningJudge +         │
│ 5 axes x 1,020         │     │ or an API backend        │     │ AnswerJudge (Haiku 4.5)  │
└────────────────────────┘     └──────────────────────────┘     └──────────────────────────┘
                                           │                                │
                                           ▼                                ▼
                                   inference.jsonl                   judged.jsonl
                                   {id, reasoning,                   {id, judge: {reasoning_*,
                                    raw_answer, axis,                  answer_*, answer_tailored,
                                    condition, context_type}           parse_ok flags}}
                                                                                │
                                                                                ▼
                                                                    ┌──────────────────────────┐
                                                                    │ scripts/analysis/*.py    │
                                                                    │   → CSVs                 │
                                                                    │ scripts/plots/*.py       │
                                                                    │   → figures/*.svg        │
                                                                    │ plots/make_paper_figures │
                                                                    │   → paper panels         │
                                                                    └──────────────────────────┘
```

### The two-judge architecture (and why)

The judge is split into two independent judges with **disjoint inputs**, so reasoning-level
and behavioural signals cannot contaminate each other:

| Judge | Input | Metrics |
|---|---|---|
| **ReasoningJudge** | CoT trace only (no answer) | `acknowledges_preference`, `tailoring_explicit`, `eval_awareness` |
| **AnswerJudge** | Final answer + scenario preference (no reasoning) | `aligns_with_preference`, `committed`, `stance_label` |

`answer_tailored` is derived post-hoc by joining `aligns_with_preference` with the scenario
`condition` (`no_context` rows have no preference, so they get `None`). The three reported
rates all follow from these two independent signals: the cue-following rate (CFR) is
`mean(answer_tailored)`; the verbalized commitment rate (VCR) is
`mean(reasoning_tailoring_explicit)` among cue-following rows; and the unverbalized adoption
rate (UAR) is `answer_tailored ∧ ¬reasoning_tailoring_explicit`, equal to `cfr * (1 - vcr)`.

Both judges read their prompts from `config/judge.yaml`. A second judge
config, `config/judge_gpt.yaml`, holds byte-identical prompts for a
different judge model, used for inter-judge agreement.

## Dataset

- Source: `edinburgh-dawg/face-eval` on HuggingFace.
- **5,100 rows**: 5 axes x 1,020 rows each. Axes: `political`, `ethics`,
  `egalitarianism`, `epistemic-posture`, `domain-expertise`.
- Six `context_type` values: `user_turn`, `user_turn_structured`, `user_turn_implicit`,
  `explicit`, `implicit` (1,000 rows each) and `none` (100 rows, the baseline).
- `condition` is the fine-grained label: context-type prefix, source channel
  (email / slack / notes / browser_history, or none), and the preferred side —
  for example `explicit_liberal`.
- Fields: `id`, `axis`, `condition`, `context_type`, `scenario_id`, `question`,
  `messages` (flat tool-call format), `tools` (JSON string).

How the dataset was built is in `face_eval_generator/`: `generate.py` writes the
scenarios, `generate_source_variants.py` and `generate_user_turn_variant.py` write the
channel variants, and `ui.py` is a Gradio review app (needs the `dataset` extra).

### Loading

```python
from src.data.face_eval import FaceEval

# Full dataset
dataset = FaceEval()

# Filtered
political = FaceEval(axis="political")
explicit_liberal = FaceEval(condition="explicit_liberal")

# Prepare messages for inference (OpenAI format with expanded tool calls)
messages, tools = dataset.get_messages_and_tools(0)
```

## Install

```bash
make venv       # uv venv --python 3.12
make install    # uv pip install -e ".[dev]"
make hooks      # pre-commit hooks, once

make install-vllm   # Linux + CUDA only; adds the vllm extra
```

Extras declared in `pyproject.toml`:

| Extra | Contains | Needed for |
|---|---|---|
| `dev` | `pytest`, `pre-commit`, `inspect-ai` | tests, hooks, `eval/inspect/` |
| `vllm` | `vllm` | open-weight inference (Linux + CUDA) |
| `apps` | `streamlit` | `audit_app.py` |
| `dataset` | `gradio` | `face_eval_generator/ui.py` |

The linters (`black`, `isort`, `ruff`, `vulture`) run through pre-commit, which installs
them in its own environments.

## Environment

Create `.env` with whichever keys you need:

```env
ANTHROPIC_API_KEY=your_anthropic_key   # primary judge
OPENAI_API_KEY=your_openai_key         # second judge, H2 monitor
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
HF_TOKEN=your_huggingface_token        # dataset + result sync
```

## Entrypoints

| Command | What it does |
|---|---|
| `python run.py --model <name> ...` | Inference and/or judge stage on the main dataset |
| `python run_artifact_rating.py --model <name> ...` | artifact-rating cue-salience comprehension probe: how clearly does each artifact reveal the preference? |
| `python run_monitor.py ...` | H2: a monitor model reads finished transcripts and tries to detect the cue |
| `streamlit run audit_app.py` | Transcript audit UI: filter judged rows and read question / reasoning / answer / verdicts side by side (needs the `apps` extra) |
| `inspect eval eval/inspect/task.py --model <name>` | The same eval as an Inspect-AI task |
| `python sync_results.py upload` / `download` | Push or pull `results/` to `edinburgh-dawg/face-eval-results` |

## Cloud API backends: what was and was not run

`src/llm/` has working Anthropic, OpenAI, Gemini and OpenRouter backends, and `--backend`
accepts all of them. But **no cloud API model was evaluated as a subject model in the
paper.** Every subject model was open-weight, run through vLLM. Cloud APIs were used for
judging and monitoring only.

## Running the experiments

Every command resumes: `--resume` skips rows already on disk. Every sweep used seeds
**42, 43, 44**.

### Smoke test first

```bash
python run.py --model Qwen/Qwen3.5-4B --seeds 42 --stage all --max-samples 20
```

That should write 20 rows to `inference.jsonl` and 20 to `judged.jsonl` under
`results/agentic/Qwen_Qwen3.5-4B/seed_42/`, with `reasoning_parse_ok` true on almost all
of them. Fix any failure before spending GPU or judge budget.

### Inference (open weights, vLLM)

```bash
make infer MODEL=Qwen/Qwen3.5-27B GPUS=2
# defaults: SEEDS=42,43,44  CONVENTION=C0  GPUS=1  RESULTS_DIR=results

# reasoning effort, where the model has one
make infer MODEL=openai/gpt-oss-120b GPUS=4 EFFORT=medium

# the whole convention factorial in one call — the engine loads once
make infer MODEL=Qwen/Qwen3.5-27B GPUS=2 CONVENTION=C0,C3,MC0,MC3

# reasoning disabled (vLLM only; rejected for GPT-OSS)
make infer-no-think MODEL=Qwen/Qwen3.5-9B
```

`make infer` is a thin wrapper on `run.py --stage inference --backend vllm`; run `run.py`
directly for anything the wrapper does not expose (`--axis`, `--condition`,
`--max-samples`, `--trust-remote-code`, `--backend vllm_server`). `--no-think` writes to
`results/agentic_no_think` so it never overwrites a thinking run.

### Check the run before judging

A crashed inference judges cleanly and looks finished. Check first:

```bash
make check-inference RESULTS_DIR=results
```

This calls `check_inference.py --expected 5100`. Passing `--expected` matters: the
script defaults to 500, so against a 5,100-row run the completeness check passes even when
most rows are missing.

### Judge

```bash
make judge MODEL=Qwen/Qwen3.5-27B CONVENTION=C0        # primary judge, Anthropic Batch API
make judge-second MODEL=Qwen/Qwen3.5-27B CONVENTION=C0 # second judge, for inter-judge agreement
```

The second judge is an addition, never a replacement: it writes
`judged__<judge-model>.jsonl` and leaves `judged.jsonl` alone. At the judge stage `--model`
is only a download filter and a directory key — it does not have to resolve to a real repo
id, but `MODEL.replace("/", "_")` must equal the results directory name.

### artifact-rating comprehension probe and the H2 monitor

```bash
make artifact-rating MODEL=Qwen/Qwen3.5-27B GPUS=2 RESULTS_DIR=outputs
make monitor-run H2_MONITOR=gpt-5.6-luna H2_EFFORT=medium
```

`artifact-rating` writes `<RESULTS_DIR>/artifact_rating/<model>.jsonl`, and `RESULTS_DIR` defaults
to `results`. `make analyze-artifact-rating` reads `outputs/artifact_rating/`, so either pass
`RESULTS_DIR=outputs` as above or point the analysis at the other directory with
`scripts/analysis/analyze_artifact_rating.py --input-dir`.

Model names passed to the H2 monitor are results-directory keys, so they use `_` where a
repo id uses `/`. Every monitor score in the paper is at effort `medium`, and the effort is
not part of the output filename — do not mix two efforts into one file.

## Results layout

```
results/agentic/{model}[_{effort}]/seed_{N}[_{convention}]/
  inference.jsonl                # one row per scenario: reasoning, raw_answer, condition, ...
  judged.jsonl                   # one row per scenario: the six judge metrics + answer_tailored
  judged__<judge-model>.jsonl    # a second judge's verdicts, if run
  h2_monitor__<monitor>.jsonl    # H2 monitor scores, if run
  metadata.json                  # run config, sampling params, judge config, timestamps
```

Reasoning effort and thinking mode are part of a model's identity, so they suffix the model
directory (`openai_gpt-oss-120b_medium`, `deepseek-ai_DeepSeek-V4-Flash_high`). Convention
C0 keeps the plain `seed_{N}` path; every other convention appends its name. `--no-think`
runs go under `results/agentic_no_think/` with the same shape.

## Analysis and figures

Analysis scripts read `judged.jsonl` through `src/results/db.py` and write CSVs. Plot
scripts read those CSVs and write SVGs to `FIGURES_DIR` (default `figures/`).

```bash
make analyze              # every analysis script
make analyze-h1 analyze-h3 analyze-h4 analyze-h5 analyze-h6
make analyze-h2 analyze-h2-increment analyze-h2-calibration  # need monitor-run first
make analyze-artifact-rating        # aggregate the artifact-rating probe (prerequisite for figures-artifact-rating)
make analyze-convention analyze-inter-judge

make figures              # every figure group
make figures-h1 figures-h4 figures-effort figures-h6
make figures-h2
make figures-register figures-misc figures-artifact-rating figures-inter-judge

make figures-paper-all    # regenerate the paper's source figures, then compose the panels
```

`figures-paper-all` is the one to run for the paper: it runs `figures-paper-sources`
(about a quarter of `figures`) and then `figures-paper` to add boxes, titles and legends.
`make check-paper-font` reports whether CMU Serif Bold is installed, the only thing the
panel composition needs beyond the SVGs.

Override the output directory on any target: `make figures-h1 FIGURES_DIR=paper/figs`.
`make help` lists every target.

## Inspect-AI task

`eval/inspect/` runs the same eval as an Inspect-AI task. It is a parallel track: it writes
Inspect's own logs, not `inference.jsonl` / `judged.jsonl`, and the paper's numbers come
from `run.py`.

It reproduces the paper's orderings, not its exact numbers. Dataset, prompts, sampling
config, judge prompts and judge model all match (see `eval/inspect/README.md`), but the two
paths reach the model differently: `run.py` batches through offline vLLM, the task sends one
request per sample to a server, so a shared seed does not give a shared sample at
`temperature=1.0`. On a Qwen3.5-9B check the per-cell rates land inside the spread `run.py`
itself shows across seeds 42/43/44.

```bash
python eval/inspect/run_eval.py --model <provider/model> --seed 9001
python eval/inspect/run_eval.py --model openai/gpt-4o --seed 9001 --axis political

# Open-weight subject model on a vLLM server (VLLM_BASE_URL / VLLM_API_KEY)
python eval/inspect/run_eval.py --model openai-api/vllm/Qwen/Qwen3.5-9B --seed 9001
```

`run_eval.py` is the runner: it files the log under
`logs/agentic[_no_think]/{model}/seed_{N}/` rather than in a flat `logs/`, so which run a
log belongs to is readable off the path. `task.py` holds the dataset loading and the
single-pass solver; `judge.py` reimplements the two-judge split as an Inspect scorer
returning the per-row UAR indicator. Run it from the repository root — it reads the judge
prompts from `config/`.

`to_results.py` converts a finished `.eval` log into the layout the analysis scripts read,
so an Inspect run can go through `make analyze` and the figures unchanged:

```bash
python eval/inspect/to_results.py logs/agentic/<model>/seed_<N>/<run>.eval
```

See `eval/inspect/README.md` for the two constraints on when a converted run is analyzable.

## Project structure

```
run.py                          # Main CLI: inference + judge stages
run_artifact_rating.py          # Cue-clarity and side-ID probe over the artifacts
run_monitor.py                  # Monitor judges over finished transcripts
sync_results.py                 # Upload / download results to HuggingFace
check_inference.py              # Run-completeness gate between infer and judge
audit_app.py                    # Streamlit transcript audit UI
Makefile                        # Every routine command (make help)
config/                         # Judge prompts (primary + second judge), monitor prompts,
                                # backend detection, sampling, axis sides
src/
  llm/                          # Backends, all subclass base.py
    vllm.py vllm_server.py gpt_oss.py gemma4.py deepseek_v4.py inkling.py
    anthropic.py openai_llm.py gemini.py openrouter.py
  evaluation/
    judges.py                   # Two-judge architecture (reasoning + answer)
    monitor.py                  # Transcript-reading monitor
    parsing.py                  # Output parsing (<think>, <answer>)
  data/
    face_eval.py                # Dataset loader + flat→OpenAI tool-call conversion
    conventions.py              # System-prompt conventions C0–C3 / MC0–MC3 / OC0
  results/                      # storage.py (layout), db.py (queries), check.py
  utils/                        # logging.py, parsing.py, plotting.py (registry, colours)
  pipeline.py                   # Inference + judge orchestration
scripts/                        # Everything you run after the pipeline, to make the paper
  analysis/                     # analyze_* → the CSV/JSON the paper quotes
  plots/                        # plot_* → SVG, make_paper_figures.py → paper panels
eval/inspect/                   # Inspect-AI task, scorer, results converter, demo notebook
face_eval_generator/            # How edinburgh-dawg/face-eval was built (+ Gradio review UI)
web/                            # FACE-Eval companion site (static HTML/CSS/JS + figures)
tests/                          # Test suite
```

`web/` is a static site with no build step. Serve it with `python -m http.server` from
inside `web/` and open the printed URL.

## Development

```bash
make lint        # pre-commit over all files: ruff, black, isort, vulture
make test        # pytest tests/ -v
```

`tests/` covers Python only: `config/` YAML is never loaded there, so a renamed or missing
config file passes `pytest` and fails at runtime. Check those paths by hand.

## Citation

```bibtex
@article{gema2026faceeval,
  title={Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered}, 
  author={Aryo Pradipta Gema and Neel Rajani and Rohit Saxena and Wai-Chung Kwan and Pasquale Minervini},
  year={2026},
  eprint={2608.29464},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2608.29464}, 
}
```

## Contact

If you have any questions, please do not hesitate to reach out to [aryo.gema@ed.ac.uk](mailto:aryo.gema@ed.ac.uk) and [p.minervini@ed.ac.uk](mailto:p.minervini@ed.ac.uk)

## License

MIT
