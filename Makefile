.DEFAULT_GOAL := help
PYTHON := .venv/bin/python
FIGURES_DIR ?= figures
# Every analyze/plot script takes --figures-dir. Passing it once as $(FIG)
# keeps FIGURES_DIR working end to end.
FIG := --figures-dir $(FIGURES_DIR)
# Second judge for the inter-judge agreement check; override per comparison.
SECOND_JUDGE ?= gpt-5.6-luna
# Convention the agreement is measured under. ALL pools; C0 speaks about the
# headline figures specifically.
JUDGE_CONVENTION ?= ALL

.PHONY: help venv install install-vllm hooks lint test analyze-h1 analyze-h3 analyze-h4 analyze-h5 \
 analyze-h6 analyze-artifact-rating analyze-convention analyze-h2 analyze-h2-increment \
 analyze-h2-calibration analyze-inter-judge figures-inter-judge analyze figures-h1 \
 figures-h4 figures-effort figures-h6 figures-register figures-misc figures-h2 \
 figures-artifact-rating figures check-paper-font figures-paper figures-paper-sources \
 figures-paper-all infer infer-no-think check-inference judge judge-second artifact-rating \
 monitor-run

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
 awk 'BEGIN {FS = ":.*## "}; {printf "  %-22s %s\n", $$1, $$2}'

venv: ## Create .venv (Python 3.12)
	uv venv --python 3.12

install: ## Install package with dev extras
	uv pip install -e ".[dev]" --python $(PYTHON)

install-vllm: ## Install with vllm extras (Linux/CUDA only)
	UV_HTTP_TIMEOUT=300 uv pip install -e ".[dev,vllm]" \
		--python $(PYTHON) \
		--extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match

hooks: ## Install pre-commit hooks (run once after venv + install)
	$(PYTHON) -m pre_commit install

lint: ## Run linters and formatters via pre-commit
	$(PYTHON) -m pre_commit run --all-files

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

# ---------------------------------------------------------------------------
# Analysis scripts (write CSVs / data files)
# ---------------------------------------------------------------------------

analyze-h1: ## Run H1 role×register analysis (writes CSV to results/)
	$(PYTHON) scripts/analysis/analyze_h1_role_register.py

analyze-h3: ## Run H3 filter-stability analysis
	$(PYTHON) scripts/analysis/analyze_h3_filter_stability.py $(FIG)

analyze-h4: ## Write H4 eval-awareness CSV
	$(PYTHON) scripts/analysis/analyze_h4_eval_awareness.py $(FIG)

analyze-h5: ## Write H5 convention delta-of-deltas CSV
	$(PYTHON) scripts/analysis/analyze_h5_convention_power.py $(FIG)

analyze-h6: ## Write H6 clarity-matched gap and per-scenario CSVs
	$(PYTHON) scripts/analysis/analyze_h6_clarity_matched.py $(FIG)

analyze-artifact-rating: ## Aggregate artifact-rating annotation outputs (prerequisite for figures-artifact-rating)
	$(PYTHON) scripts/analysis/analyze_artifact_rating.py

analyze-convention: ## Run convention-backfire analysis (writes to figures/)
	$(PYTHON) scripts/analysis/analyze_convention_backfire.py $(FIG)

analyze-h2: ## Write H2 monitor-detection AUC CSV (requires monitor-run)
	$(PYTHON) scripts/analysis/analyze_monitor_detection.py $(FIG)

analyze-h2-increment: ## Write H2 CoT-action increment + VCR CSVs (both monitor models)
	$(PYTHON) scripts/analysis/analyze_monitor_increment.py $(FIG)

analyze-h2-calibration: ## Write H2 monitor calibration + >=70 operating-point CSVs
	$(PYTHON) scripts/analysis/analyze_h2_calibration.py $(FIG)

analyze-inter-judge: ## Write inter-judge agreement + attrition CSVs (requires a second judge's judged_*.jsonl)
	$(PYTHON) scripts/analysis/analyze_inter_judge_agreement.py --judge-model $(SECOND_JUDGE) \
		--convention $(JUDGE_CONVENTION) $(FIG)

figures-inter-judge: ## Plot the inter-judge scatters, both metrics (requires analyze-inter-judge)
	$(PYTHON) scripts/plots/plot_inter_judge_agreement.py --metric vcr --judge-model $(SECOND_JUDGE) \
		--convention $(JUDGE_CONVENTION) $(FIG)
	$(PYTHON) scripts/plots/plot_inter_judge_agreement.py --metric uar --judge-model $(SECOND_JUDGE) \
		--convention $(JUDGE_CONVENTION) $(FIG)

# analyze-artifact-rating writes outputs/artifact_rating_aggregated.jsonl, which analyze-h6 reads,
# so it has to come first: make stops at the first failure, and on a box without
# that file the old order died at h6 before ever producing it.
analyze: analyze-h1 analyze-h3 analyze-h4 analyze-h5 analyze-artifact-rating analyze-h6 analyze-convention ## Run all analysis scripts

# ---------------------------------------------------------------------------
# Figure scripts (read CSVs / DB, write SVGs to FIGURES_DIR)
# ---------------------------------------------------------------------------

figures-h1: ## Plot all H1 figures
	$(PYTHON) scripts/plots/plot_h1_channel_asymmetry.py $(FIG)
	$(PYTHON) scripts/plots/plot_h1_phase_diagram.py $(FIG)

figures-h4: ## Plot H4 eval-awareness bar chart (requires analyze-h4)
	$(PYTHON) scripts/plots/plot_h4_eval_awareness.py $(FIG)

figures-effort: ## Plot the effort-vs-tokens figures (both metrics)
	$(PYTHON) scripts/plots/plot_effort_vs_tokens.py --metric verbalized $(FIG)
	$(PYTHON) scripts/plots/plot_effort_vs_tokens.py --metric covert $(FIG)

figures-h6: ## Plot the explicitness-gap vs clarity scatters (requires analyze-h6)
	$(PYTHON) scripts/plots/plot_h6_clarity_scatter.py $(FIG) --metric vcr --style gap
	$(PYTHON) scripts/plots/plot_h6_clarity_scatter.py $(FIG) --metric vcr --style raw
	$(PYTHON) scripts/plots/plot_h6_clarity_scatter.py $(FIG) --metric uar --style gap
	$(PYTHON) scripts/plots/plot_h6_clarity_scatter.py $(FIG) --metric uar --style raw

figures-register: ## Plot register/convention dumbbell figures (both metrics, all cells)
	$(PYTHON) scripts/plots/plot_convention_dumbbell.py --mode v1 --metric verbalized $(FIG)
	$(PYTHON) scripts/plots/plot_convention_dumbbell.py --mode v1 --metric covert $(FIG)
	$(PYTHON) scripts/plots/plot_convention_dumbbell.py --mode v2 --metric verbalized $(FIG)
	$(PYTHON) scripts/plots/plot_convention_dumbbell.py --mode v2 --metric covert $(FIG)
	$(PYTHON) scripts/plots/plot_register_matched_dumbbell.py --metric verbalized $(FIG)
	$(PYTHON) scripts/plots/plot_register_matched_dumbbell.py --metric covert $(FIG)
	$(PYTHON) scripts/plots/plot_register_matched_dumbbell_by_source.py --metric verbalized $(FIG)
	$(PYTHON) scripts/plots/plot_register_matched_dumbbell_by_source.py --metric covert $(FIG)
	$(PYTHON) scripts/plots/plot_h1_register.py --metric verbalized $(FIG)
	$(PYTHON) scripts/plots/plot_h1_register.py --metric covert $(FIG)
	$(PYTHON) scripts/plots/plot_h1_register.py --metric covert --convention C3 $(FIG)
	$(PYTHON) scripts/plots/plot_h1_register.py --metric covert --convention MC0 $(FIG)
	$(PYTHON) scripts/plots/plot_h1_register.py --metric covert --convention MC3 $(FIG)

figures-misc: ## Plot miscellaneous standalone figures
	$(PYTHON) scripts/plots/plot_no_context_shift.py $(FIG)

figures-h2: ## Plot the H2 monitor-capability, increment and thesis-scatter panels
	$(PYTHON) scripts/plots/plot_monitor_capability.py $(FIG)
	$(PYTHON) scripts/plots/plot_h2_increment.py $(FIG)
	$(PYTHON) scripts/plots/plot_h2_thesis_scatter.py $(FIG)

figures-artifact-rating: ## Plot the cue-clarity and position-bias figures (requires analyze-artifact-rating)
	$(PYTHON) scripts/plots/plot_cue_clarity.py $(FIG)
	$(PYTHON) scripts/plots/plot_position_bias.py $(FIG)

figures: ## Regenerate all figures
figures: figures-h1 figures-h2 figures-h4 figures-effort \
 figures-h6 figures-register figures-misc figures-artifact-rating

# Composes the SVGs already on disk — no DB, no analyze chain, seconds to run.
# PAPER_SOURCES is a search path: directories are tried in order and the first
# one holding a given SVG wins, so figures generated on a remote pod can be
# mixed with local ones. Output lands in <first dir>/paper unless PAPER_OUT is
# set. Examples:
# make figures-paper PAPER_SOURCES="figures_remote figures"
# make figures-paper PAPER_SOURCES=figures_remote PAPER_OUT=figures/paper
PAPER_SOURCES ?= $(FIGURES_DIR)
PAPER_OUT ?=

# EVAL_UNAWARE=1 rebuilds the paper on the eval-unaware subset: the source scripts
# drop rows the reasoning judge flagged evaluation-aware and stamp `_evalunaware` on
# their filenames, and the composer picks up that art. H4 closed on "awareness is
# under 2% in every (model, cell)", which holds for the open-weight sweep but not for
# the frontier additions (Inkling-Small reaches 15% on the user channel).
# Unset, everything writes the same filenames with the same numbers as before.
# Only the eight invocations whose population is the C0 conditioning set take the
# flag, so this variant skips the rest of figures-paper-sources rather than
# re-running scripts the filter cannot move.
# make figures-paper-all EVAL_UNAWARE=1 PAPER_SOURCES=figures_remote
EVAL_UNAWARE ?=

check-paper-font: ## Report whether CMU Serif Bold is installed (the only thing figures-paper needs beyond the SVGs)
	@$(PYTHON) -c "import scripts.plots.make_paper_figures as m; m.register_chrome_font(); \
 import matplotlib.font_manager as fm; \
 print('OK:', fm.findfont(fm.FontProperties(family=m.FONT, weight=m.FONT_WEIGHT, stretch=m.FONT_STRETCH)))"

figures-paper: ## Compose paper-ready panels (box + title + legend); PAPER_SOURCES sets the source search path
	$(PYTHON) scripts/plots/make_paper_figures.py --figures-dir $(PAPER_SOURCES) \
		$(if $(PAPER_OUT),--out-dir $(PAPER_OUT),) $(if $(EVAL_UNAWARE),--eval-unaware,)

# Only the scripts behind the 41 source SVGs the panels consume — a fraction of
# `make figures`. Monitor id is pinned so the two H2 filenames stay the ones
# make_paper_figures.py expects; they are otherwise named after whichever monitor
# happens to sort last.
# Prerequisites, if the CSVs are stale: analyze-artifact-rating, analyze-h6,
# analyze-h4, analyze-h2-increment, analyze-inter-judge.
PAPER_MONITOR ?= gpt-5.6-luna

# fig23 is the weak-monitor panel, so the same two scripts have to run for the
# second monitor as well. Leaving it out made `make figures-paper` die on a
# clean tree with four missing SVGs, which is how it was found.
PAPER_MONITOR_WEAK ?= gpt-4o-mini-2024-07-18

# Set when EVAL_UNAWARE is on, empty otherwise. Nine invocations are shared by
# both variants and differ only by this flag, so they are written once.
EVAL_FLAG := $(if $(EVAL_UNAWARE),--exclude-eval-aware,)

# The tail differs. With the filter on we also rebuild the VCR twins and the
# other three conventions; with it off we rebuild the panels the filter cannot
# move, which have no filtered variant to produce.
ifdef EVAL_UNAWARE
# VCR twins of the two dumbbells: the panels are covert, but the appendix
# per-source tables read the non-covert files, which had no filtered version.
# The three extra conventions back the abstract's claim that no system prompt
# closes the explicitness gap anywhere. Appendix I is about user_turn alone,
# whereas the user_explicit cell pools it with user_turn_structured.
define _paper_tail
	$(PYTHON) scripts/plots/plot_register_matched_dumbbell.py --metric verbalized $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_register_matched_dumbbell_by_source.py --metric verbalized $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_h1_register.py --metric covert --convention C3 $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_h1_register.py --metric covert --convention MC0 $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_h1_register.py --metric covert --convention MC3 $(EVAL_FLAG) $(FIG)
endef
else
# figures-inter-judge runs last on purpose: it is the one step that needs a
# second judge's judged_*.jsonl, so if that is missing on this box everything
# else has already been written.
define _paper_tail
	$(PYTHON) scripts/plots/plot_cue_clarity.py $(FIG)
	$(PYTHON) scripts/plots/plot_position_bias.py $(FIG)
	$(PYTHON) scripts/plots/plot_h6_clarity_scatter.py $(FIG) --metric vcr --style gap
	$(PYTHON) scripts/plots/plot_h4_eval_awareness.py $(FIG)
	$(PYTHON) scripts/plots/plot_monitor_capability.py $(FIG) --monitor-model $(PAPER_MONITOR)
	$(PYTHON) scripts/plots/plot_h2_thesis_scatter.py $(FIG) --monitor-model $(PAPER_MONITOR)
	$(PYTHON) scripts/plots/plot_monitor_capability.py $(FIG) --monitor-model $(PAPER_MONITOR_WEAK)
	$(PYTHON) scripts/plots/plot_h2_thesis_scatter.py $(FIG) --monitor-model $(PAPER_MONITOR_WEAK)
	$(MAKE) figures-inter-judge
endef
endif

figures-paper-sources: ## Regenerate only the source figures the paper panels use (~1/4 of `figures`)
	$(PYTHON) scripts/plots/plot_h1_channel_asymmetry.py $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_h1_phase_diagram.py $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_no_context_shift.py $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_register_matched_dumbbell.py --metric covert $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_register_matched_dumbbell_by_source.py --metric covert $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_h1_register.py --metric covert $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_convention_dumbbell.py --mode v2 --metric covert $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_effort_vs_tokens.py --metric verbalized $(EVAL_FLAG) $(FIG)
	$(PYTHON) scripts/plots/plot_effort_vs_tokens.py --metric covert $(EVAL_FLAG) $(FIG)
	$(_paper_tail)

figures-paper-all: figures-paper-sources figures-paper ## Regenerate the paper's source figures, then compose the panels

# ---------------------------------------------------------------------------
# Inference and judge stages
#
# One model at a time. Set MODEL, and SEEDS/CONVENTION/EFFORT if the defaults do
# not fit. The cluster launchers that used to fan these out over a model list are
# gone, so a sweep is a loop over MODEL in the shell.
# ---------------------------------------------------------------------------

MODEL ?=
SEEDS ?= 42,43,44
CONVENTION ?= C0
EFFORT ?=
GPUS ?= 1
RESULTS_DIR ?= results

_REQUIRE_MODEL = @[ -n "$(MODEL)" ] || { echo "Set MODEL, e.g. make $@ MODEL=Qwen/Qwen3.5-9B"; exit 1; }

infer: ## Run inference for one model (MODEL=... [SEEDS=] [CONVENTION=] [EFFORT=] [GPUS=])
	$(_REQUIRE_MODEL)
	$(PYTHON) run.py --model $(MODEL) --seeds $(SEEDS) --stage inference --backend vllm \
		--convention $(CONVENTION) --tensor-parallel-size $(GPUS) \
		$(if $(EFFORT),--reasoning-effort $(EFFORT),) --output-dir $(RESULTS_DIR) --resume

infer-no-think: ## Same, with reasoning disabled (vLLM only; not supported for GPT-OSS)
	$(_REQUIRE_MODEL)
	$(PYTHON) run.py --model $(MODEL) --seeds $(SEEDS) --stage inference --backend vllm \
		--convention $(CONVENTION) --tensor-parallel-size $(GPUS) \
		--output-dir $(RESULTS_DIR) --no-think --resume

# Checks row count and legibility before the judge spends money: a crashed
# inference judges cleanly and looks finished. --expected must be the dataset
# size; the script's own default of 500 passes every run unconditionally.
check-inference: ## Verify a run is complete enough to judge (RESULTS_DIR=...)
	$(PYTHON) check_inference.py --results $(RESULTS_DIR)/agentic --expected 5100

judge: ## Judge one model with the primary judge (MODEL=... [CONVENTION=] [SEEDS=])
	$(_REQUIRE_MODEL)
	$(PYTHON) run.py --model $(MODEL) --convention $(CONVENTION) --seeds $(SEEDS) \
		--stage judge --output-dir $(RESULTS_DIR) --batch --resume

# Writes judged__gpt-5.6-luna.jsonl and leaves judged.jsonl alone. Same prompts
# as the primary judge, so the judge model is the only thing that differs.
judge-second: ## Judge with the second judge, for inter-judge agreement (MODEL=...)
	$(_REQUIRE_MODEL)
	$(PYTHON) run.py --model $(MODEL) --convention $(CONVENTION) --seeds $(SEEDS) \
		--stage judge --judge-config config/judge_gpt.yaml \
		--output-dir $(RESULTS_DIR) --batch --resume

# No --output-dir on purpose: run_artifact_rating.py defaults to outputs/, which is
# where analyze-artifact-rating reads from. An older launcher passed results/ instead,
# so those rows sit under results/artifact_rating/ — reach them with --input-dir.
artifact-rating: ## Run the artifact-rating probe for one model (MODEL=...)
	$(_REQUIRE_MODEL)
	$(PYTHON) run_artifact_rating.py --model $(MODEL) --runs 3 --seed 42 \
		--tensor-parallel-size $(GPUS) $(if $(EFFORT),--reasoning-effort $(EFFORT),) --resume

# Models selected for H2 monitor run (Baker et al. subset: ~32B scale, diverse families)
# Appended one line at a time: a `\` continuation inside a := becomes a space,
# which would split --models into a truncated list plus a stray positional.
_H2_MODELS := Qwen_Qwen3.5-27B,google_gemma-4-31B-it,allenai_Olmo-3.1-32B-Think
_H2_MODELS := $(_H2_MODELS),openai_gpt-oss-120b_medium
_H2_MODELS := $(_H2_MODELS),deepseek-ai_DeepSeek-V4-Pro_high,thinkingmachines_Inkling-NVFP4_0.7
_H2_MODELS := $(_H2_MODELS),zai-org_GLM-5.2-FP8_high,moonshotai_Kimi-K2.6

# gpt-5.6-luna is the paper's strong monitor (make_paper_figures.py).
# Its scores are written at reasoning-effort medium, and the effort is
# not part of h2_monitor__*.jsonl's name, so changing H2_EFFORT later would mix two
# efforts inside one file with nothing recording which row is which.
#
# H2_EFFORT has to match the model's kind, and getting it wrong fails the whole batch:
# a reasoning model rejects the config's explicit temperature unless an effort is
# declared, and a non-reasoning model rejects the effort. The Baker et al. weak monitor
# is the second kind:
# make monitor-run H2_MONITOR=gpt-4o-mini-2024-07-18 H2_EFFORT=
H2_MONITOR ?= gpt-5.6-luna
H2_EFFORT ?= medium

monitor-run: ## Run H2 monitor judges (H2_MONITOR / H2_EFFORT; requires OPENAI_API_KEY)
	$(PYTHON) run_monitor.py \
		--models $(_H2_MODELS) \
		--monitor-model $(H2_MONITOR) \
		$(if $(H2_EFFORT),--reasoning-effort $(H2_EFFORT),) \
		--resume

# The inspect-ai task lives in eval/inspect/. Run it with:
# python eval/inspect/run_eval.py --model <name> --seed <N>
# which files the log under logs/agentic/{model}/seed_{N}/.
# See eval/inspect/README.md.
