# `scripts/`

Everything you run after the pipeline to turn results into paper-ready figures. `analysis/` writes
the CSV and JSON outputs that the paper quotes; `plots/` draws SVGs and composes them into paper
panels. The runners that produce results, and the gate that checks them, live at the repository
root. Run these scripts through the repository-root `Makefile` (`make help`).

Read a script's module docstring first: it names the files that script reads and writes.

## Data flow

```
results/agentic/{model}/seed_{N}/judged.jsonl   inference + two-judge scores
        |  src/results/db.py (ResultsDB)
        v
analyze_*.py  ------------------->  figures/*.csv
        |                                |
        +--------------------------> plot_*.py  --->  figures/*.svg
                                                         |
                                        make_paper_figures.py  --->  figures/paper/*.svg
```

Some plot scripts read a CSV an analyze script wrote; others skip the CSV and read `ResultsDB`
themselves. Both are normal. `make analyze` runs the analysis chain, `make figures` every figure,
`make figures-paper` the panels from SVGs already on disk.

**The target that matters most is `make figures-paper-sources`.** It regenerates only the source
figures the paper panels consume — about a quarter of `make figures`. `make figures-paper-all` runs
that and then composes the panels. `scripts/plots/make_paper_figures.py` holds `PANELS`, the authority on
which figures are in the paper and which source SVGs each panel needs.

## Three kinds of dependency

A script can have no Makefile target, no importer, and still be load-bearing. Check all three.

**1. Makefile targets.** Some scripts are invoked several times with different flags
(`plot_h1_register.py` runs 5 times in `figures-register`).

**2. Python imports.** Four modules are libraries for other scripts:

| Module | Importers | Exports used |
| --- | --- | --- |
| `_eval_aware_filter.py` | 9 | `--exclude-eval-aware` wiring, `SUFFIX` |
| `plot_register_matched_dumbbell.py` | 3 | `paired_delta` |
| `plot_h2_increment.py` | 2 | `CELL_COLORS`, `_monitor_order` |
| `analyze_monitor_detection.py` | 2 | `_compute_causal_labels`, `_label_key` |

Two plot scripts do both jobs — they draw a figure *and* export helpers. Deleting either would
break scripts that never appear next to them in the Makefile.

- `plot_no_context_shift.py` — draws the CFR figure, exports `CELL_COLORS`.
- `plot_register_matched_dumbbell.py` — draws the channel dumbbell, exports `paired_delta`.

**3. CSV artifacts.** A plot script reads a file an analyze script wrote, with no import between
them. Nothing in the source links the two; each script declares the edge in its own module
docstring (`Reads: ... (written by ...)`). The write side is `save_table` in
`src/utils/plotting.py`. The read side is either `load_metric_rows` from the same module (all the H2
plots) or a plain `csv.DictReader` (`plot_h4_eval_awareness.py`, `plot_h6_*`,
`plot_inter_judge_agreement.py`).

Counter-example: `analyze_h3_filter_stability.py` writes CSVs no plot script reads. It is a
pre-registered robustness check whose verdict is printed, not drawn.

## The scripts

### H1 — channel and register

| Script | Purpose |
| --- | --- |
| `analyze_h1_role_register.py` | 2x2 role x register analysis on L1/L3, with bootstrap CIs |
| `plot_h1_channel_asymmetry.py` | User vs tool channel bar charts, 2-bar and 4-bar |
| `plot_h1_phase_diagram.py` | Alignment rate vs verbalized commitment, 2x2 facets |
| `plot_h1_register.py` | Role x register gap-of-gaps scatter |
| `plot_no_context_shift.py` | Cue-following rate per cell (also exports `CELL_COLORS`) |
| `plot_register_matched_dumbbell.py` | User vs tool dumbbell (also exports `paired_delta`) |
| `plot_register_matched_dumbbell_by_source.py` | Same, one figure per source |

### H2 — monitor detection

| Script | Purpose |
| --- | --- |
| `analyze_monitor_detection.py` | Monitor AUC on the causal-dependence label |
| `analyze_monitor_increment.py` | CoT-vs-action increment and VCR CSVs, both monitors |
| `analyze_h2_calibration.py` | Monitor reliability and the >=70 operating point |
| `plot_monitor_capability.py` | Panel A — action-only vs action+reasoning AUROC |
| `plot_h2_increment.py` | CoT − action increment per cell (also a library, see above) |
| `plot_h2_thesis_scatter.py` | Unverbalized adoption vs CoT-monitor detection, per model-cell |

### H3–H6 — filter, awareness, conventions, clarity

| Script | Purpose |
| --- | --- |
| `analyze_h3_filter_stability.py` | Filter flip rate, VCR orderings under 4 filter variants, post-filter N |
| `analyze_h4_eval_awareness.py` | Eval-awareness rates, and the channel gap on the eval-unaware subset |
| `plot_h4_eval_awareness.py` | Eval-awareness bar chart per model and cell |
| `analyze_h5_convention_power.py` | Convention delta-of-deltas with a joint bootstrap |
| `plot_convention_dumbbell.py` | Three convention contrasts as dumbbells (modes v1/v2) |
| `analyze_convention_backfire.py` | Qwen3.5 convention-backfire tests, with CSVs and figures |
| `analyze_h6_clarity_matched.py` | Explicitness gap at clarity-matched pairs |
| `plot_h6_clarity_scatter.py` | Explicitness gap vs clarity, scatter plus regression |

### Test-time compute

| Script | Purpose |
| --- | --- |
| `plot_effort_vs_tokens.py` | Faithfulness and alignment vs mean CoT tokens, per effort-swept model |

### Artifact-rating probe

| Script | Purpose |
| --- | --- |
| `analyze_artifact_rating.py` | Aggregate raw artifact-rating annotation into `outputs/artifact_rating_aggregated.jsonl` |
| `plot_cue_clarity.py` | Side-ID accuracy and rated clarity, explicit vs implicit |
| `plot_position_bias.py` | Side-ID rate by which option is ground truth |

`analyze_artifact_rating.py` must run before `figures-artifact-rating` and before `analyze-h6`.

### Judge validation

| Script | Purpose |
| --- | --- |
| `analyze_inter_judge_agreement.py` | Agreement between the primary judge and a second judge |
| `plot_inter_judge_agreement.py` | Judge A vs judge B scatter, per model |

### Paper composition, dataset, pipeline operations

| Script | Purpose |
| --- | --- |
| `make_paper_figures.py` | Compose paper panels from the SVGs on disk; holds `PANELS` |

### Shared wiring

| Script | Purpose |
| --- | --- |
| `plots/_eval_aware_filter.py` | Shared `--exclude-eval-aware` wiring and the `_evalunaware` filename suffix |

## Direct use

Every runnable script here has a Makefile target. The dataset build-and-publish scripts live in
`face_eval_generator/` beside the data they produce.

`plots/_eval_aware_filter.py` has no target by design — its nine neighbours import it.
