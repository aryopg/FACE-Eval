# Building the FACE-Eval dataset

How `edinburgh-dawg/face-eval` was made. The published dataset is the canonical
artifact; this directory holds the code and the intermediate files that produced it.

## The chain

```
config/axes.json
config/scenario_generation_prompts.yaml
        |
        |  generate.py --axis {axis} --stage ideation
        v
data/{axis}/ideation.jsonl          scenario sketches, one file per axis
        |
        |  human review in ui.py (Review tab)
        v
data/{axis}/reviews.json            keep / discard per sketch
        |
        |  generate.py --axis {axis} --stage realization
        v
data/{axis}/scenarios.jsonl         question + the profile-source conditions
        |
        +--  generate_source_variants.py    -->  data/{axis}/source_{src}.jsonl
        |
        +--  generate_user_turn_variant.py  -->  data/{axis}/user_turn.jsonl
        |                                        data/finalized_user_turn.jsonl
        v
        |  export_full_combined.py
        v
data/finalized_combined.jsonl       the assembled dataset
        |
        |  publish.py
        v
edinburgh-dawg/face-eval
```

`export_full_combined.py` is the assembler. None of the variant scripts writes
`finalized_combined.jsonl`; they write the per-axis and per-variant files above,
and `export_full_combined.py` merges three inputs into the dataset:

- the profile-source and `no_context` rows of the *existing*
  `finalized_combined.jsonl`, carried over verbatim — question revisions were
  applied there and never back-propagated to `scenarios.jsonl`,
- the tool-channel rows built from every `data/{axis}/source_{src}.jsonl`,
- the user-turn rows of `data/finalized_user_turn.jsonl`.

`generate_saliency_matched.py` is a side branch, not a step in this chain. It
reads the `implicit_*` rows of `finalized_combined.jsonl` and writes
`data/finalized_saliency_matched.jsonl`. `export_full_combined.py` never reads
that file; those rows reach `finalized_combined.jsonl` only when the script is
run with `--append`.

## The five axes

`political`, `ethics`, `egalitarianism`, `epistemic-posture`, `domain-expertise`.
Defined in `config/axes.json`, one `data/{axis}/` directory each.

Two earlier axes, `collectivism` and `epistemology`, were dropped before the final
dataset and are not in it.

## What each script does

| Script | Role |
|---|---|
| `generate.py` | Two stages. Ideation writes scenario sketches per axis. Realization expands each kept sketch into a concrete question plus its context conditions. |
| `generate_source_variants.py` | Re-renders the canonical stance statements as artifacts in another channel: email, slack, notes, browser history. One LLM call produces four rows per scenario and source, explicit and implicit for both sides. Questions are never regenerated. |
| `generate_user_turn_variant.py` | Builds the user-message twins of every accepted `explicit_*` row: one in natural prose, one wrapped in a retrieved-looking block that matches the tool-return register. No LLM call, these are template wraps. The pair separates the channel's role from its register. |
| `generate_saliency_matched.py` | For every `implicit_*` row, appends the tool-return payload verbatim to the user question, matching length, register and position. The control for the saliency confound. |
| `ui.py` | Gradio app with three tabs: ideation, review, scenario browsing. Needs the `dataset` extra. Run with `python -m face_eval_generator.ui`. |
| `export_full_combined.py` | Assembles `data/finalized_combined.jsonl` from the three inputs listed above. Writes a `.bak` first and reports a row breakdown; `--dry-run` prints the breakdown and writes nothing. |
| `publish.py` | Builds a HuggingFace `Dataset` from `finalized_combined.jsonl` and pushes it with a freshly generated dataset card. Private by default. |
| `add_canary.py` | Adds the contamination-detection canary field to the published dataset. The UUID is passed in, never generated here, so re-runs are idempotent. |
| `select_annotation_items.py` | Selects the 100-item human-annotation subset — 50 explicit/implicit pairs matched on scenario, source and side, stratified by source × axis. |

## Acceptance gate

A scenario reaches the channel-variant stage only if **both** of its
`explicit_{side_a}` and `explicit_{side_b}` rows were marked accept during review.
Both sides must be usable, or the scenario cannot support a paired comparison.

## What is and is not in git

Tracked: the generation code, the prompt and axis configs, and the per-axis
intermediates (`ideation.jsonl`, `reviews.json`, `scenarios.jsonl`). Those are small
and they are what shows how the dataset was built and reviewed.

Not tracked: `data/finalized_combined.jsonl` (17 MB) and `data/finalized_user_turn.jsonl`
(9 MB). They are the assembled dataset, and the assembled dataset is published on
HuggingFace. Regenerate them from the intermediates with the variant scripts above, or
just load `edinburgh-dawg/face-eval`.

## Publishing

```bash
python -m face_eval_generator.export_full_combined      # assemble the dataset
python -m face_eval_generator.publish --dry-run         # build locally, push nothing
python -m face_eval_generator.publish                   # pushes private by default
```

`--axis` takes one axis or `all`. `generate.py --stage all` runs ideation and
realization together, but the review step sits between them, so the two stages are
normally run separately.

`python -m face_eval_generator.add_canary` adds the contamination canary to the
published dataset.
