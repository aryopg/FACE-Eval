"""Unified UI for agentic scenario generation pipeline.

Three tabs:
  1. Ideation — generate scenario sketches with controls
  2. Review — browse and keep/discard sketches
  3. Scenarios — browse realized scenarios

Usage:
    python -m face_eval_generator.ui
"""

import asyncio
import html
import json

import gradio as gr

from face_eval_generator.generate import (
    DEFAULT_DATA_DIR,
    SOURCE_VARIANT_NAMES,
    AnthropicLLM,
    export_all_finalized,
    export_finalized,
    export_source_finalized,
    export_user_turn_finalized,
    get_flagged_scenario_ids,
    get_kept_sketches,
    load_axes_config,
    load_axis_descriptions,
    load_jsonl,
    load_prompts,
    load_reviews,
    load_scenario_reviews,
    load_source_reviews,
    load_user_turn_reviews,
    push_to_huggingface,
    run_ideation,
    run_realization,
    run_revision,
    save_scenario_reviews,
    save_source_reviews,
    save_user_turn_reviews,
)
from face_eval_generator.generate_source_variants import (
    load_source_prompts,
    run_revision_for_axis,
    run_source_for_axis,
    source_rows_path,
)
from face_eval_generator.generate_user_turn_variant import generate_for_axis as regenerate_user_turn_for_axis

DATA_DIR = DEFAULT_DATA_DIR

CSS = """
.status-keep {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9) !important;
    border: 2px solid #4caf50 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.status-discard {
    background: linear-gradient(135deg, #ffebee, #ffcdd2) !important;
    border: 2px solid #ef5350 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.status-unreviewed {
    background: linear-gradient(135deg, #f5f5f5, #eeeeee) !important;
    border: 2px solid #bdbdbd !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.status-rerun {
    background: linear-gradient(135deg, #fff3e0, #ffe0b2) !important;
    border: 2px solid #ff9800 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.stats-bar {
    background: #263238 !important;
    border-radius: 8px !important;
    padding: 12px 20px !important;
}
.scenario-card {
    background: #fafafa !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 10px !important;
    padding: 18px !important;
    margin-bottom: 8px !important;
}
"""


def _axis_names() -> list[str]:
    return list(load_axes_config().keys())


def _format_axis_banner(axis: str) -> str:
    axes = load_axes_config()
    if axis not in axes:
        return ""
    cfg = axes[axis]
    side_a, side_b = cfg["side_a"], cfg["side_b"]
    desc = cfg["behavior_description"]
    sentences = desc.split(". ")
    short = ". ".join(sentences[:2]) + "." if len(sentences) > 2 else desc
    return (
        f'<div style="background:linear-gradient(135deg,#e3f2fd,#bbdefb);border-radius:10px;'
        f'padding:14px 18px;margin-bottom:12px;border-left:4px solid #1565c0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
        f'<strong style="color:#0d47a1;font-size:1.1em;">{axis.replace("-", " ").title()}</strong>'
        f"<div>"
        f'<span style="background:#4caf50;color:white;padding:2px 10px;border-radius:10px;font-size:0.85em;margin-right:6px;">{side_a}</span>'
        f'<span style="color:#263238;">vs</span> '
        f'<span style="background:#ef5350;color:white;padding:2px 10px;border-radius:10px;font-size:0.85em;margin-left:6px;">{side_b}</span>'
        f"</div></div>"
        f'<div style="color:#37474f;font-size:0.9em;line-height:1.5;">{short}</div>'
        f"</div>"
    )


def _format_sketch(sketch: dict, status: str | None) -> str:
    hints = sketch.get("context_hints", {})
    sketch_id = sketch.get("sketch_id", "???")

    if status == "keep":
        badge = '<span style="background:#4caf50;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">KEPT</span>'
        css_class = "status-keep"
    elif status == "discard":
        badge = '<span style="background:#ef5350;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">DISCARDED</span>'
        css_class = "status-discard"
    else:
        badge = '<span style="background:#9e9e9e;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">UNREVIEWED</span>'
        css_class = "status-unreviewed"

    return f"""<div class="{css_class}">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <h2 style="margin:0;color:#37474f;">{sketch_id}</h2>
    {badge}
</div>
<div style="margin-bottom:16px;">
    <div style="font-size:1.1em;color:#263238;margin-bottom:8px;"><strong>Scenario</strong></div>
    <div style="color:#455a64;line-height:1.6;">{sketch.get("scenario", "")}</div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">
    <div style="background:#e3f2fd;padding:10px 14px;border-radius:8px;">
        <strong style="color:#1565c0;">Topic</strong><br/>
        <span style="color:#37474f;">{sketch.get("topic", "")}</span>
    </div>
    <div style="background:#e8eaf6;padding:10px 14px;border-radius:8px;">
        <strong style="color:#263238;">Question direction</strong><br/>
        <span style="color:#37474f;">{sketch.get("question_direction", "")}</span>
    </div>
</div>
<div style="background:#fafafa;border:1px solid #e0e0e0;border-radius:8px;padding:16px;">
    <div style="font-size:1.05em;font-weight:bold;color:#37474f;margin-bottom:10px;">Context Hints</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
        <div style="background:#e8f5e9;padding:10px;border-radius:6px;border-left:3px solid #4caf50;">
            <strong style="color:#2e7d32;">Explicit A</strong><br/>
            <span style="font-size:0.9em;color:#455a64;">{hints.get("explicit_A", "N/A")}</span>
        </div>
        <div style="background:#ffebee;padding:10px;border-radius:6px;border-left:3px solid #ef5350;">
            <strong style="color:#c62828;">Explicit B</strong><br/>
            <span style="font-size:0.9em;color:#455a64;">{hints.get("explicit_B", "N/A")}</span>
        </div>
        <div style="background:#e8f5e9;padding:10px;border-radius:6px;border-left:3px solid #66bb6a;">
            <strong style="color:#2e7d32;">Implicit A</strong><br/>
            <span style="font-size:0.9em;color:#455a64;">{hints.get("implicit_A", "N/A")}</span>
        </div>
        <div style="background:#ffebee;padding:10px;border-radius:6px;border-left:3px solid #ef9a9a;">
            <strong style="color:#c62828;">Implicit B</strong><br/>
            <span style="font-size:0.9em;color:#455a64;">{hints.get("implicit_B", "N/A")}</span>
        </div>
    </div>
</div>
</div>"""


def _format_scenario_card(row: dict, status: str | None = None) -> str:
    condition = row.get("condition", "")
    context = row.get("context")
    context_type = row.get("context_type", "")

    # Status badge and CSS class
    if status == "accept":
        badge = '<span style="background:#4caf50;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">ACCEPTED</span>'
        css_class = "status-keep"
    elif status == "rerun":
        badge = '<span style="background:#ff9800;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">RERUN</span>'
        css_class = "status-rerun"
    else:
        badge = '<span style="background:#9e9e9e;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">UNREVIEWED</span>'
        css_class = "status-unreviewed"

    # Color by condition type
    if context_type == "explicit":
        border_color = "#1565c0"
        bg = "#e3f2fd"
    elif context_type == "implicit":
        border_color = "#1565c0"
        bg = "#e8eaf6"
    else:
        border_color = "#1565c0"
        bg = "#f5f5f5"

    # Format context
    if context is None:
        ctx_html = '<em style="color:#263238;">No context (control)</em>'
    elif isinstance(context, list):
        msgs = []
        for m in context:
            role = m.get("role", "user")
            color = "#1565c0" if role == "user" else "#4caf50"
            msgs.append(
                f'<div style="margin:4px 0;color:#455a64;"><strong style="color:{color};">{role}:</strong> {m.get("content", "")}</div>'
            )
        ctx_html = "".join(msgs)
    else:
        ctx_html = f'<div style="color:#455a64;">{context}</div>'

    return f"""<div class="{css_class}">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <h2 style="margin:0;color:#37474f;">{row.get("scenario_id", "")}</h2>
    <div>
        <span style="background:{border_color};color:white;padding:2px 10px;border-radius:10px;font-size:0.85em;margin-right:8px;">{condition}</span>
        {badge}
    </div>
</div>
<div style="background:white;padding:12px;border-radius:6px;margin-bottom:10px;border:1px solid #e0e0e0;">
    <strong style="color:#37474f;">Question</strong><br/>
    <span style="color:#455a64;">{row.get("question", "")}</span>
</div>
<div style="background:{bg};padding:12px;border-radius:6px;">
    <strong style="color:#37474f;">Context ({context_type})</strong><br/>
    {ctx_html}
</div>
</div>"""


def _review_stats(sketches: list[dict], reviews: dict) -> str:
    total = len(sketches)
    kept = sum(1 for s in sketches if reviews.get(s.get("sketch_id")) == "keep")
    discarded = sum(1 for s in sketches if reviews.get(s.get("sketch_id")) == "discard")
    reviewed = kept + discarded
    pct = int(reviewed / total * 100) if total else 0
    return (
        f'<div class="stats-bar">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;color:white;">'
        f"<span>Reviewed: <strong>{reviewed}/{total}</strong></span>"
        f'<span style="color:#81c784;">Kept: <strong>{kept}</strong></span>'
        f'<span style="color:#ef9a9a;">Discarded: <strong>{discarded}</strong></span>'
        f"</div>"
        f'<div style="background:#455a64;border-radius:4px;height:6px;margin-top:8px;">'
        f'<div style="background:linear-gradient(90deg,#4caf50,#66bb6a);height:100%;border-radius:4px;'
        f'width:{pct}%;transition:width 0.3s;"></div>'
        f"</div></div>"
    )


# =====================================================================
# Tab builders
# =====================================================================


def build_ideation_tab():
    """Tab 1: Run ideation with controls."""
    axis_names = _axis_names()

    with gr.Column():
        gr.HTML(
            '<div style="padding:8px 0;">'
            '<h2 style="margin:0;color:#263238;">Step 1: Generate Scenario Sketches</h2>'
            '<p style="color:#263238;margin:4px 0 0;">Generate diverse scenario sketches. Once done, proceed to Step 2 to review and curate them.</p>'
            "</div>"
        )

        with gr.Row():
            axis_dd = gr.Dropdown(choices=axis_names, value=axis_names[0], label="Axis", scale=2)
            model_dd = gr.Textbox(value="claude-opus-4-6", label="Model", scale=2)

        axis_banner = gr.HTML(_format_axis_banner(axis_names[0]))

        with gr.Row():
            batch_size = gr.Number(value=5, label="Batch size", precision=0)
            num_batches = gr.Number(value=4, label="Num batches", precision=0)

        generate_btn = gr.Button("Generate Sketches", variant="primary", size="lg")
        status_display = gr.HTML("")

        def on_axis_change(axis):
            return _format_axis_banner(axis)

        axis_dd.change(on_axis_change, inputs=[axis_dd], outputs=[axis_banner])

        def on_generate(axis, model, bs, nb):
            try:
                prompts = load_prompts()
                prompts["batch_size"] = int(bs)
                prompts["num_batches"] = int(nb)
                descriptions = load_axis_descriptions(axis)
                llm = AnthropicLLM(model=model)

                sketches = run_ideation(llm, axis, descriptions, prompts, DATA_DIR)
                return (
                    f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;">'
                    f'<strong style="color:#2e7d32;">Done!</strong> Generated {len(sketches)} sketches for <strong>{axis}</strong>. '
                    f"Proceed to <strong>Step 2</strong> to review and curate them."
                    f"</div>"
                )
            except Exception as e:
                return (
                    f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;">'
                    f'<strong style="color:#c62828;">Error:</strong> {e}'
                    f"</div>"
                )

        generate_btn.click(
            on_generate,
            inputs=[axis_dd, model_dd, batch_size, num_batches],
            outputs=[status_display],
        )


def build_review_tab():
    """Tab 2: Review and curate sketches."""
    axis_names = _axis_names()

    with gr.Column():
        gr.HTML(
            '<div style="padding:8px 0;">'
            '<h2 style="margin:0;color:#263238;">Step 2: Review & Curate Sketches</h2>'
            '<p style="color:#263238;margin:4px 0 0;">Keep or discard sketches. When satisfied, run realization on kept sketches to proceed to Step 3.</p>'
            "</div>"
        )

        initial_reviews = load_reviews(DATA_DIR, axis_names[0]) if axis_names else {}
        current_reviews = gr.State(initial_reviews)
        current_idx = gr.State(0)

        with gr.Row():
            axis_dd = gr.Dropdown(choices=axis_names, value=axis_names[0], label="Axis", scale=2)
            progress_label = gr.HTML("", scale=1)

        axis_banner = gr.HTML(_format_axis_banner(axis_names[0]))
        sketch_display = gr.HTML("")

        with gr.Row(equal_height=True):
            prev_btn = gr.Button("< Prev", size="sm", scale=1)
            discard_btn = gr.Button("Discard", variant="stop", size="lg", scale=2)
            keep_btn = gr.Button("Keep", variant="primary", size="lg", scale=2)
            next_btn = gr.Button("Next >", size="sm", scale=1)

        stats_display = gr.HTML("")

        # Realization section
        gr.HTML('<hr style="margin:20px 0;border-color:#e0e0e0;"/>')
        with gr.Row():
            realize_info = gr.HTML("")
            realize_btn = gr.Button("Run Realization on Kept Sketches", variant="primary")
        realize_status = gr.HTML("")

        def _load_sketches(axis):
            return load_jsonl(DATA_DIR / axis / "ideation.jsonl")

        def _show(axis, idx, reviews):
            sketches = _load_sketches(axis)
            if not sketches:
                return "<em>No sketches. Run ideation first.</em>", 0, "", ""
            idx = max(0, min(idx, len(sketches) - 1))
            sketch = sketches[idx]
            status = reviews.get(sketch.get("sketch_id"))
            html = _format_sketch(sketch, status)
            progress = f'<div style="text-align:center;font-size:1.3em;font-weight:bold;color:#263238;">{idx + 1} / {len(sketches)}</div>'
            stats = _review_stats(sketches, reviews)
            return html, idx, progress, stats

        def _realize_info_html(axis, reviews):
            kept = len([v for v in reviews.values() if v == "keep"])
            if kept:
                return f'<span style="color:#37474f;">Ready to realize <strong>{kept}</strong> kept sketches for <strong>{axis}</strong></span>'
            return '<span style="color:#263238;">No kept sketches yet. Review some first.</span>'

        def on_axis_change(axis):
            reviews = load_reviews(DATA_DIR, axis)
            sketches = _load_sketches(axis)
            first_unreviewed = 0
            for i, s in enumerate(sketches):
                if s.get("sketch_id") not in reviews:
                    first_unreviewed = i
                    break
            html, idx, progress, stats = _show(axis, first_unreviewed, reviews)
            banner = _format_axis_banner(axis)
            info = _realize_info_html(axis, reviews)
            return html, idx, reviews, progress, stats, banner, info

        def on_keep(axis, idx, reviews):
            sketches = _load_sketches(axis)
            if not sketches:
                return "<em>No sketches.</em>", 0, reviews, "", "", ""
            sketch_id = sketches[idx].get("sketch_id")
            reviews[sketch_id] = "keep"
            path = DATA_DIR / axis / "reviews.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(reviews, indent=2))
            next_idx = min(idx + 1, len(sketches) - 1)
            html, next_idx, progress, stats = _show(axis, next_idx, reviews)
            info = _realize_info_html(axis, reviews)
            return html, next_idx, reviews, progress, stats, info

        def on_discard(axis, idx, reviews):
            sketches = _load_sketches(axis)
            if not sketches:
                return "<em>No sketches.</em>", 0, reviews, "", "", ""
            sketch_id = sketches[idx].get("sketch_id")
            reviews[sketch_id] = "discard"
            path = DATA_DIR / axis / "reviews.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(reviews, indent=2))
            next_idx = min(idx + 1, len(sketches) - 1)
            html, next_idx, progress, stats = _show(axis, next_idx, reviews)
            info = _realize_info_html(axis, reviews)
            return html, next_idx, reviews, progress, stats, info

        def on_prev(axis, idx, reviews):
            html, idx, progress, stats = _show(axis, max(0, idx - 1), reviews)
            return html, idx, progress, stats

        def on_next(axis, idx, reviews):
            sketches = _load_sketches(axis)
            html, idx, progress, stats = _show(axis, min(idx + 1, len(sketches) - 1), reviews)
            return html, idx, progress, stats

        def on_realize(axis):
            try:
                kept = get_kept_sketches(DATA_DIR, axis)
                if not kept:
                    return (
                        '<div style="background:#fff3e0;padding:14px;border-radius:8px;border-left:4px solid #ff9800;">'
                        '<strong style="color:#e65100;">No kept sketches.</strong> Review and keep some first.'
                        "</div>"
                    )
                prompts = load_prompts()
                descriptions = load_axis_descriptions(axis)
                llm = AnthropicLLM(model=prompts["model"])

                rows = asyncio.run(run_realization(llm, kept, axis, descriptions, prompts, DATA_DIR))
                return (
                    f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;">'
                    f'<strong style="color:#2e7d32;">Done!</strong> Realized {len(rows)} scenario rows from {len(kept)} sketches. '
                    f"Proceed to <strong>Step 3</strong> to review and finalize them."
                    f"</div>"
                )
            except Exception as e:
                return (
                    f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;">'
                    f'<strong style="color:#c62828;">Error:</strong> {e}'
                    f"</div>"
                )

        axis_outputs = [
            sketch_display,
            current_idx,
            current_reviews,
            progress_label,
            stats_display,
            axis_banner,
            realize_info,
        ]
        action_outputs = [sketch_display, current_idx, current_reviews, progress_label, stats_display, realize_info]
        nav_outputs = [sketch_display, current_idx, progress_label, stats_display]

        axis_dd.change(on_axis_change, inputs=[axis_dd], outputs=axis_outputs)
        keep_btn.click(on_keep, inputs=[axis_dd, current_idx, current_reviews], outputs=action_outputs)
        discard_btn.click(on_discard, inputs=[axis_dd, current_idx, current_reviews], outputs=action_outputs)
        prev_btn.click(on_prev, inputs=[axis_dd, current_idx, current_reviews], outputs=nav_outputs)
        next_btn.click(on_next, inputs=[axis_dd, current_idx, current_reviews], outputs=nav_outputs)
        realize_btn.click(on_realize, inputs=[axis_dd], outputs=[realize_status])

    return on_axis_change, [axis_dd], axis_outputs


def build_scenarios_tab():
    """Tab 3: Review realized scenarios with accept/rerun workflow."""
    axis_names = _axis_names()

    with gr.Column():
        gr.HTML(
            '<div style="padding:8px 0;">'
            '<h2 style="margin:0;color:#263238;">Step 3: Review & Finalize Scenarios</h2>'
            '<p style="color:#263238;margin:4px 0 0;">Accept or flag scenarios for revision. Export finalized dataset when done.</p>'
            "</div>"
        )

        # Pre-load reviews for the default axis
        initial_reviews = load_scenario_reviews(DATA_DIR, axis_names[0]) if axis_names else {}
        current_reviews = gr.State(initial_reviews)
        current_idx = gr.State(0)

        with gr.Row():
            axis_dd = gr.Dropdown(choices=axis_names, value=axis_names[0], label="Axis", scale=2)
            condition_dd = gr.Dropdown(
                choices=["all", "explicit", "implicit", "no_context"],
                value="all",
                label="Filter condition type",
                scale=1,
            )
            progress_label = gr.HTML("", scale=1)

        axis_banner = gr.HTML(_format_axis_banner(axis_names[0]))
        stats_display = gr.HTML("")
        scenario_display = gr.HTML("")

        with gr.Row(equal_height=True):
            prev_btn = gr.Button("< Prev", size="sm", scale=1)
            rerun_btn = gr.Button("Rerun", variant="stop", size="lg", scale=2)
            accept_btn = gr.Button("Accept", variant="primary", size="lg", scale=2)
            next_btn = gr.Button("Next >", size="sm", scale=1)

        # Action section
        gr.HTML('<hr style="margin:20px 0;border-color:#e0e0e0;"/>')
        with gr.Row():
            revision_btn = gr.Button("Re-run Flagged", variant="secondary")
            export_btn = gr.Button("Export Finalized", variant="secondary")
        action_status = gr.HTML("")

        def _load_rows(axis):
            return load_jsonl(DATA_DIR / axis / "scenarios.jsonl")

        def _filter_rows(rows, condition_filter):
            if condition_filter == "all":
                return rows
            if condition_filter == "no_context":
                return [r for r in rows if r.get("context_type") == "none"]
            return [r for r in rows if r.get("context_type") == condition_filter]

        def _row_key(row):
            return f"{row.get('scenario_id', '')}__{row.get('condition', '')}"

        def _scenario_stats(rows, reviews):
            total = len(rows)
            accepted = sum(1 for r in rows if reviews.get(_row_key(r)) == "accept")
            rerun = sum(1 for r in rows if reviews.get(_row_key(r)) == "rerun")
            reviewed = accepted + rerun
            pct = int(reviewed / total * 100) if total else 0
            return (
                f'<div class="stats-bar">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;color:white;">'
                f"<span>Reviewed: <strong>{reviewed}/{total}</strong></span>"
                f'<span style="color:#81c784;">Accepted: <strong>{accepted}</strong></span>'
                f'<span style="color:#ffcc80;">Rerun: <strong>{rerun}</strong></span>'
                f"</div>"
                f'<div style="background:#455a64;border-radius:4px;height:6px;margin-top:8px;">'
                f'<div style="background:linear-gradient(90deg,#4caf50,#66bb6a);height:100%;border-radius:4px;'
                f'width:{pct}%;transition:width 0.3s;"></div>'
                f"</div></div>"
            )

        def _show(axis, idx, reviews, condition_filter):
            rows = _filter_rows(_load_rows(axis), condition_filter)
            if not rows:
                return "<em>No scenarios. Run realization first.</em>", 0, "", ""
            idx = max(0, min(idx, len(rows) - 1))
            row = rows[idx]
            status = reviews.get(_row_key(row))
            html = _format_scenario_card(row, status)
            progress = f'<div style="text-align:center;font-size:1.3em;font-weight:bold;color:#263238;">{idx + 1} / {len(rows)}</div>'
            stats = _scenario_stats(rows, reviews)
            return html, idx, progress, stats

        def on_axis_change(axis, condition_filter):
            reviews = load_scenario_reviews(DATA_DIR, axis)
            rows = _filter_rows(_load_rows(axis), condition_filter)
            first_unreviewed = 0
            for i, r in enumerate(rows):
                if _row_key(r) not in reviews:
                    first_unreviewed = i
                    break
            html, idx, progress, stats = _show(axis, first_unreviewed, reviews, condition_filter)
            banner = _format_axis_banner(axis)
            return html, idx, reviews, progress, stats, banner

        def on_accept(axis, idx, reviews, condition_filter):
            rows = _filter_rows(_load_rows(axis), condition_filter)
            if not rows:
                return "<em>No scenarios.</em>", 0, reviews, "", ""
            reviews[_row_key(rows[idx])] = "accept"
            save_scenario_reviews(DATA_DIR, axis, reviews)
            next_idx = min(idx + 1, len(rows) - 1)
            html, next_idx, progress, stats = _show(axis, next_idx, reviews, condition_filter)
            return html, next_idx, reviews, progress, stats

        def on_rerun(axis, idx, reviews, condition_filter):
            rows = _filter_rows(_load_rows(axis), condition_filter)
            if not rows:
                return "<em>No scenarios.</em>", 0, reviews, "", ""
            reviews[_row_key(rows[idx])] = "rerun"
            save_scenario_reviews(DATA_DIR, axis, reviews)
            next_idx = min(idx + 1, len(rows) - 1)
            html, next_idx, progress, stats = _show(axis, next_idx, reviews, condition_filter)
            return html, next_idx, reviews, progress, stats

        def on_prev(axis, idx, reviews, condition_filter):
            html, idx, progress, stats = _show(axis, max(0, idx - 1), reviews, condition_filter)
            return html, idx, progress, stats

        def on_next(axis, idx, reviews, condition_filter):
            rows = _filter_rows(_load_rows(axis), condition_filter)
            html, idx, progress, stats = _show(axis, min(idx + 1, len(rows) - 1), reviews, condition_filter)
            return html, idx, progress, stats

        def on_run_revision(axis):
            try:
                flagged = get_flagged_scenario_ids(DATA_DIR, axis)
                if not flagged:
                    return (
                        '<div style="background:#fff3e0;padding:14px;border-radius:8px;border-left:4px solid #ff9800;">'
                        '<strong style="color:#e65100;">No flagged scenarios.</strong> Mark some rows as "Rerun" first.'
                        "</div>"
                    )
                prompts = load_prompts()
                descriptions = load_axis_descriptions(axis)
                llm = AnthropicLLM(model=prompts["model"])
                rows = asyncio.run(run_revision(llm, axis, descriptions, prompts, DATA_DIR))
                return (
                    f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;">'
                    f'<strong style="color:#2e7d32;">Done!</strong> Revised {len(flagged)} flagged scenarios, {len(rows)} total rows.'
                    f"</div>"
                )
            except Exception as e:
                return (
                    f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;">'
                    f'<strong style="color:#c62828;">Error:</strong> {e}'
                    f"</div>"
                )

        def on_export(axis):
            try:
                count = export_finalized(DATA_DIR, axis)
                return (
                    f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;">'
                    f'<strong style="color:#2e7d32;">Exported {count} accepted rows</strong> to finalized.jsonl'
                    f"</div>"
                )
            except Exception as e:
                return (
                    f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;">'
                    f'<strong style="color:#c62828;">Error:</strong> {e}'
                    f"</div>"
                )

        axis_outputs = [scenario_display, current_idx, current_reviews, progress_label, stats_display, axis_banner]
        action_outputs = [scenario_display, current_idx, current_reviews, progress_label, stats_display]
        nav_outputs = [scenario_display, current_idx, progress_label, stats_display]
        action_inputs = [axis_dd, current_idx, current_reviews, condition_dd]
        nav_inputs = [axis_dd, current_idx, current_reviews, condition_dd]

        axis_dd.change(on_axis_change, inputs=[axis_dd, condition_dd], outputs=axis_outputs)
        condition_dd.change(on_axis_change, inputs=[axis_dd, condition_dd], outputs=axis_outputs)
        accept_btn.click(on_accept, inputs=action_inputs, outputs=action_outputs)
        rerun_btn.click(on_rerun, inputs=action_inputs, outputs=action_outputs)
        prev_btn.click(on_prev, inputs=nav_inputs, outputs=nav_outputs)
        next_btn.click(on_next, inputs=nav_inputs, outputs=nav_outputs)
        revision_btn.click(on_run_revision, inputs=[axis_dd], outputs=[action_status])
        export_btn.click(on_export, inputs=[axis_dd], outputs=[action_status])

    return on_axis_change, [axis_dd, condition_dd], axis_outputs


def build_bulk_actions_tab():
    """Tab: Bulk operations across all axes."""
    axis_names = _axis_names()

    with gr.Column():
        gr.HTML(
            '<div style="padding:8px 0;">'
            '<h2 style="margin:0;color:#263238;">Bulk Actions</h2>'
            '<p style="color:#263238;margin:4px 0 0;">Run operations across all axes at once.</p>'
            "</div>"
        )

        # Show summary of current state per axis
        def _axis_summary():
            rows = []
            for axis in axis_names:
                sketches = load_jsonl(DATA_DIR / axis / "ideation.jsonl")
                reviews = load_reviews(DATA_DIR, axis)
                kept = sum(1 for v in reviews.values() if v == "keep")
                scenarios = load_jsonl(DATA_DIR / axis / "scenarios.jsonl")
                s_reviews = load_scenario_reviews(DATA_DIR, axis)
                accepted = sum(1 for v in s_reviews.values() if v == "accept")
                rows.append(
                    f"<tr>"
                    f'<td style="padding:6px 12px;"><strong>{axis}</strong></td>'
                    f'<td style="padding:6px 12px;text-align:center;">{len(sketches)}</td>'
                    f'<td style="padding:6px 12px;text-align:center;color:#4caf50;">{kept}</td>'
                    f'<td style="padding:6px 12px;text-align:center;">{len(scenarios)}</td>'
                    f'<td style="padding:6px 12px;text-align:center;color:#4caf50;">{accepted}</td>'
                    f"</tr>"
                )
            return (
                '<table style="width:100%;border-collapse:collapse;margin:12px 0;">'
                '<tr style="background:#263238;color:white;">'
                '<th style="padding:8px 12px;text-align:left;">Axis</th>'
                '<th style="padding:8px 12px;">Sketches</th>'
                '<th style="padding:8px 12px;">Kept</th>'
                '<th style="padding:8px 12px;">Scenarios</th>'
                '<th style="padding:8px 12px;">Accepted</th>'
                "</tr>" + "".join(rows) + "</table>"
            )

        summary_display = gr.HTML(_axis_summary())
        refresh_btn = gr.Button("Refresh Summary", size="sm")
        refresh_btn.click(lambda: _axis_summary(), outputs=[summary_display])

        gr.HTML('<hr style="margin:20px 0;border-color:#e0e0e0;"/>')

        # --- Realize All ---
        gr.HTML(
            '<div style="margin-bottom:8px;">'
            '<strong style="color:#263238;">Realize all kept sketches</strong>'
            '<p style="color:#263238;font-size:0.9em;margin:2px 0 0;">Runs realization on kept sketches for every axis that has them. This makes API calls.</p>'
            "</div>"
        )
        with gr.Row():
            realize_confirm = gr.Checkbox(label="I confirm — run realization for all axes", value=False)
            realize_all_btn = gr.Button("Realize All Axes", variant="primary")
        realize_status = gr.HTML("")

        def on_realize_all(confirmed):
            if not confirmed:
                return '<div style="background:#fff3e0;padding:14px;border-radius:8px;border-left:4px solid #ff9800;"><strong style="color:#e65100;">Check the confirmation box first.</strong></div>'
            try:
                prompts = load_prompts()
                results = []
                for axis in axis_names:
                    kept = get_kept_sketches(DATA_DIR, axis)
                    if not kept:
                        results.append(f"{axis}: skipped (no kept sketches)")
                        continue
                    descriptions = load_axis_descriptions(axis)
                    llm = AnthropicLLM(model=prompts["model"])
                    rows = asyncio.run(run_realization(llm, kept, axis, descriptions, prompts, DATA_DIR))
                    results.append(f"{axis}: {len(rows)} rows from {len(kept)} sketches")
                summary = "<br/>".join(results)
                return (
                    f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;">'
                    f'<strong style="color:#2e7d32;">Done!</strong><br/>{summary}'
                    f"</div>"
                )
            except Exception as e:
                return (
                    f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;">'
                    f'<strong style="color:#c62828;">Error:</strong> {e}'
                    f"</div>"
                )

        realize_all_btn.click(on_realize_all, inputs=[realize_confirm], outputs=[realize_status])

        gr.HTML('<hr style="margin:20px 0;border-color:#e0e0e0;"/>')

        # --- Export & Combine All ---
        gr.HTML(
            '<div style="margin-bottom:8px;">'
            '<strong style="color:#263238;">Export & combine all finalized scenarios</strong>'
            '<p style="color:#263238;font-size:0.9em;margin:2px 0 0;">Exports per-axis finalized.jsonl files and merges them into a single finalized_combined.jsonl.</p>'
            "</div>"
        )
        with gr.Row():
            export_confirm = gr.Checkbox(label="I confirm — export and combine all axes", value=False)
            export_all_btn = gr.Button("Export & Combine All", variant="primary")
        export_status = gr.HTML("")

        def on_export_all(confirmed):
            if not confirmed:
                return '<div style="background:#fff3e0;padding:14px;border-radius:8px;border-left:4px solid #ff9800;"><strong style="color:#e65100;">Check the confirmation box first.</strong></div>'
            try:
                count, path = export_all_finalized(DATA_DIR)
                return (
                    f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;">'
                    f'<strong style="color:#2e7d32;">Done!</strong> {count} total rows exported to <code>{path}</code>'
                    f"</div>"
                )
            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                print(f"Export error:\n{tb}")
                return (
                    f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;">'
                    f'<strong style="color:#c62828;">Error:</strong> {e}<br/><pre style="font-size:0.8em;margin-top:8px;white-space:pre-wrap;color:#333;">{tb}</pre>'
                    f"</div>"
                )

        export_all_btn.click(on_export_all, inputs=[export_confirm], outputs=[export_status])

        gr.HTML('<hr style="margin:20px 0;border-color:#e0e0e0;"/>')

        # --- Push to HuggingFace ---
        gr.HTML(
            '<div style="margin-bottom:8px;">'
            '<strong style="color:#263238;">Push to HuggingFace</strong>'
            '<p style="color:#263238;font-size:0.9em;margin:2px 0 0;">Upload the combined dataset to a HuggingFace repo. Run Export & Combine first.</p>'
            "</div>"
        )
        with gr.Row():
            hf_repo = gr.Textbox(value="edinburgh-dawg/face-eval", label="Repo ID", scale=2)
            hf_private = gr.Checkbox(label="Private", value=True, scale=1)
        with gr.Row():
            hf_confirm = gr.Checkbox(label="I confirm — push to HuggingFace", value=False)
            hf_push_btn = gr.Button("Push to HuggingFace", variant="primary")
        hf_status = gr.HTML("")

        def on_push_hf(repo_id, private, confirmed):
            if not confirmed:
                return '<div style="background:#fff3e0;padding:14px;border-radius:8px;border-left:4px solid #ff9800;"><strong style="color:#e65100;">Check the confirmation box first.</strong></div>'
            try:
                url = push_to_huggingface(DATA_DIR, repo_id, private=private)
                return (
                    f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;">'
                    f'<strong style="color:#2e7d32;">Done!</strong> Dataset pushed to <a href="{url}" target="_blank">{url}</a>'
                    f"</div>"
                )
            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                print(f"Push error:\n{tb}")
                return (
                    f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;">'
                    f'<strong style="color:#c62828;">Error:</strong> {e}<br/><pre style="font-size:0.8em;margin-top:8px;white-space:pre-wrap;color:#333;">{tb}</pre>'
                    f"</div>"
                )

        hf_push_btn.click(on_push_hf, inputs=[hf_repo, hf_private, hf_confirm], outputs=[hf_status])


# =====================================================================
# Source-variant artifact rendering
# =====================================================================


def _render_artifact_list(items: list[dict], source: str) -> str:
    """Render an email/slack/notes/browser_history JSON array as HTML rows."""
    if not items:
        return '<em style="color:#263238;">empty</em>'
    out = []
    for it in items:
        if source == "email":
            head = f"<strong>{it.get('subject', '(no subject)')}</strong> &middot; {it.get('date', '')}"
            sub = f"{it.get('from', '?')} &rarr; {it.get('to', '?')}"
            body = it.get("body", "")
        elif source == "slack":
            head = f"<strong>{it.get('channel', '?')}</strong> &middot; {it.get('timestamp', '')}"
            sub = it.get("user", "?")
            body = it.get("text", "")
        elif source == "notes":
            head = f"<strong>{it.get('title', '(untitled)')}</strong> &middot; {it.get('date', '')}"
            sub = ""
            body = it.get("body", "")
        elif source == "browser_history":
            head = f"<strong>{it.get('title', '(no title)')}</strong> &middot; {it.get('last_visited', '')}"
            url = it.get("url", "")
            dwell = it.get("dwell_seconds")
            sub = f"{url} &middot; dwell={dwell}s" if dwell is not None else url
            body = ""
        else:
            head = "<strong>item</strong>"
            sub = ""
            body = json.dumps(it)
        body_html = f'<div style="color:#455a64;margin-top:4px;white-space:pre-wrap;">{body}</div>' if body else ""
        sub_html = f'<div style="font-size:0.85em;color:#263238;">{sub}</div>' if sub else ""
        out.append(
            f'<div style="background:white;border:1px solid #e0e0e0;border-radius:6px;padding:10px 12px;margin:6px 0;">'
            f'<div style="color:#37474f;">{head}</div>{sub_html}{body_html}</div>'
        )
    return "".join(out)


def _format_source_row_card(row: dict, status: str | None) -> str:
    condition = row.get("condition", "")
    context = row.get("context")
    context_type = row.get("context_type", "")
    source = row.get("source", "?")

    if status == "accept":
        badge = '<span style="background:#4caf50;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">ACCEPTED</span>'
        css = "status-keep"
    elif status == "rerun":
        badge = '<span style="background:#ff9800;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">RERUN</span>'
        css = "status-rerun"
    else:
        badge = '<span style="background:#9e9e9e;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">UNREVIEWED</span>'
        css = "status-unreviewed"

    # Parse context: implicit may be a JSON string or a list; explicit is a string.
    if context_type == "implicit":
        items: list[dict] = []
        if isinstance(context, list):
            items = context
        elif isinstance(context, str):
            try:
                items = json.loads(context)
            except Exception:
                items = []
        ctx_html = (
            _render_artifact_list(items, source)
            if items
            else f'<div style="color:#455a64;white-space:pre-wrap;">{context}</div>'
        )
    else:
        ctx_html = f'<div style="color:#455a64;white-space:pre-wrap;">{context}</div>'

    return f"""<div class="{css}">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <h2 style="margin:0;color:#37474f;">{row.get("scenario_id", "")}</h2>
    <div>
        <span style="background:#1565c0;color:white;padding:2px 10px;border-radius:10px;font-size:0.85em;margin-right:8px;">{condition}</span>
        {badge}
    </div>
</div>
<div style="background:white;padding:12px;border-radius:6px;margin-bottom:10px;border:1px solid #e0e0e0;">
    <strong style="color:#37474f;">Question</strong><br/>
    <span style="color:#455a64;">{row.get("question", "")}</span>
</div>
<div style="background:#f5f5f5;padding:12px;border-radius:6px;">
    <strong style="color:#37474f;">Context ({context_type} / {source})</strong>
    <div style="margin-top:6px;">{ctx_html}</div>
</div>
</div>"""


def _format_user_turn_card(row: dict, status: str | None) -> str:
    condition = row.get("condition", "")
    source = row.get("source", "?")
    context_type = row.get("context_type", "user_turn")
    user_content = next((m["content"] for m in row.get("messages", []) if m["role"] == "user"), "")

    if status == "accept":
        badge = '<span style="background:#4caf50;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">ACCEPTED</span>'
        css = "status-keep"
    elif status == "rerun":
        badge = '<span style="background:#ff9800;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">RERUN</span>'
        css = "status-rerun"
    else:
        badge = '<span style="background:#9e9e9e;color:white;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:0.85em;">UNREVIEWED</span>'
        css = "status-unreviewed"

    return f"""<div class="{css}">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <h2 style="margin:0;color:#37474f;">{row.get("scenario_id", "")}</h2>
    <div>
        <span style="background:#1565c0;color:white;padding:2px 10px;border-radius:10px;font-size:0.85em;margin-right:6px;">{condition}</span>
        <span style="background:#7b1fa2;color:white;padding:2px 10px;border-radius:10px;font-size:0.85em;margin-right:8px;">{context_type} &middot; {source}</span>
        {badge}
    </div>
</div>
<div style="background:white;padding:12px;border-radius:6px;border:1px solid #e0e0e0;">
    <strong>User message</strong>
    <pre style="white-space:pre-wrap;word-wrap:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.9em;margin:6px 0 0;padding:0;background:transparent;border:0;">{html.escape(user_content)}</pre>
</div>
</div>"""


# =====================================================================
# Step 4 — Channel Source Variants
# =====================================================================


def build_source_variants_tab():
    axis_names = _axis_names()
    source_names = list(SOURCE_VARIANT_NAMES)

    with gr.Column():
        gr.HTML(
            '<div style="padding:8px 0;">'
            '<h2 style="margin:0;color:#263238;">Step 4: Review Channel Source Variants</h2>'
            '<p style="color:#263238;margin:4px 0 0;">Generate, review, revise, and export email / slack / notes / browser-history renderings of accepted scenarios.</p>'
            "</div>"
        )

        current_reviews = gr.State({})
        current_idx = gr.State(0)

        with gr.Row():
            axis_dd = gr.Dropdown(choices=axis_names, value=axis_names[0], label="Axis", scale=2)
            source_dd = gr.Dropdown(choices=source_names, value=source_names[0], label="Source", scale=2)
            status_dd = gr.Dropdown(
                choices=["all", "unreviewed", "accepted", "rerun"], value="all", label="Status filter", scale=2
            )
            progress_label = gr.HTML("", scale=1)

        axis_banner = gr.HTML(_format_axis_banner(axis_names[0]))
        stats_display = gr.HTML("")
        row_display = gr.HTML("")

        with gr.Row(equal_height=True):
            prev_btn = gr.Button("< Prev", size="sm", scale=1)
            rerun_btn = gr.Button("Rerun", variant="stop", size="lg", scale=2)
            accept_btn = gr.Button("Accept", variant="primary", size="lg", scale=2)
            next_btn = gr.Button("Next >", size="sm", scale=1)

        gr.HTML('<hr style="margin:20px 0;border-color:#e0e0e0;"/>')

        # Generation block
        gr.HTML(
            '<div style="margin-bottom:8px;"><strong style="color:#263238;">Generate</strong>'
            '<p style="color:#263238;font-size:0.9em;margin:2px 0 0;">Generates 4 rows per accepted scenario (explicit/implicit × side A/B). Idempotent unless &ldquo;regenerate all&rdquo; is checked.</p></div>'
        )
        with gr.Row():
            concurrency = gr.Number(value=10, label="Concurrency", precision=0, scale=1)
            regenerate = gr.Checkbox(label="Regenerate all (overwrites existing)", value=False, scale=2)
            generate_btn = gr.Button("Generate", variant="primary", scale=2)
        generate_status = gr.HTML("")

        # Revision block
        gr.HTML('<hr style="margin:16px 0;border-color:#e0e0e0;"/>')
        gr.HTML(
            '<div style="margin-bottom:8px;"><strong style="color:#263238;">Re-run flagged</strong>'
            '<p style="color:#263238;font-size:0.9em;margin:2px 0 0;">Re-renders only conditions marked &ldquo;Rerun&rdquo;. One LLM call per flagged scenario.</p></div>'
        )
        revise_btn = gr.Button("Re-run flagged for (axis, source)", variant="secondary")
        revise_status = gr.HTML("")

        # Export block
        gr.HTML('<hr style="margin:16px 0;border-color:#e0e0e0;"/>')
        gr.HTML(
            '<div style="margin-bottom:8px;"><strong style="color:#263238;">Export</strong>'
            '<p style="color:#263238;font-size:0.9em;margin:2px 0 0;">Writes accepted rows to finalized_source_{src}.jsonl (per-axis).</p></div>'
        )
        export_btn = gr.Button("Export accepted")
        export_status = gr.HTML("")

        # ----- handlers -----
        def _load_rows(axis, source):
            return load_jsonl(source_rows_path(DATA_DIR, axis, source))

        def _row_key(row):
            return f"{row['scenario_id']}__{row['condition']}"

        def _stats(rows, reviews):
            total = len(rows)
            accepted = sum(1 for r in rows if reviews.get(_row_key(r)) == "accept")
            rerun = sum(1 for r in rows if reviews.get(_row_key(r)) == "rerun")
            reviewed = accepted + rerun
            pct = int(reviewed / total * 100) if total else 0
            return (
                f'<div class="stats-bar">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;color:white;">'
                f"<span>Reviewed: <strong>{reviewed}/{total}</strong></span>"
                f'<span style="color:#81c784;">Accepted: <strong>{accepted}</strong></span>'
                f'<span style="color:#ffcc80;">Rerun: <strong>{rerun}</strong></span>'
                f"</div>"
                f'<div style="background:#455a64;border-radius:4px;height:6px;margin-top:8px;">'
                f'<div style="background:linear-gradient(90deg,#4caf50,#66bb6a);height:100%;border-radius:4px;width:{pct}%;"></div>'
                f"</div></div>"
            )

        def _filter(rows, reviews, status_filter):
            if status_filter == "all":
                return rows
            if status_filter == "unreviewed":
                return [r for r in rows if _row_key(r) not in reviews]
            target = "accept" if status_filter == "accepted" else "rerun"
            return [r for r in rows if reviews.get(_row_key(r)) == target]

        def _show(axis, source, idx, reviews, status_filter):
            rows = _filter(_load_rows(axis, source), reviews, status_filter)
            stats = _stats(_load_rows(axis, source), reviews)
            if not rows:
                return "<em>No rows. Generate first.</em>", 0, "", stats
            idx = max(0, min(idx, len(rows) - 1))
            row = rows[idx]
            html = _format_source_row_card(row, reviews.get(_row_key(row)))
            progress = f'<div style="text-align:center;font-size:1.3em;font-weight:bold;color:#263238;">{idx + 1} / {len(rows)}</div>'
            return html, idx, progress, stats

        def on_select_change(axis, source, status_filter):
            reviews = load_source_reviews(DATA_DIR, axis, source)
            rows = _filter(_load_rows(axis, source), reviews, status_filter)
            first = 0
            for i, r in enumerate(rows):
                if _row_key(r) not in reviews:
                    first = i
                    break
            html, idx, progress, stats = _show(axis, source, first, reviews, status_filter)
            return html, idx, reviews, progress, stats, _format_axis_banner(axis)

        def on_set(axis, source, idx, reviews, status_filter, mark):
            rows = _filter(_load_rows(axis, source), reviews, status_filter)
            if not rows:
                return "<em>No rows.</em>", 0, reviews, "", ""
            reviews[_row_key(rows[idx])] = mark
            save_source_reviews(DATA_DIR, axis, source, reviews)
            next_idx = min(idx + 1, len(rows) - 1)
            html, next_idx, progress, stats = _show(axis, source, next_idx, reviews, status_filter)
            return html, next_idx, reviews, progress, stats

        def on_accept(axis, source, idx, reviews, status_filter):
            return on_set(axis, source, idx, reviews, status_filter, "accept")

        def on_rerun(axis, source, idx, reviews, status_filter):
            return on_set(axis, source, idx, reviews, status_filter, "rerun")

        def on_prev(axis, source, idx, reviews, status_filter):
            html, idx, progress, stats = _show(axis, source, max(0, idx - 1), reviews, status_filter)
            return html, idx, progress, stats

        def on_next(axis, source, idx, reviews, status_filter):
            rows = _filter(_load_rows(axis, source), reviews, status_filter)
            html, idx, progress, stats = _show(axis, source, min(idx + 1, len(rows) - 1), reviews, status_filter)
            return html, idx, progress, stats

        def on_generate(axis, source, conc, regen):
            try:
                cfg = load_source_prompts()
                llm = AnthropicLLM(model=cfg["model"])
                rows = asyncio.run(run_source_for_axis(llm, axis, source, cfg, DATA_DIR, int(conc), None, bool(regen)))
                return (
                    f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;">'
                    f'<strong style="color:#2e7d32;">Done!</strong> source_{source}.jsonl now has {len(rows)} rows for {axis}.'
                    f"</div>"
                )
            except Exception as exc:
                return f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;"><strong style="color:#c62828;">Error:</strong> {exc}</div>'

        def on_revise(axis, source):
            try:
                cfg = load_source_prompts()
                llm = AnthropicLLM(model=cfg["model"])
                n = asyncio.run(run_revision_for_axis(llm, axis, source, cfg, DATA_DIR, 10))
                return (
                    f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;">'
                    f'<strong style="color:#2e7d32;">Revised {n} scenarios.</strong>'
                    f"</div>"
                )
            except Exception as exc:
                return f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;"><strong style="color:#c62828;">Error:</strong> {exc}</div>'

        def on_export(axis, source):
            try:
                n = export_source_finalized(DATA_DIR, axis, source)
                return f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;"><strong style="color:#2e7d32;">Exported {n} accepted rows</strong> to finalized_source_{source}.jsonl</div>'
            except Exception as exc:
                return f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;"><strong style="color:#c62828;">Error:</strong> {exc}</div>'

        select_outputs = [row_display, current_idx, current_reviews, progress_label, stats_display, axis_banner]
        action_outputs = [row_display, current_idx, current_reviews, progress_label, stats_display]
        nav_outputs = [row_display, current_idx, progress_label, stats_display]
        action_inputs = [axis_dd, source_dd, current_idx, current_reviews, status_dd]

        axis_dd.change(on_select_change, inputs=[axis_dd, source_dd, status_dd], outputs=select_outputs)
        source_dd.change(on_select_change, inputs=[axis_dd, source_dd, status_dd], outputs=select_outputs)
        status_dd.change(on_select_change, inputs=[axis_dd, source_dd, status_dd], outputs=select_outputs)
        accept_btn.click(on_accept, inputs=action_inputs, outputs=action_outputs)
        rerun_btn.click(on_rerun, inputs=action_inputs, outputs=action_outputs)
        prev_btn.click(on_prev, inputs=action_inputs, outputs=nav_outputs)
        next_btn.click(on_next, inputs=action_inputs, outputs=nav_outputs)
        generate_btn.click(on_generate, inputs=[axis_dd, source_dd, concurrency, regenerate], outputs=[generate_status])
        revise_btn.click(on_revise, inputs=[axis_dd, source_dd], outputs=[revise_status])
        export_btn.click(on_export, inputs=[axis_dd, source_dd], outputs=[export_status])

    return on_select_change, [axis_dd, source_dd, status_dd], select_outputs


# =====================================================================
# Step 5 — User-turn Variants
# =====================================================================


def build_user_turn_tab():
    axis_names = _axis_names()
    source_choices = ["all", "profile"] + list(SOURCE_VARIANT_NAMES)
    variant_choices = ["all", "user_turn", "user_turn_structured", "user_turn_implicit"]

    with gr.Column():
        gr.HTML(
            '<div style="padding:8px 0;">'
            '<h2 style="margin:0;color:#263238;">Step 5: Review User-turn Variants</h2>'
            '<p style="color:#263238;margin:4px 0 0;">Regenerate (template-driven, no LLM), spot-check, and export. Re-run flagged drops flagged rows and re-derives them from current accepted explicit rows.</p>'
            "</div>"
        )

        current_reviews = gr.State({})
        current_idx = gr.State(0)

        with gr.Row():
            axis_dd = gr.Dropdown(choices=axis_names, value=axis_names[0], label="Axis", scale=2)
            source_dd = gr.Dropdown(choices=source_choices, value="all", label="Source filter", scale=2)
            variant_dd = gr.Dropdown(choices=variant_choices, value="all", label="Variant filter", scale=2)
            status_dd = gr.Dropdown(
                choices=["all", "unreviewed", "accepted", "rerun"], value="all", label="Status filter", scale=2
            )
            progress_label = gr.HTML("", scale=1)

        axis_banner = gr.HTML(_format_axis_banner(axis_names[0]))
        stats_display = gr.HTML("")
        row_display = gr.HTML("")

        with gr.Row(equal_height=True):
            prev_btn = gr.Button("< Prev", size="sm", scale=1)
            rerun_btn = gr.Button("Rerun", variant="stop", size="lg", scale=2)
            accept_btn = gr.Button("Accept", variant="primary", size="lg", scale=2)
            next_btn = gr.Button("Next >", size="sm", scale=1)

        gr.HTML('<hr style="margin:16px 0;border-color:#e0e0e0;"/>')
        with gr.Row():
            batch_accept_btn = gr.Button("Batch-accept all unreviewed in current view", variant="secondary")
            batch_status = gr.HTML("")

        gr.HTML('<hr style="margin:16px 0;border-color:#e0e0e0;"/>')
        gr.HTML(
            '<div style="margin-bottom:8px;"><strong style="color:#263238;">Regenerate</strong>'
            '<p style="color:#263238;font-size:0.9em;margin:2px 0 0;">Re-derive user_turn.jsonl from accepted explicit rows. Resets review state for rows whose id no longer exists.</p></div>'
        )
        regenerate_btn = gr.Button("Regenerate user-turn rows for this axis", variant="primary")
        regenerate_status = gr.HTML("")

        gr.HTML('<hr style="margin:16px 0;border-color:#e0e0e0;"/>')
        export_btn = gr.Button("Export accepted → finalized_user_turn.jsonl")
        export_status = gr.HTML("")

        # ----- handlers -----
        def _load_rows(axis):
            return load_jsonl(DATA_DIR / axis / "user_turn.jsonl")

        def _row_key(row):
            return row["id"]

        def _filter(rows, reviews, source_f, variant_f, status_f):
            out = rows
            if source_f != "all":
                out = [r for r in out if (r.get("source") or "profile") == source_f]
            if variant_f != "all":
                out = [r for r in out if r.get("context_type") == variant_f]
            if status_f == "unreviewed":
                out = [r for r in out if _row_key(r) not in reviews]
            elif status_f in ("accepted", "rerun"):
                target = "accept" if status_f == "accepted" else "rerun"
                out = [r for r in out if reviews.get(_row_key(r)) == target]
            return out

        def _stats(rows, reviews):
            total = len(rows)
            accepted = sum(1 for r in rows if reviews.get(_row_key(r)) == "accept")
            rerun = sum(1 for r in rows if reviews.get(_row_key(r)) == "rerun")
            reviewed = accepted + rerun
            pct = int(reviewed / total * 100) if total else 0
            return (
                f'<div class="stats-bar">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;color:white;">'
                f"<span>Reviewed: <strong>{reviewed}/{total}</strong></span>"
                f'<span style="color:#81c784;">Accepted: <strong>{accepted}</strong></span>'
                f'<span style="color:#ffcc80;">Rerun: <strong>{rerun}</strong></span>'
                f"</div>"
                f'<div style="background:#455a64;border-radius:4px;height:6px;margin-top:8px;">'
                f'<div style="background:linear-gradient(90deg,#4caf50,#66bb6a);height:100%;border-radius:4px;width:{pct}%;"></div>'
                f"</div></div>"
            )

        def _show(axis, idx, reviews, source_f, variant_f, status_f):
            rows = _filter(_load_rows(axis), reviews, source_f, variant_f, status_f)
            stats = _stats(_filter(_load_rows(axis), reviews, source_f, variant_f, "all"), reviews)
            if not rows:
                return "<em>No rows. Regenerate first.</em>", 0, "", stats
            idx = max(0, min(idx, len(rows) - 1))
            row = rows[idx]
            html = _format_user_turn_card(row, reviews.get(_row_key(row)))
            progress = f'<div style="text-align:center;font-size:1.3em;font-weight:bold;color:#263238;">{idx + 1} / {len(rows)}</div>'
            return html, idx, progress, stats

        def on_select_change(axis, source_f, variant_f, status_f):
            reviews = load_user_turn_reviews(DATA_DIR, axis)
            rows = _filter(_load_rows(axis), reviews, source_f, variant_f, status_f)
            first = 0
            for i, r in enumerate(rows):
                if _row_key(r) not in reviews:
                    first = i
                    break
            html, idx, progress, stats = _show(axis, first, reviews, source_f, variant_f, status_f)
            return html, idx, reviews, progress, stats, _format_axis_banner(axis)

        def on_set(axis, idx, reviews, source_f, variant_f, status_f, mark):
            rows = _filter(_load_rows(axis), reviews, source_f, variant_f, status_f)
            if not rows:
                return "<em>No rows.</em>", 0, reviews, "", ""
            reviews[_row_key(rows[idx])] = mark
            save_user_turn_reviews(DATA_DIR, axis, reviews)
            next_idx = min(idx + 1, len(rows) - 1)
            html, next_idx, progress, stats = _show(axis, next_idx, reviews, source_f, variant_f, status_f)
            return html, next_idx, reviews, progress, stats

        def on_accept(axis, idx, reviews, source_f, variant_f, status_f):
            return on_set(axis, idx, reviews, source_f, variant_f, status_f, "accept")

        def on_rerun(axis, idx, reviews, source_f, variant_f, status_f):
            return on_set(axis, idx, reviews, source_f, variant_f, status_f, "rerun")

        def on_prev(axis, idx, reviews, source_f, variant_f, status_f):
            html, idx, progress, stats = _show(axis, max(0, idx - 1), reviews, source_f, variant_f, status_f)
            return html, idx, progress, stats

        def on_next(axis, idx, reviews, source_f, variant_f, status_f):
            rows = _filter(_load_rows(axis), reviews, source_f, variant_f, status_f)
            html, idx, progress, stats = _show(
                axis, min(idx + 1, len(rows) - 1), reviews, source_f, variant_f, status_f
            )
            return html, idx, progress, stats

        def on_batch_accept(axis, reviews, source_f, variant_f, status_f):
            rows = _filter(_load_rows(axis), reviews, source_f, variant_f, status_f)
            unreviewed = [r for r in rows if _row_key(r) not in reviews]
            for r in unreviewed:
                reviews[_row_key(r)] = "accept"
            save_user_turn_reviews(DATA_DIR, axis, reviews)
            return (
                f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;">'
                f'<strong style="color:#2e7d32;">Batch-accepted {len(unreviewed)} rows.</strong>'
                f"</div>",
                reviews,
            )

        def on_regenerate(axis):
            try:
                rows = regenerate_user_turn_for_axis(DATA_DIR, axis)
                # Prune review state for ids no longer present
                reviews = load_user_turn_reviews(DATA_DIR, axis)
                valid_ids = {r["id"] for r in rows}
                pruned = {k: v for k, v in reviews.items() if k in valid_ids}
                if len(pruned) != len(reviews):
                    save_user_turn_reviews(DATA_DIR, axis, pruned)
                return f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;"><strong style="color:#2e7d32;">Regenerated {len(rows)} user-turn rows for {axis}.</strong></div>'
            except Exception as exc:
                return f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;"><strong style="color:#c62828;">Error:</strong> {exc}</div>'

        def on_export(axis):
            try:
                n = export_user_turn_finalized(DATA_DIR, axis)
                return f'<div style="background:#e8f5e9;padding:14px;border-radius:8px;border-left:4px solid #4caf50;"><strong style="color:#2e7d32;">Exported {n} accepted user-turn rows</strong> to finalized_user_turn.jsonl</div>'
            except Exception as exc:
                return f'<div style="background:#ffebee;padding:14px;border-radius:8px;border-left:4px solid #ef5350;"><strong style="color:#c62828;">Error:</strong> {exc}</div>'

        select_outputs = [row_display, current_idx, current_reviews, progress_label, stats_display, axis_banner]
        action_outputs = [row_display, current_idx, current_reviews, progress_label, stats_display]
        nav_outputs = [row_display, current_idx, progress_label, stats_display]
        action_inputs = [axis_dd, current_idx, current_reviews, source_dd, variant_dd, status_dd]

        axis_dd.change(on_select_change, inputs=[axis_dd, source_dd, variant_dd, status_dd], outputs=select_outputs)
        source_dd.change(on_select_change, inputs=[axis_dd, source_dd, variant_dd, status_dd], outputs=select_outputs)
        variant_dd.change(on_select_change, inputs=[axis_dd, source_dd, variant_dd, status_dd], outputs=select_outputs)
        status_dd.change(on_select_change, inputs=[axis_dd, source_dd, variant_dd, status_dd], outputs=select_outputs)
        accept_btn.click(on_accept, inputs=action_inputs, outputs=action_outputs)
        rerun_btn.click(on_rerun, inputs=action_inputs, outputs=action_outputs)
        prev_btn.click(on_prev, inputs=action_inputs, outputs=nav_outputs)
        next_btn.click(on_next, inputs=action_inputs, outputs=nav_outputs)
        batch_accept_btn.click(
            on_batch_accept,
            inputs=[axis_dd, current_reviews, source_dd, variant_dd, status_dd],
            outputs=[batch_status, current_reviews],
        )
        regenerate_btn.click(on_regenerate, inputs=[axis_dd], outputs=[regenerate_status])
        export_btn.click(on_export, inputs=[axis_dd], outputs=[export_status])

    return on_select_change, [axis_dd, source_dd, variant_dd, status_dd], select_outputs


def build_app():
    with gr.Blocks(title="Agentic Data Generation", theme=gr.themes.Soft(), css=CSS) as app:
        gr.HTML(
            '<div style="text-align:center;padding:16px 0 4px;">'
            '<h1 style="margin:0;color:#263238;">Agentic Data Generation</h1>'
            '<p style="color:#263238;margin:4px 0 0;">Step 1: Ideation &rarr; Step 2: Review Sketches &rarr; Step 3: Review Profile Scenarios &rarr; Step 4: Channel Source Variants &rarr; Step 5: User-turn Variants &rarr; Export</p>'
            "</div>"
        )

        with gr.Tab("Step 1: Ideation"):
            build_ideation_tab()

        with gr.Tab("Step 2: Review Sketches"):
            review_init, review_inputs, review_outputs = build_review_tab()

        with gr.Tab("Step 3: Review Profile Scenarios"):
            scenario_init, scenario_inputs, scenario_outputs = build_scenarios_tab()

        with gr.Tab("Step 4: Channel Source Variants"):
            sv_init, sv_inputs, sv_outputs = build_source_variants_tab()

        with gr.Tab("Step 5: User-turn Variants"):
            ut_init, ut_inputs, ut_outputs = build_user_turn_tab()

        with gr.Tab("Bulk Actions"):
            build_bulk_actions_tab()

        # Load reviews on page refresh
        app.load(review_init, inputs=review_inputs, outputs=review_outputs)
        app.load(scenario_init, inputs=scenario_inputs, outputs=scenario_outputs)
        app.load(sv_init, inputs=sv_inputs, outputs=sv_outputs)
        app.load(ut_init, inputs=ut_inputs, outputs=ut_outputs)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
