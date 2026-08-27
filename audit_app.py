"""Streamlit auditor for agentic-sycophancy results.

Run with:
    streamlit run audit_app.py

Filters records by model (single), convention, seed, axis, channel/context_type,
source, condition, stance, and unverbalized-adoption definition (aligned ∧ ¬L3 or
aligned ∧ ¬L1). Picks one record at a time and shows its question / reasoning /
answer / judge annotations side by side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from src.results.storage import discover_runs, load_merged_results

RESULTS_DIR = "results/agentic"
DATA_DIR = Path("face_eval_generator/data")
LOOKUP_FILES = (
    "scenarios.jsonl",
    "user_turn.jsonl",
    "finalized_email.jsonl",
    "finalized_slack.jsonl",
    "finalized_notes.jsonl",
    "finalized_browser_history.jsonl",
)


@st.cache_data(show_spinner=False)
def list_runs() -> list[dict[str, Any]]:
    """Return inference runs with their filesystem paths, cached for the UI."""
    out = []
    for r in discover_runs(RESULTS_DIR):
        if not r["has_inference"]:
            continue
        out.append({**r, "path": str(r["path"])})
    return out


@st.cache_data(show_spinner="Loading run…")
def load_run(path: str) -> list[dict[str, Any]]:
    """Load one run and merge its inference records with judge annotations."""
    return load_merged_results(Path(path))


@st.cache_data(show_spinner="Indexing dataset…")
def load_source_lookup() -> dict[str, dict[str, Any]]:
    """Map id (scenario_id__condition) → {source, context_type, context, messages}.

    `context` is the raw cue field from `scenarios.jsonl` (str for explicit,
    list[dict] for implicit, missing for no_context). `messages` is the
    OpenAI-style prompt with the cue baked in for the user_turn /
    finalized_{email,slack,notes,browser_history} variants. Either or both may
    be present; the renderer falls back gracefully.
    """
    lookup: dict[str, dict[str, Any]] = {}
    if not DATA_DIR.exists():
        return lookup
    for axis_dir in sorted(DATA_DIR.iterdir()):
        if not axis_dir.is_dir():
            continue
        for name in LOOKUP_FILES:
            p = axis_dir / name
            if not p.exists():
                continue
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    key = f"{d['scenario_id']}__{d['condition']}"
                    lookup[key] = {
                        "source": d.get("source"),
                        "context_type": d.get("context_type"),
                        "context": d.get("context"),
                        "messages": d.get("messages"),
                    }
    return lookup


def _is_covert(judge: dict[str, Any]) -> bool | None:
    aligned = judge.get("answer_aligns_with_preference")
    tailored = judge.get("reasoning_tailoring_explicit")
    if aligned is None or tailored is None:
        return None
    return bool(aligned) and not bool(tailored)


def _rate(records: list[dict[str, Any]], pred: Callable[[dict[str, Any]], Any]) -> str:
    vals = [pred(r) for r in records]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    k = sum(1 for v in vals if v)
    return f"{k / len(vals):.1%} ({k}/{len(vals)})"


def _judge(r: dict[str, Any]) -> dict[str, Any]:
    return r.get("judge") or {}


STANCE_OPPOSITES: dict[str, str] = {
    "liberal": "conservative",
    "conservative": "liberal",
    "deontological": "utilitarian",
    "utilitarian": "deontological",
    "egalitarian": "elitist",
    "elitist": "egalitarian",
    "deferential": "skeptical",
    "skeptical": "deferential",
    "expert": "novice",
    "novice": "expert",
}


def _condition_parts(condition: str) -> tuple[str, str] | None:
    """Split condition into (group, stance). Returns None for no_context or unrecognised stances."""
    if condition == "no_context":
        return None
    parts = condition.rsplit("_", 1)
    if len(parts) != 2 or parts[1] not in STANCE_OPPOSITES:
        return None
    return parts[0], parts[1]


def _stance_pairs(
    records: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return (label, record_A, record_B) for every stance pair in records."""
    groups: dict[tuple, dict[str, dict[str, Any]]] = {}
    for r in records:
        parsed = _condition_parts(r.get("condition", ""))
        if parsed is None:
            continue
        cond_group, stance = parsed
        key = (r.get("scenario_id"), cond_group, r.get("_model"), r.get("_seed"), r.get("_convention"))
        groups.setdefault(key, {})[stance] = r

    pairs = []
    for key, stance_map in groups.items():
        stances = sorted(stance_map.keys())
        if len(stances) != 2 or STANCE_OPPOSITES.get(stances[0]) != stances[1]:
            continue
        s_a, s_b = stances
        scenario_id, cond_group, _, seed, conv = key
        covert_a = _is_covert(_judge(stance_map[s_a]))
        covert_b = _is_covert(_judge(stance_map[s_b]))
        label = f"s{seed} {conv} | {scenario_id} | {cond_group} | covert={covert_a}/{covert_b}"
        pairs.append((label, stance_map[s_a], stance_map[s_b]))
    return sorted(pairs, key=lambda x: x[0])


def _convention_pairs(
    records: list[dict[str, Any]], conv_a: str, conv_b: str
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return (label, record_A, record_B) for same scenario/condition across two conventions."""
    by_key: dict[tuple, dict[str, dict[str, Any]]] = {}
    for r in records:
        conv = r.get("_convention")
        if conv not in (conv_a, conv_b):
            continue
        key = (r.get("scenario_id"), r.get("condition"), r.get("_model"), r.get("_seed"))
        by_key.setdefault(key, {})[conv] = r

    pairs = []
    for key, conv_map in by_key.items():
        if conv_a not in conv_map or conv_b not in conv_map:
            continue
        r_a, r_b = conv_map[conv_a], conv_map[conv_b]
        scenario_id, cond, _, seed = key
        covert_a = _is_covert(_judge(r_a))
        covert_b = _is_covert(_judge(r_b))
        label = f"s{seed} | {scenario_id} | {cond} | covert={covert_a}/{covert_b}"
        pairs.append((label, r_a, r_b))
    return sorted(pairs, key=lambda x: x[0])


def _adoption_badge(r: dict[str, Any]) -> None:
    """Show a coloured box indicating tailoring/alignment status."""
    j = _judge(r)
    aligned = j.get("answer_aligns_with_preference")
    tailored = j.get("reasoning_tailoring_explicit")
    if aligned is None:
        st.info("Judge not available")
    elif not aligned:
        st.success("Not aligned with preference")
    elif tailored:
        st.warning("Aligned — explicit tailoring in reasoning")
    else:
        st.error("Unverbalized adoption (aligned, reasoning not tailored)")


def _show_record_compact(r: dict[str, Any]) -> None:
    """Render one record vertically; safe to use inside a column."""
    j = _judge(r)
    uid = f"{r.get('id', '')}_{r.get('_convention', '')}"

    meta_cols = st.columns(2)
    meta_cols[0].markdown(f"**Axis / Source**\n\n{r.get('axis')} / {r.get('_source')}")
    meta_cols[1].markdown(f"**Context type / Condition**\n\n{r.get('context_type')} / {r.get('condition')}")

    _adoption_badge(r)

    if j:
        flags = [
            ("aligned", j.get("answer_aligns_with_preference")),
            ("L1 ack", j.get("reasoning_acknowledges_preference")),
            ("L3 tailor", j.get("reasoning_tailoring_explicit")),
            ("committed", j.get("answer_committed")),
            ("stance", j.get("answer_stance_label")),
        ]
        chip_cols = st.columns(len(flags))
        for c, (kk, v) in zip(chip_cols, flags):
            c.metric(kk, _flag(v) if isinstance(v, bool) or v is None else str(v))

    st.markdown("#### Cue")
    _render_cue(r)

    st.markdown("#### Question")
    st.write(r.get("question") or "(empty)")

    st.markdown("#### Reasoning")
    st.text_area(f"reasoning_{uid}", r.get("reasoning") or "", height=300, label_visibility="collapsed")

    st.markdown("#### Answer")
    st.text_area(f"answer_{uid}", r.get("raw_answer") or "", height=200, label_visibility="collapsed")

    if j:
        st.markdown("#### Judge — reasoning explanation")
        st.write(j.get("reasoning_explanation") or "(empty)")
        st.markdown("#### Judge — answer explanation")
        st.write(j.get("answer_explanation") or "(empty)")
        with st.expander("Full judge JSON"):
            st.json(j)
    if r.get("_cue_messages") or r.get("_cue_context") is not None:
        with st.expander("Raw cue payload"):
            st.json({"context": r.get("_cue_context"), "messages": r.get("_cue_messages")})


def main() -> None:
    st.set_page_config(page_title="Agentic Sycophancy Auditor", layout="wide")
    st.title("Agentic Sycophancy — Result Auditor")

    runs = list_runs()
    if not runs:
        st.error(f"No runs found under {RESULTS_DIR}.")
        return
    src_lookup = load_source_lookup()

    models = sorted({r["model"] for r in runs})
    conventions = sorted({r["convention"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})

    conv_a = conv_b = None
    with st.sidebar:
        view_mode = st.radio("View mode", ["Single", "Compare: Stances", "Compare: Conventions"])
        st.header("Runs")
        sel_model = st.selectbox("Model", models, index=0)
        default_conv = ["C0"] if "C0" in conventions else conventions[:1]
        sel_conv = st.multiselect("Convention", conventions, default=default_conv)
        sel_seeds = st.multiselect("Seed", seeds, default=seeds)
        if view_mode == "Compare: Conventions":
            st.header("Convention comparison")
            st.caption("Convention filter above is ignored; both A and B are loaded automatically.")
            conv_a = st.selectbox("Convention A", conventions, index=0)
            remaining = [c for c in conventions if c != conv_a]
            conv_b = st.selectbox("Convention B", remaining) if remaining else None

    if not (sel_model and sel_conv and sel_seeds):
        st.warning("Pick at least one model, convention, and seed.")
        return

    if view_mode == "Compare: Conventions" and conv_a and conv_b:
        chosen = [
            r
            for r in runs
            if r["model"] == sel_model and r["convention"] in (conv_a, conv_b) and r["seed"] in sel_seeds
        ]
    else:
        chosen = [r for r in runs if r["model"] == sel_model and r["convention"] in sel_conv and r["seed"] in sel_seeds]
    if not chosen:
        st.warning("No matching runs.")
        return

    records: list[dict[str, Any]] = []
    for run in chosen:
        for rec in load_run(run["path"]):
            meta = src_lookup.get(rec["id"], {})
            rec = {
                **rec,
                "_model": run["model"],
                "_seed": run["seed"],
                "_convention": run["convention"],
                "_source": meta.get("source"),
                "_cue_context": meta.get("context"),
                "_cue_messages": meta.get("messages"),
            }
            records.append(rec)

    axes = sorted({r["axis"] for r in records if r.get("axis")})
    context_types = sorted({r.get("context_type") for r in records if r.get("context_type")})
    sources = sorted({r["_source"] for r in records if r.get("_source")})
    conditions = sorted({r.get("condition") for r in records if r.get("condition")})
    stances = sorted(
        {_judge(r).get("answer_stance_label") for r in records if _judge(r).get("answer_stance_label") is not None}
    )

    with st.sidebar:
        st.header("Scenario")
        sel_axes = st.multiselect("Axis (domain)", axes, default=axes)
        sel_ctx = st.multiselect("Channel / context_type", context_types, default=context_types)
        sel_src = st.multiselect("Source", sources, default=sources)
        sel_cond = st.multiselect("Condition (raw)", conditions, default=conditions)
        sel_stance = st.multiselect("Stance label", stances, default=stances)

        st.header("Unverbalized adoption")
        covert_filter = st.selectbox(
            "Definition",
            ["any", "aligned ∧ ¬L3", "aligned ∧ ¬L1"],
            index=0,
        )

    def keep(r: dict[str, Any]) -> bool:
        if r.get("axis") not in sel_axes:
            return False
        if r.get("context_type") not in sel_ctx:
            return False
        if sel_src and r.get("_source") not in sel_src:
            return False
        if r.get("condition") not in sel_cond:
            return False
        j = _judge(r)
        if sel_stance and j and j.get("answer_stance_label") not in sel_stance:
            return False
        if covert_filter == "aligned ∧ ¬L3":
            if not _is_covert(j):
                return False
        elif covert_filter == "aligned ∧ ¬L1":
            aligned = j.get("answer_aligns_with_preference")
            l1 = j.get("reasoning_acknowledges_preference")
            if not (aligned is True and l1 is False):
                return False
        return True

    filtered = [r for r in records if keep(r)]

    def _is_covert_l1(r: dict[str, Any]) -> bool | None:
        j = _judge(r)
        aligned = j.get("answer_aligns_with_preference")
        l1 = j.get("reasoning_acknowledges_preference")
        if aligned is None or l1 is None:
            return None
        return bool(aligned) and not bool(l1)

    cols = st.columns(3)
    cols[0].metric("Records", f"{len(filtered)} / {len(records)}")
    cols[1].metric("Covert (aligned ∧ ¬L3)", _rate(filtered, lambda r: _is_covert(_judge(r))))
    cols[2].metric("Covert (aligned ∧ ¬L1)", _rate(filtered, _is_covert_l1))

    if not filtered:
        st.info("No matching records.")
        return

    if view_mode == "Single":
        filtered.sort(key=lambda r: (r["_seed"], r["_convention"], r["id"]))
        labels = [
            f's{r["_seed"]} {r["_convention"]} | {r["id"]} | ' f"¬L3={_is_covert(_judge(r))} ¬L1={_is_covert_l1(r)}"
            for r in filtered
        ]
        idx = st.selectbox("Pick a record", range(len(filtered)), format_func=lambda i: labels[i])
        _show(filtered[idx])

    elif view_mode == "Compare: Stances":
        pairs = _stance_pairs(filtered)
        if not pairs:
            st.info(
                "No stance pairs found. Ensure both stances (e.g. liberal/conservative) are present in the filtered records."
            )
            return
        idx = st.selectbox("Pick a pair", range(len(pairs)), format_func=lambda i: pairs[i][0])
        _, r_a, r_b = pairs[idx]
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader(r_a.get("condition"))
            _show_record_compact(r_a)
        with col_b:
            st.subheader(r_b.get("condition"))
            _show_record_compact(r_b)

    elif view_mode == "Compare: Conventions":
        if not conv_a or not conv_b:
            st.warning("Select two different conventions in the sidebar.")
            return
        pairs = _convention_pairs(filtered, conv_a, conv_b)
        if not pairs:
            st.info("No matching pairs found for the selected conventions.")
            return
        idx = st.selectbox("Pick a pair", range(len(pairs)), format_func=lambda i: pairs[i][0])
        _, r_a, r_b = pairs[idx]
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader(f"Convention: {conv_a}")
            _show_record_compact(r_a)
        with col_b:
            st.subheader(f"Convention: {conv_b}")
            _show_record_compact(r_b)


_ROLE_ICON = {"system": "⚙️", "user": "🧑", "assistant": "🤖", "tool": "🛠️"}


def _render_cue(r: dict[str, Any]) -> None:
    """Show the cue (the bit that planted the preference) using whichever
    representation the dataset row carries: `context` (str or list) from
    scenarios.jsonl, or `messages` (list of role/content/tool_call) from the
    user_turn / finalized_{source}.jsonl variants."""
    ctx = r.get("_cue_context")
    msgs = r.get("_cue_messages")
    context_type = r.get("context_type")

    if not ctx and not msgs:
        st.caption(f"No cue stored locally (context_type = `{context_type}`).")
        return

    # scenarios.jsonl: explicit cue is a string.
    if isinstance(ctx, str):
        st.markdown("**Cue — explicit context (string)**")
        st.info(ctx)
        return

    # scenarios.jsonl: implicit cue is a list of prior turns.
    if isinstance(ctx, list) and ctx:
        st.markdown("**Cue — implicit prior conversation**")
        for m in ctx:
            role = m.get("role", "?")
            icon = _ROLE_ICON.get(role, "•")
            st.markdown(f"{icon} **{role}**")
            st.markdown(f"> {m.get('content','')}")
        return

    # finalized_*.jsonl / user_turn.jsonl: cue lives inside the prompt messages.
    if isinstance(msgs, list) and msgs:
        st.markdown("**Cue — prompt messages**")
        for m in msgs:
            role = m.get("role", "?")
            if role == "system":
                continue  # boilerplate; available in raw expander
            icon = _ROLE_ICON.get(role, "•")
            tool_call = m.get("tool_call") or (
                m.get("tool_calls")[0]["function"]["name"]
                if isinstance(m.get("tool_calls"), list) and m["tool_calls"]
                else None
            )
            header = f"{icon} **{role}**"
            if tool_call:
                header += f"  ·  tool=`{tool_call}`"
            st.markdown(header)
            content = m.get("content")
            if content:
                st.markdown(f"> {content}")
        return


def _flag(value: Any) -> str:
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "—"


def _show(r: dict[str, Any]) -> None:
    j = _judge(r)
    st.subheader(r["id"])
    meta_cols = st.columns(4)
    meta_cols[0].markdown(f"**Model**\n\n{r['_model']}")
    meta_cols[1].markdown(f"**Seed / Convention**\n\n{r['_seed']} / {r['_convention']}")
    meta_cols[2].markdown(f"**Axis / Source**\n\n{r.get('axis')} / {r.get('_source')}")
    meta_cols[3].markdown(f"**Context type / Condition**\n\n{r.get('context_type')} / {r.get('condition')}")

    if j:
        flags = [
            ("aligned (CFR)", j.get("answer_aligns_with_preference")),
            ("L1 acknowledges", j.get("reasoning_acknowledges_preference")),
            ("L3 tailoring", j.get("reasoning_tailoring_explicit")),
            ("covert", _is_covert(j)),
            ("committed", j.get("answer_committed")),
            ("stance", j.get("answer_stance_label")),
        ]
        chip_cols = st.columns(len(flags))
        for c, (k, v) in zip(chip_cols, flags):
            c.metric(k, _flag(v) if isinstance(v, bool) or v is None else str(v))

    st.markdown("### Cue")
    _render_cue(r)

    left, right = st.columns(2)
    with left:
        st.markdown("### Question")
        st.write(r.get("question") or "(empty)")
        st.markdown("### Reasoning")
        st.text_area("reasoning", r.get("reasoning") or "", height=420, label_visibility="collapsed")
    with right:
        st.markdown("### Answer")
        st.text_area("raw_answer", r.get("raw_answer") or "", height=420, label_visibility="collapsed")
        if j:
            st.markdown("### Judge — reasoning explanation")
            st.write(j.get("reasoning_explanation") or "(empty)")
            st.markdown("### Judge — answer explanation")
            st.write(j.get("answer_explanation") or "(empty)")

    if j:
        with st.expander("Full judge JSON"):
            st.json(j)
    if r.get("_cue_messages") or r.get("_cue_context") is not None:
        with st.expander("Raw cue payload"):
            st.json({"context": r.get("_cue_context"), "messages": r.get("_cue_messages")})


if __name__ == "__main__":
    main()
