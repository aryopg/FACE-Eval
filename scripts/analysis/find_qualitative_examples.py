"""Find same-scenario pairs where the model committed twice but verbalized it once.

VCR is `reasoning_tailoring_explicit` conditioned on `answer_aligns_with_preference`.
A pair here is one scenario where the answer went the user's way under two conditions,
and the CoT admitted the tailoring under one but not the other. The question is held
fixed, so the only thing that moved is where the cue arrived.

By default a pair is the headline direction: the cue arrives in the user's message and
the CoT owns the tailoring, or it arrives in a tool return and the CoT goes quiet. Both
halves cue the same preference side, so the channel is the only thing that moved.
Pass --any-direction and --any-side to relax either.

Reads run directories directly rather than through ResultsDB, because the slim
projection ResultsDB defaults to drops the reasoning text these examples are made of.

Usage:
    python scripts/analysis/find_qualitative_examples.py --model Qwen_Qwen3.5-27B
    python scripts/analysis/find_qualitative_examples.py --all-families \\
        --n 80 --html analysis/review.html
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.data.face_eval import FaceEval
from src.results.storage import discover_runs, load_merged_results
from src.utils.plotting import CELL_CONTEXT_TYPES, DIR_FAMILY, FAMILY_ORDER, MODEL_PARAMS

# context_type -> cell, inverted from the plotting registry so the two stay in step.
CELL_OF = {ct: cell for cell, cts in CELL_CONTEXT_TYPES.items() for ct in cts if cell != "no_context"}


def eligible(row: dict, baseline: dict[str, bool | None]) -> bool:
    """Say whether a row is a commitment the headline analysis would count."""
    j = row.get("judge") or {}
    return (
        CELL_OF.get(row.get("context_type")) is not None
        # No prior commitment before the cue, the filter_causal_dependent baseline.
        and baseline.get(row.get("scenario_id")) is False
        and j.get("reasoning_parse_ok") is True
        and j.get("answer_parse_ok") is True
        and j.get("reasoning_eval_awareness") is not True
        and j.get("answer_aligns_with_preference") is True
    )


def _is_headline(pair: tuple[dict, dict]) -> bool:
    """Say whether the CoT admits it on the user channel and goes quiet on the tool channel."""
    v, c = pair
    return CELL_OF[v["context_type"]].startswith("user") and CELL_OF[c["context_type"]].startswith("tool")


def side(row: dict) -> str:
    """The preference side a condition cues, the last token of the condition string."""
    return row["condition"].rsplit("_", 1)[-1]


def find_pairs(rows: list[dict], same_side: bool = True, headline_only: bool = True) -> list[tuple[dict, dict]]:
    """Return one (verbalized, covert) pair per scenario.

    With headline_only, the verbalized half is a user-message cue and the covert half
    is a tool-return cue: the same cue is owned in the CoT when the user states it and
    goes unmentioned when a document carries it. Without it, the reverse direction and
    the within-channel contrasts are kept too.

    With same_side, both halves cue the same preference side, so the channel is the
    only thing that moved. Without it, a pair may contrast opposite sides too.
    """
    baseline = {
        r["scenario_id"]: (r.get("judge") or {}).get("answer_committed")
        for r in rows
        if r.get("context_type") == "none"
    }

    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if eligible(r, baseline):
            by_scenario[r["scenario_id"]].append(r)

    pairs = []
    for scenario_rows in by_scenario.values():
        verbalized = [r for r in scenario_rows if (r["judge"] or {}).get("reasoning_tailoring_explicit")]
        covert = [r for r in scenario_rows if not (r["judge"] or {}).get("reasoning_tailoring_explicit")]
        candidates = [
            (v, c)
            for v in verbalized
            for c in covert
            if CELL_OF[v["context_type"]] != CELL_OF[c["context_type"]]
            and (not same_side or side(v) == side(c))
            and (not headline_only or _is_headline((v, c)))
        ]
        # One pair per scenario, so --n 5 means five distinct questions rather than
        # five slices of the same one.
        if candidates:
            pairs.append(max(candidates, key=_is_headline))

    pairs.sort(key=lambda p: p[0]["scenario_id"])
    # Headline pairs first as a block, so a small --n never trades one away for a
    # weaker pair; within each block, round-robin the axes.
    headline = [p for p in pairs if _is_headline(p)]
    rest = [p for p in pairs if not _is_headline(p)]
    return _interleave_axes(headline) + _interleave_axes(rest)


def _interleave_axes(pairs: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    """Round-robin over axes, so a small --n spans topics instead of one axis."""
    by_axis: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for p in pairs:
        by_axis[p[0]["axis"]].append(p)

    out = []
    while by_axis:
        for axis in sorted(by_axis):
            out.append(by_axis[axis].pop(0))
        by_axis = {a: ps for a, ps in by_axis.items() if ps}
    return out


# Effort suffixes, ranked. Inkling's are continuous floats, parsed below.
_EFFORT_RANK = {"low": 0.0, "medium": 1.0, "high": 2.0, "max": 3.0}


def _params(model_dir: str) -> int | None:
    """Parameter count for a run dir, whose effort suffix MODEL_PARAMS does not carry."""
    parts = model_dir.split("_")
    for cut in range(len(parts), 0, -1):
        base = "_".join(parts[:cut])
        if base in MODEL_PARAMS:
            return MODEL_PARAMS[base]
    return None


def _effort(model_dir: str) -> float:
    """Rank a run dir's effort suffix, so a tie on size picks the hardest-thinking run."""
    suffix = model_dir.rsplit("_", 1)[-1]
    if suffix in _EFFORT_RANK:
        return _EFFORT_RANK[suffix]
    try:
        return float(suffix)
    except ValueError:
        return -1.0


def largest_per_family(results_dir: str, seed: int, convention: str) -> list[str]:
    """One run dir per family: the most parameters, then the highest effort.

    Skips run dirs the plotting registry does not map, which is how it stays in step
    with the figures — a model absent from MODEL_PARAMS or DIR_FAMILY is absent there too.
    """
    by_family: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for run in discover_runs(results_dir):
        if not (run["has_judged"] and run["seed"] == seed and run["convention"] == convention):
            continue
        model = run["model"]
        params, family = _params(model), DIR_FAMILY.get(model)
        if params is not None and family is not None:
            by_family[family].append((params, _effort(model), model))

    return [max(by_family[f])[2] for f in FAMILY_ORDER if by_family.get(f)]


def interleave_models(by_model: dict[str, list[tuple[dict, dict]]]) -> list[tuple[dict, dict]]:
    """Round-robin over models, so a small --n spans families instead of one model."""
    queues = {m: list(ps) for m, ps in by_model.items() if ps}
    out = []
    while queues:
        for model in [m for m in by_model if m in queues]:
            out.append(queues[model].pop(0))
        queues = {m: ps for m, ps in queues.items() if ps}
    return out


def context_messages(row: dict, by_id: dict[str, dict]) -> list[dict]:
    """The prompt the model saw, as [{role, name, content}].

    The cue is never in `question`: a user_turn row appends the profile to the user
    message, and a tool row returns it from the tool. Judging a pair without this
    means judging it without the thing that varies.
    """
    item = by_id.get(row["id"])
    if item is None:
        return []

    out = []
    for m in item["messages"]:
        # The dataset spells it `tool_call`, a bare name, on both the assistant turn
        # that makes the call and the tool turn that answers it.
        call = m.get("tool_call")
        content = m.get("content")
        if m["role"] == "assistant" and call:
            out.append({"role": "assistant", "name": f"calls {call}()", "content": ""})
            continue
        if content in (None, "", "None"):
            continue
        out.append({"role": m["role"], "name": call or "", "content": str(content)})
    return out


def load_selection(path: Path) -> dict[str, str]:
    """Read the review page's export: {"<model>/<scenario_id>": note} for kept pairs.

    Accepts the page's own JSON (objects carrying decision and note) or a bare list of
    ids, so a hand-written shortlist works too.
    """
    payload = json.loads(path.read_text())
    if payload and isinstance(payload[0], dict):
        return {r["id"]: (r.get("note") or "") for r in payload if r.get("decision") == "keep"}
    return {str(x): "" for x in payload}


def clip(text: str, limit: int) -> str:
    """Shorten text to limit characters. limit=0 keeps it whole."""
    text = (text or "").strip()
    return text if limit <= 0 or len(text) <= limit else text[:limit].rstrip() + " […]"


def render(pair: tuple[dict, dict], index: int, limit: int, note: str = "") -> str:
    """Render one pair as markdown."""
    v, c = pair
    out = [
        f"## {index}. `{v['scenario_id']}` ({v['axis']}) — {v['_model']}",
        "",
        f"**Question.** {v['question']}",
        "",
    ]
    if note:
        out += [f"> **Note.** {note}", ""]
    for label, r in (("Verbalized", v), ("Covert", c)):
        j = r["judge"]
        out += [
            f"### {label} — `{r['condition']}` ({CELL_OF[r['context_type']]})",
            "",
            f"**Reasoning.** {clip(r['reasoning'], limit)}",
            "",
            f"**Answer.** {clip(r['raw_answer'], limit)}",
            "",
            f"**Reasoning judge.** {j['reasoning_explanation']}",
            "",
            f"**Answer judge.** {j['answer_explanation']}",
            "",
        ]
    return "\n".join(out)


_PAGE = """<!doctype html>
<meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{font:15px/1.55 system-ui,sans-serif;margin:0;background:#f6f6f4;color:#1a1a1a}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;
        display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 header b{font-size:14px} .grow{flex:1}
 .bar{height:4px;background:#e3e3e0;width:100%} .bar i{display:block;height:100%;background:#2f6f4f}
 main{max-width:1180px;margin:0 auto;padding:16px}
 .q{background:#fff;border:1px solid #ddd;border-radius:6px;padding:14px 16px;margin-bottom:12px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
 @media(max-width:900px){.cols{grid-template-columns:1fr}}
 .col{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px 14px}
 .col.v{border-top:3px solid #2f6f4f} .col.c{border-top:3px solid #a63d40}
 h3{margin:.2em 0 .1em;font-size:14px} code{background:#eee;padding:1px 5px;border-radius:3px;font-size:12px}
 .lbl{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#666;margin-top:12px}
 pre{white-space:pre-wrap;word-wrap:break-word;font:13px/1.5 ui-monospace,monospace;margin:.3em 0;
     max-height:300px;overflow:auto;background:#fafafa;border:1px solid #eee;padding:8px;border-radius:4px}
 .judge{font-size:13px;color:#444;background:#fbf8ee;border-left:3px solid #d8c98a;padding:6px 10px;margin:.3em 0}
 .msg{margin:.35em 0;border-left:3px solid #ccc;padding-left:8px}
 .msg b{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:#777}
 .msg.user{border-left-color:#3a6ea5} .msg.tool{border-left-color:#b07d2b}
 .msg.system{border-left-color:#ddd;opacity:.6} .msg.assistant{border-left-color:#aaa;opacity:.7}
 .msg pre{max-height:220px;background:#fff}
 button{font:14px system-ui;padding:7px 16px;border-radius:5px;border:1px solid #bbb;background:#fff;cursor:pointer}
 button.keep{background:#2f6f4f;color:#fff;border-color:#2f6f4f}
 button.skip{background:#a63d40;color:#fff;border-color:#a63d40}
 .tag{font-size:12px;color:#555} .done{padding:40px;text-align:center}
 textarea{width:100%;height:140px;font:12px ui-monospace,monospace}
 .note{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px 14px;margin-top:12px}
 .note textarea{height:70px;font:13px/1.5 system-ui,sans-serif;border:1px solid #ccc;border-radius:4px;padding:8px}
 .note.filled{border-color:#8a6d1f;background:#fffdf5}
</style>
<header>
  <b id="pos"></b>
  <span class="tag" id="meta"></span>
  <span class="grow"></span>
  <button onclick="prev()">&larr; Back</button>
  <button class="skip" onclick="mark('skip')">Skip (s)</button>
  <button class="keep" onclick="mark('keep')">Keep (k)</button>
  <button onclick="showKept()">Kept <span id="nkeep">0</span> &middot; Notes <span id="nnote">0</span></button>
</header>
<div class="bar"><i id="bar"></i></div>
<main id="app"></main>
<script>
const PAIRS = __DATA__, KEY = "faceeval-review-__KEY__";
let marks = {}; try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
// Earlier versions stored a bare "keep"/"skip"; keep those decisions on upgrade.
for (const k in marks) if (typeof marks[k] === "string") marks[k] = { v: marks[k], n: "" };
const entry = id => (marks[id] ||= { v: null, n: "" });
let i = 0;
const esc = t => (t ?? "").replace(/[&<>]/g, m => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[m]));
function save(){ try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch(e){} }
function ctx(msgs){ return (msgs||[]).map(m => {
  const who = m.role + (m.name ? " " + m.name : "");
  return `<div class="msg ${m.role}"><b>${esc(who)}</b>${m.content ? `<pre>${esc(m.content)}</pre>` : ""}</div>`;
}).join(""); }
function side(r, cls, label){ return `<div class="col ${cls}">
  <h3>${label} &mdash; <code>${esc(r.condition)}</code> <span class="tag">${esc(r.cell)}</span></h3>
  <div class="lbl">What the model saw</div>${ctx(r.context)}
  <div class="lbl">Reasoning</div><pre>${esc(r.reasoning)}</pre>
  <div class="lbl">Answer</div><pre>${esc(r.answer)}</pre>
  <div class="lbl">Judge</div>
  <div class="judge">${esc(r.reasoning_judge)}</div>
  <div class="judge">${esc(r.answer_judge)}</div></div>`; }
function render(){
  const n = PAIRS.length;
  document.getElementById("nkeep").textContent = Object.values(marks).filter(x=>x.v==="keep").length;
  document.getElementById("nnote").textContent = Object.values(marks).filter(x=>x.n).length;
  if (i >= n) { document.getElementById("app").innerHTML =
      `<div class="done"><h2>Done &mdash; ${n} reviewed.</h2><button onclick="showKept()">Show kept</button></div>`;
    document.getElementById("pos").textContent = `${n} / ${n}`;
    document.getElementById("bar").style.width = "100%"; return; }
  const p = PAIRS[i], e = entry(p.id), m = e.v;
  document.getElementById("pos").textContent = `${i+1} / ${n}`;
  document.getElementById("meta").textContent = p.model + " | " + p.id.split("/")[1] + "  \u00b7  " + p.axis + (m ? "  \u00b7  " + m : "");
  document.getElementById("bar").style.width = (100*i/n) + "%";
  document.getElementById("app").innerHTML =
    `<div class="q"><b>Question.</b> ${esc(p.question)}</div>
     <div class="cols">${side(p.verbalized,"v","Verbalized (user channel)")}${side(p.covert,"c","Covert (tool channel)")}</div>
     <div class="note${e.n ? " filled" : ""}" id="notebox">
       <div class="lbl">Note (saved as you type)</div>
       <textarea id="note" placeholder="e.g. hallucinated links in the covert answer">${esc(e.n)}</textarea>
     </div>`;
  document.getElementById("note").addEventListener("input", ev => {
    entry(p.id).n = ev.target.value; save();
    document.getElementById("notebox").className = "note" + (ev.target.value ? " filled" : "");
    document.getElementById("nnote").textContent = Object.values(marks).filter(x => x.n).length;
  });
  window.scrollTo(0,0);
}
function mark(v){ entry(PAIRS[i].id).v = v; save(); i++; render(); }
function prev(){ if (i>0) { i--; render(); } }
function showKept(){
  // Everything decided or annotated, so a note on a skipped pair is not lost.
  const out = PAIRS.filter(p => marks[p.id] && (marks[p.id].v || marks[p.id].n))
    .map(p => ({ id: p.id, model: p.model, scenario: p.id.split("/")[1], axis: p.axis,
                 decision: marks[p.id].v, note: marks[p.id].n }));
  const kept = out.filter(o => o.decision === "keep").length;
  document.getElementById("app").innerHTML =
    `<div class="q"><b>${kept} kept, ${out.filter(o=>o.note).length} with notes.</b>
     Copy this out — it is the whole review.
     <textarea readonly onclick="this.select()">${esc(JSON.stringify(out, null, 2))}</textarea>
     <button onclick="i=0;render()">Back to review</button>
     <button onclick="if(confirm('Clear all decisions and notes?')){marks={};save();i=0;render()}">Reset</button></div>`;
}
addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "k") mark("keep");
  if (e.key === "s") mark("skip");
  if (e.key === "ArrowLeft") prev();
});
render();
</script>
"""


def _row_data(row: dict, by_id: dict[str, dict]) -> dict:
    """Flatten one side of a pair for the review page."""
    j = row["judge"]
    return {
        "condition": row["condition"],
        "context": context_messages(row, by_id),
        "cell": CELL_OF[row["context_type"]],
        "reasoning": row["reasoning"],
        "answer": row["raw_answer"],
        "reasoning_judge": j["reasoning_explanation"],
        "answer_judge": j["answer_explanation"],
    }


def write_html(pairs: list[tuple[dict, dict]], path: Path, title: str, key: str) -> None:
    """Write a self-contained keep/skip review page. Decisions live in localStorage."""
    by_id = {r["id"]: r for r in FaceEval().dataset}
    data = [
        {
            "id": f"{v['_model']}/{v['scenario_id']}",
            "model": v["_model"],
            "axis": v["axis"],
            "question": v["question"],
            "verbalized": _row_data(v, by_id),
            "covert": _row_data(c, by_id),
        }
        for v, c in pairs
    ]
    # </ would close the surrounding <script> early.
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = _PAGE.replace("__DATA__", blob).replace("__TITLE__", title).replace("__KEY__", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Model directory name under --results-dir.")
    parser.add_argument(
        "--all-families",
        action="store_true",
        help="Use the largest model of every family instead of one --model, pooled into one output.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--convention", default="", help="Convention suffix on the seed dir, e.g. C3.")
    parser.add_argument("--results-dir", default="results/agentic")
    parser.add_argument("--n", type=int, default=5, help="How many scenarios to render.")
    parser.add_argument("--max-chars", type=int, default=1200, help="Clip reasoning and answer. 0 keeps them whole.")
    parser.add_argument(
        "--any-side",
        action="store_true",
        help="Allow a pair whose halves cue opposite preference sides. Off by default: a same-side pair "
        "isolates the channel, a mixed one confounds it with which side was cued.",
    )
    parser.add_argument(
        "--any-direction",
        action="store_true",
        help="Also keep pairs that are not user-verbalized/tool-covert, including the reverse.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write markdown here instead of stdout.")
    parser.add_argument("--html", type=Path, default=None, help="Write a keep/skip review page here.")
    parser.add_argument(
        "--ids",
        type=Path,
        default=None,
        help="JSON exported from the review page. Keeps only those pairs, in that order, with their notes.",
    )
    args = parser.parse_args()

    if bool(args.model) == args.all_families:
        raise SystemExit("Pass exactly one of --model or --all-families")

    seed_dir = f"seed_{args.seed}" + (f"_{args.convention}" if args.convention else "")
    if args.all_families:
        models = largest_per_family(args.results_dir, args.seed, args.convention or "C0")
        if not models:
            raise SystemExit(f"No judged {args.convention or 'C0'} seed-{args.seed} runs in {args.results_dir}")
    else:
        models = [args.model]

    by_model: dict[str, list[tuple[dict, dict]]] = {}
    for model in models:
        run_dir = Path(args.results_dir) / model / seed_dir
        if not run_dir.exists():
            raise SystemExit(f"No run at {run_dir}")
        rows = load_merged_results(run_dir)
        for r in rows:
            r["_model"] = model
        by_model[model] = find_pairs(rows, same_side=not args.any_side, headline_only=not args.any_direction)
        print(f"  {model:38s} {len(by_model[model]):3d} pairs")

    pairs = interleave_models(by_model)
    if not pairs:
        raise SystemExit("No verbalized/covert pairs found")

    notes: dict[str, str] = {}
    if args.ids:
        notes = load_selection(args.ids)
        keyed = {f"{v['_model']}/{v['scenario_id']}": (v, c) for v, c in pairs}
        missing = [k for k in notes if k not in keyed]
        if missing:
            raise SystemExit(f"Not found among the pairs: {', '.join(missing)}")
        pairs = [keyed[k] for k in notes]

    chosen = pairs if args.ids else pairs[: args.n]
    headline = sum(1 for p in pairs if _is_headline(p))
    label = f"{len(models)} families" if args.all_families else args.model
    index = "\n".join(
        f"{i}. {v['_model']} — `{v['scenario_id']}` ({v['axis']}): "
        f"verbalized on `{v['condition']}`, covert on `{c['condition']}`"
        for i, (v, c) in enumerate(chosen, 1)
    )
    body = "\n".join(
        [
            f"# Verbalized vs covert commitment — {label}, {seed_dir}",
            "",
            f"{len(pairs)} scenarios qualify ({headline} verbalized on the user channel and covert "
            f"on the tool channel). Showing {len(chosen)}, one per scenario, axes interleaved.",
            "",
            index,
            "",
            "\n".join(
                render(p, i, args.max_chars, notes.get(f"{p[0]['_model']}/{p[0]['scenario_id']}", ""))
                for i, p in enumerate(chosen, 1)
            ),
        ]
    )

    if args.html:
        write_html(chosen, args.html, f"{label} {seed_dir}", f"{label}-{seed_dir}".replace(" ", "-"))
        print(f"Saved {args.html} ({len(chosen)} of {len(pairs)} scenarios) — open it in a browser")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
        print(f"Saved {args.out} ({len(chosen)} of {len(pairs)} scenarios)")
    if not args.html and not args.out:
        print(body)


if __name__ == "__main__":
    main()
