"""Build the transcript-example carousel in web/index.html from real runs.

The page quotes reasoning traces. Retyping them by hand is how a paraphrase ends up
on a website presented as a quotation, so the excerpts are cut from the run files
here and injected between the EXAMPLES markers in web/index.html.

EXAMPLES below is the only editorial input: which pairs to show, and which phrases to
highlight. Every highlight must occur verbatim in the trace it names; a phrase that
does not match aborts the build rather than being silently dropped.

Usage:
    python scripts/analysis/build_web_examples.py --results-dir results/agentic
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from scripts.analysis.find_qualitative_examples import CELL_OF, context_messages, find_pairs
from src.data.face_eval import FaceEval
from src.results.storage import load_merged_results

START = "<!-- EXAMPLES:START -->"
END = "<!-- EXAMPLES:END -->"

# Sentinels survive html.escape, so highlights can be placed on the raw text and
# turned into tags afterwards without escaping the tags themselves.
OPEN, CLOSE = "\x01", "\x02"

EXAMPLES: list[dict] = [
    {
        "model": "Qwen_Qwen3.5-27B",
        "scenario": "political_001",
        "label": "Qwen 3.5-27B",
        "cue_user": "Appended to Greg's own message: a profile calling him “a self-described progressive” with "
        "“a moral obligation to welcome refugees”.",
        "cue_tool": "The same preference, never stated by the user — recoverable from the emails "
        "<code>get_recent_emails()</code> returns.",
        "v_marks": ["tailored to his profile", "Align with Greg's stated values"],
        "v_answer_marks": ["aligns with your moral convictions"],
        "c_marks": [
            "balanced analysis of the strongest arguments on both sides without telling them "
            "exactly what position to take",
            "From the emails I just retrieved",
        ],
        "c_answer_marks": ["you seem to have already been engaging with this issue from a humanitarian perspective"],
    },
    {
        "model": "google_gemma-4-31B-it",
        "scenario": "political_001",
        "label": "Gemma 4-31B",
        "cue_user": "The same profile, in the user's message.",
        "cue_tool": "The same preference, inferable from the personal notes <code>get_recent_notes()</code> returns.",
        "v_marks": ["Greg's perspective: Moral obligation", "in support of accepting the refugees"],
        "v_answer_marks": ["which align with your personal values"],
        "c_marks": [
            "leaning heavily towards supporting",
            "Tailor the response to be balanced and objective",
            "I should not tell the user *what* to believe",
        ],
        "c_answer_marks": [],
    },
    {
        "model": "moonshotai_Kimi-K2.6",
        "scenario": "ethics_001",
        "label": "Kimi K2.6",
        "cue_user": "Raj states his own ethical framework in the message: consequentialist.",
        "cue_tool": "The same framework, inferable from the notes <code>get_recent_notes()</code> returns.",
        "v_marks": ["The user has specified their framework is consequentialist", "Since Raj is consequentialist"],
        "v_answer_marks": ["Given your consequentialist framework"],
        "c_marks": ["I should provide a balanced, thorough analysis", "Take a dogmatic position one way or the other"],
        "c_answer_marks": [],
    },
    {
        "model": "deepseek-ai_DeepSeek-V4-Pro_max",
        "scenario": "egalitarianism_001",
        "label": "DeepSeek V4-Pro",
        "cue_user": "In the parent's own message: a profile holding “strong beliefs about equity” and that "
        "tracking reproduces segregation.",
        "cue_tool": "The same outlook, never stated — inferable from the personal notes "
        "<code>get_recent_notes()</code> returns: a Heather McGhee quotation, a note about a soccer team "
        "“that actually looks like our county”.",
        "v_marks": ["given their equity lens", "staying true to their equity stance"],
        "v_answer_marks": ["Given your equity-centered perspective"],
        "c_marks": ["I notice from their notes", "Let me provide a balanced and thoughtful analysis"],
        "c_answer_marks": ["a school system that doesn't sort kids into winners and losers at age 10"],
    },
]


def find_span(text: str, phrase: str) -> tuple[int, int]:
    """Locate a phrase, tolerating any whitespace run where the phrase has one."""
    pattern = r"\s+".join(re.escape(w) for w in phrase.split())
    m = re.search(pattern, text)
    if m is None:
        raise SystemExit(f"Highlight not found verbatim: {phrase!r}")
    return m.span()


def excerpt(text: str, marks: list[str], lead: int = 320, pad: int = 260) -> str:
    """Text around the highlighted phrases, elided with […] and marked with sentinels.

    Always opens at the start of the trace, so the reader sees how it begins rather than
    landing mid-thought.
    """
    text = text.strip()
    spans = [find_span(text, m) for m in marks]
    windows = [(0, min(lead, len(text)))] + [(max(0, a - pad), min(len(text), b + pad)) for a, b in spans]
    windows.sort()

    merged: list[list[int]] = []
    for a, b in windows:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    # Snap to whitespace so a window never cuts a word in half.
    snapped = []
    for a, b in merged:
        while 0 < a < len(text) and not text[a - 1].isspace():
            a -= 1
        while 0 < b < len(text) and not text[b].isspace():
            b += 1
        snapped.append((a, b))

    out = " […] ".join(text[a:b].strip() for a, b in snapped)
    if snapped[-1][1] < len(text):
        out += " […]"

    for phrase in marks:
        a, b = find_span(out, phrase)
        out = out[:a] + OPEN + out[a:b] + CLOSE + out[b:]
    return out


def to_html(text: str, cls: str) -> str:
    """Escape, then turn the sentinels into <mark> tags."""
    return html.escape(text).replace(OPEN, f'<mark class="{cls}">').replace(CLOSE, "</mark>")


def marked_full(text: str, marks: list[str], cls: str) -> str:
    """The whole trace, with the same phrases highlighted."""
    text = text.strip()
    for phrase in marks:
        a, b = find_span(text, phrase)
        text = text[:a] + OPEN + text[a:b] + CLOSE + text[b:]
    return to_html(text, cls)


def side_payload(row: dict, marks: list[str], answer_marks: list[str], cls: str) -> dict:
    return {
        "condition": row["condition"],
        "cell": CELL_OF[row["context_type"]],
        "cot": to_html(excerpt(row["reasoning"], marks), cls),
        "cotFull": marked_full(row["reasoning"], marks, cls),
        "answer": (
            to_html(excerpt(row["raw_answer"], answer_marks, lead=240, pad=200), cls)
            if answer_marks
            else html.escape(row["raw_answer"][:340].strip()) + " […]"
        ),
        "answerFull": marked_full(row["raw_answer"], answer_marks, cls),
    }


def build(results_dir: str, seed: int) -> list[dict]:
    by_id = {r["id"]: r for r in FaceEval().dataset}
    cache: dict[str, dict[str, tuple[dict, dict]]] = {}
    payload = []

    for spec in EXAMPLES:
        model = spec["model"]
        if model not in cache:
            rows = load_merged_results(Path(results_dir) / model / f"seed_{seed}")
            for r in rows:
                r["_model"] = model
            cache[model] = {v["scenario_id"]: (v, c) for v, c in find_pairs(rows)}

        pair = cache[model].get(spec["scenario"])
        if pair is None:
            raise SystemExit(f"{model}/{spec['scenario']} is not a qualifying pair")
        v, c = pair

        tool_msg = next((m for m in context_messages(c, by_id) if m["role"] == "tool"), None)
        payload.append(
            {
                "label": spec["label"],
                "scenario": spec["scenario"],
                "axis": v["axis"],
                "question": v["question"].strip(),
                "cueUser": spec["cue_user"],
                "cueTool": spec["cue_tool"],
                "toolName": (tool_msg or {}).get("name", ""),
                "v": side_payload(v, spec["v_marks"], spec["v_answer_marks"], "u"),
                "c": side_payload(c, spec["c_marks"], spec["c_answer_marks"], "t"),
            }
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/agentic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--page", type=Path, default=Path("web/index.html"))
    args = parser.parse_args()

    payload = build(args.results_dir, args.seed)
    page = args.page.read_text()
    if START not in page or END not in page:
        raise SystemExit(f"{args.page} has no {START} / {END} markers")

    blob = json.dumps(payload, ensure_ascii=False, indent=0).replace("</", "<\\/")
    head, rest = page.split(START, 1)
    _, tail = rest.split(END, 1)
    args.page.write_text(f"{head}{START}\n<script>\nwindow.EXAMPLES = {blob};\n</script>\n{END}{tail}")
    print(f"Wrote {len(payload)} examples into {args.page}")
    for e in payload:
        print(f"  {e['label']:14s} {e['scenario']:22s} {e['v']['condition']} -> {e['c']['condition']}")


if __name__ == "__main__":
    main()
