"""Tests for adding a second judge alongside the pre-registered Anthropic one.

The GPT judge is an addition, not a replacement: the Anthropic judge keeps writing
judged.jsonl with an unchanged request payload, and every default stays put.
"""

from __future__ import annotations

import json

import pytest

from src.results.storage import DEFAULT_JUDGE_MODEL, judged_filename

_VERDICT = '{"acknowledges_preference": true, "tailoring_explicit": true, "eval_awareness": false, "explanation": "x"}'
_ANSWER_VERDICT = '{"aligns_with_preference": true, "committed": true, "stance_label": "A", "explanation": "x"}'

_ROWS = [
    {
        "id": "row_0",
        "axis": "political",
        "condition": "explicit_liberal",
        "question": "Which policy?",
        "reasoning": "user prefers liberal, align",
        "raw_answer": "I support policy A.",
    },
    {
        "id": "row_1",
        "axis": "political",
        "condition": "explicit_liberal",
        "question": "Which policy?",
        "reasoning": "weigh both sides",
        "raw_answer": "It depends.",
    },
]


def _judge_config(tmp_path, name: str, model: str, backend: str | None = None, reasoning_effort: str | None = None):
    backend_line = f'backend: "{backend}"\n' if backend else ""
    if reasoning_effort:
        backend_line += f'reasoning_effort: "{reasoning_effort}"\n'
    cfg = tmp_path / name
    cfg.write_text(
        f"""
model: "{model}"
{backend_line}max_tokens: 128
temperature: 0.0

reasoning_judge:
  system_prompt: |
    Judge the reasoning trace.
  user_prompt_template: |
    Q: {{question}}
    Reasoning: {{reasoning}}

answer_judge:
  system_prompt: |
    Judge the final answer.
  user_prompt_template: |
    Q: {{question}}
    Answer: {{answer}}
"""
    )
    return cfg


class _RecordingBatchLLM:
    """Stand-in for AnthropicLLM: records submitted chunks, answers every request."""

    submitted: list[list[dict]] = []

    def __init__(self, model, use_batch=False, **kwargs):
        self.model = model
        type(self).submitted = []

    def create_batch(self, requests):
        type(self).submitted.append(requests)
        return f"batch_{len(type(self).submitted)}"

    def poll_batch(self, batch_id):
        out = {}
        for chunk in type(self).submitted:
            for req in chunk:
                cid = req["custom_id"]
                out[cid] = _VERDICT if cid.startswith("r__") else _ANSWER_VERDICT
        return out


class _RecordingOpenAILLM(_RecordingBatchLLM):
    """Stand-in for OpenAILLM, including its /v1/responses request shape."""

    last_reasoning_effort: str | None = None

    def __init__(self, model, use_batch=False, reasoning_effort=None, **kwargs):
        super().__init__(model, use_batch=use_batch, **kwargs)
        type(self).last_reasoning_effort = reasoning_effort

    def build_batch_request(self, custom_id, messages, max_tokens, temperature, tools=None):
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": self.model, "input": messages, "max_output_tokens": max_tokens},
        }


class TestJudgedFilename:
    def test_default_judge_keeps_the_unsuffixed_file(self):
        assert judged_filename(DEFAULT_JUDGE_MODEL) == "judged.jsonl"

    def test_other_judges_are_keyed_by_model(self):
        assert judged_filename("gpt-5.6-luna") == "judged__gpt-5.6-luna.jsonl"

    def test_slashes_are_sanitised(self):
        assert judged_filename("openai/gpt-5.6") == "judged__openai_gpt-5.6.jsonl"

    def test_two_judges_never_collide(self):
        names = {judged_filename(m) for m in (DEFAULT_JUDGE_MODEL, "gpt-5.6-luna", "gemini-3-pro")}
        assert len(names) == 3


class TestBackendDispatch:
    def test_anthropic_is_the_default_when_no_backend_key(self, tmp_path, monkeypatch):
        from src.evaluation import judges as mod

        monkeypatch.setattr(mod, "AnthropicLLM", _RecordingBatchLLM)
        monkeypatch.setattr(mod, "OpenAILLM", _RecordingOpenAILLM)
        cfg = _judge_config(tmp_path, "judge.yaml", DEFAULT_JUDGE_MODEL)

        mod.run_judges(_ROWS, config_path=str(cfg), use_batch=True)

        assert _RecordingBatchLLM.submitted, "Anthropic backend should have been used"

    def test_openai_backend_routes_to_openai_client(self, tmp_path, monkeypatch):
        from src.evaluation import judges as mod

        monkeypatch.setattr(mod, "AnthropicLLM", _RecordingBatchLLM)
        monkeypatch.setattr(mod, "OpenAILLM", _RecordingOpenAILLM)
        cfg = _judge_config(tmp_path, "judge_gpt.yaml", "gpt-5.6-luna", backend="openai")

        entries = mod.run_judges(_ROWS, config_path=str(cfg), use_batch=True)

        assert _RecordingOpenAILLM.submitted, "OpenAI backend should have been used"
        assert len(entries) == 2

    def test_openai_backend_without_batch_fails_loudly(self, tmp_path):
        from src.evaluation import judges as mod

        cfg = _judge_config(tmp_path, "judge_gpt.yaml", "gpt-5.6-luna", backend="openai")
        with pytest.raises(ValueError, match="batch-only"):
            mod.run_judges(_ROWS, config_path=str(cfg), use_batch=False)


class TestAnthropicPayloadUnchanged:
    """The refactor that added a second backend must not alter the first one's requests."""

    def test_request_shape_is_preserved(self, tmp_path, monkeypatch):
        from src.evaluation import judges as mod

        monkeypatch.setattr(mod, "AnthropicLLM", _RecordingBatchLLM)
        cfg = _judge_config(tmp_path, "judge.yaml", DEFAULT_JUDGE_MODEL)

        mod.run_judges(_ROWS, config_path=str(cfg), use_batch=True)

        reqs = [r for chunk in _RecordingBatchLLM.submitted for r in chunk]
        assert [r["custom_id"] for r in reqs] == ["r__0", "r__1", "a__0", "a__1"]
        params = reqs[0]["params"]
        assert set(params) == {"model", "max_tokens", "temperature", "system", "messages"}
        assert params["model"] == DEFAULT_JUDGE_MODEL
        assert params["max_tokens"] == 128
        assert params["temperature"] == 0.0
        assert params["system"] == "Judge the reasoning trace.\n"
        assert params["messages"][0]["role"] == "user"
        assert "user prefers liberal, align" in params["messages"][0]["content"]

    def test_answer_requests_carry_the_answer_judge_system_prompt(self, tmp_path, monkeypatch):
        from src.evaluation import judges as mod

        monkeypatch.setattr(mod, "AnthropicLLM", _RecordingBatchLLM)
        cfg = _judge_config(tmp_path, "judge.yaml", DEFAULT_JUDGE_MODEL)

        mod.run_judges(_ROWS, config_path=str(cfg), use_batch=True)

        answer_reqs = [r for chunk in _RecordingBatchLLM.submitted for r in chunk if r["custom_id"].startswith("a__")]
        assert all(r["params"]["system"] == "Judge the final answer.\n" for r in answer_reqs)


class TestOpenAIPayload:
    def test_system_and_user_are_sent_as_input_messages(self, tmp_path, monkeypatch):
        from src.evaluation import judges as mod

        monkeypatch.setattr(mod, "OpenAILLM", _RecordingOpenAILLM)
        cfg = _judge_config(tmp_path, "judge_gpt.yaml", "gpt-5.6-luna", backend="openai")

        mod.run_judges(_ROWS, config_path=str(cfg), use_batch=True)

        reqs = [r for chunk in _RecordingOpenAILLM.submitted for r in chunk]
        assert [r["custom_id"] for r in reqs] == ["r__0", "r__1", "a__0", "a__1"]
        messages = reqs[0]["body"]["input"]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == "Judge the reasoning trace.\n"

    def test_reasoning_effort_from_config_reaches_the_client(self, tmp_path, monkeypatch):
        """Without an effort the client sends temperature, which reasoning models reject."""
        from src.evaluation import judges as mod

        monkeypatch.setattr(mod, "OpenAILLM", _RecordingOpenAILLM)
        cfg = _judge_config(tmp_path, "j.yaml", "gpt-5.6-luna", backend="openai", reasoning_effort="none")

        mod.run_judges(_ROWS, config_path=str(cfg), use_batch=True)

        assert _RecordingOpenAILLM.last_reasoning_effort == "none"

    def test_absent_reasoning_effort_is_passed_as_none(self, tmp_path, monkeypatch):
        from src.evaluation import judges as mod

        monkeypatch.setattr(mod, "OpenAILLM", _RecordingOpenAILLM)
        cfg = _judge_config(tmp_path, "j.yaml", "gpt-5.6-luna", backend="openai")

        mod.run_judges(_ROWS, config_path=str(cfg), use_batch=True)

        assert _RecordingOpenAILLM.last_reasoning_effort is None

    def test_a_typo_in_reasoning_effort_fails_before_spending(self, tmp_path, monkeypatch):
        """An unrecognised effort is ignored downstream, which would 400 every request."""
        from src.evaluation import judges as mod

        monkeypatch.setattr(mod, "OpenAILLM", _RecordingOpenAILLM)
        cfg = _judge_config(tmp_path, "j.yaml", "gpt-5.6-luna", backend="openai", reasoning_effort="minimal")

        with pytest.raises(ValueError, match="Unknown reasoning_effort"):
            mod.run_judges(_ROWS, config_path=str(cfg), use_batch=True)

    def test_reasoning_summary_dict_is_reduced_to_content(self, tmp_path, monkeypatch):
        """A model that returns a reasoning summary yields a dict, not a string."""
        from src.evaluation import judges as mod

        class _SummaryLLM(_RecordingOpenAILLM):
            def poll_batch(self, batch_id):
                return {
                    cid: {
                        "reasoning": "thinking out loud",
                        "content": _VERDICT if cid.startswith("r__") else _ANSWER_VERDICT,
                    }
                    for chunk in type(self).submitted
                    for cid in (r["custom_id"] for r in chunk)
                }

        monkeypatch.setattr(mod, "OpenAILLM", _SummaryLLM)
        cfg = _judge_config(tmp_path, "judge_gpt.yaml", "gpt-5.6-luna", backend="openai")

        entries = mod.run_judges(_ROWS, config_path=str(cfg), use_batch=True)

        assert all(e["judge"]["reasoning_parse_ok"] for e in entries)
        assert all(e["judge"]["answer_parse_ok"] for e in entries)


class TestRunJudgeStageCoexistence:
    """Both judges write into the same run dir without touching each other's file."""

    def _run(self, tmp_path, monkeypatch, cfg, llm_attr, stub):
        from src.evaluation import judges as mod
        from src.pipeline import run_judge_stage

        monkeypatch.setattr(mod, llm_attr, stub)
        return run_judge_stage(
            model="test/model",
            seed=42,
            output_dir=str(tmp_path),
            judge_config_path=str(cfg),
            use_batch=True,
        )

    def test_gpt_judge_writes_a_separate_file_and_leaves_the_first_intact(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "test_model" / "seed_42"
        run_dir.mkdir(parents=True)
        with open(run_dir / "inference.jsonl", "w") as f:
            for row in _ROWS:
                f.write(json.dumps(row) + "\n")

        anthropic_cfg = _judge_config(tmp_path, "judge.yaml", DEFAULT_JUDGE_MODEL)
        self._run(tmp_path, monkeypatch, anthropic_cfg, "AnthropicLLM", _RecordingBatchLLM)

        judged = run_dir / "judged.jsonl"
        assert judged.exists()
        before = judged.read_bytes()

        gpt_cfg = _judge_config(tmp_path, "judge_gpt.yaml", "gpt-5.6-luna", backend="openai")
        self._run(tmp_path, monkeypatch, gpt_cfg, "OpenAILLM", _RecordingOpenAILLM)

        assert (run_dir / "judged__gpt-5.6-luna.jsonl").exists()
        assert judged.read_bytes() == before, "the Anthropic judge's file must not be touched"

    def test_each_judge_records_its_own_provenance(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "test_model" / "seed_42"
        run_dir.mkdir(parents=True)
        with open(run_dir / "inference.jsonl", "w") as f:
            for row in _ROWS:
                f.write(json.dumps(row) + "\n")

        self._run(
            tmp_path,
            monkeypatch,
            _judge_config(tmp_path, "j.yaml", DEFAULT_JUDGE_MODEL),
            "AnthropicLLM",
            _RecordingBatchLLM,
        )
        self._run(
            tmp_path,
            monkeypatch,
            _judge_config(tmp_path, "jg.yaml", "gpt-5.6-luna", backend="openai"),
            "OpenAILLM",
            _RecordingOpenAILLM,
        )

        meta = json.loads((run_dir / "metadata.json").read_text())
        assert meta["judged"]["judge_model"] == DEFAULT_JUDGE_MODEL
        assert meta["judged__gpt-5.6-luna"]["judge_model"] == "gpt-5.6-luna"

    def test_resume_reads_only_its_own_judge_file(self, tmp_path, monkeypatch):
        """The GPT judge must not treat the Anthropic verdicts as already-done work."""
        run_dir = tmp_path / "test_model" / "seed_42"
        run_dir.mkdir(parents=True)
        with open(run_dir / "inference.jsonl", "w") as f:
            for row in _ROWS:
                f.write(json.dumps(row) + "\n")

        from src.evaluation import judges as mod
        from src.pipeline import run_judge_stage

        monkeypatch.setattr(mod, "AnthropicLLM", _RecordingBatchLLM)
        run_judge_stage(
            model="test/model",
            seed=42,
            output_dir=str(tmp_path),
            judge_config_path=str(_judge_config(tmp_path, "j.yaml", DEFAULT_JUDGE_MODEL)),
            use_batch=True,
        )

        monkeypatch.setattr(mod, "OpenAILLM", _RecordingOpenAILLM)
        entries = run_judge_stage(
            model="test/model",
            seed=42,
            output_dir=str(tmp_path),
            judge_config_path=str(_judge_config(tmp_path, "jg.yaml", "gpt-5.6-luna", backend="openai")),
            use_batch=True,
            resume=True,
        )

        assert len(entries) == 2
        assert _RecordingOpenAILLM.submitted, "resume must not skip rows the other judge already scored"


class TestResultsDBSelectsJudge:
    def test_load_all_reads_the_requested_judge_file(self, tmp_path):
        from src.results.db import ResultsDB

        run_dir = tmp_path / "test_model" / "seed_42"
        run_dir.mkdir(parents=True)
        with open(run_dir / "inference.jsonl", "w") as f:
            f.write(json.dumps({"id": "row_0", "condition": "explicit_liberal", "context_type": "explicit"}) + "\n")
        with open(run_dir / "judged.jsonl", "w") as f:
            f.write(json.dumps({"id": "row_0", "judge": {"answer_tailored": True}}) + "\n")
        with open(run_dir / "judged__gpt-5.6-luna.jsonl", "w") as f:
            f.write(json.dumps({"id": "row_0", "judge": {"answer_tailored": False}}) + "\n")

        default_db = ResultsDB.load_all(str(tmp_path), slim=False)
        gpt_db = ResultsDB.load_all(str(tmp_path), slim=False, judged_file="judged__gpt-5.6-luna.jsonl")

        assert default_db.records[0]["judge"]["answer_tailored"] is True
        assert gpt_db.records[0]["judge"]["answer_tailored"] is False

    def test_each_judge_gets_its_own_slim_cache(self, tmp_path):
        from src.results.db import ResultsDB

        run_dir = tmp_path / "test_model" / "seed_42"
        run_dir.mkdir(parents=True)
        with open(run_dir / "inference.jsonl", "w") as f:
            f.write(json.dumps({"id": "row_0", "condition": "explicit_liberal", "context_type": "explicit"}) + "\n")
        # answer_committed survives the slim projection; answer_tailored is derived later.
        with open(run_dir / "judged.jsonl", "w") as f:
            f.write(json.dumps({"id": "row_0", "judge": {"answer_committed": True}}) + "\n")
        with open(run_dir / "judged__gpt-5.6-luna.jsonl", "w") as f:
            f.write(json.dumps({"id": "row_0", "judge": {"answer_committed": False}}) + "\n")

        ResultsDB.load_all(str(tmp_path))
        ResultsDB.load_all(str(tmp_path), judged_file="judged__gpt-5.6-luna.jsonl")

        # Reloading must come from each judge's own cache, not the other's.
        assert ResultsDB.load_all(str(tmp_path)).records[0]["judge"]["answer_committed"] is True
        assert (
            ResultsDB.load_all(str(tmp_path), judged_file="judged__gpt-5.6-luna.jsonl").records[0]["judge"][
                "answer_committed"
            ]
            is False
        )
