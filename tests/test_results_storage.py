from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.results.storage import (
    discover_runs,
    get_run_dir,
    load_merged_results,
    load_results,
    save_results,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# get_run_dir
# ---------------------------------------------------------------------------


class TestGetRunDir:
    def test_creates_directory(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "my-model", 42)
        assert run_dir.exists()
        assert run_dir.is_dir()

    def test_correct_path_structure(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "my-model", 42)
        assert run_dir == tmp_dir / "my-model" / "seed_42"

    def test_slash_in_model_name_sanitized(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "Qwen/Qwen3-4B", 123)
        assert run_dir == tmp_dir / "Qwen_Qwen3-4B" / "seed_123"
        assert run_dir.exists()

    def test_multiple_slashes_sanitized(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "org/sub/model", 1)
        assert run_dir == tmp_dir / "org_sub_model" / "seed_1"

    def test_idempotent(self, tmp_dir: Path):
        """Calling twice with same args returns same path without error."""
        run_dir_a = get_run_dir(str(tmp_dir), "model", 42)
        run_dir_b = get_run_dir(str(tmp_dir), "model", 42)
        assert run_dir_a == run_dir_b

    def test_different_seeds_produce_different_dirs(self, tmp_dir: Path):
        dir_a = get_run_dir(str(tmp_dir), "model", 1)
        dir_b = get_run_dir(str(tmp_dir), "model", 2)
        assert dir_a != dir_b
        assert dir_a.parent == dir_b.parent  # same model dir

    def test_reasoning_effort_appended_to_model_dir(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "openai/gpt-oss-20b", 42, reasoning_effort="high")
        assert run_dir == tmp_dir / "openai_gpt-oss-20b_high" / "seed_42"
        assert run_dir.exists()

    def test_different_efforts_produce_different_dirs(self, tmp_dir: Path):
        dir_low = get_run_dir(str(tmp_dir), "openai/gpt-oss-20b", 42, reasoning_effort="low")
        dir_high = get_run_dir(str(tmp_dir), "openai/gpt-oss-20b", 42, reasoning_effort="high")
        assert dir_low != dir_high

    def test_no_effort_unaffected(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "Qwen/Qwen3-4B", 42)
        assert run_dir == tmp_dir / "Qwen_Qwen3-4B" / "seed_42"


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_writes_jsonl_file(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        results = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
        save_results(run_dir, results, "inference")

        jsonl_path = run_dir / "inference.jsonl"
        assert jsonl_path.exists()

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": "a", "value": 1}
        assert json.loads(lines[1]) == {"id": "b", "value": 2}

    def test_writes_judged_stage(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(run_dir, [{"id": "x"}], "judged")
        assert (run_dir / "judged.jsonl").exists()

    def test_writes_metadata_json(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(run_dir, [{"id": "a"}], "inference")

        meta_path = run_dir / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "inference_completed_at" in meta
        assert meta["inference_total_samples"] == 1

    def test_metadata_includes_custom_fields(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(
            run_dir,
            [{"id": "a"}],
            "inference",
            metadata={"model": "m", "seed": 1},
        )
        meta = json.loads((run_dir / "metadata.json").read_text())
        assert meta["model"] == "m"
        assert meta["seed"] == 1

    def test_metadata_accumulates_across_stages(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(run_dir, [{"id": "a"}], "inference", metadata={"model": "m"})
        save_results(run_dir, [{"id": "a", "judge": "ok"}], "judged")

        meta = json.loads((run_dir / "metadata.json").read_text())
        assert "inference_completed_at" in meta
        assert "judged_completed_at" in meta
        assert meta["model"] == "m"

    def test_empty_results_list(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(run_dir, [], "inference")

        jsonl_path = run_dir / "inference.jsonl"
        assert jsonl_path.exists()
        assert jsonl_path.read_text() == ""

        meta = json.loads((run_dir / "metadata.json").read_text())
        assert meta["inference_total_samples"] == 0

    def test_non_serializable_fields_use_default_str(self, tmp_dir: Path):
        """json.dumps(default=str) should handle non-serializable types."""
        from datetime import datetime

        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        results = [{"id": "a", "ts": datetime(2025, 1, 1)}]
        save_results(run_dir, results, "inference")

        loaded = json.loads((run_dir / "inference.jsonl").read_text().strip().split("\n")[0])
        assert loaded["id"] == "a"
        assert "2025" in loaded["ts"]


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------


class TestLoadResults:
    def test_roundtrip(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        original = [{"id": "a", "x": 1}, {"id": "b", "x": 2}]
        save_results(run_dir, original, "inference")

        loaded = load_results(run_dir, "inference")
        assert loaded == original

    def test_missing_file_raises(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        with pytest.raises(FileNotFoundError):
            load_results(run_dir, "inference")

    def test_skips_blank_lines(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        jsonl_path = run_dir / "inference.jsonl"
        jsonl_path.write_text('{"id": "a"}\n\n{"id": "b"}\n\n')

        loaded = load_results(run_dir, "inference")
        assert len(loaded) == 2
        assert loaded[0]["id"] == "a"
        assert loaded[1]["id"] == "b"

    def test_malformed_json_raises(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        jsonl_path = run_dir / "inference.jsonl"
        jsonl_path.write_text('{"id": "a"}\nnot-json\n')

        with pytest.raises(json.JSONDecodeError):
            load_results(run_dir, "inference")


# ---------------------------------------------------------------------------
# discover_runs
# ---------------------------------------------------------------------------


class TestDiscoverRuns:
    def test_empty_directory(self, tmp_dir: Path):
        assert discover_runs(str(tmp_dir)) == []

    def test_nonexistent_directory(self, tmp_dir: Path):
        assert discover_runs(str(tmp_dir / "nope")) == []

    def test_finds_single_run(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "model-a", 42)
        save_results(run_dir, [{"id": "1"}], "inference")

        runs = discover_runs(str(tmp_dir))
        assert len(runs) == 1
        assert runs[0]["model"] == "model-a"
        assert runs[0]["seed"] == 42
        assert runs[0]["has_inference"] is True
        assert runs[0]["has_judged"] is False

    def test_finds_multiple_models_and_seeds(self, tmp_dir: Path):
        for model in ["model-a", "model-b"]:
            for seed in [1, 2]:
                run_dir = get_run_dir(str(tmp_dir), model, seed)
                save_results(run_dir, [{"id": "x"}], "inference")

        runs = discover_runs(str(tmp_dir))
        assert len(runs) == 4

        models = [(r["model"], r["seed"]) for r in runs]
        assert ("model-a", 1) in models
        assert ("model-a", 2) in models
        assert ("model-b", 1) in models
        assert ("model-b", 2) in models

    def test_sorted_by_model_then_seed(self, tmp_dir: Path):
        # Create in reverse order
        for model, seed in [("z", 9), ("a", 2), ("a", 1), ("z", 1)]:
            run_dir = get_run_dir(str(tmp_dir), model, seed)
            save_results(run_dir, [], "inference")

        runs = discover_runs(str(tmp_dir))
        keys = [(r["model"], r["seed"]) for r in runs]
        assert keys == [("a", 1), ("a", 2), ("z", 1), ("z", 9)]

    def test_detects_judged_file(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(run_dir, [{"id": "x"}], "inference")
        save_results(run_dir, [{"id": "x", "judge": "ok"}], "judged")

        runs = discover_runs(str(tmp_dir))
        assert runs[0]["has_inference"] is True
        assert runs[0]["has_judged"] is True

    def test_includes_metadata(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(run_dir, [{"id": "x"}], "inference", metadata={"model": "m"})

        runs = discover_runs(str(tmp_dir))
        assert runs[0]["metadata"]["model"] == "m"

    def test_missing_metadata_returns_empty_dict(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        # Create seed dir with inference file but no metadata.json
        (run_dir / "inference.jsonl").write_text('{"id": "x"}\n')

        runs = discover_runs(str(tmp_dir))
        assert len(runs) == 1
        assert runs[0]["metadata"] == {}

    def test_skips_non_seed_directories(self, tmp_dir: Path):
        model_dir = tmp_dir / "model"
        model_dir.mkdir()
        (model_dir / "seed_1").mkdir()
        (model_dir / "not_a_seed").mkdir()
        (model_dir / "random_dir").mkdir()

        runs = discover_runs(str(tmp_dir))
        assert len(runs) == 1
        assert runs[0]["seed"] == 1

    def test_skips_files_in_model_dir(self, tmp_dir: Path):
        model_dir = tmp_dir / "model"
        model_dir.mkdir()
        (model_dir / "seed_1").mkdir()
        (model_dir / "some_file.txt").write_text("hi")

        runs = discover_runs(str(tmp_dir))
        assert len(runs) == 1

    def test_path_field_is_correct(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        runs = discover_runs(str(tmp_dir))
        assert runs[0]["path"] == run_dir


# ---------------------------------------------------------------------------
# load_merged_results
# ---------------------------------------------------------------------------


class TestLoadMergedResults:
    def test_merge_inference_and_judged(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(
            run_dir,
            [{"id": "a", "answer": 1}, {"id": "b", "answer": 2}],
            "inference",
        )
        save_results(
            run_dir,
            [{"id": "a", "judge": "correct"}, {"id": "b", "judge": "incorrect"}],
            "judged",
        )

        merged = load_merged_results(run_dir)
        assert len(merged) == 2

        by_id = {r["id"]: r for r in merged}
        assert by_id["a"]["answer"] == 1
        assert by_id["a"]["judge"] == "correct"
        assert by_id["b"]["answer"] == 2
        assert by_id["b"]["judge"] == "incorrect"

    def test_missing_judged_returns_inference_only(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(run_dir, [{"id": "a", "answer": 1}], "inference")

        merged = load_merged_results(run_dir)
        assert len(merged) == 1
        assert merged[0]["id"] == "a"
        assert merged[0]["answer"] == 1
        assert "judge" not in merged[0]

    def test_inference_record_not_in_judged_gets_none(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(
            run_dir,
            [{"id": "a", "answer": 1}, {"id": "b", "answer": 2}],
            "inference",
        )
        save_results(
            run_dir,
            [{"id": "a", "judge": "correct"}],
            "judged",
        )

        merged = load_merged_results(run_dir)
        by_id = {r["id"]: r for r in merged}
        assert by_id["a"]["judge"] == "correct"
        assert by_id["b"]["judge"] is None

    def test_missing_inference_raises(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        with pytest.raises(FileNotFoundError):
            load_merged_results(run_dir)

    def test_preserves_all_inference_fields(self, tmp_dir: Path):
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        record = {"id": "a", "question": "q", "answer": 42, "reasoning": "because"}
        save_results(run_dir, [record], "inference")
        save_results(run_dir, [{"id": "a", "judge": "ok"}], "judged")

        merged = load_merged_results(run_dir)
        assert merged[0]["question"] == "q"
        assert merged[0]["answer"] == 42
        assert merged[0]["reasoning"] == "because"
        assert merged[0]["judge"] == "ok"

    def test_judge_field_is_extracted_from_judge_key(self, tmp_dir: Path):
        """load_merged_results uses entry.get('judge'), not the whole entry."""
        run_dir = get_run_dir(str(tmp_dir), "m", 1)
        save_results(run_dir, [{"id": "a"}], "inference")
        judge_data = {"uses_clue": True, "label": "unfaithful"}
        save_results(run_dir, [{"id": "a", "judge": judge_data}], "judged")

        merged = load_merged_results(run_dir)
        assert merged[0]["judge"] == judge_data


def test_get_run_dir_c0_has_no_suffix(tmp_path):
    from src.results.storage import get_run_dir

    run_dir = get_run_dir(str(tmp_path), "Qwen/Qwen3.5-4B", seed=42, convention="C0")
    assert run_dir.name == "seed_42"


def test_get_run_dir_c3_adds_suffix(tmp_path):
    from src.results.storage import get_run_dir

    run_dir = get_run_dir(str(tmp_path), "Qwen/Qwen3.5-4B", seed=42, convention="C3")
    assert run_dir.name == "seed_42_C3"


def test_get_run_dir_default_is_c0(tmp_path):
    from src.results.storage import get_run_dir

    run_dir = get_run_dir(str(tmp_path), "Qwen/Qwen3.5-4B", seed=42)
    assert run_dir.name == "seed_42"


def test_discover_runs_parses_convention_suffix(tmp_path):
    from src.results.storage import discover_runs, get_run_dir

    for conv in ("C0", "C1", "C3"):
        d = get_run_dir(str(tmp_path), "model_x", seed=42, convention=conv)
        (d / "inference.jsonl").write_text("")
        (d / "metadata.json").write_text("{}")

    runs = discover_runs(str(tmp_path))
    conventions = sorted(r["convention"] for r in runs)
    assert conventions == ["C0", "C1", "C3"]
    # Seed parsed independently of suffix.
    assert all(r["seed"] == 42 for r in runs)
