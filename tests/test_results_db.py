from __future__ import annotations

import pickle
import sys
from types import SimpleNamespace

import pytest

from src.results import db as db_module
from src.results.db import ResultsDB, _get_nested, _hf_repo_id, _tokens_of

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "id": "q1",
        "variant_type": "control",
        "answer": 42,
        "_model": "modelA",
        "_seed": 1,
        "judge": {"uses_clue": True, "confidence": 0.9},
    },
    {
        "id": "q2",
        "variant_type": "simple_incorrect",
        "answer": 7,
        "_model": "modelA",
        "_seed": 1,
        "judge": {"uses_clue": False, "confidence": 0.3},
    },
    {
        "id": "q3",
        "variant_type": "simple_incorrect",
        "answer": 7,
        "_model": "modelA",
        "_seed": 2,
        "judge": {"uses_clue": True, "confidence": 0.8},
    },
    {
        "id": "q4",
        "variant_type": "control",
        "answer": 10,
        "_model": "modelB",
        "_seed": 1,
        "judge": {"uses_clue": False, "confidence": 0.1},
    },
    {
        "id": "q5",
        "variant_type": "complex_incorrect",
        "answer": 99,
        "_model": "modelB",
        "_seed": 2,
        "judge": None,
    },
]


@pytest.fixture
def db():
    return ResultsDB(SAMPLE_RECORDS)


@pytest.fixture
def empty_db():
    return ResultsDB([])


# ===========================================================================
# _get_nested
# ===========================================================================


class TestGetNested:
    def test_simple_key(self):
        assert _get_nested({"a": 1}, "a") == 1

    def test_dotted_key(self):
        assert _get_nested({"judge": {"uses_clue": True}}, "judge.uses_clue") is True

    def test_missing_key_returns_none(self):
        assert _get_nested({"a": 1}, "b") is None

    def test_missing_nested_key_returns_none(self):
        assert _get_nested({"judge": {"uses_clue": True}}, "judge.missing") is None

    def test_deeply_nested_key(self):
        d = {"a": {"b": {"c": {"d": 42}}}}
        assert _get_nested(d, "a.b.c.d") == 42

    def test_intermediate_not_dict_returns_none(self):
        d = {"a": "not_a_dict"}
        assert _get_nested(d, "a.b") is None

    def test_empty_dict(self):
        assert _get_nested({}, "a") is None

    def test_none_value(self):
        """Accessing a key whose value is None returns None (not an error)."""
        d = {"judge": None}
        assert _get_nested(d, "judge") is None

    def test_nested_through_none_value(self):
        """Traversing through None returns None."""
        d = {"judge": None}
        assert _get_nested(d, "judge.uses_clue") is None


# ===========================================================================
# ResultsDB.filter
# ===========================================================================


class TestFilter:
    def test_filter_by_simple_key(self, db):
        result = db.filter(variant_type="control")
        assert result.count() == 2
        assert all(r["variant_type"] == "control" for r in result.records)

    def test_filter_by_nested_key(self, db):
        result = db.filter(**{"judge.uses_clue": True})
        assert result.count() == 2
        ids = {r["id"] for r in result.records}
        assert ids == {"q1", "q3"}

    def test_filter_no_matches(self, db):
        result = db.filter(variant_type="nonexistent")
        assert result.count() == 0

    def test_filter_returns_new_db(self, db):
        result = db.filter(variant_type="control")
        assert result is not db
        assert db.count() == 5  # original unchanged

    def test_chained_filter(self, db):
        result = db.filter(variant_type="control").filter(_model="modelA")
        assert result.count() == 1
        assert result.records[0]["id"] == "q1"

    def test_filter_multiple_kwargs(self, db):
        result = db.filter(variant_type="simple_incorrect", _seed=2)
        assert result.count() == 1
        assert result.records[0]["id"] == "q3"

    def test_filter_on_empty_db(self, empty_db):
        result = empty_db.filter(variant_type="control")
        assert result.count() == 0


# ===========================================================================
# ResultsDB.exclude
# ===========================================================================


class TestExclude:
    def test_exclude_by_value(self, db):
        result = db.exclude(variant_type="control")
        assert result.count() == 3
        assert all(r["variant_type"] != "control" for r in result.records)

    def test_exclude_by_nested_key(self, db):
        result = db.exclude(**{"judge.uses_clue": True})
        # q2 (False), q4 (False), q5 (judge is None so uses_clue is None) remain
        assert result.count() == 3
        ids = {r["id"] for r in result.records}
        assert ids == {"q2", "q4", "q5"}

    def test_exclude_no_matches(self, db):
        result = db.exclude(variant_type="nonexistent")
        assert result.count() == 5

    def test_exclude_returns_new_db(self, db):
        result = db.exclude(variant_type="control")
        assert result is not db
        assert db.count() == 5

    def test_exclude_multiple_kwargs_uses_any(self, db):
        """exclude removes records where ANY field matches (not all)."""
        result = db.exclude(variant_type="control", _model="modelB")
        # q1: control -> excluded
        # q2: modelA, simple_incorrect -> kept
        # q3: modelA, simple_incorrect -> kept
        # q4: control AND modelB -> excluded
        # q5: modelB -> excluded
        assert result.count() == 2
        ids = {r["id"] for r in result.records}
        assert ids == {"q2", "q3"}

    def test_exclude_on_empty_db(self, empty_db):
        result = empty_db.exclude(variant_type="control")
        assert result.count() == 0

    def test_chain_filter_then_exclude(self, db):
        result = db.filter(_model="modelA").exclude(variant_type="control")
        assert result.count() == 2
        assert all(r["_model"] == "modelA" for r in result.records)
        assert all(r["variant_type"] != "control" for r in result.records)


# ===========================================================================
# ResultsDB.has
# ===========================================================================


class TestHas:
    def test_key_exists(self, db):
        result = db.has("judge")
        # q5 has judge=None, so it should be excluded
        assert result.count() == 4

    def test_key_does_not_exist(self, db):
        result = db.has("nonexistent_field")
        assert result.count() == 0

    def test_nested_key(self, db):
        result = db.has("judge.uses_clue")
        # q5 has judge=None, so judge.uses_clue is None
        assert result.count() == 4

    def test_has_on_empty_db(self, empty_db):
        result = empty_db.has("judge")
        assert result.count() == 0

    def test_has_returns_new_db(self, db):
        result = db.has("judge")
        assert result is not db


# ===========================================================================
# ResultsDB.group_by
# ===========================================================================


class TestGroupBy:
    def test_group_by_simple_key(self, db):
        groups = db.group_by("_model")
        assert set(groups.keys()) == {("modelA",), ("modelB",)}
        assert groups[("modelA",)].count() == 3
        assert groups[("modelB",)].count() == 2

    def test_group_by_nested_key(self, db):
        groups = db.group_by("judge.uses_clue")
        assert (True,) in groups
        assert (False,) in groups
        assert (None,) in groups  # q5
        assert groups[(True,)].count() == 2
        assert groups[(False,)].count() == 2
        assert groups[(None,)].count() == 1

    def test_group_by_multiple_keys(self, db):
        groups = db.group_by("_model", "_seed")
        assert ("modelA", 1) in groups
        assert ("modelA", 2) in groups
        assert ("modelB", 1) in groups
        assert ("modelB", 2) in groups
        assert groups[("modelA", 1)].count() == 2
        assert groups[("modelA", 2)].count() == 1

    def test_group_by_returns_resultsdb_values(self, db):
        groups = db.group_by("_model")
        for v in groups.values():
            assert isinstance(v, ResultsDB)

    def test_group_by_empty_db(self, empty_db):
        groups = empty_db.group_by("_model")
        assert groups == {}


# ===========================================================================
# ResultsDB.count
# ===========================================================================


class TestCount:
    def test_count_non_empty(self, db):
        assert db.count() == 5

    def test_count_empty(self, empty_db):
        assert empty_db.count() == 0

    def test_count_after_filter(self, db):
        assert db.filter(_model="modelA").count() == 3


# ===========================================================================
# ResultsDB.fraction
# ===========================================================================


class TestFraction:
    def test_basic_fraction(self, db):
        # judge.uses_clue is True for q1, q3 -> 2 out of 5
        frac = db.fraction("judge.uses_clue")
        assert frac == pytest.approx(2 / 5)

    def test_fraction_zero_denominator(self, empty_db):
        assert empty_db.fraction("judge.uses_clue") == 0.0

    def test_fraction_all_match(self):
        records = [
            {"flag": True},
            {"flag": True},
            {"flag": True},
        ]
        assert ResultsDB(records).fraction("flag") == pytest.approx(1.0)

    def test_fraction_none_match(self):
        records = [
            {"flag": False},
            {"flag": False},
        ]
        assert ResultsDB(records).fraction("flag") == pytest.approx(0.0)

    def test_fraction_missing_field_is_falsy(self):
        records = [
            {"a": 1},
            {"a": 1},
        ]
        # "flag" is missing -> None -> falsy
        assert ResultsDB(records).fraction("flag") == pytest.approx(0.0)

    def test_fraction_truthy_values(self):
        """Non-boolean truthy values count."""
        records = [
            {"val": 1},
            {"val": 0},
            {"val": "yes"},
            {"val": ""},
        ]
        # truthy: 1, "yes" -> 2 out of 4
        assert ResultsDB(records).fraction("val") == pytest.approx(0.5)


# ===========================================================================
# ResultsDB.mean_sem
# ===========================================================================


class TestMeanSem:
    def test_multiple_seeds(self):
        """Two seeds with different fractions produce correct mean and SEM."""
        records = [
            # Seed 1: 2 out of 4 uses_clue=True -> fraction = 0.5
            {"_seed": 1, "judge": {"uses_clue": True}},
            {"_seed": 1, "judge": {"uses_clue": True}},
            {"_seed": 1, "judge": {"uses_clue": False}},
            {"_seed": 1, "judge": {"uses_clue": False}},
            # Seed 2: 3 out of 4 uses_clue=True -> fraction = 0.75
            {"_seed": 2, "judge": {"uses_clue": True}},
            {"_seed": 2, "judge": {"uses_clue": True}},
            {"_seed": 2, "judge": {"uses_clue": True}},
            {"_seed": 2, "judge": {"uses_clue": False}},
        ]
        db = ResultsDB(records)
        mean, sem = db.mean_sem("judge.uses_clue")

        expected_mean = (0.5 + 0.75) / 2  # 0.625
        # std with ddof=1: sqrt(((0.5-0.625)^2 + (0.75-0.625)^2) / 1) = sqrt(0.03125) ~ 0.1768
        # sem = std / sqrt(2) ~ 0.125
        assert mean == pytest.approx(expected_mean)
        assert sem == pytest.approx(0.125)

    def test_single_seed(self):
        """Single seed => SEM is 0.0."""
        records = [
            {"_seed": 42, "judge": {"uses_clue": True}},
            {"_seed": 42, "judge": {"uses_clue": False}},
        ]
        db = ResultsDB(records)
        mean, sem = db.mean_sem("judge.uses_clue")
        assert mean == pytest.approx(0.5)
        assert sem == 0.0

    def test_empty_db(self, empty_db):
        mean, sem = empty_db.mean_sem("judge.uses_clue")
        assert mean == 0.0
        assert sem == 0.0

    def test_three_seeds(self):
        records = [
            # Seed 1: 1/1 = 1.0
            {"_seed": 1, "flag": True},
            # Seed 2: 0/1 = 0.0
            {"_seed": 2, "flag": False},
            # Seed 3: 1/2 = 0.5
            {"_seed": 3, "flag": True},
            {"_seed": 3, "flag": False},
        ]
        db = ResultsDB(records)
        mean, sem = db.mean_sem("flag")

        fractions = [1.0, 0.0, 0.5]
        expected_mean = sum(fractions) / 3
        import numpy as np

        arr = np.array(fractions)
        expected_sem = float(arr.std(ddof=1) / np.sqrt(3))

        assert mean == pytest.approx(expected_mean)
        assert sem == pytest.approx(expected_sem)

    def test_all_seeds_same_fraction(self):
        """When all seeds have the same fraction, SEM should be 0."""
        records = [
            {"_seed": 1, "flag": True},
            {"_seed": 1, "flag": False},
            {"_seed": 2, "flag": True},
            {"_seed": 2, "flag": False},
        ]
        db = ResultsDB(records)
        mean, sem = db.mean_sem("flag")
        assert mean == pytest.approx(0.5)
        assert sem == pytest.approx(0.0)


# ===========================================================================
# ResultsDB.load_all (mocked filesystem)
# ===========================================================================


class TestLoadAll:
    def test_load_all_basic(self, tmp_path):
        """load_all delegates to discover_runs and load_merged_results."""
        import json

        # Set up a fake results directory
        model_dir = tmp_path / "modelX"
        seed_dir = model_dir / "seed_42"
        seed_dir.mkdir(parents=True)

        records = [
            {"id": "r1", "answer": 1},
            {"id": "r2", "answer": 2},
        ]
        with open(seed_dir / "inference.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        (seed_dir / "metadata.json").write_text(json.dumps({"total_samples": 2}))

        db = ResultsDB.load_all(results_dir=str(tmp_path))
        assert db.count() == 2
        assert all(r["_model"] == "modelX" for r in db.records)
        assert all(r["_seed"] == 42 for r in db.records)

    def test_load_all_require_judged(self, tmp_path):
        """load_all with require_judged=True skips unjudged runs."""
        import json

        model_dir = tmp_path / "modelY"
        seed_dir = model_dir / "seed_1"
        seed_dir.mkdir(parents=True)

        with open(seed_dir / "inference.jsonl", "w") as f:
            f.write(json.dumps({"id": "r1"}) + "\n")
        (seed_dir / "metadata.json").write_text(json.dumps({}))

        # No judged.jsonl -> should be skipped
        db = ResultsDB.load_all(results_dir=str(tmp_path), require_judged=True)
        assert db.count() == 0

    def test_load_all_empty_dir(self, tmp_path):
        db = ResultsDB.load_all(results_dir=str(tmp_path))
        assert db.count() == 0


# ===========================================================================
# ResultsDB.__repr__
# ===========================================================================


class TestRepr:
    def test_repr(self, db):
        assert repr(db) == "ResultsDB(5 records)"

    def test_repr_empty(self, empty_db):
        assert repr(empty_db) == "ResultsDB(0 records)"


# ===========================================================================
# ResultsDB.records property
# ===========================================================================


class TestRecords:
    def test_records_returns_list(self, db):
        assert isinstance(db.records, list)
        assert len(db.records) == 5


# ===========================================================================
# ResultsDB.cluster_mean_sem
# ===========================================================================


class TestClusterMeanSem:
    def test_empty_db_returns_zeros(self, empty_db):
        mean, se = empty_db.cluster_mean_sem("flag")
        assert mean == 0.0
        assert se == 0.0

    def test_single_cluster_se_is_zero(self):
        records = [{"scenario_id": "s1", "flag": True}, {"scenario_id": "s1", "flag": True}]
        mean, se = ResultsDB(records).cluster_mean_sem("flag")
        assert mean == pytest.approx(1.0)
        assert se == pytest.approx(0.0)

    def test_all_false_returns_zero_mean(self):
        records = [{"scenario_id": f"s{i}", "flag": False} for i in range(4)]
        mean, se = ResultsDB(records).cluster_mean_sem("flag")
        assert mean == pytest.approx(0.0)

    def test_all_true_returns_one_mean(self):
        records = [{"scenario_id": f"s{i}", "flag": True} for i in range(4)]
        mean, se = ResultsDB(records).cluster_mean_sem("flag")
        assert mean == pytest.approx(1.0)

    def test_point_estimate_matches_plain_fraction(self):
        records = [
            {"scenario_id": "s1", "flag": True},
            {"scenario_id": "s2", "flag": True},
            {"scenario_id": "s3", "flag": False},
            {"scenario_id": "s4", "flag": False},
        ]
        db = ResultsDB(records)
        mean, _ = db.cluster_mean_sem("flag", n_boot=100, seed=0)
        assert mean == pytest.approx(0.5)

    def test_se_positive_with_multiple_clusters(self):
        records = [
            {"scenario_id": "s1", "flag": True},
            {"scenario_id": "s2", "flag": False},
            {"scenario_id": "s3", "flag": True},
            {"scenario_id": "s4", "flag": False},
        ]
        _, se = ResultsDB(records).cluster_mean_sem("flag", n_boot=200, seed=42)
        assert se > 0.0

    def test_n_boot_less_than_2_returns_zero_se(self):
        records = [{"scenario_id": "s1", "flag": True}, {"scenario_id": "s2", "flag": False}]
        _, se = ResultsDB(records).cluster_mean_sem("flag", n_boot=1)
        assert se == pytest.approx(0.0)

    def test_no_clusters_returns_zeros(self):
        records = [{"flag": True}]  # no scenario_id key → all map to None
        mean, se = ResultsDB(records).cluster_mean_sem("flag")
        assert mean == pytest.approx(1.0)  # one cluster with one truthy value


def test_load_all_tags_records_with_convention(tmp_path):
    import json

    from src.results.db import ResultsDB
    from src.results.storage import get_run_dir

    for conv in ("C0", "C2"):
        d = get_run_dir(str(tmp_path), "model_x", seed=42, convention=conv)
        (d / "inference.jsonl").write_text(json.dumps({"id": f"row-{conv}"}) + "\n")
        (d / "metadata.json").write_text(json.dumps({"convention": conv}))

    db = ResultsDB.load_all(str(tmp_path))
    conventions = sorted({r["_convention"] for r in db.records})
    assert conventions == ["C0", "C2"]


class TestHfRepoId:
    """Dir name -> repo id. A wrong mapping degrades silently to the shared tokenizer."""

    def test_plain_model_dir(self):
        assert _hf_repo_id("Qwen_Qwen3.5-9B") == "Qwen/Qwen3.5-9B"
        assert _hf_repo_id("allenai_Olmo-3.1-32B-Think") == "allenai/Olmo-3.1-32B-Think"

    def test_named_effort_suffix_is_dropped(self):
        assert _hf_repo_id("openai_gpt-oss-20b_high") == "openai/gpt-oss-20b"
        assert _hf_repo_id("zai-org_GLM-5.2-FP8_max") == "zai-org/GLM-5.2-FP8"

    def test_float_effort_suffix_is_dropped(self):
        assert _hf_repo_id("thinkingmachines_Inkling-NVFP4_0.99") == "thinkingmachines/Inkling-NVFP4"

    def test_version_number_in_the_name_is_not_an_effort(self):
        # "Kimi-K2.6" ends in a decimal but is not preceded by an underscore.
        assert _hf_repo_id("moonshotai_Kimi-K2.6") == "moonshotai/Kimi-K2.6"

    def test_empty_text_needs_no_tokenizer(self):
        assert _tokens_of("", "moonshotai_Kimi-K2.6") == 0
        assert _tokens_of(None, None) == 0


class TestTokenizerGating:
    """Only effort-swept dirs get their own tokenizer; the rest must not reach the hub."""

    @pytest.fixture(autouse=True)
    def counters(self, monkeypatch):
        """Per-model counters are module globals, so one test would poison the next."""
        monkeypatch.setattr(db_module, "_COUNTERS", {})
        monkeypatch.setattr(db_module, "_FALLBACK_WARNED", set())
        monkeypatch.setattr(db_module, "_shared_counter", lambda: len)

    @pytest.fixture
    def loads(self, monkeypatch):
        """Records the repo ids a load was attempted for, without touching the network."""
        attempted: list[str] = []

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(repo, **kwargs):
                attempted.append(repo)
                raise RuntimeError("offline\nsecond line dropped from the warning")

        monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))
        return attempted

    def test_effort_dir_loads_its_own_tokenizer(self, loads):
        _tokens_of("hi", "deepseek-ai_DeepSeek-V4-Flash_max")
        assert loads == ["deepseek-ai/DeepSeek-V4-Flash"]

    def test_non_effort_dir_never_loads_one(self, loads):
        assert _tokens_of("hi", "Qwen_Qwen3.5-9B") == 2
        assert _tokens_of("hi", "allenai_Olmo-3.1-32B-Think") == 2
        assert loads == []

    def test_fallback_names_the_model_and_the_cause(self, loads, capsys):
        _tokens_of("hi", "zai-org_GLM-5.2-FP8_high")
        out = capsys.readouterr().out
        assert "zai-org/GLM-5.2-FP8" in out
        assert "RuntimeError: offline" in out
        assert "second line" not in out

    def test_fallback_warns_once_per_model(self, loads, capsys):
        _tokens_of("hi", "zai-org_GLM-5.2-FP8_high")
        _tokens_of("there", "zai-org_GLM-5.2-FP8_high")
        assert capsys.readouterr().out.count("could not load") == 1
        assert loads == ["zai-org/GLM-5.2-FP8"]


class TestCacheOnlyLoad:
    """FACE_DB_CACHE_ONLY: read the slim cache on a box that holds no run directories."""

    @pytest.fixture
    def cached(self, tmp_path, monkeypatch):
        monkeypatch.setenv(db_module._CACHE_ONLY_ENV, "1")
        monkeypatch.setattr(db_module, "discover_runs", lambda *a, **k: pytest.fail("run directories must not be read"))
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()

        def write(require_judged=True, judged_file="judged.jsonl", projection=None):
            stamp = (
                db_module._PROJECTION_ID if projection is None else projection,
                require_judged,
                judged_file,
                (("run_a", 1_700_000_000.0, 1_700_000_001.0),),
            )
            # Ask the implementation for the names, so the fixture cannot drift from it.
            cache_rel, stamp_rel = db_module._cache_relpaths(judged_file)
            (tmp_path / cache_rel).write_bytes(pickle.dumps(SAMPLE_RECORDS))
            (tmp_path / stamp_rel).write_bytes(pickle.dumps(stamp))

        return write

    def test_loads_the_records(self, cached, tmp_path, capsys):
        cached()
        db = ResultsDB.load_all(str(tmp_path), require_judged=True)
        assert len(db.records) == len(SAMPLE_RECORDS)
        assert "Run directories were not read" in capsys.readouterr().out

    def test_missing_cache_points_at_the_sync_command(self, tmp_path, monkeypatch):
        monkeypatch.setenv(db_module._CACHE_ONLY_ENV, "1")
        with pytest.raises(SystemExit, match="download --cache-only"):
            ResultsDB.load_all(str(tmp_path))

    def test_require_judged_mismatch_is_refused(self, cached, tmp_path):
        cached(require_judged=False)
        with pytest.raises(SystemExit, match="require_judged"):
            ResultsDB.load_all(str(tmp_path), require_judged=True)

    def test_stale_schema_is_refused(self, cached, tmp_path):
        # A cache built when _KEEP_TOP / _KEEP_JUDGE held a different field set.
        cached(projection="0badc0de")
        with pytest.raises(SystemExit, match="was built for"):
            ResultsDB.load_all(str(tmp_path), require_judged=True)

    def test_unset_env_reads_run_dirs_as_before(self, cached, tmp_path, monkeypatch):
        cached()
        monkeypatch.delenv(db_module._CACHE_ONLY_ENV)
        monkeypatch.setattr(db_module, "discover_runs", lambda *a, **k: [])
        assert ResultsDB.load_all(str(tmp_path), require_judged=True).records == []
