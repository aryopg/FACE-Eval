"""Lightweight query interface for faithfulness evaluation results."""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.results.storage import DEFAULT_JUDGED_FILE, discover_runs, load_merged_results

# Set to load the slim cache alone, on a box that holds no run directories — see
# `_load_cache_without_run_dirs`.
_CACHE_ONLY_ENV = "FACE_DB_CACHE_ONLY"


def _cache_relpaths(judged_file: str) -> tuple[str, str]:
    """Cache and stamp paths, kept separate per judge so alternating judges don't thrash."""
    stem = f".cache/db_slim_{_PROJECTION_ID}"
    if judged_file != DEFAULT_JUDGED_FILE:
        stem = f"{stem}__{judged_file.removesuffix('.jsonl')}"
    return f"{stem}.pkl", f"{stem}.stamp"


def cache_glob() -> str:
    """Glob matching the current cache files inside a substrate dir, for sync_results."""
    return f".cache/db_slim_{_PROJECTION_ID}*"


# Top-level inference fields kept by the slim projection. Everything else
# (question, reasoning, raw_answer, ...) gets dropped at load time.
_KEEP_TOP: frozenset[str] = frozenset(
    {
        "id",
        "axis",
        "condition",
        "context_type",
        "scenario_id",
        "source",
        "_model",
        "_seed",
        "_convention",
    }
)
# Judge fields read by plotting / analysis scripts. The fat text fields
# (reasoning_explanation, raw_*_judge, etc.) are dropped.
_KEEP_JUDGE: frozenset[str] = frozenset(
    {
        "reasoning_acknowledges_preference",
        "reasoning_tailoring_explicit",
        "reasoning_parse_ok",
        "answer_aligns_with_preference",
        "answer_committed",
        "answer_parse_ok",
        "answer_stance_label",
        # Dropping this one silently emptied any filter keyed on it: the field is
        # in judged.jsonl, so the filter looked right, and the figure came out all
        # zeros instead of failing.
        "answer_tailored",
        "reasoning_eval_awareness",
    }
)

# The cache holds the projection above, not the whole record, and the stamp only covers
# run mtimes -- it cannot tell that the field list changed. Naming the file after a
# fingerprint of that list closes the gap without anyone having to remember: edit either
# set and the cache lands under a new name, leaving the old one readable for a rollback.
_PROJECTION_ID = hashlib.sha256(repr((sorted(_KEEP_TOP), sorted(_KEEP_JUDGE))).encode()).hexdigest()[:8]

# One token counter per model, built lazily so the download is paid only on a cold
# cache build. `reasoning_tokens` is a mixed ruler: effort-swept models get their own
# tokenizer, everything else is counted on o200k_harmony. Widen this before comparing
# CoT length across the whole model set.
_COUNTERS: dict[str, Callable[[str], int]] = {}
_FALLBACK_WARNED: set[str] = set()
# Run dirs suffix the reasoning effort; the repo id is the part before it.
_EFFORT_SUFFIX = re.compile(r"_(?:low|medium|high|max|\d+(?:\.\d+)?)$")


def _hf_repo_id(model_dir: str) -> str:
    """Results dir name -> HuggingFace repo id (org names carry no underscore)."""
    return _EFFORT_SUFFIX.sub("", model_dir).replace("_", "/", 1)


def _shared_counter() -> Callable[[str], int]:
    """o200k_harmony: gpt-oss's own encoding, and the fallback for everything else."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_harmony")
    return lambda text: len(enc.encode(text, disallowed_special=()))


def _counter_for(model_dir: str | None) -> Callable[[str], int]:
    """An effort-swept model's own token counter; o200k_harmony for everything else.

    A failed download or import falls back to the shared encoding and names the
    cause, because that silently puts an H4 x-axis on the wrong ruler.
    """
    key = model_dir or ""
    if key in _COUNTERS:
        return _COUNTERS[key]

    counter = None
    if model_dir and _EFFORT_SUFFIX.search(model_dir):
        repo = _hf_repo_id(model_dir)
        try:
            from transformers import AutoTokenizer

            # trust_remote_code=False also stops transformers prompting on stdin,
            # which would hang an unattended cache build.
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
            counter = lambda text: len(tok.encode(text, add_special_tokens=False))  # noqa: E731
        except Exception as exc:  # noqa: BLE001 - any failure means fall back, loudly
            if key not in _FALLBACK_WARNED:
                _FALLBACK_WARNED.add(key)
                first_line = str(exc).strip().split("\n")[0]
                reason = f"{type(exc).__name__}: {first_line}"
                print(
                    f"reasoning_tokens: could not load the {repo} tokenizer ({reason}); counting "
                    f"{model_dir} on o200k_harmony instead, so its CoT lengths are an estimate"
                )
    _COUNTERS[key] = counter or _shared_counter()
    return _COUNTERS[key]


def _tokens_of(text: str | None, model_dir: str | None) -> int:
    if not text:
        return 0
    return _counter_for(model_dir)(text)


_SOURCE_TOKENS: tuple[str, ...] = ("browser_history", "email", "slack", "notes")


def _source_from_id(record_id: str | None) -> str | None:
    """Derive source type from record id for runs that pre-date source logging."""
    if not record_id:
        return None
    for src in _SOURCE_TOKENS:
        if f"_{src}_" in record_id:
            return src
    return "profile"


def _project(record: dict) -> dict:
    slim: dict[str, Any] = {k: record.get(k) for k in _KEEP_TOP if k in record}
    slim["reasoning_tokens"] = _tokens_of(record.get("reasoning"), record.get("_model"))
    # source was not written to inference.jsonl in older runs; derive from id
    if slim.get("source") is None and slim.get("context_type") != "none":
        slim["source"] = _source_from_id(record.get("id"))
    judge = record.get("judge")
    if isinstance(judge, dict):
        slim["judge"] = {k: judge.get(k) for k in _KEEP_JUDGE if k in judge}
    else:
        slim["judge"] = judge
    return slim


def _cache_stamp(runs: list[dict], require_judged: bool, judged_file: str) -> tuple:
    entries: list[tuple] = []
    for run in runs:
        run_path = Path(run["path"])
        inf = run_path / "inference.jsonl"
        jud = run_path / judged_file
        entries.append(
            (
                str(run_path),
                inf.stat().st_mtime if inf.exists() else 0.0,
                jud.stat().st_mtime if jud.exists() else 0.0,
            )
        )
    return (_PROJECTION_ID, require_judged, judged_file, tuple(sorted(entries)))


def _try_load_cache(results_dir: str, runs: list[dict], require_judged: bool, judged_file: str) -> list[dict] | None:
    cache_rel, stamp_rel = _cache_relpaths(judged_file)
    cache_path = Path(results_dir) / cache_rel
    stamp_path = Path(results_dir) / stamp_rel
    if not (cache_path.exists() and stamp_path.exists()):
        return None
    try:
        cached_stamp = pickle.loads(stamp_path.read_bytes())
        if cached_stamp != _cache_stamp(runs, require_judged, judged_file):
            return None
        return pickle.loads(cache_path.read_bytes())
    except (pickle.UnpicklingError, EOFError, OSError):
        return None


def _load_cache_without_run_dirs(results_dir: str, require_judged: bool, judged_file: str) -> list[dict]:
    """Load the slim cache on a box that holds the cache but not the runs behind it.

    The normal path stamps the cache with every run's mtimes, which is the one thing such
    a box cannot check — and `discover_runs` would find nothing there anyway. The rest of
    the stamp (schema version, require_judged, judged file) is still checked, so what is
    taken on trust is only that the cache is current with the results it was built from.
    """
    cache_rel, stamp_rel = _cache_relpaths(judged_file)
    cache_path = Path(results_dir) / cache_rel
    stamp_path = Path(results_dir) / stamp_rel
    if not (cache_path.exists() and stamp_path.exists()):
        raise SystemExit(
            f"{_CACHE_ONLY_ENV} is set but {cache_path} is not there. "
            "Fetch it with `python sync_results.py download --cache-only`, or unset "
            f"{_CACHE_ONLY_ENV} to read the run directories."
        )
    projection, cached_require_judged, cached_judged_file, entries = pickle.loads(stamp_path.read_bytes())
    if (projection, cached_require_judged, cached_judged_file) != (_PROJECTION_ID, require_judged, judged_file):
        raise SystemExit(
            f"{cache_path} was built for ({projection}, require_judged={cached_require_judged}, "
            f"{cached_judged_file}) but this caller asked for ({_PROJECTION_ID}, "
            f"require_judged={require_judged}, {judged_file}). Rebuild it where the run directories are."
        )
    records = pickle.loads(cache_path.read_bytes())
    newest = max((max(inf, jud) for _path, inf, jud in entries), default=0.0)
    built_from = datetime.fromtimestamp(newest).strftime("%Y-%m-%d") if newest else "unknown"
    print(
        f"{_CACHE_ONLY_ENV}: {len(records)} records over {len(entries)} runs from {cache_path}, "
        f"newest result {built_from}. Run directories were not read."
    )
    return records


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write `payload` to `path` so a reader sees either the old file or the whole new one."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _save_cache(
    results_dir: str, runs: list[dict], require_judged: bool, judged_file: str, records: list[dict]
) -> None:
    cache_rel, stamp_rel = _cache_relpaths(judged_file)
    cache_path = Path(results_dir) / cache_rel
    stamp_path = Path(results_dir) / stamp_rel
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a per-process temp file and rename it into place. A plain write to
        # cache_path leaves a partial file under the real name for as long as the ~350MB
        # takes to land, and the stamp that follows would then certify it as current --
        # a corrupt cache reads as a valid one, which is silent rather than loud. The
        # pid keeps two concurrent builders off each other's temp file.
        _write_atomic(cache_path, pickle.dumps(records, protocol=pickle.HIGHEST_PROTOCOL))
        # Stamp last: it is what marks the cache usable, so it must not outrun the data.
        _write_atomic(
            stamp_path, pickle.dumps(_cache_stamp(runs, require_judged, judged_file), protocol=pickle.HIGHEST_PROTOCOL)
        )
    except OSError:
        pass  # cache is best-effort; a failed write shouldn't break the loader


def _get_nested(record: dict, key: str) -> Any:
    """Get a value from a dict using dotted key notation."""
    parts = key.split(".")
    val = record
    for part in parts:
        if not isinstance(val, dict):
            return None
        val = val.get(part)
    return val


def paired_rate_ci(
    arm_a: list[dict],
    arm_b: list[dict],
    field: str,
    cluster_key: str = "scenario_id",
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Joint cluster bootstrap on rate(B) - rate(A). Returns (point, ci_lo, ci_hi).

    One scenario resample indexes both arms, so the correlation between them survives.
    Differencing two independently bootstrapped intervals instead treats paired arms as
    independent and inflates the width; that is why a marginal-overlap check is a
    conservative stand-in for this, not an equivalent of it. Clusters missing from an arm
    contribute (0, 0) to it, which is what makes the union of keys the right index set.

    `field` must already be a plain 0/1 on each record, and the caller must have
    restricted the records to the rate's denominator (for a conditional rate, pass only
    the conditioning rows). Precomputing the indicator is what lets the resample run
    vectorised rather than re-aggregating dicts n_boot times.
    """

    def _stats(records: list[dict], keys: list) -> np.ndarray:
        by_cluster: dict = {}
        for r in records:
            by_cluster.setdefault(_get_nested(r, cluster_key), []).append(r[field])
        return np.array([(sum(by_cluster.get(k, ())), len(by_cluster.get(k, ()))) for k in keys], dtype=float)

    keys = list({_get_nested(r, cluster_key) for r in arm_a} | {_get_nested(r, cluster_key) for r in arm_b})
    if not keys:
        return float("nan"), float("nan"), float("nan")
    a_stats, b_stats = _stats(arm_a, keys), _stats(arm_b, keys)

    def _rate(stats: np.ndarray, axis: int | None = None):
        s, c = stats[..., 0].sum(axis=axis), stats[..., 1].sum(axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(c > 0, s / c, np.nan)

    point = float(_rate(b_stats) - _rate(a_stats))
    if len(keys) < 2 or n_boot < 2:
        return point, point, point

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    deltas = _rate(b_stats[idx], axis=1) - _rate(a_stats[idx], axis=1)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size < 2:
        return point, point, point
    return point, float(np.quantile(deltas, alpha / 2)), float(np.quantile(deltas, 1 - alpha / 2))


class ResultsDB:
    """Chainable query interface over a list of result dicts."""

    def __init__(self, records: list[dict]):
        self._records = records

    @classmethod
    def load_all(
        cls,
        results_dir: str = "results/agentic",
        require_judged: bool = False,
        slim: bool = True,
        judged_file: str = DEFAULT_JUDGED_FILE,
    ) -> ResultsDB:
        """Load all results across models and seeds.

        Args:
            results_dir: Path to results directory.
            require_judged: If True, only load runs that have the judge file.
            judged_file: Which judge's verdicts to merge in. Defaults to the
              pre-registered judge; pass `judged_filename(model)` for another.
            slim: If True (default), project records to the fields actually
              used by plotting/analysis (~120 bytes/record) and precompute
              `reasoning_tokens` (see `_counter_for`). Cuts
              memory ~50x and enables an mtime-stamped pickle cache at
              `<results_dir>/.cache/db_slim_v{N}.pkl`. Pass slim=False for
              tools that need raw reasoning / raw_answer / judge text
              (transcript viewers, audits).

        Setting FACE_DB_CACHE_ONLY loads that cache on its own, for a box that has the
        cache but not the ~5GB of run directories it was built from.
        """
        if slim and os.environ.get(_CACHE_ONLY_ENV):
            return cls(_load_cache_without_run_dirs(results_dir, require_judged, judged_file))

        runs = [
            run
            for run in discover_runs(results_dir, judged_file=judged_file)
            if run["has_inference"] and (not require_judged or run["has_judged"])
        ]

        if slim:
            cached = _try_load_cache(results_dir, runs, require_judged, judged_file)
            if cached is not None:
                return cls(cached)

        all_records: list[dict] = []
        for run in runs:
            results = load_merged_results(Path(run["path"]), judged_file=judged_file)
            for r in results:
                r["_model"] = run["model"]
                r["_seed"] = run["seed"]
                r["_convention"] = run.get("convention", "C0")
                all_records.append(_project(r) if slim else r)

        if slim:
            _save_cache(results_dir, runs, require_judged, judged_file, all_records)

        return cls(all_records)

    @property
    def records(self) -> list[dict]:
        return self._records

    def filter(self, **kwargs) -> ResultsDB:
        """Keep records where all specified fields match the given values."""

        def matches(r):
            return all(_get_nested(r, k) == v for k, v in kwargs.items())

        return ResultsDB([r for r in self._records if matches(r)])

    def filter_in(self, field: str, values) -> ResultsDB:
        """Keep records where ``field``'s value is in the given iterable.

        Used for cell-membership filters where a single cell pools multiple
        context_types (e.g. ``user_explicit`` = {user_turn, user_turn_structured}).
        """
        wanted = set(values)
        return ResultsDB([r for r in self._records if _get_nested(r, field) in wanted])

    def exclude(self, **kwargs) -> ResultsDB:
        """Remove records where any specified field matches the given value."""

        def matches(r):
            return any(_get_nested(r, k) == v for k, v in kwargs.items())

        return ResultsDB([r for r in self._records if not matches(r)])

    def has(self, field: str) -> ResultsDB:
        """Keep records where field is not None."""
        return ResultsDB([r for r in self._records if _get_nested(r, field) is not None])

    def group_by(self, *keys: str) -> dict[tuple, ResultsDB]:
        """Group records by one or more keys. Returns dict of tuple -> ResultsDB."""
        groups: dict[tuple, list[dict]] = {}
        for r in self._records:
            key = tuple(_get_nested(r, k) for k in keys)
            groups.setdefault(key, []).append(r)
        return {k: ResultsDB(v) for k, v in groups.items()}

    def count(self) -> int:
        return len(self._records)

    def fraction(self, field: str) -> float:
        """Fraction of records where field is truthy."""
        if not self._records:
            return 0.0
        truthy = sum(1 for r in self._records if _get_nested(r, field))
        return truthy / len(self._records)

    def mean_sem(self, field: str) -> tuple[float, float]:
        """Mean and SEM of fraction(field) across seeds.

        Groups by _seed, computes fraction per seed, returns (mean, sem).
        """
        seed_groups = self.group_by("_seed")
        fractions = [g.fraction(field) for g in seed_groups.values()]
        if not fractions:
            return 0.0, 0.0
        arr = np.array(fractions)
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, sem

    def cluster_mean_sem(
        self,
        field: str,
        cluster_key: str = "scenario_id",
        n_boot: int = 2000,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Cluster-robust mean and SE over `cluster_key` via nonparametric bootstrap.

        Point estimate: fraction of records where field is truthy. SE: resample
        clusters with replacement, recompute fraction, take std. Captures both
        within-cluster (seed) and between-cluster (scenario) variance.

        Prefer `cluster_mean_ci` for new code — 95% CIs are what we plot.
        """
        point, b_means = self._cluster_bootstrap(field, cluster_key, n_boot, seed)
        if b_means is None:
            return point, 0.0
        return point, float(np.std(b_means, ddof=1))

    def cluster_mean_ci(
        self,
        field: str,
        cluster_key: str = "scenario_id",
        alpha: float = 0.05,
        n_boot: int = 2000,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """Cluster-robust mean and percentile bootstrap CI over `cluster_key`.

        Returns (point, ci_lo, ci_hi). The default alpha=0.05 gives a 95% CI.
        Replication unit is `cluster_key` (default scenario_id) — seeds within a
        scenario are correlated, so iid Wald SEs underestimate uncertainty by
        ~2x in this codebase.
        """
        point, b_means = self._cluster_bootstrap(field, cluster_key, n_boot, seed)
        if b_means is None:
            return point, point, point
        lo = float(np.quantile(b_means, alpha / 2))
        hi = float(np.quantile(b_means, 1 - alpha / 2))
        return point, lo, hi

    def cluster_bootstrap_ci(
        self,
        agg: Callable[[list[dict]], float | None],
        cluster_key: str = "scenario_id",
        alpha: float = 0.05,
        n_boot: int = 2000,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """Cluster bootstrap on an arbitrary per-resample aggregator.

        `agg(records)` returns a scalar (e.g., mean tokens, ratio AC/T,
        predicate fraction, paired contrast user-tool). For binary
        truthy/falsy fields use the faster `cluster_mean_ci` instead.

        Returns (point, ci_lo, ci_hi). nan-tuple if agg returns None on the
        full sample. (point, point, point) if fewer than 2 clusters or boot
        replicates collapse.
        """
        if not self._records:
            return 0.0, 0.0, 0.0
        by_cluster: dict = {}
        for r in self._records:
            by_cluster.setdefault(_get_nested(r, cluster_key), []).append(r)
        if not by_cluster:
            return 0.0, 0.0, 0.0
        cluster_lists = list(by_cluster.values())
        point = agg([r for cl in cluster_lists for r in cl])
        if point is None:
            return float("nan"), float("nan"), float("nan")
        if n_boot < 2 or len(cluster_lists) < 2:
            return float(point), float(point), float(point)
        rng = np.random.default_rng(seed)
        boot: list[float] = []
        n = len(cluster_lists)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            resample = [r for i in idx for r in cluster_lists[i]]
            v = agg(resample)
            if v is not None:
                boot.append(float(v))
        if len(boot) < 2:
            return float(point), float(point), float(point)
        arr = np.asarray(boot)
        return float(point), float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))

    def _cluster_bootstrap(
        self,
        field: str,
        cluster_key: str,
        n_boot: int,
        seed: int,
    ) -> tuple[float, np.ndarray | None]:
        """Shared bootstrap engine. Returns (point, bootstrap_means or None)."""
        if not self._records:
            return 0.0, None
        by_cluster: dict = {}
        for r in self._records:
            k = _get_nested(r, cluster_key)
            by_cluster.setdefault(k, []).append(1 if _get_nested(r, field) else 0)
        if not by_cluster:
            return 0.0, None
        stats = np.array([(sum(v), len(v)) for v in by_cluster.values()], dtype=float)
        total_sum = float(stats[:, 0].sum())
        total_count = float(stats[:, 1].sum())
        if total_count == 0:
            return 0.0, None
        point = total_sum / total_count
        if n_boot < 2 or len(stats) < 2:
            return point, None
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(stats), size=(n_boot, len(stats)))
        b_sum = stats[idx, 0].sum(axis=1)
        b_count = stats[idx, 1].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            b_means = np.where(b_count > 0, b_sum / b_count, 0.0)
        return point, b_means

    def filter_causal_dependent(self) -> ResultsDB:
        """Keep only cued rows where (a) the answer shifted stance vs. no_context
        AND (b) the no_context baseline for the same (scenario_id, _model, _seed)
        shows committed==False (model had no prior commitment before seeing the cue).

        No_context rows must still be present in self when called — this method
        reads them for the baseline lookup and excludes them from the output.
        """
        baseline: dict[tuple, bool | None] = {}
        for r in self._records:
            if r.get("context_type") == "none":
                key = (r.get("scenario_id"), r.get("_model"), r.get("_seed"))
                baseline[key] = (r.get("judge") or {}).get("answer_committed")

        def keep(r: dict) -> bool:
            if r.get("context_type") == "none":
                return False
            stance = (r.get("judge") or {}).get("answer_stance_label")
            if stance in (None, "", "none"):
                return False
            key = (r.get("scenario_id"), r.get("_model"), r.get("_seed"))
            return baseline.get(key) is False

        return ResultsDB([r for r in self._records if keep(r)])

    def filter_eval_unaware(self) -> ResultsDB:
        """Drop rows whose CoT the reasoning judge flagged as evaluation-aware.

        Only an explicit True is dropped, so rows the judge left unset survive —
        the same polarity analyze_h4_eval_awareness.py uses for its unaware subset.
        Call this after filter_causal_dependent(): that method reads the no_context
        rows for its baseline lookup, and thinning them on an unrelated judge field
        first would drop cued rows for want of a baseline.
        """

        def keep(r: dict) -> bool:
            return (r.get("judge") or {}).get("reasoning_eval_awareness") is not True

        return ResultsDB([r for r in self._records if keep(r)])

    def __repr__(self) -> str:
        return f"ResultsDB({len(self._records)} records)"
