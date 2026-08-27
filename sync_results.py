"""Sync results to/from a private HuggingFace repo.

Usage:
    python sync_results.py upload                  # judged*.jsonl + h2_monitor__*.jsonl + metadata.json (default)
    python sync_results.py upload --monitor-only   # h2_monitor__*.jsonl only (nothing else touched)
    python sync_results.py upload --full           # upload everything (use after inference)
    python sync_results.py upload --cache-only     # the slim query cache only, for figure-only boxes
    python sync_results.py download --cache-only   # ...and read it back with FACE_DB_CACHE_ONLY=1
    python sync_results.py upload --full --dry-run # list what would/would not be uploaded
    python sync_results.py download                # download inference results to results/
    python sync_results.py download --annotation   # download artifact_rating annotation results to outputs/
    python sync_results.py upload --repo edinburgh-dawg/face-eval-results --results-dir results
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, filter_repo_objects

from src.results.db import cache_glob


def _print_upload_plan(results_dir: str, allow_patterns: list[str] | None, ignore_patterns: list[str]) -> None:
    """List which files under results_dir would and would not be uploaded."""
    folder = Path(results_dir)
    # .git/ and .cache/huggingface/ are skipped by upload_large_folder itself.
    paths = sorted(
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file() and not p.relative_to(folder).as_posix().startswith((".git/", ".cache/huggingface/"))
    )
    included = set(filter_repo_objects(paths, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns))
    excluded = [p for p in paths if p not in included]

    print(f"WOULD UPLOAD ({len(included)} files, allow_patterns={allow_patterns}):")
    for p in paths:
        if p in included:
            print(f"  + {p}")
    print(f"\nWOULD SKIP ({len(excluded)} files):")
    for p in excluded:
        print(f"  - {p}")


_MONITOR_PATTERN = "**/h2_monitor__*.jsonl"
# The slim query cache, ~300MB against the ~5GB of inference.jsonl it was built from.
# A box that only makes figures can take this instead — see FACE_DB_CACHE_ONLY.
_CACHE_PATTERN = f"**/{cache_glob()}"


def upload(
    repo_id: str,
    results_dir: str,
    full: bool = False,
    monitor_only: bool = False,
    cache_only: bool = False,
    dry_run: bool = False,
) -> None:
    """Upload selected result artifacts to a private HuggingFace dataset repo.

    By default this sends judge outputs, monitor outputs, and metadata. Use
    ``full``, ``monitor_only``, or ``cache_only`` to select a different
    artifact set; ``dry_run`` prints the exact selection without uploading.
    """
    api = HfApi()
    # judged*.jsonl, not judged.jsonl: a second judge writes judged__{model}.jsonl
    # alongside the first, and a literal name would silently leave it behind.
    # h2_monitor__*.jsonl is the H2 monitor pass, written by run_monitor.py.
    # It is scored output like judged*.jsonl, and analyze_monitor_increment /
    # analyze_h2_calibration read nothing else, so leaving it out stranded the
    # H2 analysis on whichever box happened to run the monitor.
    # --monitor-only narrows to just that pass, for topping up a repo whose
    # judged*.jsonl are already newer than the local copies.
    if monitor_only:
        allow_patterns = [_MONITOR_PATTERN]
    elif cache_only:
        allow_patterns = [_CACHE_PATTERN]
    elif full:
        allow_patterns = None
    else:
        allow_patterns = ["**/judged*.jsonl", "**/metadata.json", _MONITOR_PATTERN]
    # *.partial.jsonl is a crashed run's leftover, which judged*.jsonl would otherwise match.
    ignore_patterns = ["**/*.bak", "**/*.partial.jsonl"]
    if dry_run:
        _print_upload_plan(results_dir, allow_patterns, ignore_patterns)
        print(f"\nDry run: nothing uploaded to {repo_id}")
        return
    try:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=results_dir,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    except HfHubHTTPError as e:
        raise RuntimeError(
            f"Failed to upload {results_dir!r} to {repo_id!r}. " "Check that HF_TOKEN is set and has write access."
        ) from e
    print(f"Uploaded {results_dir} -> {repo_id}")


def download(
    repo_id: str,
    results_dir: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
    convention: str | None = None,
    no_think: bool = False,
    annotation: bool = False,
    cache_only: bool = False,
) -> None:
    """Download results, one model slice, annotations, or the query cache.

    Model filters use the on-disk directory convention (``/`` becomes ``_``).
    Set ``annotation`` for artifact-rating data, or ``cache_only`` for a
    figure-making machine that does not need the underlying run directories.
    """
    if annotation:
        # artifact-rating annotation outputs live at artifact_rating/ and artifact_rating_no_think/ in the
        # HF repo, uploaded from the pod's results/ dir. Locally they go to
        # outputs/, which is where run_artifact_rating.py writes them and where
        # analyze_artifact_rating and every SAL/H6/H7 script looks for them.
        # A broken symlink at the destination makes snapshot_download fail deep
        # inside mkdir with a bare FileExistsError: pathlib's exist_ok check
        # calls is_dir(), which is False for a dangling link.
        for name in ("artifact_rating", "artifact_rating_no_think"):
            dest = Path("outputs") / name
            if dest.is_symlink() and not dest.exists():
                raise SystemExit(
                    f"{dest} is a symlink to {os.readlink(dest)!r}, which does not exist.\n"
                    "Annotations download straight into outputs/ now, so the symlink is "
                    f"redundant — remove it with `rm {dest}` and re-run."
                )
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir="outputs",
                allow_patterns=["artifact_rating/**", "artifact_rating_no_think/**", "ab_assignment.json"],
            )
        except HfHubHTTPError as e:
            raise RuntimeError(
                f"Failed to download annotation results from {repo_id!r}. "
                "Check that HF_TOKEN is set and has read access."
            ) from e
        print(f"Downloaded annotation results {repo_id}:artifact_rating/ -> outputs/artifact_rating/")
        return

    allow_patterns = None
    ignore_patterns = None
    if cache_only:
        allow_patterns = [f"{'agentic_no_think' if no_think else 'agentic'}/{cache_glob()}"]
    elif model is not None:
        model_name = model.replace("/", "_")
        if reasoning_effort is not None:
            model_name = f"{model_name}_{reasoning_effort}"
        substrate = "agentic_no_think" if no_think else "agentic"
        if convention is None:
            allow_patterns = [f"{substrate}/{model_name}/**"]
        elif convention == "C0":
            allow_patterns = [f"{substrate}/{model_name}/seed_*/**"]
            ignore_patterns = [f"{substrate}/{model_name}/seed_*_*/**"]
        else:
            allow_patterns = [f"{substrate}/{model_name}/seed_*_{convention}/**"]

    # snapshot_download hands the filtered file list to tqdm's thread_map, which
    # dies with a bare "min() iterable argument is empty" when nothing matched.
    # Say what was actually looked for instead.
    if allow_patterns is not None:
        repo_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        if not list(filter_repo_objects(repo_files, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)):
            raise RuntimeError(
                f"No files in {repo_id!r} matched allow_patterns={allow_patterns} "
                f"(ignore_patterns={ignore_patterns}). The repo holds {len(repo_files)} file(s). "
                "Check the --model spelling: the directory name is the model with '/' replaced by '_', "
                "including any reasoning-effort suffix."
            )

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=results_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    except HfHubHTTPError as e:
        raise RuntimeError(
            f"Failed to download {repo_id!r} to {results_dir!r}. " "Check that HF_TOKEN is set and has read access."
        ) from e
    print(f"Downloaded {repo_id} -> {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync results with HuggingFace")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--repo", default="edinburgh-dawg/face-eval-results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Upload all result files including inference.jsonl (use after inference runs)",
    )
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Upload only h2_monitor__*.jsonl, leaving judged*.jsonl and metadata.json on the repo untouched",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Sync only the slim query cache, for a box that makes figures but holds no run dirs "
        "(read it back with FACE_DB_CACHE_ONLY=1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Upload only: list the files that would and would not be uploaded, then exit",
    )
    parser.add_argument("--model", default=None, help="Filter download to a specific model (e.g. Qwen/Qwen3-4B)")
    parser.add_argument(
        "--reasoning-effort", default=None, help="Reasoning effort suffix used in the model directory name"
    )
    parser.add_argument(
        "--convention",
        default=None,
        choices=["C0", "C1", "C2", "C3", "MC0", "MC3"],
        help="Filter download to a specific convention arm (C0/C1/C2/C3/MC0/MC3)",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Filter model download to agentic_no_think/ instead of agentic/",
    )
    parser.add_argument(
        "--annotation",
        action="store_true",
        help="Download artifact-rating annotation results (artifact_rating/**, artifact_rating_no_think/**) into outputs/",
    )
    args = parser.parse_args()

    if args.action == "upload":
        if sum((args.full, args.monitor_only, args.cache_only)) > 1:
            raise SystemExit("--full, --monitor-only and --cache-only are mutually exclusive")
        upload(
            args.repo,
            args.results_dir,
            full=args.full,
            monitor_only=args.monitor_only,
            cache_only=args.cache_only,
            dry_run=args.dry_run,
        )
    else:
        download(
            args.repo,
            args.results_dir,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            convention=args.convention,
            no_think=args.no_think,
            annotation=args.annotation,
            cache_only=args.cache_only,
        )


if __name__ == "__main__":
    main()
