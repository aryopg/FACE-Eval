"""The Makefile must not name things that do not exist.

Both checks here caught real drift during the scripts/ reorganisation: recipe
lines pointing at moved scripts, and a hand-maintained .PHONY list that fell
behind the targets. Neither shows up in a normal test run, and `make` itself
reports a missing script only when you actually run the target.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAKEFILE = (ROOT / "Makefile").read_text()

# A target definition, excluding `VAR := value`.
_TARGET = re.compile(r"^([a-zA-Z0-9_-]+):(?!=)", re.M)
_PHONY = re.compile(r"^\.PHONY:((?:[^\n]*\\\n)*[^\n]*)\n", re.M)


def test_every_script_path_exists() -> None:
    """Every .py a recipe invokes is really there."""
    paths = sorted(set(re.findall(r"(?:scripts/[A-Za-z0-9_/]+|\$\(PYTHON\) [a-z_]+)\.py", MAKEFILE)))
    missing = [p for p in (x.replace("$(PYTHON) ", "") for x in paths) if not (ROOT / p).exists()]
    assert not missing, f"Makefile invokes scripts that do not exist: {missing}"


def test_phony_matches_the_targets() -> None:
    """.PHONY is hand-maintained, so it drifts. Nothing else notices."""
    declared = set(_PHONY.search(MAKEFILE).group(1).replace("\\", "").split())
    real = set(_TARGET.findall(MAKEFILE))
    assert real - declared == set(), f"targets missing from .PHONY: {sorted(real - declared)}"
    assert declared - real == set(), f".PHONY names targets that no longer exist: {sorted(declared - real)}"


def test_help_text_on_every_public_target() -> None:
    """`make help` reads the ## comments, so a target without one is invisible."""
    undocumented = [
        t
        for t in _TARGET.findall(MAKEFILE)
        if not t.startswith("_") and not re.search(rf"^{re.escape(t)}:[^\n]*##", MAKEFILE, re.M)
    ]
    assert not undocumented, f"targets absent from `make help`: {undocumented}"
