"""Every module in scripts/ must import cleanly.

A deleted or moved module leaves a dangling import behind. Nothing else catches
that until someone runs the script, so this test imports all of them.

The walk is recursive, because scripts/ has subpackages. A non-recursive glob
would still find the handful of modules at the root, so it would keep passing
while covering almost nothing -- hence the floor in ``test_every_script_is_covered``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

# scripts/adhoc/ is gitignored one-off scratch, absent on a fresh clone. Including
# it would make the collected set depend on what happens to be lying around.
_SKIP_DIRS = {"adhoc", "__pycache__"}

# Floor per directory, not one total: a single total lets plots/ vanish while
# analysis/ grows. Update these when adding or removing a script -- the diff on
# this line is the reminder the repo otherwise lacks.
EXPECTED_MIN = {"plots": 18, "analysis": 11}


def _iter_scripts() -> list[tuple[str, str]]:
    """(dotted module path, top-level subdirectory) for every module under scripts/."""
    found = []
    for path in sorted(SCRIPTS_DIR.rglob("*.py")):
        rel = path.relative_to(SCRIPTS_DIR)
        if _SKIP_DIRS.intersection(rel.parts) or path.name == "__init__.py":
            continue
        dotted = ".".join(("scripts", *rel.with_suffix("").parts))
        found.append((dotted, rel.parts[0] if len(rel.parts) > 1 else ""))
    return found


SCRIPT_MODULES = [m for m, _ in _iter_scripts()]


@pytest.mark.parametrize("name", SCRIPT_MODULES)
def test_script_imports(name: str) -> None:
    """Import one module from scripts/."""
    importlib.import_module(name)


def test_every_script_is_covered() -> None:
    """Fail if the walk stopped seeing a directory, which a glob does silently."""
    seen: dict[str, int] = {}
    for _, subdir in _iter_scripts():
        seen[subdir] = seen.get(subdir, 0) + 1
    short = {d: (seen.get(d, 0), n) for d, n in EXPECTED_MIN.items() if seen.get(d, 0) < n}
    assert not short, "directories under-covered (found, expected at least): " + repr(short)
