"""The repo-root entry points must import cleanly.

These are the modules a user actually invokes, and until now nothing imported
them in the suite -- a dangling import in run.py would have surfaced only when
someone ran it. tests/test_script_imports.py covers scripts/; this covers the
four runners plus the audit UI, which live at the root because they produce data
rather than read it.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# vulture_whitelist.py is lint config, not an entry point.
ENTRY_POINTS = sorted(p.stem for p in ROOT.glob("*.py") if p.stem != "vulture_whitelist")

EXPECTED_MIN = 5


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_entry_point_imports(name: str) -> None:
    """Import one repo-root entry point."""
    importlib.import_module(name)


def test_every_entry_point_is_covered() -> None:
    """Fail if the glob stopped seeing the root, which it would do silently."""
    assert len(ENTRY_POINTS) >= EXPECTED_MIN, f"only found {ENTRY_POINTS}"
