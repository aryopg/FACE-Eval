"""Every module in face_eval_generator/ must import cleanly.

The dataset-provenance scripts moved here out of scripts/, which
tests/test_script_imports.py globs. Without this they lose import coverage.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "face_eval_generator"

PACKAGE_MODULES = sorted(p.stem for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__")


@pytest.mark.parametrize("name", PACKAGE_MODULES)
def test_face_eval_generator_imports(name: str) -> None:
    """Import one module from face_eval_generator/."""
    importlib.import_module(f"face_eval_generator.{name}")
