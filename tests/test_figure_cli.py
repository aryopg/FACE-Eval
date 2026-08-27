"""Every script that writes into figures/ must accept --figures-dir.

Most scripts take the flag. Some hardcode a module-level ``FIGURES_DIR = Path("figures")``
instead, so ``--figures-dir`` does nothing for them. This test names those scripts.

Detection reads the source with ``ast``. Running each script with ``--help`` is not an
option: some build no parser at all and start loading results on any argv.
"""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

# The one module under plots/ that draws nothing: it is the shared
# --exclude-eval-aware wiring its nine neighbours import. Named individually rather
# than exempting a directory, so a figure script that forgets the flag still fails.
NON_FIGURE_SCRIPTS = {
    "_eval_aware_filter",
}

# The analysis scripts that write outside figures/, so --figures-dir would mean
# nothing to them. The other nine do take the flag.
NON_FIGURE_ANALYSIS_SCRIPTS = {
    "analyze_artifact_rating",
    "analyze_h1_role_register",
    "find_qualitative_examples",
    "build_web_examples",
}

EXEMPT_SCRIPTS = NON_FIGURE_SCRIPTS | NON_FIGURE_ANALYSIS_SCRIPTS

# rglob, not glob: scripts/ has subpackages, and a flat glob would quietly check
# only the three root modules -- all of them exempt -- leaving nothing at all.
FIGURE_SCRIPTS = sorted(
    p
    for p in SCRIPTS_DIR.rglob("*.py")
    if "adhoc" not in p.relative_to(SCRIPTS_DIR).parts and p.stem not in EXEMPT_SCRIPTS
)

# scripts/plots/ carries no exemptions at all, so this floor also pins the split.
EXPECTED_MIN_FIGURE_SCRIPTS = 26


def declares_figures_dir_flag(path: Path) -> bool:
    """Say whether the script registers a --figures-dir argument."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if any(isinstance(a, ast.Constant) and a.value == "--figures-dir" for a in node.args):
            return True
    return False


def test_figure_scripts_accept_figures_dir() -> None:
    """Check that every figure-writing script takes --figures-dir."""
    assert len(FIGURE_SCRIPTS) >= EXPECTED_MIN_FIGURE_SCRIPTS, (
        f"only {len(FIGURE_SCRIPTS)} scripts found under {SCRIPTS_DIR}; the walk has " "stopped seeing a subdirectory"
    )
    offenders = sorted(p.name for p in FIGURE_SCRIPTS if not declares_figures_dir_flag(p))
    assert not offenders, (
        f"{len(offenders)} of {len(FIGURE_SCRIPTS)} figure scripts hardcode their output "
        "directory and ignore --figures-dir:\n  " + "\n  ".join(offenders)
    )
