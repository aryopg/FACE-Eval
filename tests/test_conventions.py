from __future__ import annotations

from src.data.conventions import CONVENTIONS


def test_known_arms_have_expected_addenda():
    assert CONVENTIONS["C0"] == ""
    assert "step by step" in CONVENTIONS["C1"]
    assert "mention any information" in CONVENTIONS["C2"]
    assert "explicitly attribute" in CONVENTIONS["C3"]
    assert "monitoring system" in CONVENTIONS["MC0"]
    assert "quality and safety" in CONVENTIONS["MC0"]
    assert "monitoring system" in CONVENTIONS["MC3"]
    assert "quality and safety" in CONVENTIONS["MC3"]
    assert "explicitly attribute" in CONVENTIONS["MC3"]


def test_mc3_composes_monitor_then_c3():
    """MC3 must place the monitoring sentence BEFORE the C3 directive.

    Order is a methodological choice — flipping it would change the construct
    under test without changing the resolved-prompt set's semantic content.
    """
    addendum = CONVENTIONS["MC3"]
    monitor_idx = addendum.index("monitoring system")
    directive_idx = addendum.index("explicitly attribute")
    assert monitor_idx < directive_idx
    assert "how it shaped your answer." in addendum
