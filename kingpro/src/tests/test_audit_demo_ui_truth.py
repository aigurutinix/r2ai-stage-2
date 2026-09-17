from __future__ import annotations

from scripts.audit_demo_ui_truth import FORBIDDEN_DEMO_TEXT, REQUIRED_DEMO_TEXT


def test_canonical_ui_markers_are_competition_relevant_and_low_clutter() -> None:
    assert "KINGPRO" in REQUIRED_DEMO_TEXT
    assert "FINANCIAL QA" in REQUIRED_DEMO_TEXT
    assert "Chat" in REQUIRED_DEMO_TEXT
    assert "Batch" in REQUIRED_DEMO_TEXT
    assert "1012 verified programs" in REQUIRED_DEMO_TEXT
    assert "Nguồn kiểm chứng" in REQUIRED_DEMO_TEXT
    assert "PUBLIC SCORE · V217 · ID 3696" not in REQUIRED_DEMO_TEXT
    assert "PUBLIC SCORE · V217 · ID 3696" not in FORBIDDEN_DEMO_TEXT
    assert "PUBLIC SCORE · V276 · ID 3742" in FORBIDDEN_DEMO_TEXT
    assert "PUBLIC SCORE · V290 · ID 3745" in FORBIDDEN_DEMO_TEXT
