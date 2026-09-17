from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_unqualified_component_total import (  # noqa: E402
    matched_family,
    unqualified_component_candidate,
)


def provision_frame() -> pd.DataFrame:
    return pd.DataFrame([
        ["", "Số cuối năm"],
        ["Dự phòng rủi ro cho vay khách hàng tại Việt Nam", "39.850.765"],
        ["Dự phòng rủi ro cho vay khách hàng tại các thị trường nước ngoài", "618.295"],
        ["", "40.469.060"],
    ])


def test_detects_unqualified_geographic_component() -> None:
    finding = unqualified_component_candidate(
        provision_frame(),
        1,
        1,
        "Số dư dự phòng rủi ro cho vay khách hàng là bao nhiêu?",
    )
    assert finding is not None
    assert finding["qualifier_family"] == "geography"
    assert finding["candidate_raw"] == "40.469.060"


def test_skips_explicit_geographic_component() -> None:
    assert unqualified_component_candidate(
        provision_frame(),
        1,
        1,
        "Dự phòng rủi ro cho vay khách hàng tại Việt Nam là bao nhiêu?",
    ) is None


def test_skips_when_siblings_are_different_metrics() -> None:
    frame = pd.DataFrame([
        ["Tiền gửi trong nước", "100"],
        ["Cho vay nước ngoài", "20"],
        ["", "120"],
    ])
    assert unqualified_component_candidate(frame, 0, 1, "Tiền gửi là bao nhiêu?") is None


def test_recognizes_counterparty_family() -> None:
    assert matched_family("Phải thu khách hàng bên thứ ba") == "counterparty"
