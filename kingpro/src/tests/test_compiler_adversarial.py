from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from audit_compiler_adversarial import audit


def test_compiler_adversarial_suite_is_fail_closed() -> None:
    report = audit(
        root=ROOT,
        registry=ROOT
        / "sub_top123_candidate_v184_mbb_credit_provision_ratio"
        / "submission.json",
    )
    assert report["positive_failures"] == []
    assert report["adversarial_cases"] >= 29
    assert report["adversarial_rejected"] == report["adversarial_cases"]
    assert report["negative_acceptances"] == []
    assert report["passed"] is True
