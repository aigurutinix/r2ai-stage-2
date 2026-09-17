from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scripts.audit_column_header_contracts as audit_module


def test_voting_contract_prefers_voting_sibling(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            ["", "Tỷ lệ lợi ích", "Quyền biểu quyết"],
            ["Cộng", "21,29%", "21,34%"],
        ]
    )
    monkeypatch.setattr(audit_module, "parsed_table", lambda _: frame)

    siblings = audit_module.sibling_headers("doc|1", 1)

    assert siblings[1]["header_folded"] == "quyen bieu quyet"


def test_total_contract_does_not_accept_maturity_bucket() -> None:
    contract = next(row for row in audit_module.CONTRACTS if row["name"] == "total_bucket")
    selected = "den 1 thang"
    total = "tong cong"

    assert not audit_module.contains_any(selected, contract["wanted_header_cues"])
    assert audit_module.contains_any(total, contract["wanted_header_cues"])


def test_current_term_cue_does_not_match_ngan_hang_substring() -> None:
    question = "tien gui cua khach hang tai ngan hang cuoi nam"

    assert not audit_module.contains_any(question, ("ngan han",))
    assert audit_module.contains_any("du no ngan han cuoi nam", ("ngan han",))


def test_punctuation_cue_still_matches_percentage_header() -> None:
    assert audit_module.contains_any("ty le %", ("%",))


def test_total_contract_ignores_component_qualified_total_value() -> None:
    contract = next(row for row in audit_module.CONTRACTS if row["name"] == "total_bucket")

    assert not audit_module.contains_any(
        "tong gia tri con lai cua tai san co dinh huu hinh khac",
        contract["question_cues"],
    )
    assert audit_module.contains_any("tong so du phong phai tra", contract["question_cues"])


def test_total_contract_distinguishes_total_from_company_name() -> None:
    contract = next(row for row in audit_module.CONTRACTS if row["name"] == "total_bucket")

    assert audit_module.question_matches_contract(
        contract, "tong cong", "Tổng cộng"
    )
    assert not audit_module.question_matches_contract(
        contract, "tong cong ty van tai", "Tổng Công ty Vận tải"
    )
