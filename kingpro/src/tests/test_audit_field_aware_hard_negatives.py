import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_field_aware_hard_negatives.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_field_aware_hard_negatives", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(MODULE)


def test_governance_role_is_a_separate_question_facet() -> None:
    question = "Thù lao thành viên HĐQT Chu Thị Bình năm 2021 là bao nhiêu?"
    assert MODULE.requested_facets(question) == {"governance_role": "board"}
    assert MODULE.effective_group_facets(
        "Chu Thị Bình", "Ban Giám đốc", "governance_role"
    ) == {"management"}


def test_same_person_label_remains_same_metric_after_entity_terms_are_removed() -> None:
    question = "Thù lao thành viên HĐQT Chu Thị Bình năm 2021 là bao nhiêu?"
    assert MODULE.labels_share_metric(question, "Chu Thị Bình", "Chu Thị Bình")
    assert not MODULE.labels_share_metric(question, "Chu Thị Bình", "Lê Văn Quang")


def test_load_rows_preserves_merged_parent_role(tmp_path, monkeypatch) -> None:
    report_dir = tmp_path / "MPC_report"
    report_dir.mkdir()
    (report_dir / "table.csv").write_text(
        "0,1,2\n"
        ",2021VND,2020VND\n"
        "Hội đồng Quản trị,Hội đồng Quản trị,Hội đồng Quản trị\n"
        "Chu Thị Bình,150.000.000,150.000.000\n"
        "Ban Giám đốc,Ban Giám đốc,Ban Giám đốc\n"
        "Chu Thị Bình,1.150.851.285,1.099.739.984\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "TABLES", tmp_path)
    catalog = {
        "MPC_report": [
            {
                "table_ref": "MPC_report|1",
                "csv_path": "MPC_report/table.csv",
            }
        ]
    }

    rows = MODULE.load_rows("MPC_report", catalog)

    assert rows[0]["row_path"] == ["Hội đồng Quản trị", "Chu Thị Bình"]
    assert rows[1]["row_path"] == ["Ban Giám đốc", "Chu Thị Bình"]
    assert MODULE.effective_group_facets(
        rows[0]["label"], rows[0]["row_context"], "governance_role"
    ) == {"board"}
    assert MODULE.effective_group_facets(
        rows[1]["label"], rows[1]["row_context"], "governance_role"
    ) == {"management"}
