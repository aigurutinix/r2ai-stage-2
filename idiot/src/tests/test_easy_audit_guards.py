from __future__ import annotations

import unittest

from road2ai_vifinqa.easy_solver import EasyCandidate, _audit_change_guard


def _candidate(
    candidate_id: str,
    *,
    row_label: str,
    column_header: str,
    raw_number: float,
    answer_value: float,
    source_scale: float = 1.0,
    requested_scale: float = 1.0,
) -> EasyCandidate:
    return EasyCandidate(
        candidate_id=candidate_id,
        ticker="ABC",
        report_year=2023,
        scope="consolidated",
        doc_id="ABC_financial_statements_2023_consolidated",
        table_id=1,
        table_rows=10,
        row_idx=1,
        col_idx=1,
        row_label=row_label,
        section="",
        column_header=column_header,
        table_context="",
        raw_value=str(raw_number),
        raw_number=raw_number,
        source_scale=source_scale,
        requested_scale=requested_scale,
        answer_value=answer_value,
        retrieval_score=1.0,
    )


class EasyCompactAuditGuardTest(unittest.TestCase):
    def test_keeps_first_when_duplicate_raw_value_has_conflicting_scale(self) -> None:
        first = _candidate(
            "e000001",
            row_label="Chi phi hoat dong",
            column_header="Nam nay | Trieu dong",
            raw_number=1_041_601.0,
            answer_value=1_041_601.0,
            source_scale=1_000_000.0,
            requested_scale=1_000_000.0,
        )
        audited = _candidate(
            "e000002",
            row_label="Chi phi hoat dong",
            column_header="Nam 2019",
            raw_number=1_041_601.0,
            answer_value=1.041601,
            requested_scale=1_000_000.0,
        )
        self.assertEqual(
            _audit_change_guard(
                "Chi phi hoat dong nam 2019 la bao nhieu trieu dong?",
                (first,),
                (audited,),
            ),
            "same_raw_value_conflicting_normalized_scale",
        )

    def test_keeps_explicit_total_over_component(self) -> None:
        total = _candidate(
            "e000001",
            row_label="TONG TAI SAN CO",
            column_header="31/12/2023",
            raw_number=31_500_625.0,
            answer_value=31_500_625.0,
        )
        component = _candidate(
            "e000002",
            row_label="Tien mat, vang bac, da quy",
            column_header="31/12/2023 | A. TAI SAN",
            raw_number=163_234.0,
            answer_value=163_234.0,
        )
        self.assertEqual(
            _audit_change_guard(
                "Tong tai san hop nhat cua SGB den ngay 31/12/2023 la bao nhieu?",
                (total,),
                (component,),
            ),
            "audit_drops_explicit_total",
        )

    def test_keeps_explicit_report_unit_when_duplicate_raw_is_reused(self) -> None:
        first = _candidate(
            "e000001",
            row_label="Cho vay doi voi cac to chuc, ca nhan nuoc ngoai",
            column_header="31/12/2025 | Trieu dong",
            raw_number=222_172.0,
            answer_value=222_172_000.0,
            source_scale=1_000_000.0,
            requested_scale=1_000.0,
        )
        audited = _candidate(
            "e000002",
            row_label="Nuoc ngoai",
            column_header="Cho vay khach hang | 31/12/2025",
            raw_number=222_172.0,
            answer_value=222_172.0,
            source_scale=1_000.0,
            requested_scale=1_000.0,
        )
        self.assertEqual(
            _audit_change_guard(
                "So tien cho vay khach hang nuoc ngoai nam 2025 la bao nhieu nghin dong?",
                (first,),
                (audited,),
            ),
            "same_raw_value_conflicting_normalized_scale",
        )

    def test_rejects_previous_period_when_question_names_current_year(self) -> None:
        first = _candidate(
            "e000001",
            row_label="Tien va cac khoan tuong duong tien",
            column_header="So dau nam",
            raw_number=3_111.0,
            answer_value=3_111.0,
        )
        previous = _candidate(
            "e000002",
            row_label="Tien va tuong duong tien dau nam",
            column_header="Nam truoc",
            raw_number=1_073.0,
            answer_value=1_073.0,
        )
        self.assertEqual(
            _audit_change_guard(
                "Tien va tuong duong tien dau nam 2022 cua NLG la bao nhieu?",
                (first,),
                (previous,),
            ),
            "audit_switches_to_previous_period",
        )

    def test_does_not_block_audit_that_adds_an_explicit_total(self) -> None:
        component = _candidate(
            "e000001",
            row_label="Gia von chuyen nhuong bat dong san",
            column_header="Nam 2024",
            raw_number=958.0,
            answer_value=958.0,
        )
        total = _candidate(
            "e000002",
            row_label="Tong gia von",
            column_header="Nam 2024",
            raw_number=1_097.0,
            answer_value=1_097.0,
        )
        self.assertIsNone(
            _audit_change_guard(
                "Tong gia von hang ban cua HPX nam 2024 la bao nhieu?",
                (component,),
                (total,),
            )
        )

    def test_company_legal_name_does_not_turn_metric_into_total_request(self) -> None:
        first = _candidate(
            "e000001",
            row_label="Tai ngay cuoi nam",
            column_header="Tong | Gia tri con lai",
            raw_number=215.0,
            answer_value=215.0,
        )
        audited = _candidate(
            "e000002",
            row_label="Tai san co dinh vo hinh",
            column_header="So cuoi nam",
            raw_number=215.0,
            answer_value=215.0,
        )
        self.assertIsNone(
            _audit_change_guard(
                "Gia tri con lai cua tai san co dinh vo hinh cua Tong Cong ty VGC "
                "cuoi nam 2025 la bao nhieu?",
                (first,),
                (audited,),
            )
        )


if __name__ == "__main__":
    unittest.main()
