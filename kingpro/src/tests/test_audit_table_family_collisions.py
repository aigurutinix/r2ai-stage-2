from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_table_family_collisions import provision_family_collision  # noqa: E402


def test_flags_provision_columns_for_unrelated_metric() -> None:
    descriptor = "Dự phòng chung triệu đồng Dự phòng cụ thể triệu đồng"
    assert provision_family_collision("Số dư chứng khoán nợ là bao nhiêu?", descriptor)


def test_allows_explicit_provision_question() -> None:
    descriptor = "Dự phòng chung triệu đồng Dự phòng cụ thể triệu đồng"
    assert not provision_family_collision(
        "Dự phòng rủi ro cho vay khách hàng là bao nhiêu?", descriptor
    )


def test_ignores_non_provision_table() -> None:
    descriptor = "Số cuối năm VND Số đầu năm VND"
    assert not provision_family_collision("Số dư chứng khoán nợ là bao nhiêu?", descriptor)
