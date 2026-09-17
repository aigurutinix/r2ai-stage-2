from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from kingpro.product.service import ProductService, RefusalPolicy
from kingpro.retrieval.structural_fusion import (
    guarded_structural_rescue,
    structural_evidence,
)


QUESTION = (
    "Tổng số dư tương đương đồng ngoại tệ của VGT cuối năm 2022 "
    "là bao nhiêu trăm tỷ đồng?"
)
TARGET = "VGT_financial_statements_2022_consolidated|1741"


def q312_context() -> dict:
    return {
        "table_ref": TARGET,
        "section_title": (
            "32. Các khoản mục ngoài Bảng cân đối kế toán | (b) Ngoại tệ các loại"
        ),
        "header_text": (
            "31/12/2022 > Nguyên tệ | 31/12/2022 > Tương đương VND | "
            "1/1/2022 > Nguyên tệ | 1/1/2022 > Tương đương VND"
        ),
        "inferred_row_text": (
            "Tổng cộng > 31/12/2022 > Tương đương VND | "
            "Tổng cộng > 1/1/2022 > Tương đương VND"
        ),
    }


def hits(prefix: str, count: int = 8) -> list[dict]:
    return [{"table_ref": f"{prefix}|{index}", "score": 10 - index} for index in range(count)]


def test_q312_strong_total_context_replaces_only_weak_tail() -> None:
    baseline = hits("BASE")
    structural = [{"table_ref": TARGET, "score": 12.0}]
    catalog = {TARGET: q312_context()}

    fused, trace = guarded_structural_rescue(
        QUESTION, baseline, structural, catalog, limit=8
    )

    assert [row["table_ref"] for row in fused[:7]] == [
        row["table_ref"] for row in baseline[:7]
    ]
    assert fused[7]["table_ref"] == TARGET
    assert fused[7]["structural_rescue"] is True
    assert trace["rescues"][0]["removed_table_ref"] == "BASE|7"


def test_non_additive_or_header_only_candidate_cannot_rescue() -> None:
    baseline = hits("BASE")
    weak = q312_context()
    weak["inferred_row_text"] = ""

    fused, trace = guarded_structural_rescue(
        QUESTION,
        baseline,
        [{"table_ref": TARGET, "score": 12.0}],
        {TARGET: weak},
    )

    assert fused == baseline
    assert trace["rescues"] == []
    assert trace["rejections"][0]["has_proven_total"] is False


def test_company_word_does_not_count_as_total_request() -> None:
    evidence = structural_evidence(
        "Công ty VGT có bao nhiêu ngoại tệ cuối năm 2022?", q312_context()
    )
    assert evidence.question_requests_total is False
    assert evidence.strong is False


def test_multi_entity_selector_cannot_be_short_circuited_by_a_component_total() -> None:
    evidence = structural_evidence(
        (
            "Trong ba mã BSR, PLX và PVT, doanh nghiệp tăng doanh thu cao nhất "
            "có tổng chi phí bán hàng và quản lý tăng bao nhiêu phần trăm?"
        ),
        q312_context(),
    )
    assert evidence.single_entity_direct is False
    assert evidence.strong is False


def test_baseline_target_rank_is_preserved_when_tail_is_rescued() -> None:
    baseline = hits("BASE")
    baseline[2] = {"table_ref": "LOCKED_TARGET|3", "score": 8.0}
    fused, trace = guarded_structural_rescue(
        QUESTION,
        baseline,
        [{"table_ref": TARGET, "score": 12.0}],
        {TARGET: q312_context()},
    )

    assert fused[2]["table_ref"] == "LOCKED_TARGET|3"
    assert trace["rescues"][0]["position"] == 8


def test_rescue_is_capped_at_one_candidate_per_question() -> None:
    second = {**q312_context(), "table_ref": "VGT|SECOND"}
    structural = [
        {"table_ref": TARGET, "score": 12.0},
        {"table_ref": "VGT|SECOND", "score": 11.0},
    ]
    fused, trace = guarded_structural_rescue(
        QUESTION,
        hits("BASE"),
        structural,
        {TARGET: q312_context(), "VGT|SECOND": second},
    )
    assert len(trace["rescues"]) == 1
    assert sum(bool(row.get("structural_rescue")) for row in fused) == 1


def test_product_service_wires_opt_in_sidecar_without_reordering_baseline() -> None:
    class FakeSidecar:
        catalog = {TARGET: q312_context()}

        def tables_in_reports(self, *_args, **_kwargs):
            return [{"table_ref": TARGET, "score": 12.0}]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tables_root = root / "build" / "tables"
        tables_root.mkdir(parents=True)
        catalog = {}
        baseline = hits("BASE")
        for index, hit in enumerate(baseline):
            csv_path = f"base-{index}.csv"
            pd.DataFrame({"0": ["Nợ vay", "Phải trả"]}).to_csv(
                tables_root / csv_path, index=False, encoding="utf-8-sig"
            )
            catalog[hit["table_ref"]] = {
                "table_ref": hit["table_ref"],
                "report_id": "VGT_REPORT",
                "ticker": "VGT",
                "year": "2022",
                "scope": "hợp nhất",
                "csv_path": csv_path,
                "search_text": "Nợ vay",
            }
        pd.DataFrame({"0": ["USD", "EUR", ""]}).to_csv(
            tables_root / "target.csv", index=False, encoding="utf-8-sig"
        )
        catalog[TARGET] = {
            "table_ref": TARGET,
            "report_id": "VGT_REPORT",
            "ticker": "VGT",
            "year": "2022",
            "scope": "hợp nhất",
            "csv_path": "target.csv",
            "search_text": "USD | EUR",
        }
        FakeSidecar.catalog[TARGET] = {
            **q312_context(),
            "csv_path": "target.csv",
        }
        service = ProductService(
            root=root,
            llm_fn=lambda _system, _user: "result = 0",
            policy=RefusalPolicy(base_tables=8, max_tables=8, table_rerank_pool=8),
            structural_sidecar=FakeSidecar(),
        )
        service._catalog = catalog
        facets = {
            "tickers": ["VGT"],
            "years": ["2022"],
            "scope": "hợp nhất",
            "analytic": False,
        }
        with patch(
            "kingpro.product.service.retrieve_decomposed",
            return_value=[
                {
                    "table_ref": "VGT_REPORT|1",
                    "ticker": "VGT",
                    "year": "2022",
                }
            ],
        ), patch("kingpro.product.service.tables_in_reports", return_value=baseline):
            _facets, _docs, ranked, checks = service._retrieve(QUESTION, facets)

        assert [row["table_ref"] for row in ranked[:7]] == [
            row["table_ref"] for row in baseline[:7]
        ]
        assert ranked[7]["table_ref"] == TARGET
        assert checks["structural_rescue"]["enabled"] is True
        assert checks["structural_rescue"]["rescues"][0]["position"] == 8
