"""Stress the deterministic compiler across schema and locale mutations.

The suite rebuilds the statement cube for each structural variant.  This is
the meaningful contract for unseen reports: column/row order and header names
may change before ingestion, then the compiler must bind to the newly resolved
cells.  A separate stale-index case mutates a source after compilation and
requires execution to fail closed instead of returning another row silently.

This synthetic suite is a bounded regression oracle, not a private-score or
unrestricted-schema claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kingpro.answering.sandbox import run_pandas_code  # noqa: E402
from kingpro.financial.statement_cube import build_statement_cube  # noqa: E402
from kingpro.product.deterministic_compiler import (  # noqa: E402
    DeterministicFinancialCompiler,
)


CDKT_ROWS: tuple[tuple[str, str, float, float], ...] = (
    ("Tài sản ngắn hạn", "100", 200, 180),
    ("Tiền và tương đương tiền", "110", 60, 55),
    ("Đầu tư tài chính ngắn hạn", "120", 30, 25),
    ("Các khoản phải thu ngắn hạn", "130", 70, 65),
    ("Hàng tồn kho", "140", 40, 35),
    ("Tài sản dài hạn", "200", 300, 270),
    ("Tổng cộng tài sản", "270", 500, 450),
    ("Nợ phải trả", "300", 250, 230),
    ("Nợ ngắn hạn", "310", 100, 90),
    ("Vay và nợ thuê tài chính ngắn hạn", "320", 25, 20),
    ("Vốn chủ sở hữu", "400", 250, 220),
    ("Vốn góp của chủ sở hữu", "411", 70, 70),
)

KQKD_ROWS: tuple[tuple[str, str, float, float], ...] = (
    ("Doanh thu bán hàng và cung cấp dịch vụ", "01", 1_000, 900),
    ("Doanh thu thuần", "10", 1_000, 900),
    ("Giá vốn hàng bán", "11", 400, 360),
    ("Lợi nhuận gộp", "20", 600, 540),
    ("Doanh thu hoạt động tài chính", "21", 10, 8),
    ("Chi phí tài chính", "22", 50, 45),
    ("Chi phí bán hàng", "25", 20, 18),
    ("Chi phí quản lý doanh nghiệp", "26", 80, 72),
    ("Thu nhập khác", "31", 12, 10),
    ("Chi phí khác", "32", 20, 16),
    ("Lợi nhuận kế toán trước thuế", "50", 200, 180),
    ("Chi phí thuế TNDN hiện hành", "51", 15, 14),
    ("Lợi nhuận sau thuế", "60", 100, 90),
)

FACETS = {
    "tickers": ["VNM"],
    "years": ["2023"],
    "scope": "hợp nhất",
    "analytic": False,
}
TOTAL_ASSETS_QUESTION = (
    "Tổng tài sản của VNM năm 2023 là bao nhiêu triệu đồng?"
)
CURRENT_RATIO_QUESTION = (
    "Hệ số thanh toán hiện hành của VNM năm 2023 là bao nhiêu lần?"
)
COGS_QUESTION = (
    "Giá vốn hàng bán của VNM năm 2023 là bao nhiêu triệu đồng?"
)


def _format_number(value_million: float, *, unit: str, negative: bool = False) -> str:
    value = value_million * 1_000_000 if unit == "vnd" else value_million
    if float(value).is_integer():
        rendered = f"{int(value):,}".replace(",", ".") if unit == "vnd" else str(int(value))
    else:
        rendered = ("{0:.6f}".format(value)).rstrip("0").rstrip(".").replace(".", ",")
    return "({0})".format(rendered) if negative else rendered


def _write_table(
    path: Path,
    rows: tuple[tuple[str, str, float, float], ...],
    *,
    unit: str,
    row_permutation: bool = False,
    column_permutation: bool = False,
    relative_headers: bool = False,
    missing_code: str | None = None,
    decimal_total: bool = False,
    negative_cogs: bool = False,
) -> None:
    current_header = "Năm nay VND" if relative_headers else "2023 VND"
    previous_header = "Năm trước VND" if relative_headers else "2022 VND"
    if unit == "million":
        current_header = current_header.replace("VND", "triệu VND")
        previous_header = previous_header.replace("VND", "triệu VND")
    header = ["Chỉ tiêu" if not relative_headers else "Khoản mục", "Mã số" if not relative_headers else "Mã", "TM", current_header, previous_header]
    data: list[list[str]] = []
    for label, code, current, previous in rows:
        if code == missing_code:
            continue
        if decimal_total and code == "270":
            current = 1234.5
        data.append(
            [
                label,
                code,
                "",
                _format_number(
                    current,
                    unit=unit,
                    negative=negative_cogs and code == "11",
                ),
                _format_number(previous, unit=unit),
            ]
        )
    if row_permutation:
        data.reverse()
    if column_permutation:
        order = [3, 1, 0, 4, 2]
        header = [header[index] for index in order]
        data = [[row[index] for index in order] for row in data]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([str(index) for index in range(len(header))])
        writer.writerow(header)
        writer.writerows(data)


def _build_root(
    root: Path,
    *,
    unit: str = "vnd",
    row_permutation: bool = False,
    column_permutation: bool = False,
    relative_headers: bool = False,
    missing_code: str | None = None,
    missing_kind: str | None = None,
    decimal_total: bool = False,
    negative_cogs: bool = False,
) -> tuple[DeterministicFinancialCompiler, dict[str, Path]]:
    report_id = "VNM_financial_statements_2023_consolidated"
    tables_dir = root / "build" / "tables" / report_id
    paths = {
        "cdkt": tables_dir / "table_1_line100.csv",
        "kqkd": tables_dir / "table_2_line200.csv",
    }
    _write_table(
        paths["cdkt"],
        CDKT_ROWS,
        unit=unit,
        row_permutation=row_permutation,
        column_permutation=column_permutation,
        relative_headers=relative_headers,
        missing_code=missing_code if missing_kind == "cdkt" else None,
        decimal_total=decimal_total,
    )
    _write_table(
        paths["kqkd"],
        KQKD_ROWS,
        unit=unit,
        row_permutation=row_permutation,
        column_permutation=column_permutation,
        relative_headers=relative_headers,
        missing_code=missing_code if missing_kind == "kqkd" else None,
        negative_cogs=negative_cogs,
    )
    catalog_rows = []
    for kind, line in (("cdkt", 100), ("kqkd", 200)):
        catalog_rows.append(
            {
                "table_ref": "{0}|{1}".format(report_id, line),
                "report_id": report_id,
                "ticker": "VNM",
                "year": "2023",
                "scope": "hợp nhất",
                "line": line,
                "page": 10 if kind == "cdkt" else 12,
                "n_cols": 5,
                "csv_path": "{0}/{1}".format(report_id, paths[kind].name),
                "search_text": " ".join(row[0] for row in (CDKT_ROWS if kind == "cdkt" else KQKD_ROWS)),
            }
        )
    catalog_path = root / "build" / "catalog.jsonl"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in catalog_rows),
        encoding="utf-8",
    )
    cube = build_statement_cube(
        catalog_path,
        root / "build" / "tables",
        max_page=30,
    )
    cube.write_jsonl(root / "build" / "statement_cube.jsonl")
    return DeterministicFinancialCompiler(root), paths


def _compile_and_replay(
    compiler: DeterministicFinancialCompiler,
    question: str,
    expected: float,
) -> dict[str, Any]:
    compiled = compiler.compile(question, FACETS)
    if compiled is None:
        return {"ok": False, "reason": "compiler_refused_positive"}
    replay = run_pandas_code(compiled.pandas_query, compiled.csv_paths, timeout=5)
    value = replay.get("result")
    matched = bool(
        replay.get("ok")
        and isinstance(value, (int, float))
        and math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-9)
    )
    return {
        "ok": matched,
        "metric": compiled.metric,
        "compiled_answer": compiled.answer,
        "replay": replay,
        "expected": expected,
    }


def _positive_case(name: str, **kwargs: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        compiler, _paths = _build_root(Path(directory), **kwargs)
        question = COGS_QUESTION if kwargs.get("negative_cogs") else TOTAL_ASSETS_QUESTION
        expected = 400.0 if kwargs.get("negative_cogs") else 1234.5 if kwargs.get("decimal_total") else 500.0
        result = _compile_and_replay(compiler, question, expected)
    return {"name": name, "expectation": "answer", **result}


def _missing_case(
    name: str,
    *,
    missing_code: str,
    missing_kind: str,
    question: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        compiler, _paths = _build_root(
            Path(directory),
            missing_code=missing_code,
            missing_kind=missing_kind,
        )
        compiled = compiler.compile(question, FACETS)
    return {
        "name": name,
        "expectation": "refusal",
        "ok": compiled is None,
        "compiled_metric": None if compiled is None else compiled.metric,
    }


def _stale_case() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        compiler, paths = _build_root(Path(directory))
        compiled = compiler.compile(TOTAL_ASSETS_QUESTION, FACETS)
        if compiled is None:
            return {
                "name": "stale_source_after_compile",
                "expectation": "fail_closed",
                "ok": False,
                "reason": "compiler_refused_baseline",
            }
        with paths["cdkt"].open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        asset_index = next(index for index, row in enumerate(rows) if "270" in row)
        other_index = next(index for index, row in enumerate(rows) if "100" in row)
        rows[asset_index], rows[other_index] = rows[other_index], rows[asset_index]
        with paths["cdkt"].open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
        replay = run_pandas_code(compiled.pandas_query, compiled.csv_paths, timeout=5)
    return {
        "name": "stale_source_after_compile",
        "expectation": "fail_closed",
        "ok": replay.get("ok") is False and replay.get("result") is None,
        "replay": replay,
    }


def audit() -> dict[str, Any]:
    cases = [
        _positive_case("baseline"),
        _positive_case("row_permutation", row_permutation=True),
        _positive_case("column_permutation", column_permutation=True),
        _positive_case("relative_header_alias", relative_headers=True),
        _positive_case("million_unit", unit="million"),
        _positive_case("vietnamese_decimal_locale", unit="million", decimal_total=True),
        _positive_case("parenthesized_negative_cost", unit="million", negative_cogs=True),
        _missing_case(
            "missing_direct_operand",
            missing_code="270",
            missing_kind="cdkt",
            question=TOTAL_ASSETS_QUESTION,
        ),
        _missing_case(
            "missing_ratio_denominator",
            missing_code="310",
            missing_kind="cdkt",
            question=CURRENT_RATIO_QUESTION,
        ),
        _stale_case(),
    ]
    passed = sum(bool(case.get("ok")) for case in cases)
    return {
        "schema_version": 1,
        "case_count": len(cases),
        "passed_count": passed,
        "failed_count": len(cases) - passed,
        "passed": passed == len(cases),
        "cases": cases,
        "coverage": {
            "row_order": True,
            "column_order": True,
            "relative_header_alias": True,
            "currency_scale": ["VND", "triệu VND"],
            "locale": ["dot_thousands", "comma_decimal", "parenthesized_negative"],
            "missing_operand_refusal": True,
            "stale_source_fail_closed": True,
        },
        "claim_limit": (
            "Finite synthetic regression over named schema classes. It does not "
            "prove accuracy on arbitrary unseen tables or reveal private labels."
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-failure", action="store_true")
    args = parser.parse_args()
    report = audit()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        output = args.out.resolve() if args.out.is_absolute() else (ROOT / args.out).resolve()
        try:
            output.relative_to(ROOT)
        except ValueError as exc:
            raise SystemExit("output must remain inside the project") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if args.fail_on_failure and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
