"""Reproducible verification report for a generated reviewed-template batch."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from vifinqa.generation.validation.pandas_check import check_answer


@dataclass(frozen=True, slots=True)
class RowVerification:
    row_id: int
    template_id: str
    query_ok: bool
    dependency_ok: bool
    actual: object
    expected: object
    accessed_table_count: int
    declared_table_count: int
    missing_tables: tuple[str, ...]
    extra_tables: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class BatchVerification:
    input_path: str
    row_count: int
    distinct_template_count: int
    structural_ok: bool
    query_pass_count: int
    dependency_pass_count: int
    rows: tuple[RowVerification, ...]

    @property
    def ok(self) -> bool:
        return (
            self.structural_ok
            and self.row_count > 0
            and self.distinct_template_count == self.row_count
            and self.query_pass_count == self.row_count
            and self.dependency_pass_count == self.row_count
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "summary": {
                "input_path": self.input_path,
                "row_count": self.row_count,
                "distinct_template_count": self.distinct_template_count,
                "structural_ok": self.structural_ok,
                "query_pass_count": self.query_pass_count,
                "dependency_pass_count": self.dependency_pass_count,
                "all_pass": self.ok,
            },
            "rows": [asdict(row) for row in self.rows],
        }


def verify_template_batch(input_path: Path) -> BatchVerification:
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    structural_ok = True
    rows: list[RowVerification] = []
    template_ids: list[str] = []
    for index, record in enumerate(records, start=1):
        template_id = str(record.get("template_id", ""))
        template_ids.append(template_id)
        relevant_tables = tuple(record.get("relevant_tables", ()))
        csv_paths = tuple(record.get("csv_path", ()))
        answer = record.get("answer")
        structural_ok = structural_ok and all(
            (
                record.get("id") == index,
                bool(record.get("question", "").strip()),
                record.get("difficulty") == "hard",
                bool(template_id),
                len(relevant_tables) == len(csv_paths),
                len(set(relevant_tables)) == len(relevant_tables),
                isinstance(answer, (int, float))
                and not isinstance(answer, bool)
                and math.isfinite(float(answer)),
            )
        )
        if len(relevant_tables) != len(csv_paths):
            rows.append(
                RowVerification(
                    index,
                    template_id,
                    False,
                    False,
                    None,
                    answer,
                    0,
                    len(relevant_tables),
                    relevant_tables,
                    (),
                    "relevant_tables/csv_path length mismatch",
                )
            )
            continue
        paths = {
            table_ref: Path(csv_path)
            for table_ref, csv_path in zip(relevant_tables, csv_paths, strict=True)
        }
        result = check_answer(record.get("pandas_query", ""), paths, answer)
        declared = set(relevant_tables)
        accessed = set(result.accessed_refs)
        missing = tuple(sorted(declared - accessed))
        extra = tuple(sorted(accessed - declared))
        rows.append(
            RowVerification(
                index,
                template_id,
                result.ok,
                not missing and not extra,
                result.actual,
                answer,
                len(accessed),
                len(declared),
                missing,
                extra,
                result.detail,
            )
        )
    return BatchVerification(
        input_path=str(input_path),
        row_count=len(records),
        distinct_template_count=len(set(template_ids)),
        structural_ok=structural_ok,
        query_pass_count=sum(row.query_ok for row in rows),
        dependency_pass_count=sum(row.dependency_ok for row in rows),
        rows=tuple(rows),
    )


def render_verification_markdown(report: BatchVerification) -> str:
    lines = [
        "# Hard template batch verification",
        "",
        f"- Input: `{report.input_path}`",
        f"- Rows / distinct template IDs: {report.row_count} / {report.distinct_template_count}",
        f"- Structural contract: {'PASS' if report.structural_ok else 'FAIL'}",
        f"- Query replay: {report.query_pass_count}/{report.row_count}",
        f"- Dependency tracking: {report.dependency_pass_count}/{report.row_count}",
        f"- Overall: {'PASS' if report.ok else 'FAIL'}",
        "",
        "| id | template_id | query | dependency | accessed/declared | detail |",
        "|---:|---|---|---|---:|---|",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.row_id} | {row.template_id} | {'PASS' if row.query_ok else 'FAIL'} | "
            f"{'PASS' if row.dependency_ok else 'FAIL'} | "
            f"{row.accessed_table_count}/{row.declared_table_count} | {row.detail} |"
        )
    return "\n".join(lines) + "\n"


def write_verification_report(
    report: BatchVerification, *, json_out: Path, markdown_out: Path
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(report.to_payload(), ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    markdown_out.write_text(render_verification_markdown(report), encoding="utf-8")
