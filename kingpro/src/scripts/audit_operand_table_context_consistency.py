"""Find same-metric operands selected from incompatible table contexts.

Coordinate and runtime checks prove that a number exists; they do not prove
that two numbers with the same label mean the same thing.  q769 is the control:
``Quyền sử dụng đất`` is read from intangible PPE for VSC but from investment
property for ACV.  This read-only audit classifies table context independently
from the compact metric key and reports cross-operand family disagreements.

The classifier is deliberately conservative.  A family must dominate the
table title/physical lineage; mentions in movement rows (for example
``Tăng từ bất động sản đầu tư`` inside a tangible-PPE table) are weak evidence.
Findings are review hints and are never automatic repairs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_legacy_source_scope import fold


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "build" / "catalog_enriched.jsonl"

FAMILY_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "asset_class": {
        "investment_property": ("bat dong san dau tu",),
        "intangible_fixed_assets": (
            "tai san co dinh vo hinh",
            "tscd vo hinh",
        ),
        "tangible_fixed_assets": (
            "tai san co dinh huu hinh",
            "tscd huu hinh",
        ),
    },
    "governance_role": {
        "board": ("hoi dong quan tri", "hdqt"),
        "management": ("ban giam doc", "ban dieu hanh"),
        "supervisory": ("ban kiem soat",),
    },
    "provision_type": {
        "general": ("du phong chung",),
        "specific": ("du phong cu the",),
    },
    "tax_basis": {
        "current": ("thue thu nhap doanh nghiep hien hanh", "thue tndn hien hanh"),
        "deferred": ("thue thu nhap doanh nghiep hoan lai", "thue tndn hoan lai"),
    },
    "term": {
        "short_term": ("ngan han", "duoi 12 thang", "den 12 thang"),
        "long_term": ("dai han", "tren 12 thang", "sau 12 thang"),
    },
    "valuation_basis": {
        "gross": ("nguyen gia", "gia goc", "truoc du phong"),
        "net": ("gia tri con lai", "gia tri thuan", "sau du phong"),
    },
    "share_basis": {
        "authorized": ("duoc duyet", "duoc phep phat hanh"),
        "issued": ("da phat hanh",),
        "outstanding": ("dang luu hanh", "luu hanh"),
    },
    "eps_basis": {
        "basic": ("lai co ban tren co phieu", "lai co ban"),
        "diluted": ("lai suy giam tren co phieu", "lai suy giam"),
    },
    "geography": {
        "domestic": ("trong nuoc", "noi dia"),
        "foreign": ("nuoc ngoai", "xuat khau"),
    },
    "counterparty": {
        "related_party": ("ben lien quan",),
        "third_party": ("ben thu ba", "ben khac"),
    },
    "cash_flow_scope": {
        "operating": ("luu chuyen tien tu hoat dong kinh doanh",),
        "investing": ("luu chuyen tien tu hoat dong dau tu",),
        "financing": ("luu chuyen tien tu hoat dong tai chinh",),
    },
    "reporting_scope": {
        "group": ("tap doan", "hop nhat"),
        "parent": ("cong ty me", "bao cao rieng"),
    },
}


def phrase_score(text: object, phrases: tuple[str, ...]) -> int:
    normalized = fold(text)
    return sum(normalized.count(phrase) for phrase in phrases)


def last_family(
    text: object, families: dict[str, tuple[str, ...]]
) -> str | None:
    """Return the family of the last explicit heading phrase in ``text``."""

    normalized = fold(text)
    positions = {
        family: max((normalized.rfind(phrase) for phrase in phrases), default=-1)
        for family, phrases in families.items()
    }
    maximum = max(positions.values(), default=-1)
    winners = [family for family, position in positions.items() if position == maximum and position >= 0]
    return winners[0] if len(winners) == 1 else None


def classify_context(
    cell: dict[str, Any],
    metadata: dict[str, Any],
    families: dict[str, tuple[str, ...]],
) -> tuple[str | None, dict[str, int]]:
    """Classify from independent context with title/lineage dominance.

    Section titles and the selected cell's physical header path are strong.
    Full catalog search text is weak because it includes all movement rows and
    can mention a source/destination asset class without changing table type.
    """

    header = " | ".join(str(value) for value in cell.get("header_path", []))
    source_context = cell.get("source_context", "")
    section_title = metadata.get("section_title", "")
    strong_parts = [section_title, header, source_context]
    weak = metadata.get("search_text", "")
    scores = {
        family: 4 * sum(phrase_score(part, phrases) for part in strong_parts)
        + phrase_score(weak, phrases)
        for family, phrases in families.items()
    }
    # Physical column lineage is closest to the selected cell.  Otherwise use
    # the last explicit heading: OCR context often carries the previous
    # section first (q889), then introduces the actual section last.
    for strong in (header, source_context, section_title):
        winner = last_family(strong, families)
        if winner is not None:
            return winner, scores
    return None, scores


def load_catalog(required_refs: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with CATALOG.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            table_ref = str(row.get("table_ref", ""))
            if table_ref in required_refs:
                result[table_ref] = row
    return result


def audit(lineage: dict[str, Any]) -> dict[str, Any]:
    required_refs = {
        str(cell["source_table"])
        for record in lineage.get("records", [])
        for cell in record.get("cells", [])
        if cell.get("source_table")
    }
    catalog = load_catalog(required_refs)
    findings: list[dict[str, Any]] = []
    groups_checked = 0

    for record in lineage.get("records", []):
        by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cell in record.get("cells", []):
            by_metric[str(cell.get("metric_key", ""))].append(cell)
        for metric_key, cells in by_metric.items():
            if len(cells) < 2:
                continue
            for group, families in FAMILY_GROUPS.items():
                classified: list[dict[str, Any]] = []
                for cell in cells:
                    table_ref = str(cell.get("source_table", ""))
                    family, scores = classify_context(
                        cell,
                        catalog.get(table_ref, {}),
                        families,
                    )
                    classified.append(
                        {
                            "ticker": cell.get("ticker"),
                            "year": cell.get("year"),
                            "table_ref": table_ref,
                            "family": family,
                            "scores": scores,
                            "source_label": cell.get("source_label"),
                            "header_path": cell.get("header_path"),
                            "source_context": cell.get("source_context"),
                        }
                    )
                known = Counter(item["family"] for item in classified if item["family"])
                if not known:
                    continue
                groups_checked += 1
                if len(known) <= 1:
                    continue
                findings.append(
                    {
                        "id": int(record["id"]),
                        "metric_key": metric_key,
                        "context_group": group,
                        "families": dict(known),
                        "question": record.get("question"),
                        "answer": record.get("answer"),
                        "operands": classified,
                    }
                )

    return {
        "kind": "operand_table_context_consistency_review_queue",
        "lineage_submission": lineage.get("submission"),
        "lineage_records": len(lineage.get("records", [])),
        "catalog_refs_required": len(required_refs),
        "catalog_refs_resolved": len(catalog),
        "same_metric_context_groups_checked": groups_checked,
        "finding_count": len(findings),
        "question_count": len({item["id"] for item in findings}),
        "policy": "Review only; a context disagreement must be resolved against exact physical tables before mutation.",
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lineage_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lineage = json.loads(args.lineage_report.read_text(encoding="utf-8"))
    payload = audit(lineage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "findings"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
