"""Flag source-audited programs that use a broader or adjacent BCTC metric.

This is a read-only semantic gate.  It compares unambiguous phrases in each
Vietnamese question with the exact metric keys recorded in ``source_audit``.
The gate deliberately reports candidates for manual review; it never changes
an answer by itself.  A note-disclosure metric is accepted when either its key
or its verified source-row label names the requested component.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fold(text: object) -> str:
    value = unicodedata.normalize("NFD", str(text).lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


@dataclass(frozen=True)
class Rule:
    name: str
    phrases: tuple[str, ...]
    accepted_metrics: tuple[str, ...]
    accepted_evidence: tuple[str, ...] = ()
    excluded_phrases: tuple[str, ...] = ()
    rationale: str = ""

    def matches_question(self, question: str) -> bool:
        padded = f" {question} "
        return (
            any(f" {phrase} " in padded for phrase in self.phrases)
            and not any(f" {phrase} " in padded for phrase in self.excluded_phrases)
        )

    def accepts(self, metrics: set[str], evidence: str) -> bool:
        if any(metric in metrics for metric in self.accepted_metrics):
            return True
        return any(token in evidence for token in self.accepted_evidence)


# Prefer narrow, accounting-code-bearing phrases.  Broad words such as
# "doanh thu", "chi phi" or "tien" are intentionally absent because they are
# not sufficient to distinguish a primary-statement subtotal from a note row.
RULES = (
    Rule(
        "tangible_fixed_assets",
        ("tai san co dinh huu hinh",),
        ("cdkt:221",),
        ("tangible fixed asset", "tai san co dinh huu hinh", "tscd huu hinh"),
        rationale="Tangible fixed-assets NBV is code 221; code 220 is total fixed assets.",
    ),
    Rule(
        "intangible_fixed_assets",
        ("tai san co dinh vo hinh",),
        ("cdkt:227",),
        ("intangible fixed asset", "intangible nbv", "intangible total nbv", "tai san co dinh vo hinh", "tscd vo hinh"),
        rationale="Intangible fixed-assets NBV is code 227.",
    ),
    Rule(
        "construction_in_progress",
        ("xay dung co ban do dang", "xay dung co ban dang do dang"),
        ("cdkt:242",),
        ("construction in progress", "xay dung co ban do dang", "xay dung co ban dang do dang"),
        rationale="Construction in progress is code 242; code 240 is its parent subtotal.",
    ),
    Rule(
        "short_term_supplier_advances",
        ("tra truoc cho nguoi ban ngan han",),
        ("cdkt:132",),
        ("short term supplier advance", "advances to supplier", "tra truoc cho nguoi ban ngan han"),
        rationale="Short-term advances to suppliers are code 132; code 130 is total receivables.",
    ),
    Rule(
        "short_term_trade_receivables",
        ("phai thu ngan han cua khach hang", "phai thu khach hang ngan han"),
        ("cdkt:131",),
        ("short term trade receivable", "phai thu ngan han cua khach hang", "phai thu khach hang ngan han"),
        rationale="Short-term trade receivables are code 131.",
    ),
    Rule(
        "other_short_term_receivables",
        ("phai thu ngan han khac", "phai thu khac ngan han"),
        ("cdkt:136",),
        ("other short term receivable", "short term other receivable", "related other current receivable", "phai thu ngan han khac", "phai thu khac ngan han"),
        rationale="Other short-term receivables are code 136.",
    ),
    Rule(
        "doubtful_short_term_receivables_provision",
        ("du phong phai thu ngan han kho doi",),
        ("cdkt:137",),
        ("doubtful receivable provision", "du phong phai thu ngan han kho doi"),
        rationale="Provision for doubtful short-term receivables is code 137.",
    ),
    Rule(
        "long_term_loan_receivables",
        ("cho vay dai han",),
        ("cdkt:215",),
        ("long term loan receivable", "long term lending", "cho vay dai han"),
        rationale="Long-term loan receivables are code 215.",
    ),
    Rule(
        "cash_on_hand",
        ("tien mat",),
        ("cdkt:111",),
        ("cash on hand", "cash ending", "tien mat", "cash and gold", "cash gold precious"),
        excluded_phrases=("luu chuyen tien mat",),
        rationale="Cash on hand is code 111; code 110 also includes cash equivalents.",
    ),
    Rule(
        "cash_equivalents_only",
        ("cac khoan tuong duong tien", "khoan tuong duong tien"),
        ("cdkt:112",),
        ("cash equivalent", "khoan tuong duong tien"),
        excluded_phrases=(
            "tien va cac khoan tuong duong tien",
            "tien va khoan tuong duong tien",
            "tien va tuong duong tien",
            "tien va cac khoan tuong duong",
        ),
        rationale="Cash equivalents alone are code 112.",
    ),
    Rule(
        "inventory_net",
        ("tong gia tri hang ton kho", "gia tri thuan cua hang ton kho", "gia tri hang ton kho cuoi"),
        ("cdkt:140",),
        ("inventory net", "net inventory", "hang ton kho thuan"),
        excluded_phrases=("gia goc",),
        rationale="Net inventory is code 140; code 141 is gross inventory before provision.",
    ),
    Rule(
        "investment_property",
        ("bat dong san dau tu",),
        ("cdkt:230",),
        ("investment property", "bat dong san dau tu"),
        rationale="Investment-property NBV is code 230.",
    ),
    Rule(
        "long_term_financial_investments",
        ("dau tu tai chinh dai han",),
        ("cdkt:250",),
        ("long term financial investment", "dau tu tai chinh dai han"),
        rationale="Total long-term financial investments are code 250.",
    ),
    Rule(
        "investment_in_subsidiaries",
        ("dau tu vao cong ty con",),
        ("cdkt:251",),
        ("subsidiary investment", "investment in subsidiar", "dau tu vao cong ty con"),
        rationale="Gross investment in subsidiaries is code 251; net wording may require a provision adjustment.",
    ),
    Rule(
        "investment_in_associates",
        ("dau tu vao cong ty lien ket",),
        ("cdkt:252",),
        ("associate investment", "investment in associate", "dau tu vao cong ty lien ket"),
        rationale="Investment in associates/joint ventures is code 252.",
    ),
    Rule(
        "short_term_prepaid_expense",
        ("chi phi tra truoc ngan han",),
        ("cdkt:151",),
        ("short term prepaid expense", "chi phi tra truoc ngan han"),
        rationale="Short-term prepaid expense is code 151.",
    ),
    Rule(
        "long_term_prepaid_expense",
        ("chi phi tra truoc dai han",),
        ("cdkt:261",),
        ("long term prepaid expense", "chi phi tra truoc dai han"),
        rationale="Long-term prepaid expense is code 261.",
    ),
    Rule(
        "current_assets",
        ("tai san ngan han",),
        ("cdkt:100",),
        ("current assets", "tai san ngan han"),
        rationale="Total current assets are code 100.",
    ),
    Rule(
        "long_term_assets",
        ("tai san dai han",),
        ("cdkt:200",),
        ("long term assets", "non current assets", "tai san dai han"),
        rationale="Total long-term assets are code 200.",
    ),
    Rule(
        "short_term_customer_advances",
        ("nguoi mua tra tien truoc ngan han", "khach hang tra truoc ngan han", "tra truoc ngan han cua khach hang"),
        ("cdkt:312",),
        ("short term customer advance", "short term advances from customer", "nguoi mua tra tien truoc ngan han", "khach hang tra truoc ngan han"),
        rationale="Short-term advances from customers are code 312.",
    ),
    Rule(
        "short_term_trade_payables",
        ("phai tra nguoi ban ngan han", "no phai tra nguoi ban ngan han"),
        ("cdkt:311",),
        ("short term trade payable", "related trade payable", "phai tra nguoi ban ngan han"),
        excluded_phrases=("lai vay phai tra nguoi ban ngan han",),
        rationale="Short-term trade payables are code 311.",
    ),
    Rule(
        "taxes_and_state_payables",
        ("thue va cac khoan phai nop nha nuoc", "thue va cac khoan phai nop"),
        ("cdkt:313",),
        ("taxes and state payables", "thue va cac khoan phai nop"),
        rationale="Taxes and amounts payable to the State are code 313.",
    ),
    Rule(
        "short_term_unearned_revenue",
        ("doanh thu chua thuc hien ngan han",),
        ("cdkt:318",),
        ("short term unearned revenue", "doanh thu chua thuc hien ngan han"),
        rationale="Short-term unearned revenue is code 318.",
    ),
    Rule(
        "short_term_borrowings",
        ("vay ngan han", "no vay ngan han"),
        ("cdkt:320",),
        ("short term borrowing", "short term loan", "vay ngan han"),
        excluded_phrases=("cho vay ngan han", "chi phi lai vay ngan han"),
        rationale="Short-term borrowings are code 320 in the normalized statement schema.",
    ),
    Rule(
        "current_liabilities",
        ("tong no ngan han",),
        ("cdkt:310",),
        ("current liabilities", "tong no ngan han"),
        rationale="Total current liabilities are code 310.",
    ),
    Rule(
        "long_term_liabilities",
        ("tong no dai han", "no dai han"),
        ("cdkt:330",),
        ("long term liabilities", "non current liabilities", "tong no dai han"),
        excluded_phrases=("vay dai han", "no vay dai han"),
        rationale="Total long-term liabilities are code 330.",
    ),
    Rule(
        "total_liabilities",
        ("tong no phai tra",),
        ("cdkt:300",),
        ("total liabilities", "total financial liabilities", "tong no phai tra"),
        rationale="Total liabilities are code 300.",
    ),
    Rule(
        "total_assets",
        ("tong tai san", "tong cong tai san"),
        ("cdkt:270",),
        ("total assets", "tong tai san", "tong cong tai san"),
        excluded_phrases=("tong tai san co dinh",),
        rationale="Total assets are code 270.",
    ),
    Rule(
        "equity",
        ("von chu so huu",),
        ("cdkt:400",),
        ("shareholders equity", "owners equity", "von chu so huu"),
        rationale="Total equity is code 400.",
    ),
    Rule(
        "share_capital",
        ("von co phan", "von gop cua chu so huu"),
        ("cdkt:411",),
        ("share capital", "owner invested capital", "von co phan", "von gop cua chu so huu"),
        rationale="Share/owner-invested capital is code 411.",
    ),
    Rule(
        "construction_investment_capital",
        ("nguon von dau tu xay dung co ban",),
        ("cdkt:422",),
        ("construction investment capital", "nguon von dau tu xay dung co ban"),
        rationale="Construction-investment capital is balance-sheet code 422.",
    ),
    Rule(
        "development_investment_fund",
        ("quy dau tu phat trien",),
        ("cdkt:418",),
        ("development investment fund", "quy dau tu phat trien"),
        rationale="Development investment fund is code 418.",
    ),
    Rule(
        "retained_earnings",
        ("loi nhuan sau thue chua phan phoi", "loi nhuan chua phan phoi"),
        ("cdkt:421",),
        ("retained earnings", "undistributed profit", "loi nhuan chua phan phoi"),
        rationale="Retained earnings are code 421.",
    ),
    Rule(
        "total_capital_sources",
        ("tong nguon von",),
        ("cdkt:440", "cdkt:270"),
        ("total capital sources", "tong nguon von", "total assets"),
        excluded_phrases=("nguon von dau tu xay dung co ban",),
        rationale="Total capital sources are code 440 and reconcile to total assets code 270.",
    ),
    Rule(
        "net_revenue",
        ("doanh thu thuan",),
        ("kqkd:10",),
        ("net revenue", "lpg revenue", "doanh thu thuan"),
        rationale="Net revenue is income-statement code 10.",
    ),
    Rule(
        "cost_of_goods_sold",
        ("gia von hang ban",),
        ("kqkd:11",),
        ("cost of goods sold", "cost of sales", "gia von hang ban"),
        rationale="Cost of goods sold is code 11.",
    ),
    Rule(
        "gross_profit",
        ("loi nhuan gop",),
        ("kqkd:20",),
        ("gross profit", "loi nhuan gop"),
        rationale="Gross profit is code 20.",
    ),
    Rule(
        "finance_income",
        ("doanh thu hoat dong tai chinh", "doanh thu tai chinh"),
        ("kqkd:21",),
        ("finance income", "financial income", "financial revenue", "doanh thu hoat dong tai chinh"),
        rationale="Finance income is code 21.",
    ),
    Rule(
        "finance_cost",
        ("chi phi tai chinh",),
        ("kqkd:22",),
        ("finance cost", "financial expense", "chi phi tai chinh"),
        rationale="Total finance cost is code 22.",
    ),
    Rule(
        "interest_expense",
        ("chi phi lai vay",),
        ("kqkd:23",),
        ("interest expense", "borrowing cost", "chi phi lai vay"),
        rationale="Interest expense is code 23 or a specific borrowing-cost disclosure.",
    ),
    Rule(
        "selling_expense",
        ("chi phi ban hang",),
        ("kqkd:25",),
        ("selling expense", "chi phi ban hang"),
        rationale="Selling expense is code 25.",
    ),
    Rule(
        "administrative_expense",
        ("chi phi quan ly doanh nghiep",),
        ("kqkd:26",),
        ("administrative expense", "chi phi quan ly doanh nghiep"),
        rationale="Administrative expense is code 26.",
    ),
    Rule(
        "operating_profit",
        ("loi nhuan thuan tu hoat dong kinh doanh",),
        ("kqkd:30",),
        ("operating profit", "profit from operating activities", "loi nhuan thuan tu hoat dong kinh doanh"),
        rationale="Net profit from operating activities is code 30.",
    ),
    Rule(
        "other_income",
        ("thu nhap khac",),
        ("kqkd:31",),
        ("other income", "asset liquidation income", "thu nhap khac"),
        excluded_phrases=("thu nhap khac thuan",),
        rationale="Other income is code 31.",
    ),
    Rule(
        "other_expense",
        ("chi phi khac",),
        ("kqkd:32",),
        ("other expense", "chi phi khac"),
        rationale="Other expense is code 32.",
    ),
    Rule(
        "other_profit",
        ("loi nhuan khac", "thu nhap khac thuan"),
        ("kqkd:40",),
        ("other profit", "other result", "loi nhuan khac"),
        rationale="Other profit/result is code 40.",
    ),
    Rule(
        "profit_before_tax",
        ("loi nhuan truoc thue", "tong loi nhuan ke toan truoc thue"),
        ("kqkd:50",),
        ("profit before tax", "pbt", "loi nhuan truoc thue"),
        rationale="Profit before tax is code 50.",
    ),
    Rule(
        "current_income_tax",
        ("thue thu nhap doanh nghiep hien hanh", "chi phi thue tndn hien hanh"),
        ("kqkd:51",),
        ("current income tax", "current tax expense", "thue tndn hien hanh"),
        rationale="Current income-tax expense is code 51.",
    ),
    Rule(
        "deferred_income_tax",
        ("thue thu nhap doanh nghiep hoan lai", "chi phi thue tndn hoan lai"),
        ("kqkd:52",),
        ("deferred income tax", "deferred tax expense", "thue tndn hoan lai"),
        rationale="Deferred income-tax expense is code 52.",
    ),
    Rule(
        "net_profit_after_tax",
        ("loi nhuan sau thue",),
        ("kqkd:60",),
        ("net profit after tax", "npat", "loi nhuan sau thue"),
        excluded_phrases=(
            "thuoc ve co dong",
            "cua co dong cong ty me",
            "loi nhuan sau thue chua phan phoi",
        ),
        rationale="Total net profit after tax is code 60.",
    ),
    Rule(
        "parent_shareholder_profit",
        ("loi nhuan sau thue cua co dong cong ty me", "loi nhuan sau thue thuoc ve co dong cong ty me"),
        ("kqkd:61",),
        ("profit attributable to parent", "parent shareholder profit", "co dong cong ty me"),
        rationale="Profit attributable to owners of the parent is code 61 or its direct disclosure.",
    ),
    Rule(
        "non_controlling_profit",
        ("loi nhuan sau thue cua co dong khong kiem soat", "loi nhuan sau thue thuoc ve co dong khong kiem soat"),
        ("kqkd:62",),
        ("non controlling interest profit", "non controlling shareholder profit", "co dong khong kiem soat"),
        rationale="Profit attributable to non-controlling interests is code 62.",
    ),
    Rule(
        "basic_eps",
        ("lai co ban tren co phieu", "eps co ban"),
        ("kqkd:70",),
        ("basic eps", "lai co ban tren co phieu"),
        rationale="Basic EPS is code 70.",
    ),
    Rule(
        "diluted_eps",
        ("lai suy giam tren co phieu", "eps suy giam"),
        ("kqkd:71",),
        ("diluted eps", "lai suy giam tren co phieu"),
        rationale="Diluted EPS is code 71.",
    ),
    Rule(
        "operating_cash_flow",
        ("luu chuyen tien thuan tu hoat dong kinh doanh",),
        ("lctt:20",),
        ("net operating cash flow", "luu chuyen tien thuan tu hoat dong kinh doanh"),
        rationale="Net operating cash flow is cash-flow code 20.",
    ),
    Rule(
        "investing_cash_flow",
        ("luu chuyen tien thuan tu hoat dong dau tu",),
        ("lctt:30",),
        ("net investing cash flow", "luu chuyen tien thuan tu hoat dong dau tu"),
        rationale="Net investing cash flow is cash-flow code 30.",
    ),
    Rule(
        "financing_cash_flow",
        ("luu chuyen tien thuan tu hoat dong tai chinh",),
        ("lctt:40",),
        ("net financing cash flow", "luu chuyen tien thuan tu hoat dong tai chinh"),
        rationale="Net financing cash flow is cash-flow code 40.",
    ),
)


# Exceptions require a source-table reconciliation, not merely a leaderboard
# result.  Keep them visible in the report so a future data change cannot turn
# an undocumented suppression into a silent error.
KNOWN_EXCEPTIONS = {
    (135, "short_term_trade_receivables"): (
        "The question asks for the third-party subset of GAS parent-company "
        "short-term customer receivables.  Balance-sheet code 131 is the combined "
        "total; the cited note table reconciles that total into third parties and "
        "related parties, so the explicit third-party note cell is the more precise "
        "source for the requested answer."
    ),
    (622, "investment_in_subsidiaries"): (
        "DXG code 250 equals subsidiary gross investment (251) less its code-254 "
        "provision in both requested years; no associate or other-investment "
        "component is present, so the requested net balance is reconciled."
    ),
}


def load_audit_rows(submission_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in ("source_audit.json", "panel_source_audit.json"):
        path = submission_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            if not isinstance(row, dict):
                continue
            if row.get("sources"):
                rows.append(row)
                continue
            # Panel manifests summarize evidence counts to stay compact.  The
            # packaged source CSV is the authoritative dependency list, so
            # reconstruct just the fields required by this semantic gate.
            qid = int(row.get("id", -1))
            source_path = submission_dir / "data" / f"q{qid}_source_cells.csv"
            if not source_path.exists():
                continue
            with source_path.open(encoding="utf-8-sig", newline="") as handle:
                source_rows = list(csv.DictReader(handle))
            if not source_rows:
                continue
            enriched = dict(row)
            enriched["sources"] = [
                {
                    "metric": source.get("metric_key", ""),
                    "table_ref": source.get("source_table", ""),
                    "csv": source.get("source_csv", ""),
                    "raw": source.get("raw", ""),
                    "source_row_labels": [],
                }
                for source in source_rows
            ]
            rows.append(enriched)
    return rows


DIRECT_CODE_FILTER = re.compile(
    r"\b(?P<var>df\d+)\s*\[\s*(?P=var)\s*\[\s*['\"](?P<column>[^'\"]+)['\"]\s*\]"
    r"\s*\.astype\(str\)\s*\.str\.strip\(\)\s*==\s*['\"](?P<code>\d+[a-z]?)['\"]\s*\]",
    flags=re.IGNORECASE,
)


def evidence_csv_to_table_ref(csv_path: str) -> str | None:
    """Recover ``report|line`` from a packaged original-table CSV name."""

    stem = Path(csv_path).stem
    report, separator, line = stem.rpartition("_")
    if not separator or not report or not line.isdigit():
        return None
    return f"{report}|{line}"


def load_cube_table_metrics() -> dict[tuple[str, str], set[str]]:
    """Index primary-statement metric families by exact source table and code."""

    cube_path = ROOT / "build" / "statement_cube.jsonl"
    result: dict[tuple[str, str], set[str]] = {}
    with cube_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metric = str(row.get("metric_key", ""))
            if not metric.startswith(("cdkt:", "kqkd:", "lctt:")):
                continue
            code = str(row.get("ma_so", "")).strip().lower()
            table_ref = str(row.get("table_ref", ""))
            if code and table_ref:
                result.setdefault((table_ref, code), set()).add(metric)
                result.setdefault((table_ref, "*"), set()).add(metric.split(":", 1)[0])
    return result


def infer_legacy_query_rows(
    submission: dict[int, dict[str, object]],
    already_audited: set[int],
) -> list[dict[str, object]]:
    """Infer metrics for legacy code-filter programs from their actual CSV table.

    Older programs often filter a statement code directly but have no
    ``source_audit`` record.  Looking only at the numeric code is unsafe because
    code 60 means NPAT on an income statement and ending cash on a cash-flow
    statement.  The statement cube disambiguates the family using the exact
    evidence table read by the query.
    """

    table_metrics = load_cube_table_metrics()
    rows: list[dict[str, object]] = []
    for qid, item in submission.items():
        if qid in already_audited:
            continue
        evidence = [entry for entry in item.get("evidence", []) if isinstance(entry, dict)]
        query = str(item.get("pandas_query", ""))
        inferred_sources: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for match in DIRECT_CODE_FILTER.finditer(query):
            variable = match.group("var").lower()
            variable_match = re.fullmatch(r"df(\d+)", variable)
            if not variable_match:
                continue
            evidence_index = int(variable_match.group(1)) - 1
            if evidence_index < 0 or evidence_index >= len(evidence):
                continue
            csv_path = str(evidence[evidence_index].get("csv_path", ""))
            table_ref = evidence_csv_to_table_ref(csv_path)
            if not table_ref:
                continue
            code = match.group("code").lower()
            key = (table_ref, code)
            if key in seen:
                continue
            seen.add(key)
            metrics = table_metrics.get(key, set())
            if not metrics:
                # OCR can merge/drop an individual statement row even though
                # the table family is unambiguous.  Preserve the requested code
                # while deriving its family from neighboring cube cells.
                metrics = {
                    f"{family}:{code}"
                    for family in table_metrics.get((table_ref, "*"), set())
                }
            for metric in sorted(metrics):
                inferred_sources.append(
                    {
                        "metric": metric,
                        "table_ref": table_ref,
                        "csv": Path(csv_path).name,
                        "label": "",
                        "source_row_labels": [],
                    }
                )
        if inferred_sources:
            rows.append({"id": qid, "sources": inferred_sources, "inferred_from_query": True})
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()

    submission = {
        int(row["id"]): row
        for row in json.loads(
            (args.submission_dir / "submission.json").read_text(encoding="utf-8")
        )
    }
    audit_rows = load_audit_rows(args.submission_dir)
    source_audited_ids = {int(row["id"]) for row in audit_rows}
    inferred_rows = infer_legacy_query_rows(submission, source_audited_ids)
    audit_rows.extend(inferred_rows)
    findings: list[dict[str, object]] = []
    exceptions: list[dict[str, object]] = []
    checked_pairs = 0
    for audit_row in audit_rows:
        qid = int(audit_row["id"])
        question = str(submission[qid].get("question", ""))
        question_folded = fold(question)
        sources = [source for source in audit_row.get("sources", []) if isinstance(source, dict)]
        metrics = {str(source.get("metric", "")) for source in sources}
        evidence_parts = list(metrics)
        for source in sources:
            evidence_parts.append(str(source.get("label", "")))
            evidence_parts.extend(str(value) for value in source.get("source_row_labels", []))
        evidence = fold(" ".join(evidence_parts))
        for rule in RULES:
            if not rule.matches_question(question_folded):
                continue
            checked_pairs += 1
            if rule.accepts(metrics, evidence):
                continue
            finding = {
                "id": qid,
                "rule": rule.name,
                "question": question,
                "answer": submission[qid].get("answer"),
                "metrics": sorted(metrics),
                "source_labels": list(dict.fromkeys(
                    str(label)
                    for source in sources
                    for label in source.get("source_row_labels", [])
                    if str(label).strip()
                )),
                "expected_metrics": list(rule.accepted_metrics),
                "rationale": rule.rationale,
            }
            exception = KNOWN_EXCEPTIONS.get((qid, rule.name))
            if exception:
                finding["exception_evidence"] = exception
                exceptions.append(finding)
            else:
                findings.append(finding)

    findings.sort(key=lambda row: (int(row["id"]), str(row["rule"])))
    result = {
        "submission": str(args.submission_dir),
        "audited_questions_with_sources": len(source_audited_ids),
        "legacy_query_inferred_questions": len(inferred_rows),
        "total_questions_with_metric_evidence": len(audit_rows),
        "checked_rule_question_pairs": checked_pairs,
        "documented_exception_count": len(exceptions),
        "documented_exceptions": exceptions,
        "finding_count": len(findings),
        "findings": findings,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.fail_on_findings and findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
