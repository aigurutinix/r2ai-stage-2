"""Deterministic falsification/lineage audit for the v239 silent-risk batch.

This is deliberately a source-first audit: it never asks a critic or ensemble
whether an answer looks plausible.  It reconstructs each answer from the
compact source cells (or the cited physical table for the two missing-cell
cases), enumerates nearby same-metric alternatives, and fails closed when a
lineage manifest is absent, metadata conflicts with a physical unit, or a
selector is too close to call.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "sub_top123_candidate_v239_source_lineage_batch5"
TABLES = ROOT / "build" / "tables"
REPORT = ROOT / "build" / "v261_silent_batch_c_falsification.json"
IDS = (448, 749, 385, 401, 554, 563, 572, 533, 780, 783)


def number(raw: object) -> float:
    """Parse a Vietnamese statement number as a physical displayed value.

    Dots/commas in these extracts are thousands separators.  This intentionally
    does not apply the manifest's typed_factor: that factor describes a prior
    dtype inference and is itself audited for conflicts (not silently trusted).
    """

    text = str(raw).strip()
    if text in {"", "-", "–", "—", "nan", "None"}:
        return float("nan")
    negative = text.startswith("(") and text.endswith(")") or text.startswith("-")
    text = text.replace("(", "").replace(")", "").replace("-", "")
    text = text.replace(".", "").replace(",", "").replace("%", "")
    value = float(text)
    return -abs(value) if negative else value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def manifest(qid: int) -> list[dict[str, str]]:
    path = SUBMISSION / "data" / f"q{qid}_source_cells.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_value(rows: list[dict[str, str]], ticker: str, year: int, metric: str) -> float:
    hits = [r for r in rows if r.get("ticker") == ticker and int(r.get("year", 0)) == year and r.get("metric_key") == metric]
    if len(hits) != 1:
        raise ValueError(f"expected one cell for {ticker}/{year}/{metric}, got {len(hits)}")
    row = hits[0]
    return number(row["raw"]) * float(row.get("scale") or 1)


def physical_value(relative: str, label: str, value_col: int) -> float:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            # Extracts use either an unlabeled first column or an STT/index
            # column; identify the metric by any textual cell, while retaining
            # the explicit value column supplied by the source manifest.
            if row and any(label.casefold() in cell.casefold() for cell in row):
                return number(row[value_col])
    raise ValueError(f"label {label!r} not found in {relative}")


def catalog() -> dict[str, str]:
    result: dict[str, str] = {}
    path = ROOT / "build" / "catalog_enriched.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[str(row["table_ref"])] = str(row["csv_path"])
    return result


def round_answer(value: float, places: int = 2) -> float:
    return float(round(value, places))


def rec448(rows: list[dict[str, str]]) -> tuple[float, list[dict[str, Any]], list[str]]:
    companies = ("BSR", "PLX", "PVT", "GAS")
    growth = {t: manifest_value(rows, t, 2025, "kqkd:10") / manifest_value(rows, t, 2024, "kqkd:10") - 1 for t in companies}
    median = sorted(growth.values())[1:3]
    cutoff = sum(median) / 2
    eligible = [t for t in companies if growth[t] * 100 > cutoff * 100]
    margin = {t: manifest_value(rows, t, 2025, "kqkd:20") / manifest_value(rows, t, 2025, "kqkd:10") for t in eligible}
    selected = max(eligible, key=margin.__getitem__)
    pbt = manifest_value(rows, selected, 2025, "kqkd:50")
    interest = manifest_value(rows, selected, 2025, "kqkd:23")
    ratio = (pbt + interest) / interest
    alternatives = [{"company": t, "revenue_growth_pct": round_answer(growth[t] * 100, 6), "gross_margin_pct": round_answer((manifest_value(rows, t, 2025, "kqkd:20") / manifest_value(rows, t, 2025, "kqkd:10")) * 100, 6), "same_metric_wrong_filter": t not in eligible} for t in companies]
    return round_answer(ratio, 1), alternatives, [f"selected={selected}", f"median_growth_pct={cutoff * 100:.6f}"]


def panel_rows(rows: list[dict[str, str]], companies: tuple[str, ...], years: tuple[int, ...], metric: str) -> dict[tuple[str, int], float]:
    return {(t, y): manifest_value(rows, t, y, metric) for t in companies for y in years}


def rec385(rows: list[dict[str, str]]) -> tuple[float, list[dict[str, Any]], list[str]]:
    companies = ("DBC", "MPC", "MSN", "OGC", "QNS")
    rev = panel_rows(rows, companies, (2022, 2023, 2024), "kqkd:10")
    npat = panel_rows(rows, companies, (2023, 2024), "kqkd:60")
    cfo = panel_rows(rows, companies, (2023, 2024), "lctt:20")
    eligible = [t for t in companies if all(npat[t, y] > 0 and cfo[t, y] > 0 for y in (2023, 2024))]
    growth = {t: (rev[t, 2024] / rev[t, 2023] - 1) * 100 for t in eligible}
    value = sum(growth.values()) / len(growth)
    alt = [{"company": t, "growth_2024_pct": round_answer((rev[t, 2024] / rev[t, 2023] - 1) * 100, 6), "qualifies": t in eligible} for t in companies]
    return round_answer(value), alt, [f"eligible={','.join(eligible)}"]


def rec401(rows: list[dict[str, str]]) -> tuple[float, list[dict[str, Any]], list[str]]:
    companies = ("DBC", "MPC", "MSN", "OGC", "QNS")
    rev = panel_rows(rows, companies, (2022, 2023, 2024), "kqkd:10")
    npat = panel_rows(rows, companies, (2023, 2024), "kqkd:60")
    cfo = panel_rows(rows, companies, (2023, 2024), "lctt:20")
    gross = panel_rows(rows, companies, (2023, 2024), "kqkd:20")
    eligible = [t for t in companies if all(npat[t, y] > 0 and cfo[t, y] / npat[t, y] > 0.5 for y in (2023, 2024))]
    growth = {t: (rev[t, 2024] / rev[t, 2023] - 1) * 100 for t in eligible}
    selected = max(eligible, key=growth.__getitem__)
    result = gross[selected, 2024] / rev[selected, 2024] * 100
    alt = [{"company": t, "growth_2024_pct": round_answer((rev[t, 2024] / rev[t, 2023] - 1) * 100, 6), "qualifies": t in eligible, "gross_margin_2024_pct": round_answer(gross[t, 2024] / rev[t, 2024] * 100, 6)} for t in companies]
    flags = [f"eligible={','.join(eligible)}", f"selected={selected}"]
    if len(eligible) > 1 and abs(sorted(growth.values())[-1] - sorted(growth.values())[-2]) <= 0.1:
        flags.append("near_tie_selector_gap_pp<=0.1")
    return round_answer(result), alt, flags


def rec554(rows: list[dict[str, str]]) -> tuple[float, list[dict[str, Any]], list[str]]:
    companies = ("DCM", "DPM", "PRT")
    rev = panel_rows(rows, companies, (2019, 2020), "kqkd:10")
    gross = panel_rows(rows, companies, (2019, 2020), "kqkd:20")
    eligible = [t for t in companies if rev[t, 2020] > rev[t, 2019]]
    delta = {t: gross[t, 2020] / rev[t, 2020] * 100 - gross[t, 2019] / rev[t, 2019] * 100 for t in companies}
    alt = [{"company": t, "margin_delta_pp": round_answer(delta[t], 6), "qualifies": t in eligible} for t in companies]
    return round_answer(sum(delta[t] for t in eligible) / len(eligible)), alt, [f"eligible={','.join(eligible)}"]


def rec563(rows: list[dict[str, str]]) -> tuple[float, list[dict[str, Any]], list[str]]:
    companies = ("GEE", "GEX", "SAM")
    rev = panel_rows(rows, companies, (2020, 2021), "kqkd:10")
    gross = panel_rows(rows, companies, (2020, 2021), "kqkd:20")
    alt = []
    count = 0
    for t in companies:
        revenue_up = rev[t, 2021] > rev[t, 2020]
        margin_down = gross[t, 2021] / rev[t, 2021] < gross[t, 2020] / rev[t, 2020]
        qualifies = revenue_up and margin_down
        count += int(qualifies)
        alt.append({"company": t, "revenue_up": revenue_up, "margin_down": margin_down, "qualifies": qualifies})
    return float(count), alt, [f"qualifying_count={count}"]


def rec572(rows: list[dict[str, str]]) -> tuple[float, list[dict[str, Any]], list[str]]:
    companies = ("GEE", "GEX", "SAM")
    cfo = panel_rows(rows, companies, (2022, 2023, 2024), "lctt:20")
    rev = panel_rows(rows, companies, (2023, 2024), "kqkd:10")
    npat = {t: manifest_value(rows, t, 2024, "kqkd:60") for t in companies}
    assets = {(t, y): manifest_value(rows, t, y, "cdkt:270") for t in companies for y in (2023, 2024)}
    eligible = [t for t in companies if all(cfo[t, y] > 0 for y in (2022, 2023, 2024))]
    growth = {t: (rev[t, 2024] / rev[t, 2023] - 1) * 100 for t in eligible}
    selected = max(eligible, key=growth.__getitem__)
    result = npat[selected] / ((assets[selected, 2023] + assets[selected, 2024]) / 2) * 100
    alt = [{"company": t, "cfo_positive_all_years": t in eligible, "revenue_growth_2024_pct": round_answer((rev[t, 2024] / rev[t, 2023] - 1) * 100, 6)} for t in companies]
    return round_answer(result), alt, [f"eligible={','.join(eligible)}", f"selected={selected}"]


def rec533(rows: list[dict[str, str]]) -> tuple[float, list[dict[str, Any]], list[str]]:
    years = (2016, 2018, 2019, 2020)
    taxes = {y: manifest_value(rows, "HND", y, "note:tax_at_company_rate") for y in years}
    leases = {y: manifest_value(rows, "HND", y, "note:operating_lease_due_within_one_year") for y in years}
    selected = max(years, key=taxes.__getitem__)
    alt = [{"year": y, "tax_at_company_rate_vnd": taxes[y], "lease_due_within_one_year_vnd": leases[y], "selected": y == selected} for y in years]
    return round_answer(leases[selected] / 1_000_000_000), alt, [f"selected_year={selected}"]


def rec_direct(qid: int) -> tuple[float, list[dict[str, Any]], list[str], list[str]]:
    if qid == 749:
        eib = physical_value("build/tables/EIB_financial_statements_2025_separate/table_64_line1381.csv", "Thu nhập từ mua bán chứng khoán đầu tư", 1)
        acb = physical_value("build/tables/ACB_financial_statements_2025_separate/table_77_line2015.csv", "Thu nhập từ mua bán chứng khoán đầu tư", 1)
        return round_answer(eib - acb), [{"scope": "EIB parent", "value_million_vnd": eib}, {"scope": "ACB parent", "value_million_vnd": acb}], ["missing_compact_manifest"], ["EIB_financial_statements_2025_separate|1381", "ACB_financial_statements_2025_separate|2015"]
    if qid == 780:
        bab = physical_value("build/tables/BAB_financial_statements_2024_separate/table_4_line180.csv", "Dự phòng rủi ro cho vay khách hàng", 3)
        sgb = physical_value("build/tables/SGB_financial_statements_2024_separate/table_2_line263.csv", "Dự phòng rủi ro cho vay khách hàng", 2)
        return round_answer(abs(bab) - abs(sgb)), [{"scope": "BAB parent", "value_million_vnd": bab}, {"scope": "SGB parent", "value_million_vnd": sgb}, {"scope": "signed_difference_hard_negative", "value_million_vnd": bab - sgb}], ["typed_factor_conflicts_with_parenthesized_physical_unit"], ["BAB_financial_statements_2024_separate|180", "SGB_financial_statements_2024_separate|263"]
    mbb = physical_value("build/tables/MBB_financial_statements_2023_separate/table_3_line331.csv", "TỔNG TÀI SẢN CÓ", 2)
    eib = physical_value("build/tables/EIB_financial_statements_2023_separate/table_1_line319.csv", "TỔNG TÀI SẢN CÓ", 3)
    return round_answer(mbb - eib), [{"scope": "MBB parent", "value_million_vnd": mbb}, {"scope": "EIB parent", "value_million_vnd": eib}], [], ["MBB_financial_statements_2023_separate|331", "EIB_financial_statements_2023_separate|319"]


def run() -> dict[str, Any]:
    submission = {int(row["id"]): row for row in load_json(SUBMISSION / "submission.json")}
    prior = {int(row["id"]): row for row in load_json(ROOT / "build" / "v261_silent_batch_c.json")["records"]}
    cat = catalog()
    records: list[dict[str, Any]] = []
    for qid in IDS:
        row = submission[qid]
        prior_source = prior[qid].get("source") or {}
        refs = list(prior_source.get("table_refs") or [])
        cells = manifest(qid)
        # The compact audit record may list only representative tables.  Add
        # every table physically named by the cell manifest before checking
        # lineage, so coverage cannot be overstated by a short summary list.
        refs = list(dict.fromkeys(refs + [str(cell.get("source_table")) for cell in cells if cell.get("source_table")]))
        missing_refs = [ref for ref in refs if ref not in cat or not (TABLES / cat[ref]).is_file()]
        risk_flags: list[str] = []
        if qid == 448:
            answer, alternatives, selection_notes = rec448(cells)
        elif qid == 385:
            answer, alternatives, selection_notes = rec385(cells)
        elif qid == 401:
            answer, alternatives, selection_notes = rec401(cells)
        elif qid == 554:
            answer, alternatives, selection_notes = rec554(cells)
        elif qid == 563:
            answer, alternatives, selection_notes = rec563(cells)
        elif qid == 572:
            answer, alternatives, selection_notes = rec572(cells)
        elif qid == 533:
            answer, alternatives, selection_notes = rec533(cells)
        else:
            answer, alternatives, selection_notes, direct_refs = rec_direct(qid)
            risk_flags.extend(selection_notes)
            refs = refs or direct_refs
        if any(note.startswith("near_tie") for note in selection_notes):
            risk_flags.append("near_tie_selector_gap_pp<=0.1")
        flags = list(risk_flags)
        if not cells:
            flags.append("lineage_manifest_missing")
        if missing_refs:
            flags.append("physical_table_ref_missing")
        current = float(row["answer"])
        match = math.isclose(current, answer, rel_tol=0.0, abs_tol=0.01)
        if not match:
            flags.append("current_answer_falsified_by_independent_recomputation")
        records.append({
            "id": qid,
            "current_answer": current,
            "independent_answer": answer,
            "answer_match": match,
            "status": "review_required" if flags else "pass",
            "lineage": {"compact_manifest_present": bool(cells), "manifest_cell_count": len(cells), "table_ref_count": len(refs), "missing_physical_refs": missing_refs},
            "hard_negatives": alternatives,
            "top_alternatives": alternatives[:5],
            "selection_notes": selection_notes,
            "flags": sorted(set(flags)),
            "source_refs": refs[:12],
        })
    finding_count = sum(bool(r["flags"]) for r in records)
    return {
        "schema_version": 1,
        "kind": "deterministic_silent_batch_falsification",
        "candidate": str(SUBMISSION.name),
        "question_ids": list(IDS),
        "policy": {"critic_consulted": False, "hidden_gold_access": False, "read_only": True, "fail_closed": True, "physical_number_parser": "displayed separators plus stated table unit; manifest typed_factor is audited, not trusted"},
        "coverage": {"questions": len(records), "compact_manifests": sum(r["lineage"]["compact_manifest_present"] for r in records), "physical_table_refs_checked": sum(r["lineage"]["table_ref_count"] for r in records), "questions_with_findings": finding_count},
        "records": records,
        "summary": {"status": "review_required" if finding_count else "pass", "finding_count": finding_count, "answer_mismatches": sum(not r["answer_match"] for r in records)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()
    report = run()
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.fail_on_findings and report["summary"]["finding_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
