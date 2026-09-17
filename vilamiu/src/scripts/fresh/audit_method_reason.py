"""Self-audit method_v1 changes: question ↔ layer meta ↔ answer shape.

Does NOT treat hardhop agreement as truth. Flags:
  - unit/scale smells (million vs raw, answer→0)
  - op/filter language mismatch vs question
  - dual_tab when new answer is 0
  - catalog ROA/ROE: verify avg formula cells exist
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import hard_hop as hh  # noqa: E402
from answer_gate import verdict  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

OP_CUES: dict[str, list[str]] = {
    "pred_count": [r"có bao nhiêu", r"số lượng", r"bao nhiêu doanh nghiệp"],
    "count_years": [r"bao nhiêu năm", r"số năm"],
    "argmax_lookup": [r"cao nhất", r"lớn nhất", r"nhiều nhất"],
    "argmin_lookup": [r"thấp nhất", r"nhỏ nhất", r"ít nhất"],
    "year_argmax": [r"năm nào", r"năm .+ cao", r"cao nhất"],
    "year_argmin": [r"năm nào", r"năm .+ thấp", r"thấp nhất"],
    "below_median_mean": [r"dưới trung vị", r"thấp hơn trung vị", r"nhỏ hơn trung vị"],
    "above_median_mean": [r"trên trung vị", r"cao hơn trung vị", r"lớn hơn trung vị"],
    "median_mean_gap": [r"trung vị", r"khoảng cách", r"chênh lệch"],
    "subgroup_share": [r"tỷ trọng", r"chiếm", r"%", r"phần trăm"],
    "median_then_rank_lookup": [r"trung vị", r"thứ"],
    "pair_delta": [r"chênh", r"so với", r"khác biệt"],
    "argext_yoy_delta": [r"giảm", r"tăng", r"YoY|yoy|năm trước"],
}

METRIC_CUES: dict[str, list[str]] = {
    "quick_ratio": [r"thanh toán nhanh", r"quick"],
    "current_ratio": [r"thanh toán hiện hành", r"current"],
    "gross_margin": [r"biên lợi nhuận gộp", r"gộp"],
    "net_margin": [r"biên lợi nhuận ròng", r"biên lợi nhuận thuần", r"\bROS\b"],
    "roa_end": [r"\bROA\b", r"sinh lời trên tổng tài sản"],
    "roe_end": [r"\bROE\b", r"sinh lời trên vốn chủ"],
    "roa_avg": [r"\bROA\b", r"sinh lời trên tổng tài sản"],
    "roe_avg": [r"\bROE\b", r"sinh lời trên vốn chủ"],
    "debt_to_equity": [r"D/E", r"nợ.*vốn chủ", r"nợ phải trả trên vốn"],
    "debt_to_assets": [r"nợ.*tài sản", r"nợ trên tổng"],
    "interest_coverage": [r"khả năng thanh toán lãi", r"lãi vay"],
    "inventory": [r"hàng tồn kho"],
    "inv_to_cl": [r"tồn kho.*nợ ngắn", r"hàng tồn kho trên nợ"],
    "cfo": [r"CFO", r"dòng tiền.*hoạt động"],
    "cfo_margin": [r"CFO margin", r"biên dòng tiền"],
    "cfo_to_ni": [r"CFO.*LNST", r"CFO trên lợi nhuận", r"dòng tiền.*lợi nhuận"],
    "cfo_to_cl": [r"CFO.*nợ ngắn", r"dòng tiền.*nợ ngắn"],
    "sga_intensity": [r"bán hàng và quản lý", r"SG&A", r"chi phí bán hàng"],
    "rev_growth": [r"tăng trưởng.*doanh thu", r"doanh thu.*tăng"],
    "days_inventory": [r"ngày tồn kho", r"vòng quay.*tồn"],
    "net_income": [r"lợi nhuận sau thuế", r"LNST"],
    "net_revenue": [r"doanh thu thuần", r"doanh thu"],
    "inventory_to_assets": [r"tồn kho.*tài sản"],
}


def cue_hit(question: str, pats: list[str]) -> bool:
    return any(re.search(p, question, re.I) for p in pats)


def load_qs() -> dict[int, str]:
    path = ROOT / "data/questions/questions.jsonl"
    return {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_zip(path: Path) -> tuple[dict[int, dict], dict[str, bytes]]:
    with zipfile.ZipFile(path) as z:
        rows = {r["id"]: r for r in json.loads(z.read("submission.json"))}
        blobs = {n: z.read(n) for n in z.namelist() if n.startswith("data/")}
    return rows, blobs


def reexec(rec: dict, blobs: dict[str, bytes]) -> tuple[bool, float | None, str]:
    import pandas as pd

    ns: dict = {"pd": pd}
    try:
        for ev in rec.get("evidence") or []:
            key = ev.get("csv_path") or ev.get("path")
            if not key or key not in blobs:
                return False, None, f"missing:{key}"
            ns[ev.get("variable") or "df"] = pd.read_csv(
                io.BytesIO(blobs[key]), dtype=str, keep_default_na=False)
        exec(rec["pandas_query"], ns, ns)  # noqa: S102
        return True, float(ns["result"]), "ok"
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)[:120]


def reason_row(m: dict, q: str, row: dict, blobs: dict[str, bytes],
               book: hh.CellBook, resolver: TickerResolver) -> dict:
    flags: list[str] = []
    notes: list[str] = []
    layer = m["layer"]
    old, new = float(m["old_answer"]), float(m["new_answer"])
    meta = m.get("meta") or {}

    ok, got, err = reexec(row, blobs)
    if not ok:
        flags.append(f"reexec_fail:{err}")
    elif got is not None and abs(got - new) > 0.02:
        flags.append(f"reexec_mismatch:{got}")

    gate = verdict(q, new)
    if gate:
        flags.append(f"gate:{gate}")

    if abs(new) < 1e-9 and abs(old) > 1:
        flags.append("answer_zero")

    # unit smell: vote3 looks like VND raw, new looks like triệu
    if abs(old) > 1e6 and 0.01 < abs(new) < 1e4 and abs(old / new - 1e6) < 5e4:
        flags.append("likely_million_scale")
    if abs(old) > 1e6 and 0.01 < abs(new) < 1e4 and abs(old / new - 1e3) < 50:
        flags.append("likely_thousand_scale")

    if layer == "dual_tab":
        if abs(new) < 1e-9:
            flags.append("dual_zero_suspect")
        if "likely_million_scale" in flags:
            notes.append("dual may have picked wrong unit column")

    if layer == "maso_plan":
        code = str(meta.get("code") or "")
        # ask if question mentions mã số / chỉ tiêu matching code
        if code and not re.search(rf"\b{re.escape(code)}\b|mã số|chỉ tiêu", q, re.I):
            # soft: many questions describe line item without code
            notes.append(f"maso code={code} score={meta.get('score')}")
        if float(meta.get("score") or 0) < 0.7:
            flags.append("maso_low_score")

    if layer in ("cellbook_dag", "catalog_ratio"):
        op = meta.get("op") or ""
        filt = meta.get("filter") or ""
        target = meta.get("target") or ""
        if op in OP_CUES and not cue_hit(q, OP_CUES[op]):
            flags.append(f"op_cue_miss:{op}")
        for key in (filt, target):
            if key in METRIC_CUES and not cue_hit(q, METRIC_CUES[key]):
                flags.append(f"metric_cue_miss:{key}")

        # catalog ROA/ROE: recompute avg independently
        if target in ("roa_avg", "roe_avg") or filt in (
                "npat_to_ending_assets", "npat_to_ending_equity"):
            tickers = hh.resolve_cohort(q, resolver)
            years = hh.years_of(q)
            if len(tickers) == 1 and len(years) == 1:
                metric = "roa_avg" if "asset" in filt or target == "roa_avg" else "roe_avg"
                if "equity" in filt or target == "roe_avg":
                    metric = "roe_avg"
                got2 = hh.compute(book, tickers[0], years[0], "consolidated", metric)
                if got2 is None:
                    got2 = hh.compute(book, tickers[0], years[0], "separate", metric)
                if got2 is None:
                    flags.append("avg_formula_no_cells")
                else:
                    val, _ = got2
                    if abs(val - new) > 0.05:
                        flags.append(f"avg_mismatch:{val:.4g}")
                    else:
                        notes.append(f"avg_ok:{val:.4g}")

    # question asks count but answer not integer-ish
    if re.search(r"có bao nhiêu|bao nhiêu năm|bao nhiêu doanh nghiệp", q, re.I):
        if abs(new - round(new)) > 0.05 or abs(new) > 50:
            flags.append("count_shape_bad")

    # question asks year but answer looks like ratio
    if re.search(r"năm nào", q, re.I) and not (1990 <= new <= 2030):
        flags.append("year_shape_bad")

    verdict_label = "OK"
    if any(f.startswith("reexec") or f.startswith("gate") for f in flags):
        verdict_label = "FAIL"
    elif flags:
        hard = {"answer_zero", "dual_zero_suspect", "year_shape_bad", "count_shape_bad",
                "likely_million_scale", "avg_formula_no_cells", "avg_mismatch"}
        if any(f.split(":")[0] in hard or f in hard for f in flags):
            verdict_label = "SUSPECT"
        elif any(f.startswith("op_cue_miss") or f.startswith("metric_cue_miss")
                 or f == "maso_low_score" for f in flags):
            verdict_label = "WEAK"
        else:
            verdict_label = "SUSPECT"

    return {
        "id": m["id"],
        "layer": layer,
        "verdict": verdict_label,
        "old": old,
        "new": new,
        "flags": flags,
        "notes": notes,
        "q": re.sub(r"\s+", " ", q)[:160],
        "meta": meta,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qs = load_qs()
    rows, blobs = load_zip(ROOT / "submissions/method_v1.zip")
    man = [
        json.loads(line)
        for line in (ROOT / "artifacts/fresh/method_manifest.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    book = hh.CellBook()
    resolver = TickerResolver()

    reports = [
        reason_row(m, qs[m["id"]], rows[m["id"]], blobs, book, resolver)
        for m in man
    ]

    by = Counter(r["verdict"] for r in reports)
    print(f"audited {len(reports)}  {dict(by)}")
    out = ROOT / "artifacts/fresh/method_reason_audit.jsonl"
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in reports) + "\n",
        encoding="utf-8",
    )
    print(f"-> {out}")

    for label in ("FAIL", "SUSPECT", "WEAK", "OK"):
        bucket = [r for r in reports if r["verdict"] == label]
        if not bucket:
            continue
        print(f"\n=== {label} ({len(bucket)}) ===")
        for r in bucket:
            print(
                f"id={r['id']:4d} [{r['layer']}] {r['old']:.4g} -> {r['new']:.4g}  "
                f"flags={r['flags']}  meta={r['meta']}"
            )
            print(f"      Q: {r['q']}")


if __name__ == "__main__":
    main()
