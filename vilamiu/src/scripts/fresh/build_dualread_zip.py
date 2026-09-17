"""Dual-read splices onto vote3_hardhop — tab quote check + CoT k3/rev agreement.

Without a reverse table pass (`tab_plan_rev.jsonl`), cross-validation uses:
  * tab same cell / same figure — when both tab plans exist
  * tab verified address + CoT k3 and rev agree on the answer
  * CoT k3 + rev agree — cell located via tab plan or table scan

Only single-cell question blocks are touched; every splice passes `answer_gate`
and re-executes its `pandas_query` against bundled CSVs.

Usage:
  python scripts/fresh/build_dualread_zip.py
  python scripts/fresh/build_dualread_zip.py --tier moderate
  python scripts/fresh/build_dualread_zip.py --tier tab_all   # 96 verified tab, riskier
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import parse_statements as ps  # noqa: E402
from answer_gate import verdict  # noqa: E402
from build_submission import MAGNITUDE_CODES, PROGRAM, unit_of  # noqa: E402
from num_helper import SOURCE as NUM_SOURCE  # noqa: E402

SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)

COT_PROGRAM = """{helper}
# Chain-of-thought dual-read located this figure.
_cell = _num(df1.iloc[{row}, {col}])
result = round(_cell * {scale!r} / {unit!r}, 2)
"""


def load_jsonl(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            out[record["id"]] = record
    return out


def load_cot_answers(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("answer") is None:
            continue
        try:
            out[record["id"]] = float(record["answer"])
        except (TypeError, ValueError):
            pass
    return out


def tab_answer(entry: dict, question: str) -> float | None:
    _name, unit = unit_of(question)
    if not unit:
        return None
    raw = ps.parse_vn_number(str(entry.get("quoted", "")))
    if raw is None:
        return None
    scale = float(entry.get("scale", 1.0))
    kind = entry.get("kind", "")
    code = str(entry.get("code", "")).lstrip("t")
    if kind in MAGNITUDE_CODES and code in MAGNITUDE_CODES.get(kind, set()):
        raw = abs(raw)
    return round(raw * scale / unit, 2)


def build_tab_patch(entry: dict, question: str) -> tuple[dict, dict[str, bytes]] | None:
    _name, unit = unit_of(question)
    if not unit:
        return None
    source = ROOT / entry["csv"]
    if not source.exists():
        return None
    name = f"{entry['doc']}_table_{entry['table_id']}.csv"
    blob = source.read_bytes()
    kind = entry.get("kind", "")
    code = str(entry.get("code", "")).lstrip("t")
    wrap = "abs" if code in MAGNITUDE_CODES.get(kind, set()) else ""
    label = entry.get("label") or entry.get("code", "")
    code = PROGRAM.format(
        row=entry["row"],
        col=entry["col"],
        code=label,
        kind=kind,
        wrap=wrap,
        scale=entry["scale"],
        unit=unit,
    )
    record = {
        "answer": tab_answer(entry, question),
        "relevant_docs": [entry["doc"]],
        "relevant_tables": [entry["table_ref"]],
        "evidence": [{"variable": "df", "csv_path": f"data/{name}"}],
        "pandas_query": code,
    }
    return record, {f"data/{name}": blob}


def table_refs(meta_entry: dict | None) -> list[dict]:
    if not meta_entry:
        return []
    refs = list(meta_entry.get("refs") or [])
    if refs:
        return refs
    ticker = meta_entry.get("ticker", "")
    year = str(meta_entry.get("year", ""))
    doc = meta_entry.get("doc", "")
    out = []
    for table in meta_entry.get("tables") or []:
        table_id = table["table_id"]
        doc_name = table.get("doc") or doc
        out.append({
            "ticker": ticker,
            "year": year,
            "doc": doc_name,
            "table_id": table_id,
            "table_ref": f"{doc_name}|table_{table_id}",
        })
    return out


def locate_cot_cell(
    qid: int,
    value: float,
    question: str,
    meta: dict,
    tab_meta: dict,
    tab_entry: dict | None,
) -> tuple[dict, dict[str, bytes]] | None:
    """Prefer the verified tab address; otherwise scan offered tables."""

    _name, unit = unit_of(question)
    if not unit:
        return None

    if tab_entry is not None:
        ans = tab_answer(tab_entry, question)
        if ans is not None and abs(ans - value) <= 0.01:
            return build_tab_patch(tab_entry, question)

    seen: set[str] = set()
    refs: list[dict] = []
    for source in (meta.get(qid), tab_meta.get(qid)):
        for ref in table_refs(source):
            key = f"{ref['doc']}|{ref['table_id']}"
            if key not in seen:
                seen.add(key)
                refs.append(ref)

    for ref in refs:
        path = (
            ROOT / "data" / "official_corpus" / ref["ticker"] / ref["year"]
            / ref["doc"] / f"{ref['doc']}_extracted_tables"
            / f"table_{ref['table_id']}.csv"
        )
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                grid = list(csv_mod.reader(handle))
        except OSError:
            continue
        for r_index, grid_row in enumerate(grid[1:]):
            for c_index, cell in enumerate(grid_row):
                raw = str(cell).strip()
                if not raw:
                    continue
                parsed = ps.parse_vn_number(raw)
                if parsed is None:
                    continue
                for scale in SCALES:
                    if abs(abs(parsed) * scale / unit - abs(value)) <= 0.01:
                        name = f"data/{ref['doc']}_table_{ref['table_id']}.csv"
                        blob = path.read_bytes()
                        record = {
                            "answer": round(value, 2),
                            "relevant_docs": [ref["doc"]],
                            "relevant_tables": [f"{ref['doc']}|table_{ref['table_id']}"],
                            "evidence": [{"variable": "df1", "csv_path": name}],
                            "pandas_query": COT_PROGRAM.format(
                                helper=NUM_SOURCE.rstrip(),
                                row=r_index,
                                col=c_index,
                                scale=scale,
                                unit=unit,
                            ),
                        }
                        return record, {name: blob}
    return None


def reexec_ok(record: dict, blobs: dict[str, bytes]) -> bool:
    namespace: dict = {"pd": pd}
    try:
        for item in record["evidence"]:
            blob = blobs.get(item["csv_path"])
            if blob is None:
                return False
            namespace[item["variable"]] = pd.read_csv(
                io.BytesIO(blob), dtype=str, keep_default_na=False,
            )
        exec(record["pandas_query"], namespace, namespace)  # noqa: S102
        return abs(float(record["answer"]) - float(namespace["result"])) <= 0.01
    except Exception:
        return False


def tab_tiers(first: dict[int, dict], second: dict[int, dict]) -> dict[str, set[int]]:
    tiers: dict[str, set[int]] = {
        "cung o": set(),
        "cung con so": set(),
        "mot lan doc": set(),
    }
    for qid, entry in first.items():
        other = second.get(qid)
        if other is None:
            tiers["mot lan doc"].add(qid)
        elif (entry["table_id"], entry["row"], entry["col"]) == (
                other["table_id"], other["row"], other["col"]):
            tiers["cung o"].add(qid)
        else:
            left = ps.parse_vn_number(str(entry.get("quoted", "")))
            right = ps.parse_vn_number(str(other.get("quoted", "")))
            if (left is not None and right is not None
                    and abs(left - right) <= 0.01):
                tiers["cung con so"].add(qid)
    return tiers


def pick_candidates(
    tier: str,
    single: set[int],
    tab: dict[int, dict],
    tab_rev: dict[int, dict],
    k3: dict[int, float],
    rev: dict[int, float],
    qs: dict[int, str],
    base: dict[int, float],
) -> dict[int, str]:
    """Return qid -> reason label for splices."""

    chosen: dict[int, str] = {}

    def eligible(qid: int, answer: float) -> bool:
        if qid not in single:
            return False
        reason = verdict(qs[qid], answer)
        if reason:
            return False
        if abs(answer - base.get(qid, 0.0)) <= 0.01:
            return False
        return True

    if tab_rev:
        tiers = tab_tiers(tab, tab_rev)
        for qid in tiers["cung o"] | tiers["cung con so"]:
            entry = tab.get(qid)
            if entry is None:
                continue
            ans = tab_answer(entry, qs[qid])
            if ans is None or not eligible(qid, ans):
                continue
            label = "tab_cung_o" if qid in tiers["cung o"] else "tab_cung_so"
            chosen[qid] = label

    cot_dual: set[int] = set()
    for qid in set(k3) & set(rev):
        if abs(k3[qid] - rev[qid]) <= 0.01:
            cot_dual.add(qid)

    for qid, entry in tab.items():
        if qid not in single:
            continue
        ans = tab_answer(entry, qs[qid])
        if ans is None or not eligible(qid, ans):
            continue
        has_k3 = qid in k3 and abs(k3[qid] - ans) <= 0.01
        has_rev = qid in rev and abs(rev[qid] - ans) <= 0.01
        dual = qid in cot_dual
        if tier == "tab_all":
            chosen.setdefault(qid, "tab_verified")
        elif tier == "moderate":
            if dual and has_k3:
                chosen.setdefault(qid, "tab_cot_dual")
            elif has_k3 or has_rev:
                chosen.setdefault(qid, "tab_one_cot")
        elif tier == "strict":
            if dual and has_k3:
                chosen.setdefault(qid, "tab_cot_dual")

    for qid in cot_dual:
        ans = k3[qid]
        if not eligible(qid, ans):
            continue
        if tier in ("strict", "moderate"):
            chosen.setdefault(qid, "cot_dual")

    return chosen


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/vote3_hardhop.zip")
    parser.add_argument("--dest", default="submissions/vote3_dual.zip")
    parser.add_argument("--tier", default="moderate",
                        choices=("strict", "moderate", "tab_all"))
    parser.add_argument("--cot-k3", default="artifacts/fresh/cot_k3.jsonl")
    parser.add_argument("--cot-rev", default="artifacts/fresh/cot_rev_results.jsonl")
    parser.add_argument("--tab", default="artifacts/fresh/tab_plan.jsonl")
    parser.add_argument("--tab-rev", default="artifacts/fresh/tab_plan_rev.jsonl")
    parser.add_argument("--blocks", default="artifacts/fresh/blocks.json")
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_progall.jsonl")
    args = parser.parse_args()

    blocks = json.loads((ROOT / args.blocks).read_text(encoding="utf-8"))
    single = set(blocks.get("tien — MOT O", [])) | set(
        blocks.get("khac — don gian", []))

    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    meta = {
        record["id"]: record["meta"]
        for record in load_jsonl(ROOT / args.prompts).values()
        if record.get("meta")
    }
    tab_meta = {
        record["id"]: record["meta"]
        for record in load_jsonl(ROOT / "artifacts/fresh/prompts_tab.jsonl").values()
        if record.get("meta")
    }
    tab = load_jsonl(ROOT / args.tab)
    tab_rev = load_jsonl(ROOT / args.tab_rev)
    k3 = load_cot_answers(ROOT / args.cot_k3)
    rev = load_cot_answers(ROOT / args.cot_rev)

    with zipfile.ZipFile(ROOT / args.base) as src:
        rows = {r["id"]: r for r in json.loads(src.read("submission.json"))}
        files = {n: src.read(n) for n in src.namelist() if n != "submission.json"}
        base_ans = {qid: float(row.get("answer") or 0) for qid, row in rows.items()}

    candidates = pick_candidates(
        args.tier, single, tab, tab_rev, k3, rev, qs, base_ans)

    stats: dict[str, int] = {}
    spliced = dropped = reexec_fail = 0
    for qid in sorted(candidates):
        reason = candidates[qid]
        question = qs[qid]
        entry = tab.get(qid)
        if entry is not None and reason.startswith("tab"):
            value = tab_answer(entry, question)
            built = build_tab_patch(entry, question)
        else:
            value = k3[qid]
            built = locate_cot_cell(qid, value, question, meta, tab_meta, entry)

        if built is None or value is None:
            dropped += 1
            print(qid, reason, "NO_PATCH")
            continue
        patch, new_files = built
        if not reexec_ok(patch, new_files):
            reexec_fail += 1
            dropped += 1
            print(qid, reason, "reexec FAIL")
            continue

        old = float(rows[qid].get("answer") or 0)
        rows[qid].update(patch)
        rows[qid]["question"] = question
        files.update(new_files)
        spliced += 1
        stats[reason] = stats.get(reason, 0) + 1
        print(f"{qid} {reason} {patch['answer']} (was {old})")

    dest = ROOT / args.dest
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)

    print(f"\ntier={args.tier}  candidates={len(candidates)}  "
          f"spliced={spliced}  dropped={dropped}  reexec_fail={reexec_fail}")
    for name, count in sorted(stats.items()):
        print(f"  {count:4d}  {name}")
    if not tab_rev:
        print("  (tab_plan_rev.jsonl missing — tab dual-cell tier skipped)")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
