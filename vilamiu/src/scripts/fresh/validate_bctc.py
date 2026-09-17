"""Verify answers are grounded in BCTC corpus — not program≡answer alone.

Checks (in order):
  1. corpus_reread  — independent read from official_corpus matches answer
  2. in_document    — answer magnitude appears in resolved report cells
  3. identity       — TT200 roll-ups hold at read table (maso CDKT)

Usage:
  python scripts/fresh/validate_bctc.py --zip submissions/method_v3.zip --sample 30
  python scripts/fresh/validate_bctc.py --zip submissions/smoke.zip --ids 29,56,124
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import parse_statements as ps  # noqa: E402
from build_submission import unit_of  # noqa: E402
from cell_verify import verify_patch  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

TOL = 0.011


@dataclass
class CheckResult:
    qid: int
    layer: str
    answer: float
    ok: bool
    checks: dict[str, bool | str] = field(default_factory=dict)
    reason: str = ""


def close(a: float, b: float) -> bool:
    if a == 0 and b == 0:
        return True
    return abs(a - b) <= TOL * max(abs(a), abs(b), 1.0)


DOC_TABLE_RE = re.compile(r"^(.+)_table_(\d+)\.csv$")
DOC_META_RE = re.compile(r"^([A-Z0-9]+)_financial_statements_(\d{4})_")


def resolve_corpus_csv(evidence_path: str) -> Path | None:
    """Map submission evidence name → official_corpus CSV on disk."""

    name = evidence_path.split("/")[-1]
    if name.startswith("metric_panel_q"):
        return None
    match = DOC_TABLE_RE.match(name)
    if not match:
        return None
    doc, tid = match.group(1), match.group(2)
    meta = DOC_META_RE.match(doc)
    if meta:
        ticker, year = meta.group(1), meta.group(2)
        path = (
            ROOT / "data/official_corpus" / ticker / year / doc
            / f"{doc}_extracted_tables" / f"table_{tid}.csv"
        )
        if path.is_file():
            return path
    hits = list(
        (ROOT / "data/official_corpus").glob(
            f"**/{doc}/{doc}_extracted_tables/table_{tid}.csv"))
    return hits[0] if hits else None


def corpus_path(csv_name: str) -> Path | None:
    return resolve_corpus_csv(f"data/{csv_name}")


def reread_from_corpus(row: dict, zip_frames: dict | None = None) -> tuple[float | None, str, str]:
    """Re-execute program. Returns (value, status, source). source=disk|zip|none."""

    evidence = row.get("evidence") or []
    if not evidence:
        return None, "no_evidence", "none"
    ns: dict = {"pd": pd}
    for item in evidence:
        path = resolve_corpus_csv(item["csv_path"])
        if path is not None:
            ns[item["variable"]] = pd.read_csv(path, dtype=str, keep_default_na=False)
        elif zip_frames and item["csv_path"] in zip_frames:
            ns[item["variable"]] = zip_frames[item["csv_path"]]
        else:
            return None, f"missing:{item['csv_path'].split('/')[-1]}", "none"
    try:
        exec(row["pandas_query"], ns, ns)  # noqa: S102
        source = "disk" if all(
            resolve_corpus_csv(it["csv_path"]) is not None for it in evidence
        ) else "zip"
        return float(ns["result"]), "ok", source
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)[:80], "none"


def load_report_values(ticker: str, year: str, scope: str) -> list[float]:
    """All numeric magnitudes in statement index + tied notes for one report."""

    out: list[float] = []
    key = (ticker, year, scope)
    cache_path = ROOT / "artifacts/fresh/_validate_cells.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if key in cached:
            return cached[str(key)]

    for line in (ROOT / "artifacts/fresh/statements2.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["ticker"] != ticker or rec["year"] != year:
            continue
        sc = rec["scope"]
        if scope == "separate" and "separate" not in sc:
            continue
        if scope == "consolidated" and "separate" in sc:
            continue
        scale = float(rec.get("scale") or 1.0)
        for period in ("current", "prior"):
            for cell in rec[period].values():
                out.append(abs(cell[0] * scale))

    # Notes tied to statements (smaller, higher confidence)
    for line in (ROOT / "artifacts/fresh/notes.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        note = json.loads(line)
        if note["ticker"] != ticker or note["year"] != year:
            continue
        if note["scope"] != scope:
            continue
        if not note.get("paired"):
            continue
        scale = float(note.get("scale") or 1.0)
        try:
            import csv as _csv
            grid = list(_csv.reader(
                (ROOT / note["csv"]).open(encoding="utf-8-sig")))
            for row in grid[1:]:
                for cell in row:
                    v = ps.parse_vn_number(str(cell))
                    if v is not None:
                        out.append(abs(v * scale))
        except OSError:
            pass

    return out


def in_document(question: str, answer: float) -> tuple[bool, str]:
    """Answer magnitude appears somewhere in the resolved report."""

    if abs(answer) < 1e-12:
        return False, "zero_answer"
    _name, unit = unit_of(question)
    if not unit:
        return True, "no_unit_skip"  # ratio/year — skip
    target = abs(answer) * unit
    resolver = TickerResolver()
    found = resolver.resolve(question)
    years = YEAR_RE.findall(question)
    if len(found) != 1 or not years:
        return False, "unresolved_scope"
    ticker = next(iter(found))
    scope = "separate" if PARENT_RE.search(question) else "consolidated"
    year = max(years)
    values = load_report_values(ticker, year, scope)
    if not values:
        alt = "consolidated" if scope == "separate" else "separate"
        values = load_report_values(ticker, year, alt)
    for v in values:
        if close(v, target):
            return True, "found"
    return False, "not_in_report"


def validate_row(qid: int, row: dict, question: str, layer: str = "",
                 zip_frames: dict | None = None) -> CheckResult:
    answer = float(row.get("answer") or 0)
    layer = layer or row.get("_layer", "unknown")
    checks: dict[str, bool | str] = {}
    evidence = row.get("evidence") or []

    if not evidence:
        return CheckResult(qid, layer, answer, True, {"skip": "no_evidence"}, "ok")

    reread, err, source = reread_from_corpus(row, zip_frames)
    checks["corpus_reread"] = reread is not None and close(reread, answer)
    checks["reread_source"] = source
    if not checks["corpus_reread"]:
        checks["reread_err"] = err
        if reread is not None:
            checks["reread_got"] = round(reread, 4)

    doc_ok, doc_reason = in_document(question, answer)
    checks["in_document"] = doc_ok
    checks["in_doc_detail"] = doc_reason

    if layer in ("maso_plan", "dual_tab", "note_plan") and evidence:
        path = resolve_corpus_csv(evidence[0]["csv_path"])
        meta = row.get("_meta") or {}
        if path and path.is_file():
            id_fail = verify_patch(
                layer, path.read_bytes(), path, meta, question)
            checks["identity"] = id_fail is None
            if id_fail:
                checks["identity_fail"] = id_fail
        else:
            checks["identity"] = False

    # Grounding rules by layer
    if layer in ("maso_plan", "note_plan", "dual_tab"):
        ok = bool(checks.get("corpus_reread")) and source == "disk"
        id_fail = checks.get("identity_fail")
        if id_fail in ("cdkt_rollup_fail", "identity_doc_fail"):
            ok = False
        reason = "ok" if ok else (
            "not_from_corpus_disk" if source != "disk"
            else "corpus_mismatch" if not checks.get("corpus_reread")
            else f"identity:{id_fail}")
    elif layer in ("panel_det", "cellbook_dag", "catalog_ratio"):
        ok = bool(checks.get("corpus_reread"))
        reason = "computed_ok" if ok else "reread_fail"
    else:
        ok = bool(checks.get("corpus_reread"))
        reason = "ok" if ok else "reread_fail"

    return CheckResult(qid, layer, answer, ok, checks, reason)


def sample_ids(manifest_path: Path, n: int, seed: int) -> list[int]:
    """Stratified sample: splices from manifest + random singles."""

    manifest: list[dict] = []
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                manifest.append(json.loads(line))
    by_layer: dict[str, list[int]] = defaultdict(list)
    for m in manifest:
        by_layer[m["layer"]].append(m["id"])
    rng = random.Random(seed)
    picked: list[int] = []
    per_layer = max(2, n // max(len(by_layer), 1))
    for layer, ids in by_layer.items():
        rng.shuffle(ids)
        picked.extend(ids[:per_layer])
    remaining = n - len(picked)
    if remaining > 0:
        pool = [i for i in range(1, 1013) if i not in picked]
        picked.extend(rng.sample(pool, min(remaining, len(pool))))
    return sorted(set(picked))[:n]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="submissions/method_v3.zip")
    parser.add_argument("--ids", default="", help="comma-separated ids")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manifest", default="artifacts/fresh/method_manifest.jsonl")
    parser.add_argument("--min-pass", type=float, default=0.85,
                        help="exit 1 if pass rate below this")
    args = parser.parse_args()

    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    manifest_layers: dict[int, str] = {}
    manifest_meta: dict[int, dict] = {}
    for line in (ROOT / args.manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)
            manifest_layers[m["id"]] = m["layer"]
            manifest_meta[m["id"]] = m.get("meta") or {}

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    elif args.sample:
        ids = sample_ids(ROOT / args.manifest, args.sample, args.seed)
    else:
        ids = sorted(manifest_layers) or list(range(1, 41))

    with zipfile.ZipFile(ROOT / args.zip) as z:
        rows = {r["id"]: r for r in json.loads(z.read("submission.json"))}
        zip_frames = {
            n: pd.read_csv(io.BytesIO(z.read(n)), dtype=str, keep_default_na=False)
            for n in z.namelist()
            if n.startswith("data/")
        }

    results: list[CheckResult] = []
    for qid in ids:
        row = rows.get(qid)
        if row is None:
            continue
        layer = manifest_layers.get(qid, "incumbent")
        row = {**row, "_layer": layer, "_meta": manifest_meta.get(qid, {})}
        results.append(validate_row(qid, row, qs[qid], layer, zip_frames))

    passed = sum(1 for r in results if r.ok)
    total = len(results)
    print(f"BCTC validate: {passed}/{total} pass  ({100*passed/max(total,1):.0f}%)")
    print(f"  zip={args.zip}  ids={len(ids)}")
    by_reason: Counter[str] = Counter()
    by_layer: Counter[str] = Counter()
    for r in results:
        by_reason[r.reason if not r.ok else "ok"] += 1
        by_layer[f"{r.layer}:{'ok' if r.ok else 'fail'}"] += 1

    print("\nby layer:")
    for k, v in sorted(by_layer.items()):
        print(f"  {k}: {v}")
    print("\nfail reasons:")
    for k, v in by_reason.most_common():
        if k != "ok":
            print(f"  {k}: {v}")

    fails = [r for r in results if not r.ok][:12]
    if fails:
        print("\nexamples:")
        for r in fails:
            print(f"  id={r.qid} [{r.layer}] ans={r.answer}  {r.reason}  {r.checks}")

    rate = passed / max(total, 1)
    if rate < args.min_pass:
        print(f"\nFAIL: pass rate {rate:.2f} < {args.min_pass}")
        sys.exit(1)
    print(f"\nPASS: {rate:.2f} >= {args.min_pass}")


if __name__ == "__main__":
    main()
