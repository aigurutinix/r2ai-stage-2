"""Rerank shortlist.jsonl CDKT candidates by identity score."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from answer_gate import verdict  # noqa: E402
from build_notcell_zip import reexec_ok  # noqa: E402
from identity_topk_rerank import exec_binding, score_code_on_path  # noqa: E402
from identity_topk_scoped import doc_scope  # noqa: E402
import table_norm as tn  # noqa: E402
import method_pipeline as mp  # noqa: E402
from blocks import block_of  # noqa: E402
import hard_hop as hh  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    shortlist = {}
    for line in (ROOT / "artifacts/fresh/shortlist.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            shortlist[int(rec["id"])] = rec
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    with zipfile.ZipFile(ROOT / "submissions/method_v1.zip") as z:
        base = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    resolver = TickerResolver()

    upgrades = []
    for qid, rec in sorted(shortlist.items()):
        cands = [c for c in rec.get("candidates") or []
                 if str(c.get("kind") or "").lower() == "cdkt"
                 and str(c.get("code") or "").lstrip("t").isdigit()]
        if len(cands) < 2:
            continue
        question = qs[qid]
        n_co = len(hh.resolve_cohort(question, resolver))
        if mp.shape_of(block_of(question, companies=max(1, n_co))) != "single":
            continue
        scored = []
        for c in cands:
            raw = c.get("csv") or ""
            path = Path(raw) if Path(raw).is_file() else ROOT / raw
            if not path.is_file():
                path = tn.resolve_csv(raw, ROOT)
            if path is None:
                continue
            if not tn.scope_matches(question, str(c.get("doc") or "")):
                continue
            hit = score_code_on_path(path, str(c["code"]).lstrip("t"))
            if hit is None:
                continue
            scored.append((hit["score"], hit["n_codes"], c, path, hit))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda t: (-t[0], -t[1], -float(t[2].get("score") or 0)))
        best_score, best_n, best_c, best_path, best_hit = scored[0]
        # Compare to first shortlist CDKT (greedy order)
        first = cands[0]
        first_path = Path(first["csv"]) if Path(first["csv"]).is_file() else ROOT / first["csv"]
        if not first_path.is_file():
            first_path = tn.resolve_csv(first["csv"], ROOT) or first_path
        if best_path.resolve() == first_path.resolve():
            continue
        if best_score < 1.0:
            continue
        # first identity
        first_hit = score_code_on_path(first_path, str(first["code"]).lstrip("t"))
        first_own = float(first_hit["score"]) if first_hit else -1.0
        if first_own >= best_score:
            continue
        ran = exec_binding(
            question, best_path, str(best_c["code"]).lstrip("t"), "cdkt",
            int(best_hit["row"]), int(best_hit["col"]), float(best_hit["scale"]),
            period=str(rec.get("period") or "current"),
        )
        if ran is None:
            continue
        answer, program, blob = ran
        if verdict(question, answer):
            continue
        old = float(base[qid].get("answer") or 0)
        if abs(answer - old) <= 0.01:
            continue
        if not mp.magnitude_ok("maso_plan", answer, old):
            continue
        csv_name = f"{best_c['doc']}_table_{best_c['table_id']}.csv"
        record = {
            "answer": answer,
            "pandas_query": program,
            "evidence": [{"variable": "df", "csv_path": f"data/{csv_name}"}],
        }
        if not reexec_ok(record, {f"data/{csv_name}": blob}):
            continue
        upgrades.append({
            "id": qid,
            "from": first.get("doc"),
            "to": best_c.get("doc"),
            "code": best_c.get("code"),
            "first_own": first_own,
            "best_score": best_score,
            "old": old,
            "new": answer,
            "lex_first": first.get("score"),
            "lex_best": best_c.get("score"),
        })

    print(f"shortlist identity upgrades: {len(upgrades)}")
    for u in upgrades:
        print(u)


if __name__ == "__main__":
    main()
