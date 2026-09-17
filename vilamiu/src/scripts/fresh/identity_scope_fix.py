"""Count identity top-k upgrades when question scope mismatches greedy doc."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from answer_gate import verdict  # noqa: E402
from build_notcell_zip import reexec_ok  # noqa: E402
from identity_topk_rerank import (  # noqa: E402
    exec_binding,
    load_topk,
    rerank_candidates,
    resolve_ref,
    score_code_on_path,
)
import table_norm as tn  # noqa: E402
from blocks import block_of  # noqa: E402
import hard_hop as hh  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402
import method_pipeline as mp  # noqa: E402
from identity_topk_scoped import doc_scope, doc_ticker_year  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    greedy = mp.load_jsonl(ROOT / "artifacts/fresh/greedy_plan.jsonl")
    topk = load_topk(ROOT / "artifacts/fresh/dense_topk.json", k=9)
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    with zipfile.ZipFile(ROOT / "submissions/method_v1.zip") as z:
        base = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    resolver = TickerResolver()

    hits = []
    for qid, entry in sorted(greedy.items()):
        if str(entry.get("kind") or "").lower() != "cdkt":
            continue
        code = str(entry.get("code") or "").lstrip("t")
        if not code.isdigit():
            continue
        question = qs[qid]
        want_sep = tn.wants_separate(question)
        want_cons = tn.wants_consolidated(question)
        if not want_sep and not want_cons:
            continue
        n_co = len(hh.resolve_cohort(question, resolver))
        if mp.shape_of(block_of(question, companies=max(1, n_co))) != "single":
            continue
        raw = entry.get("csv") or ""
        gpath = Path(raw) if Path(raw).is_file() else ROOT / raw
        if not gpath.is_file():
            resolved = tn.resolve_csv(raw, ROOT)
            if resolved is None:
                continue
            gpath = resolved
        greedy_doc = str(entry.get("doc") or gpath.parent.parent.name)
        gs = doc_scope(greedy_doc)
        # Need mismatch: question wants X but greedy is Y
        if want_sep and gs == "separate":
            continue
        if want_cons and gs == "consolidated":
            continue
        if want_sep and gs != "consolidated":
            continue
        if want_cons and gs != "separate":
            continue

        # Build pool: top-k + siblings matching desired scope
        target = "separate" if want_sep else "consolidated"
        gt, gy = doc_ticker_year(greedy_doc)
        pool: list[Path] = []
        seen: set[Path] = set()
        for ref in topk.get(qid) or []:
            path = resolve_ref(ref)
            if path is None:
                continue
            doc = path.parent.parent.name
            if doc_scope(doc) != target:
                continue
            ct, cy = doc_ticker_year(doc)
            if gt and ct and ct != gt:
                continue
            if gy and cy and cy != gy:
                continue
            for p in [path, *sorted(path.parent.glob("table_*.csv"))]:
                key = p.resolve()
                if p.is_file() and key not in seen:
                    seen.add(key)
                    pool.append(p)
        if not pool:
            continue
        winner = rerank_candidates(
            code, pool, gpath,
            require_unique_perfect=False,
            min_score=1.0,
        )
        if winner is None:
            continue
        ran = exec_binding(
            question, winner["path"], code, "cdkt",
            int(winner["row"]), int(winner["col"]), float(winner["scale"]),
            period=str(entry.get("period") or "current"),
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
        pick_doc = winner["path"].parent.parent.name
        csv_name = f"{pick_doc}_table_{winner['table_id']}.csv"
        record = {
            "answer": answer,
            "pandas_query": program,
            "evidence": [{"variable": "df", "csv_path": f"data/{csv_name}"}],
        }
        if not reexec_ok(record, {f"data/{csv_name}": blob}):
            continue
        hits.append({
            "id": qid,
            "code": code,
            "want": target,
            "greedy_doc": greedy_doc,
            "pick_doc": pick_doc,
            "old": old,
            "new": answer,
            "lex": float(entry.get("score") or 0),
            "w_score": winner["score"],
            "n_cand": winner["n_cand"],
        })

    print(f"scope-mismatch identity flips: {len(hits)}")
    for h in hits:
        print(h)
    out = ROOT / "artifacts/fresh/identity_scope_fix.jsonl"
    out.write_text(
        "\n".join(json.dumps(h, ensure_ascii=False) for h in hits) + "\n",
        encoding="utf-8",
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
