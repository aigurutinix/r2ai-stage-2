"""Rerank shortlist CDKT candidates that share the top lexical code."""

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
        if not cands:
            continue
        question = qs[qid]
        n_co = len(hh.resolve_cohort(question, resolver))
        if mp.shape_of(block_of(question, companies=max(1, n_co))) != "single":
            continue
        # Anchor code = best lexical CDKT candidate that passes scope.
        cands_scoped = [
            c for c in cands
            if tn.scope_matches(question, str(c.get("doc") or ""))
        ]
        if not cands_scoped:
            continue
        cands_scoped.sort(key=lambda c: -float(c.get("score") or 0))
        anchor = cands_scoped[0]
        code = str(anchor["code"]).lstrip("t")
        same = [c for c in cands_scoped if str(c["code"]).lstrip("t") == code]
        if len(same) < 2:
            # Also allow sibling tables of the anchor doc with same code.
            raw = anchor.get("csv") or ""
            apath = Path(raw) if Path(raw).is_file() else ROOT / raw
            if not apath.is_file():
                apath = tn.resolve_csv(raw, ROOT)
            if apath is None:
                continue
            pool_paths = [apath, *sorted(apath.parent.glob("table_*.csv"))]
            scored = []
            for path in pool_paths:
                hit = score_code_on_path(path, code)
                if hit is None:
                    continue
                scored.append((hit["score"], hit["n_codes"], path, hit, anchor))
            if len(scored) < 2:
                continue
            scored.sort(key=lambda t: (-t[0], -t[1]))
            best = scored[0]
            first = next(s for s in scored if s[2].resolve() == apath.resolve())
            if best[2].resolve() == apath.resolve():
                continue
            if best[0] < 1.0 or first[0] >= best[0]:
                continue
            best_path, best_hit = best[2], best[3]
            first_own = first[0]
            best_c = {**anchor, "table_id": best_hit["table_id"],
                      "doc": best_path.parent.parent.name}
        else:
            scored = []
            for c in same:
                raw = c.get("csv") or ""
                path = Path(raw) if Path(raw).is_file() else ROOT / raw
                if not path.is_file():
                    path = tn.resolve_csv(raw, ROOT)
                if path is None:
                    continue
                hit = score_code_on_path(path, code)
                if hit is None:
                    continue
                scored.append((hit["score"], hit["n_codes"], float(c.get("score") or 0),
                               c, path, hit))
            if len(scored) < 2:
                continue
            scored.sort(key=lambda t: (-t[0], -t[1], -t[2]))
            best = scored[0]
            # lexical first among same code
            lex_first = max(scored, key=lambda t: t[2])
            if best[4].resolve() == lex_first[4].resolve():
                continue
            if best[0] < 1.0 or lex_first[0] >= best[0]:
                continue
            best_c, best_path, best_hit = best[3], best[4], best[5]
            first_own = lex_first[0]

        ran = exec_binding(
            question, best_path, code, "cdkt",
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
        csv_name = f"{best_c['doc']}_table_{best_hit['table_id']}.csv"
        record = {
            "answer": answer,
            "pandas_query": program,
            "evidence": [{"variable": "df", "csv_path": f"data/{csv_name}"}],
        }
        if not reexec_ok(record, {f"data/{csv_name}": blob}):
            continue
        upgrades.append({
            "id": qid,
            "code": code,
            "first_own": first_own,
            "best_score": best_hit["score"] if "score" in best_hit else best[0],
            "old": old,
            "new": answer,
            "doc": best_c["doc"],
            "table": best_path.name,
            "lex": float(anchor.get("score") or 0),
            "csv_name": csv_name,
            "blob_path": str(best_path),
            "row": int(best_hit["row"]),
            "col": int(best_hit["col"]),
            "scale": float(best_hit["scale"]),
            "program": program,
        })

    print(f"same-code shortlist/sibling upgrades: {len(upgrades)}")
    for u in upgrades:
        print({k: v for k, v in u.items() if k not in ("program", "blob_path")})
    out = ROOT / "artifacts/fresh/identity_shortlist_flips.jsonl"
    out.write_text(
        "\n".join(
            json.dumps({k: v for k, v in u.items() if k != "program"},
                       ensure_ascii=False)
            for u in upgrades
        ) + "\n",
        encoding="utf-8",
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
