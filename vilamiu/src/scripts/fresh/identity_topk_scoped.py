"""Identity top-k flips with scope/ticker/year gates (presentable)."""

from __future__ import annotations

import json
import re
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


def doc_scope(doc: str) -> str:
    d = doc.casefold()
    if "separate" in d or "riêng" in d:
        return "separate"
    if "consolidated" in d or "hợp nhất" in d:
        return "consolidated"
    return "unknown"


def doc_ticker_year(doc: str) -> tuple[str | None, str | None]:
    # FIT_financial_statements_2015_separate
    m = re.match(r"^([A-Z0-9]+)_financial_statements_(\d{4})_", doc)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def scope_allowed(question: str, greedy_doc: str, cand_doc: str) -> bool:
    """Keep scope unless the question explicitly demands the other."""

    gs, cs = doc_scope(greedy_doc), doc_scope(cand_doc)
    if gs == cs:
        return True
    if tn.wants_separate(question) and cs == "separate":
        return True
    if tn.wants_consolidated(question) and cs == "consolidated":
        return True
    # No explicit scope demand → do not cross cons↔sep.
    return False


def slim_pool(qid: int, gpath: Path, greedy_doc: str, question: str,
              topk: dict[int, list[dict]]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        key = path.resolve()
        if path.is_file() and key not in seen:
            seen.add(key)
            out.append(path)

    for path in [gpath, *sorted(gpath.parent.glob("table_*.csv"))]:
        add(path)
    gt, gy = doc_ticker_year(greedy_doc)
    for ref in topk.get(qid) or []:
        path = resolve_ref(ref)
        if path is None:
            continue
        doc = path.parent.parent.name
        if not scope_allowed(question, greedy_doc, doc):
            continue
        ct, cy = doc_ticker_year(doc)
        if gt and ct and ct != gt:
            continue
        if gy and cy and cy != gy:
            continue
        add(path)
        # same-doc siblings of top-k hit (page-split BS)
        if doc_scope(doc) == doc_scope(greedy_doc) or tn.wants_separate(question) \
                or tn.wants_consolidated(question):
            for sib in sorted(path.parent.glob("table_*.csv")):
                add(sib)
    return out


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

    safe: list[dict] = []
    for qid, entry in sorted(greedy.items()):
        if str(entry.get("kind") or "").lower() != "cdkt":
            continue
        code = str(entry.get("code") or "").lstrip("t")
        if not code.isdigit():
            continue
        question = qs[qid]
        n_co = len(hh.resolve_cohort(question, resolver))
        block = block_of(question, companies=max(1, n_co))
        shape = mp.shape_of(block)
        # Only single-cell money questions — not screens.
        if shape != "single":
            continue
        raw = entry.get("csv") or ""
        gpath = Path(raw) if Path(raw).is_file() else ROOT / raw
        if not gpath.is_file():
            resolved = tn.resolve_csv(raw, ROOT)
            if resolved is None:
                continue
            gpath = resolved
        ghit = score_code_on_path(gpath, code)
        g_own = float(ghit["score"]) if ghit else -1.0
        if g_own >= 1.0:
            continue
        greedy_doc = str(entry.get("doc") or gpath.parent.parent.name)
        pool = slim_pool(qid, gpath, greedy_doc, question, topk)
        winner = rerank_candidates(
            code, pool, gpath,
            require_unique_perfect=False,
            min_score=1.0,
        )
        if winner is None or not winner["flipped"]:
            continue
        pick_doc = winner["path"].parent.parent.name
        if not scope_allowed(question, greedy_doc, pick_doc):
            continue
        gt, gy = doc_ticker_year(greedy_doc)
        pt, py = doc_ticker_year(pick_doc)
        if gt and pt and gt != pt:
            continue
        if gy and py and gy != py:
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
        if abs(old) > 1e-6 and (
                abs(answer) / abs(old) > 50 or abs(old) / abs(answer) > 50):
            continue
        csv_name = (
            f"{pick_doc}_table_{winner['table_id']}.csv")
        record = {
            "answer": answer,
            "pandas_query": program,
            "evidence": [{"variable": "df", "csv_path": f"data/{csv_name}"}],
        }
        files = {f"data/{csv_name}": blob}
        if not reexec_ok(record, files):
            continue
        # Also pass method magnitude vs incumbent
        if not mp.magnitude_ok("maso_plan", answer, old):
            continue
        safe.append({
            "id": qid,
            "code": code,
            "g_own": g_own,
            "lex_score": float(entry.get("score") or 0),
            "w_score": winner["score"],
            "n_codes": winner["n_codes"],
            "n_cand": winner["n_cand"],
            "old": old,
            "new": answer,
            "greedy_doc": greedy_doc,
            "pick_doc": pick_doc,
            "pick": winner["path"].name,
            "csv_name": csv_name,
            "blob_path": str(winner["path"]),
            "program": program,
            "row": int(winner["row"]),
            "col": int(winner["col"]),
            "scale": float(winner["scale"]),
        })

    print(f"scoped safe flips: {len(safe)}")
    for item in safe:
        print({k: v for k, v in item.items()
               if k not in ("program", "blob_path")})
    out = ROOT / "artifacts/fresh/identity_topk_scoped.jsonl"
    # store without program blob for audit; rebuild at apply time
    out.write_text(
        "\n".join(
            json.dumps({k: v for k, v in s.items() if k != "program"},
                       ensure_ascii=False)
            for s in safe
        ) + "\n",
        encoding="utf-8",
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
