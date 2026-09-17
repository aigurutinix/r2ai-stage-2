"""Conservative identity top-k flips: only when greedy identity is broken."""

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


def slim_pool(qid: int, gpath: Path, topk: dict[int, list[dict]]) -> list[Path]:
    """Greedy doc siblings + exact top-k tables (no other-doc sibling flood)."""

    out: list[Path] = []
    seen: set[Path] = set()
    for path in [gpath, *sorted(gpath.parent.glob("table_*.csv"))]:
        key = path.resolve()
        if path.is_file() and key not in seen:
            seen.add(key)
            out.append(path)
    for ref in topk.get(qid) or []:
        path = resolve_ref(ref)
        if path is None:
            continue
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    greedy: dict[int, dict] = {}
    for line in (ROOT / "artifacts/fresh/greedy_plan.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            greedy[int(rec["id"])] = rec
    topk = load_topk(ROOT / "artifacts/fresh/dense_topk.json", k=9)
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    with zipfile.ZipFile(ROOT / "submissions/method_v1.zip") as z:
        base = {r["id"]: r for r in json.loads(z.read("submission.json"))}

    safe: list[dict] = []
    for qid, entry in sorted(greedy.items()):
        if str(entry.get("kind") or "").lower() != "cdkt":
            continue
        code = str(entry.get("code") or "").lstrip("t")
        if not code.isdigit():
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
        # Only upgrade when greedy roll-up is not perfect.
        if g_own >= 1.0:
            continue
        pool = slim_pool(qid, gpath, topk)
        winner = rerank_candidates(
            code, pool, gpath,
            require_unique_perfect=False,
            min_score=1.0,
        )
        if winner is None or not winner["flipped"]:
            continue
        question = qs[qid]
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
                abs(answer) / abs(old) > 100 or abs(old) / abs(answer) > 100):
            continue
        csv_name = (
            f"{winner['path'].parent.parent.name}_table_{winner['table_id']}.csv")
        record = {
            "answer": answer,
            "pandas_query": program,
            "evidence": [{"variable": "df", "csv_path": f"data/{csv_name}"}],
        }
        files = {f"data/{csv_name}": blob}
        if not reexec_ok(record, files):
            continue
        safe.append({
            "id": qid,
            "code": code,
            "g_own": g_own,
            "w_score": winner["score"],
            "n_codes": winner["n_codes"],
            "n_cand": winner["n_cand"],
            "old": old,
            "new": answer,
            "pick": winner["path"].name,
            "doc": winner["path"].parent.parent.name,
            "program": program,
            "csv_name": csv_name,
            "blob_path": str(winner["path"]),
        })

    print(f"safe flips: {len(safe)}")
    for item in safe:
        print({k: v for k, v in item.items() if k not in ("program", "blob_path")})
    out = ROOT / "artifacts/fresh/identity_topk_safe.jsonl"
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
