"""One workflow for all 1012 questions — no hand-editing per row.

Every question runs the same pipeline:

  vote3 incumbent
       ↓
  [1] rule_hop      deterministic multi-hop on CellBook (hard_hop)
       ↓ if no change
  [2] tab_prog      tab quote verified + program reexec agree
       ↓
  [3] address_plan  Mã số / greedy / model / ratio plans (build_submission)
       ↓
  [4] tab_verified  table reader with quote↔cell check (plan_tab)
       ↓ if nothing passes
  keep incumbent

Each layer must pass before it can replace the incumbent:
  * answer_gate.verdict  — wrong shape/year/range is rejected
  * reexec               — pandas_query on bundled CSV must match answer
  * magnitude guard      — layers 2–4 cannot change the answer by >3× (scale-fix trap)

Outputs:
  submissions/vote3_pipeline.zip
  artifacts/fresh/pipeline_manifest.jsonl   — one row per changed question
  artifacts/fresh/pipeline_report.md          — narrative for private review

Usage:
  python scripts/fresh/pipeline.py
  python scripts/fresh/pipeline.py --base submissions/vote3.zip
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

import parse_statements as ps  # noqa: E402
from answer_gate import verdict  # noqa: E402
from score_model import parse  # noqa: E402
from blocks import block_of  # noqa: E402
from build_dualread_zip import (  # noqa: E402
    build_tab_patch,
    load_jsonl,
    tab_answer,
)
from build_notcell_zip import prog_row, reexec_ok  # noqa: E402
from hard_hop import CellBook, TickerResolver, confidence_hit, solve_one  # noqa: E402

MAG_MIN, MAG_MAX = 1.0 / 3.0, 3.0

SINGLE_CELL_BLOCKS = ("tien — MOT O", "khac — don gian")
ADDRESS_BLOCKS = (
    "tien — MOT O", "khac — don gian", "ty le — mot nam", "nam nao",
    "tien — nhieu nam", "ty le — nhieu nam", "khac — nhieu nam",
    "tong hop nhieu o", "dem cong ty",
)

LAYER_DOC = {
    "rule_hop": "Multi-hop xác định trên CellBook (argmin/max, lọc, median, …)",
    "tab_prog": "Model chỉ ô + chương trình đọc CSV; hai nguồn trùng số",
    "tab_dual": "Đọc bảng 2 chiều (thuận + nghịch); cùng ô hoặc cùng số trích dẫn",
    "address_plan": "Địa chỉ Mã số / greedy / model đã verify bằng build_submission",
    "tab_verified": "Model chỉ ô; số trích dẫn khớp ô thật trong CSV",
    "incumbent": "Giữ vote3 — không layer nào vượt qua kiểm tra",
}


@dataclass
class Patch:
    layer: str
    answer: float
    pandas_query: str
    evidence: list
    relevant_docs: list
    relevant_tables: list
    files: dict[str, bytes]
    checks: dict[str, object] = field(default_factory=dict)


def mag_ratio(new: float, old: float) -> float:
    if old == 0 or new == 0:
        return 999.0
    return max(abs(new / old), abs(old / new))


def ensure_wide_zip(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/fresh/build_submission.py"),
            "--greedy-plan", "artifacts/fresh/greedy_plan.jsonl",
            "--model-plan", "artifacts/fresh/model_plan.jsonl",
            "--cohort-plan", "artifacts/fresh/cohort_plan.jsonl",
            "--out", str(path.relative_to(ROOT)),
        ],
        cwd=ROOT,
        check=True,
    )


def patch_from_hop(hit: dict) -> Patch:
    files = {f"data/{name}": text.encode("utf-8")
             for name, text in hit["csv_payloads"].items()}
    return Patch(
        layer="rule_hop",
        answer=float(hit["answer"]),
        pandas_query=hit["pandas_query"],
        evidence=hit["evidence"],
        relevant_docs=hit["relevant_docs"],
        relevant_tables=hit["relevant_tables"],
        files=files,
        checks={"op": hit.get("op"), "filter": hit.get("filter"),
                "target": hit.get("target")},
    )


def patch_from_row(layer: str, row: dict, blobs: dict[str, bytes]) -> Patch | None:
    evidence = row.get("evidence") or []
    if not evidence or not row.get("pandas_query"):
        return None
    files = {item["csv_path"]: blobs[item["csv_path"]]
             for item in evidence if item["csv_path"] in blobs}
    if len(files) != len(evidence):
        return None
    return Patch(
        layer=layer,
        answer=float(row["answer"]),
        pandas_query=row["pandas_query"],
        evidence=evidence,
        relevant_docs=row.get("relevant_docs") or [],
        relevant_tables=row.get("relevant_tables") or [],
        files=files,
    )


def patch_from_tab(layer: str, entry: dict, question: str) -> Patch | None:
    built = build_tab_patch(entry, question)
    if built is None:
        return None
    record, files = built
    return Patch(
        layer=layer,
        answer=float(record["answer"]),
        pandas_query=record["pandas_query"],
        evidence=record["evidence"],
        relevant_docs=record["relevant_docs"],
        relevant_tables=record["relevant_tables"],
        files=files,
        checks={"quoted": entry.get("quoted"), "table_ref": entry.get("table_ref")},
    )


def try_patch(
    patch: Patch | None,
    question: str,
    incumbent: float,
    *,
    mag_limit: bool,
) -> tuple[Patch | None, str | None]:
    """Return (patch, reject_reason)."""

    if patch is None:
        return None, "no_candidate"
    gate = verdict(question, patch.answer)
    if gate:
        return None, f"gate:{gate}"
    if abs(patch.answer - incumbent) <= 0.01:
        return None, "same_as_incumbent"
    if mag_limit:
        ratio = mag_ratio(patch.answer, incumbent)
        if not (MAG_MIN <= ratio <= MAG_MAX):
            return None, f"magnitude:{ratio:.2g}x"
    record = {
        "answer": patch.answer,
        "pandas_query": patch.pandas_query,
        "evidence": patch.evidence,
    }
    if not reexec_ok(record, patch.files):
        return None, "reexec_fail"
    return patch, None


def render_report(
    stats: Counter,
    manifest: list[dict],
    dest: Path,
    base_name: str,
) -> None:
    by_layer = Counter(row["layer"] for row in manifest)
    by_block = Counter(row["block"] for row in manifest)
    lines = [
        "# Pipeline report — trình bày vòng private",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Submission: `{dest.name}` (base `{base_name}`)",
        "",
        "## Ý tưởng (30 giây với giám khảo)",
        "",
        "Mỗi câu hỏi đi qua **cùng một pipeline cố định**. Không sửa tay từng dòng.",
        "Đáp số cuối luôn kèm **`pandas_query` đọc CSV gốc** trong `evidence` — "
        "BTC chạy lại program và so với `answer`.",
        "",
        "Pipeline **từ chối** đáp án sai kiểu (năm nào / % / tiền), sai reexec, "
        "hoặc đổi scale >3× so với incumbent (bẫy đã gây crash 0.32–0.42).",
        "",
        "Tab reader chỉ áp block **một ô** (`tien — MOT O`, `khac — don gian`). "
        "Câu sàng lọc / nhiều năm do rule_hop hoặc address_plan xử lý.",
        "",
        "## Các lớp (theo thứ tự ưu tiên)",
        "",
    ]
    for i, (name, doc) in enumerate(LAYER_DOC.items(), start=1):
        if name == "incumbent":
            continue
        count = by_layer.get(name, 0)
        lines.append(f"{i}. **{name}** ({count} câu) — {doc}")
    lines.extend([
        "",
        "## Thống kê",
        "",
        f"- Tổng thay đổi so base: **{len(manifest)}** / 1012",
        f"- Giữ nguyên incumbent: **{1012 - len(manifest)}**",
        "",
        "### Theo lớp",
        "",
    ])
    for name, count in by_layer.most_common():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "### Theo block câu hỏi", ""])
    for name, count in by_block.most_common():
        lines.append(f"- `{name}`: {count}")
    lines.extend([
        "",
        "## Verify checklist (mỗi dòng thay đổi)",
        "",
        "1. `answer_gate` — kiểu đáp án khớp câu hỏi",
        "2. `reexec` — chạy `pandas_query` trên CSV bundle → trùng `answer` (±0.01)",
        "3. `magnitude` — lớp 2–4: |Δ| ≤ 3× (tránh sửa scale mù)",
        "4. Tab: `quoted` copy y nguyên từ ô CSV (plan_tab)",
        "",
        "## File manifest",
        "",
        "`artifacts/fresh/pipeline_manifest.jsonl` — một JSON/câu thay đổi: "
        "id, block, layer, old/new answer, checks, tables.",
        "",
        "## Lệnh reproduce",
        "",
        "```bash",
        "python scripts/fresh/pipeline.py --full",
        "python scripts/fresh/validate_submission.py --zip submissions/vote3_pipeline.zip",
        "```",
        "",
    ])
    report_path = ROOT / "artifacts/fresh/pipeline_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def ensure_tab_rev() -> Path:
    """Run tab reverse pass + plan_tab when artifacts are missing."""

    replies = ROOT / "artifacts/fresh/replies_tab_rev.jsonl"
    prompts = ROOT / "artifacts/fresh/prompts_tab_rev.jsonl"
    plan = ROOT / "artifacts/fresh/tab_plan_rev.jsonl"
    need = sum(1 for line in prompts.read_text(encoding="utf-8").splitlines()
               if line.strip())
    have = 0
    good = 0
    if replies.exists():
        for line in replies.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            have += 1
            rep = json.loads(line).get("reply") or ""
            if rep.strip() and parse(rep):
                good += 1
    if good < need * 0.5:
        if have >= need and good < need * 0.1:
            print(f"tab rev: {good}/{need} hop le — replies lo (rong JSON). "
                  f"can --retry-empty", flush=True)
        if good < need:
            print(f"tab rev LLM: {good}/{need} hop le — goi OpenRouter...", flush=True)
            smoke = subprocess.run(
                [sys.executable, str(ROOT / "scripts/fresh/run_tab.py"),
                 "--prompts", str(prompts.relative_to(ROOT)),
                 "--out", str(replies.relative_to(ROOT)),
                 "--smoke", "5"],
                cwd=ROOT,
            )
            if smoke.returncode != 0:
                raise SystemExit("tab rev smoke test FAIL — khong goi batch")
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/fresh/run_tab.py"),
                 "--prompts", str(prompts.relative_to(ROOT)),
                 "--out", str(replies.relative_to(ROOT)),
                 "--retry-empty",
                 "--workers", "8"],
                cwd=ROOT,
                check=True,
            )
    if not plan.exists() or plan.stat().st_size == 0:
        print("plan_tab rev...", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/fresh/plan_tab.py"),
             "--replies", str(replies.relative_to(ROOT)),
             "--prompts", str(prompts.relative_to(ROOT)),
             "--out", str(plan.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
        )
    return plan


def tab_dual_tier(qid: int, forward: dict, reverse: dict) -> str | None:
    left, right = forward.get(qid), reverse.get(qid)
    if left is None or right is None:
        return None
    if (left["table_id"], left["row"], left["col"]) == (
            right["table_id"], right["row"], right["col"]):
        return "cung_o"
    lv = ps.parse_vn_number(str(left.get("quoted", "")))
    rv = ps.parse_vn_number(str(right.get("quoted", "")))
    if lv is not None and rv is not None and abs(lv - rv) <= 0.01:
        return "cung_so"
    return None


def run_pipeline(
    base_path: Path,
    dest_path: Path,
    wide_path: Path,
    tab_path: Path,
    prog_path: Path,
    tab_rev_path: Path | None = None,
) -> tuple[Counter, list[dict]]:
    ensure_wide_zip(wide_path)

    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }

    with zipfile.ZipFile(base_path) as src:
        rows = {r["id"]: r for r in json.loads(src.read("submission.json"))}
        files = {n: src.read(n) for n in src.namelist() if n != "submission.json"}

    with zipfile.ZipFile(wide_path) as wide_archive:
        wide_rows = {r["id"]: r for r in json.loads(wide_archive.read("submission.json"))}
        wide_blobs = {n: wide_archive.read(n) for n in wide_archive.namelist()
                      if n.startswith("data/")}

    tab = load_jsonl(tab_path)
    tab_rev = load_jsonl(tab_rev_path) if tab_rev_path else {}
    prog = {}
    for line in prog_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            prog[json.loads(line)["id"]] = json.loads(line)

    book, resolver = CellBook(), TickerResolver()
    stats: Counter = Counter()
    manifest: list[dict] = []

    for qid in sorted(qs):
        question = qs[qid]
        row = rows[qid]
        incumbent = float(row.get("answer") or 0)
        block = block_of(question)
        winner: Patch | None = None
        reject_log: list[str] = []

        # Layer 1 — rule hop (no magnitude limit)
        try:
            hit = solve_one(book, question, resolver)
        except Exception as exc:  # noqa: BLE001
            hit = None
            reject_log.append(f"rule_hop:error:{exc}")
        if hit is not None and confidence_hit(question, hit):
            candidate = patch_from_hop(hit)
            ok, why = try_patch(candidate, question, incumbent, mag_limit=False)
            if ok:
                winner = ok
            elif why:
                reject_log.append(f"rule_hop:{why}")

        # Layer 2 — tab + prog agree (single-cell blocks only)
        if (winner is None and qid in tab and qid in prog
                and block in SINGLE_CELL_BLOCKS):
            ta = tab_answer(tab[qid], question)
            if ta is not None:
                pa = prog[qid]
                built = prog_row(pa, question)
                if (built and reexec_ok(built[0], built[1])
                        and abs(float(pa["answer"]) - ta) <= 0.01):
                    candidate = patch_from_tab("tab_prog", tab[qid], question)
                    ok, why = try_patch(candidate, question, incumbent, mag_limit=True)
                    if ok:
                        winner = ok
                    elif why:
                        reject_log.append(f"tab_prog:{why}")

        # Layer 3 — tab dual-read (forward + reverse agree)
        if winner is None and block in SINGLE_CELL_BLOCKS and qid in tab:
            tier = tab_dual_tier(qid, tab, tab_rev)
            if tier is not None:
                candidate = patch_from_tab(f"tab_dual_{tier}", tab[qid], question)
                if candidate:
                    candidate.checks["dual_tier"] = tier
                ok, why = try_patch(candidate, question, incumbent, mag_limit=True)
                if ok:
                    winner = ok
                elif why:
                    reject_log.append(f"tab_dual:{why}")

        # Layer 4 — address plan from wide build_submission
        if winner is None and qid in wide_rows and block in ADDRESS_BLOCKS:
            wide_ans = float(wide_rows[qid].get("answer") or 0)
            if wide_ans != 0:
                candidate = patch_from_row("address_plan", wide_rows[qid], wide_blobs)
                ok, why = try_patch(candidate, question, incumbent, mag_limit=True)
                if ok:
                    winner = ok
                elif why:
                    reject_log.append(f"address_plan:{why}")

        # Layer 5 — tab verified alone (single-cell blocks only)
        if winner is None and qid in tab and block in SINGLE_CELL_BLOCKS:
            candidate = patch_from_tab("tab_verified", tab[qid], question)
            ok, why = try_patch(candidate, question, incumbent, mag_limit=True)
            if ok:
                winner = ok
            elif why:
                reject_log.append(f"tab_verified:{why}")

        if winner is None:
            stats["incumbent"] += 1
            continue

        old = incumbent
        row.update({
            "answer": winner.answer,
            "pandas_query": winner.pandas_query,
            "evidence": winner.evidence,
            "relevant_docs": winner.relevant_docs,
            "relevant_tables": winner.relevant_tables,
            "question": question,
        })
        files.update(winner.files)
        stats[winner.layer] += 1
        manifest.append({
            "id": qid,
            "block": block,
            "layer": winner.layer,
            "old_answer": old,
            "new_answer": winner.answer,
            "mag_ratio": round(mag_ratio(winner.answer, old), 4),
            "checks": winner.checks,
            "relevant_tables": winner.relevant_tables,
            "rejected": reject_log,
        })
        print(f"{qid} [{block}] {winner.layer} {winner.answer} (was {old})")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)

    manifest_path = ROOT / "artifacts/fresh/pipeline_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
        encoding="utf-8",
    )
    return stats, manifest


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Unified submission pipeline")
    parser.add_argument("--base", default="submissions/vote3.zip")
    parser.add_argument("--dest", default="submissions/vote3_pipeline.zip")
    parser.add_argument("--wide-zip", default="submissions/_wide_maso.zip")
    parser.add_argument("--tab", default="artifacts/fresh/tab_plan.jsonl")
    parser.add_argument("--prog", default="artifacts/fresh/prog_results.jsonl")
    parser.add_argument("--tab-rev", default="artifacts/fresh/tab_plan_rev.jsonl")
    parser.add_argument("--full", action="store_true",
                        help="run tab reverse LLM + plan_tab if artifacts missing")
    args = parser.parse_args()

    tab_rev_path = ROOT / args.tab_rev
    if args.full:
        tab_rev_path = ensure_tab_rev()

    stats, manifest = run_pipeline(
        ROOT / args.base,
        ROOT / args.dest,
        ROOT / args.wide_zip,
        ROOT / args.tab,
        ROOT / args.prog,
        tab_rev_path if tab_rev_path.exists() else None,
    )
    render_report(stats, manifest, ROOT / args.dest, Path(args.base).name)

    print(f"\n=== pipeline summary ===")
    print(f"changed: {len(manifest)}  kept: {1012 - len(manifest)}")
    for name, count in stats.most_common():
        print(f"  {count:4d}  {name}")
    print(f"-> {ROOT / args.dest}")
    print(f"-> {ROOT / 'artifacts/fresh/pipeline_manifest.jsonl'}")
    print(f"-> {ROOT / 'artifacts/fresh/pipeline_report.md'}")


if __name__ == "__main__":
    main()
