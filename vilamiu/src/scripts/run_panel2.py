"""Let the model interpret the question; keep the execution ours.

The dead zone is 330 questions served by cell plans (7.2%), best-effort fallback
(6%) and 46 that ship no program at all — about 160 graded answers, roughly nine
of them right. Every lexical mechanism that could serve them has been built and
measured.

A first attempt had the model write pandas over a clean panel. Nine of twelve
programs died on a style rule (comprehensions are unsafe under py37's `exec`
scoping) and the rest on the model's own reasoning text arriving as content.
Nothing about the *reasoning* was wrong; the format was.

So the model now returns a small JSON plan — which phrase is the filter, which is
the answer, what operation, over which axis — and `plan_json` executes it and
writes the pandas. Malformed syntax, hard-coded constants and py37 breakage stop
being possible rather than being caught.

Usage:
  PYTHONPATH=src python scripts/run_panel2.py --workers 8
  PYTHONPATH=src python scripts/run_panel2.py --local-url http://localhost:18000/v1 \
      --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import panel2, plan_json  # noqa: E402
from vifin.answering.panel2 import PANEL_HEADER  # noqa: E402
from vifin.answering.sandbox import portability_problems, run_query  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402

UNIT_TEXT = {
    "ty": "tỷ đồng",
    "trieu": "triệu đồng",
    "nghin": "nghìn đồng",
    "nghin_ty": "nghìn tỷ đồng",
    "tram_ty": "trăm tỷ đồng",
    "dong": "đồng",
    "phan_tram": "phần trăm",
    "lan": "lần",
    "vong": "vòng",
}

SYSTEM = """Bạn đọc một câu hỏi tài chính tiếng Việt và một bảng dữ liệu đã trích sẵn,
rồi trả về MỘT đối tượng JSON mô tả cách tính. Bạn KHÔNG viết code.

Bảng có các cột: ma (mã công ty), nam (năm), chi_tieu (tên chỉ tiêu),
gia_tri_dong (giá trị, đơn vị đồng — có thể âm).

Các trường JSON:
- "op": một trong
    "value"  - lấy một giá trị
    "max"    - giá trị lớn nhất của chi_tieu
    "min"    - giá trị nhỏ nhất
    "sum"    - tổng
    "avg"    - trung bình
    "diff"   - chênh lệch giữa mốc đầu và mốc cuối (trị tuyệt đối)
    "growth" - tăng trưởng phần trăm từ mốc đầu đến mốc cuối
    "ratio"  - chia "metric" cho "denominator"
    "screen" - chọn mốc có "filter_metric" lớn/nhỏ nhất, rồi lấy "metric" tại mốc đó
               (nếu có "denominator" thì lấy tỷ lệ metric/denominator tại mốc thắng)
    "count"  - đếm số mốc (công ty/năm) thỏa điều kiện dấu trên filter_metric
- "metric": tên chi_tieu CHÍNH XÁC như trong danh sách / bảng
- "axis": "nam" nếu so sánh giữa các năm, "ma" nếu so sánh giữa các công ty
- "denominator": mẫu số (ratio, hoặc screen-ratio, hoặc vế phải khi count kèm so sánh)
- "filter_metric": chỉ tiêu dùng để chọn/đếm
- "filter_take": "max"|"min" (screen) hoặc "pos"|"neg" (count: >0 hoặc <0)

BẮT BUỘC: mọi tên chi_tieu phải khớp một mục trong <chi_tieu_co_trong_bang>.

Ví dụ 1 — screen:
{"op": "screen", "metric": "chi phí lãi vay", "axis": "nam", "filter_metric": "tiền mặt", "filter_take": "max"}

Ví dụ 2 — ratio:
{"op": "ratio", "metric": "lợi nhuận gộp", "denominator": "doanh thu thuần", "axis": "nam"}

Ví dụ 3 — count CFO dương:
{"op": "count", "axis": "ma", "filter_metric": "cfo", "filter_take": "pos"}

Ví dụ 4 — count CFO dương và TSNH < NNH:
{"op": "count", "axis": "ma", "filter_metric": "cfo", "filter_take": "pos",
 "metric": "tài sản ngắn hạn", "denominator": "nợ ngắn hạn"}

Ví dụ 5 — screen rồi lấy biên (ratio tại winner):
{"op": "screen", "metric": "lợi nhuận gộp", "denominator": "doanh thu thuần",
 "axis": "ma", "filter_metric": "cfo", "filter_take": "max"}

Chỉ in ra JSON, không giải thích, không markdown."""


def panel_metrics(panel_rows) -> list[str]:
    seen: list[str] = []
    for row in panel_rows[1:]:
        if row[2] not in seen:
            seen.append(row[2])
    return seen


def build_user(question, panel_rows) -> str:
    lines = [",".join(PANEL_HEADER)]
    for row in panel_rows[1:]:
        lines.append(",".join(str(cell).replace(",", " ") for cell in row))
    unit = UNIT_TEXT.get(question.target_unit, question.target_unit or "số")
    metrics = "\n".join("- " + m for m in panel_metrics(panel_rows))
    table = "\n".join(lines)
    return (
        "<cau_hoi>\n" + question.question + "\n</cau_hoi>\n"
        "<don_vi>" + unit + "</don_vi>\n"
        "<chi_tieu_co_trong_bang>\n" + metrics + "\n</chi_tieu_co_trong_bang>\n\n"
        "<bang>\n" + table + "\n</bang>\n\n"
        # Qwen3 reasons by default and the reasoning is billed against the same
        # token budget as the answer; at 700 tokens nothing usable ever arrived.
        "/no_think"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--local-url", default="")
    parser.add_argument("--cache", default="artifacts/panel2_scale.jsonl")
    parser.add_argument("--build", default="artifacts/panel2_build_scale.jsonl",
                        help="Prebuilt panels JSONL (id, rows, keys, …)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    parsed = {q.id: q for q in parse_all(
        root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")}
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    build_path = root / args.build
    if not build_path.exists():
        build_path = root / "artifacts" / "panel2_build.jsonl"
    panels = {}
    for line in build_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            panels[row["id"]] = row
    print(f"loaded {len(panels)} panels from {build_path.name}")

    cache = Path(args.cache)
    done = set()
    if cache.exists():
        done = {json.loads(l)["id"]
                for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}

    todo = [i for i in sorted(panels) if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} panels to plan ({len(done)} cached), model={args.model}")

    if args.local_url:
        client = ChatClient.local(args.model, args.local_url, max_tokens=500)
    else:
        client = ChatClient.from_env(root, args.model, max_tokens=500)

    handle = cache.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters: collections.Counter = collections.Counter()
    started = time.time()

    def work(qid: int) -> None:
        question = parsed[qid]
        panel = panels[qid]
        prebuilt = panel["rows"]
        try:
            reply = client.complete(SYSTEM, build_user(question, prebuilt))
        except RuntimeError:
            with lock:
                counters["transport"] += 1
            return

        row = {"id": qid, "ok": False}
        # Prefer the prebuilt panel (OCR + Circular-200 inject). Snapping the
        # model's phrases onto its chi_tieu list avoids a second OCR pass on
        # every question; rebuild only when the named metric is absent.
        metrics = panel_metrics(prebuilt)
        plan = plan_json.parse(reply, metrics=metrics) or plan_json.parse(reply)
        if plan is None:
            row["error"] = "no usable plan"
            row["reply"] = reply[-300:]
            with lock:
                counters["no plan"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            return

        row["plan"] = {"op": plan.op, "metric": plan.metric, "axis": plan.axis,
                       "denominator": plan.denominator,
                       "filter_metric": plan.filter_metric,
                       "filter_take": plan.filter_take}

        use_rows = prebuilt
        use_keys = panel.get("keys") or []
        source_panel = "prebuilt"
        done_pair = plan_json.execute(plan, use_rows, question.target_unit or "")
        if done_pair is None:
            wanted = [plan.metric]
            for extra in (plan.filter_metric, plan.denominator):
                if extra and extra not in wanted:
                    wanted.append(extra)
            rebuilt = panel2.build(question, store, retriever, phrases=wanted)
            if rebuilt is None:
                row["error"] = "panel empty for the named metrics"
                with lock:
                    counters["no panel"] += 1
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                return
            use_rows = rebuilt.rows
            use_keys = [[k.doc_name, k.table_id] for k in rebuilt.keys]
            source_panel = "rebuilt"
            plan2 = plan_json.parse(
                json.dumps(row["plan"]), metrics=panel_metrics(use_rows)
            ) or plan
            plan = plan2
            done_pair = plan_json.execute(plan, use_rows, question.target_unit or "")
            if done_pair is None:
                row["error"] = "plan did not resolve on the panel"
                with lock:
                    counters["unresolved"] += 1
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                return

        row["panel_rows"] = use_rows
        row["keys"] = use_keys
        expected, used = done_pair

        code = plan_json.compile_query(plan, question.target_unit or "")
        problems = portability_problems(code)
        outcome = run_query(code, {"df": use_rows})
        agrees = outcome.ok and abs(outcome.value - expected) <= 2e-4 * max(
            abs(outcome.value), abs(expected), 1.0
        )
        row.update(ok=bool(agrees and not problems), value=outcome.value,
                   expected=expected, code=code, used=used,
                   error=("; ".join(problems) or outcome.error or "")[:160],
                   source="panel2_llm", mode="panel",
                   shape=plan.op, variables=["df"])
        with lock:
            counters[source_panel] += 1
            if problems:
                counters["py37"] += 1
            elif not outcome.ok:
                counters["crash:" + (outcome.error or "").split(":")[0]] += 1
            elif not agrees:
                counters["mismatch"] += 1
            else:
                counters["ok"] += 1
                counters["op:" + plan.op] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            total = (counters["ok"] + counters["no plan"] + counters["no panel"]
                     + counters["unresolved"] + counters["py37"] + counters["mismatch"])
            total += sum(v for k, v in counters.items() if k.startswith("crash"))
            if total % 20 == 0:
                print(f"  {total}/{len(todo)}  ok={counters['ok']}  "
                      f"prebuilt={counters['prebuilt']}  "
                      f"{time.time() - started:.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    handle.close()

    total = sum(v for k, v in counters.items() if not k.startswith("op:")) or 1
    print(f"\nusable {counters['ok']}/{total} ({counters['ok'] / total:.1%})")
    for key, count in counters.most_common(14):
        print(f"  {count:4d}  {key}")
    print(f"elapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
