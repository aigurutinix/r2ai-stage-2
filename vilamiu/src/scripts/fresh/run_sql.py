"""Answer the same questions in SQL, as a second formalism rather than a second sample.

Every attempt to improve this submission by agreement has used two samples of one model
answering one way, and the published result for that is discouraging: majority voting over a
small model's own samples backfires on 56-66% of hard problems, because the samples
concentrate on the same wrong answer. Our own two-pass filter gained four questions on
fifty-nine, and its clearest failure was both passes reading the same empty cell.

A second FORMALISM is a different thing. The SemEval-2025 Task 8 systems measured it
directly on the same kind of task — questions answered by generating code over tables:

    pandas only                                  72%
    pandas and SQL, an orchestrator choosing      84%
    two code models and the orchestrator          88%
    GPT-4o, for reference                         74%

Twelve points from adding SQL. SQL fails differently from pandas: it forces the columns to
be named, it has no `.iloc`, and a wrong join shows up as an empty result rather than a
plausible number. So where the two agree, they agree about the cell rather than about a
habit.

The submission ships `pandas_query`, so SQL is used to CHECK, not to answer: a pandas answer
that SQL reproduces is kept, and the rest is left to the standing build.

Tables reach SQLite exactly as they reach pandas — every value TEXT, no type inference — and
a `num()` function does the Vietnamese number parsing, so the two paths differ in the query
language and nowhere else.
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = """Bạn trả lời câu hỏi về báo cáo tài chính Việt Nam bằng MỘT câu lệnh SQLite.

Quy ước dữ liệu:
- Mỗi bảng đã được nạp sẵn vào SQLite với tên `t0`, `t1`, ... theo đúng thứ tự liệt kê.
- Cột tên là `c0`, `c1`, ... đúng như nhãn in trong bảng. Dòng `r0`, `r1`, ... là thứ tự
  dòng, truy cập bằng cột `rid` (số nguyên, bắt đầu từ 0).
- MỌI giá trị đều là TEXT. Dùng hàm `num(x)` để đổi một số kiểu Việt Nam thành số thực:
  dấu chấm phân cách hàng nghìn, dấu phẩy là thập phân, ngoặc là số âm, dấu gạch là 0.
- Mỗi bảng có đơn vị riêng (đồng / nghìn / triệu / tỷ). Đọc đơn vị từ tiêu đề bảng rồi quy
  đổi kết quả cuối về đơn vị mà CÂU HỎI yêu cầu.

Quy ước ngữ nghĩa:
- Chênh lệch không nêu chiều thì lấy trị tuyệt đối; có nêu chiều thì giữ dấu.
- Không làm tròn ở bước trung gian.

Định dạng đầu ra: chỉ MỘT câu lệnh SELECT trả về đúng một giá trị số, đặt trong khối
```sql ... ```. Không giải thích gì thêm sau khối đó."""

SQL_BLOCK = re.compile(r"```sql\s*(.+?)```", re.S | re.I)
TABLE_REF = re.compile(r"table_ref:\s*\"([^\"]+)\"")


def api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPEN_ROUTER_KEY") and "=" in line:
            return line.split("=", 1)[1].strip()
    raise SystemExit("khong tim thay OPEN_ROUTER_KEY")


def vn_number(raw: object) -> float:
    """The `num()` SQLite sees — the same parse the pandas path uses."""

    text = str(raw).strip().replace("%", "").replace(" ", "")
    if not text or text in ("-", "–", "—", "None", "nan"):
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace(".", "").replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return -value if negative else value


_path_cache: dict[str, Path | None] = {}


def csv_for(ref: str, catalogue: dict[str, str]) -> Path | None:
    """The CSV a `doc|table_N` reference names, from the catalogue or the corpus."""

    relative = catalogue.get(ref)
    if relative and (ROOT / relative).exists():
        return ROOT / relative
    if ref in _path_cache:
        return _path_cache[ref]
    doc, _, raw = ref.rpartition("|table_")
    found = list((ROOT / "data/official_corpus").glob(f"*/*/{doc}"))
    path = None
    if found:
        candidate = found[0] / f"{doc}_extracted_tables" / f"table_{raw}.csv"
        if candidate.exists():
            path = candidate
    _path_cache[ref] = path
    return path


def load_database(refs: list[str], catalogue: dict[str, str]) -> sqlite3.Connection:
    """One in-memory database holding every table the prompt showed.

    Each table is registered under BOTH names it might be called by: `t0`..`tN` in the
    order the prompt lists them, and the `doc|table_N` reference itself. The prompt tells
    the model that `dfs` is keyed by the reference — that wording belongs to the pandas
    path and is what the model followed, so seven of the first fourteen queries asked for
    a table named `"HDB_financial_statements_2021_separate|table_32"`. Registering both
    names costs nothing and removes the disagreement.

    A reference whose CSV cannot be found still gets an empty table, so a query that names
    `t3` fails on its logic rather than on a gap in the loader.
    """

    connection = sqlite3.connect(":memory:")
    connection.create_function("num", 1, vn_number)
    for index, ref in enumerate(refs):
        path = csv_for(ref, catalogue)
        grid: list[list[str]] = []
        if path is not None:
            try:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    grid = list(csv_mod.reader(handle))
            except OSError:
                grid = []
        width = max((len(row) for row in grid), default=1)
        columns = ", ".join(f"c{i} TEXT" for i in range(width))
        connection.execute(f"CREATE TABLE t{index} (rid INTEGER, {columns})")
        placeholders = ", ".join("?" * (width + 1))
        # The first CSV line is the header, exactly as `pd.read_csv` treats it, so `rid`
        # counts data rows and matches `iloc` on the pandas side.
        for number, row in enumerate(grid[1:]):
            padded = list(row) + [""] * (width - len(row))
            connection.execute(f"INSERT INTO t{index} VALUES ({placeholders})",
                               [number, *padded])
        alias = ref.replace('"', '')
        connection.execute(f'CREATE VIEW "{alias}" AS SELECT * FROM t{index}')
        # And under the table's own number. The prompt prints `doc|table_29`, so the model
        # writes `t29` — a third of the failed queries asked for a table index that is the
        # reference's number rather than its position in the list. Both readings are
        # reasonable; registering both costs one view.
        _doc, _, raw = ref.rpartition("|table_")
        if raw.isdigit() and int(raw) != index:
            try:
                connection.execute(
                    f'CREATE VIEW "t{raw}" AS SELECT * FROM t{index}')
            except sqlite3.OperationalError:
                # Two references with the same number in different documents; the first
                # one keeps the short name and the full reference still resolves.
                pass
    return connection


def scalar(connection: sqlite3.Connection, sql: str) -> float | None:
    cursor = connection.execute(sql)
    row = cursor.fetchone()
    if row is None or row[0] is None:
        return None
    try:
        value = float(row[0])
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_progall.jsonl")
    parser.add_argument("--programs", default="artifacts/fresh/progA_results.jsonl",
                        help="supplies the csv catalogue per question")
    parser.add_argument("--ids-file", default="")
    parser.add_argument("--out", default="artifacts/fresh/sql_results.jsonl")
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--failures", default="",
                        help="write queries that would not run, for diagnosis")
    args = parser.parse_args()

    key = api_key()
    catalogues: dict[int, dict[str, str]] = {}
    for line in (ROOT / args.programs).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            catalogues[int(row["id"])] = row.get("csvs") or {}

    rows = [json.loads(line) for line
            in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.ids_file:
        wanted = set(json.loads((ROOT / args.ids_file).read_text(encoding="utf-8")))
        rows = [r for r in rows if r["id"] in wanted]

    target = ROOT / args.out
    done = set()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    rows = [r for r in rows if r["id"] not in done]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} cau can chay, {len(done)} da co", flush=True)

    lock = threading.Lock()
    handle = target.open("a", encoding="utf-8")
    failures = ((ROOT / args.failures).open("a", encoding="utf-8")
                if args.failures else None)
    counters = {"ok": 0, "sql loi": 0, "khong co sql": 0, "loi goi": 0}
    started = time.time()

    def work(record: dict) -> None:
        body = json.dumps({
            "model": args.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": record["user"]}],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        }).encode("utf-8")
        payload = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    ENDPOINT, data=body,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    method="POST")
                with urllib.request.urlopen(request, timeout=240) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except Exception as exc:  # noqa: BLE001
                detail = ""
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        detail = exc.read().decode("utf-8", "replace")[:200]
                    except Exception:  # noqa: BLE001
                        detail = ""
                    if exc.code in (400, 401, 402, 403):
                        with lock:
                            counters["loi goi"] += 1
                            if counters["loi goi"] <= 2:
                                print(f"  HTTP {exc.code} {detail}", flush=True)
                        return
                if attempt == 2:
                    with lock:
                        counters["loi goi"] += 1
                        if counters["loi goi"] <= 2:
                            print(f"  {type(exc).__name__} {exc} {detail}", flush=True)
                    return
                time.sleep(3 * (attempt + 1))
        content = ((payload.get("choices") or [{}])[0].get("message") or {}).get(
            "content") or ""
        match = SQL_BLOCK.search(content)
        if not match:
            with lock:
                counters["khong co sql"] += 1
            return
        sql = match.group(1).strip().rstrip(";")
        refs = TABLE_REF.findall(record["user"])
        problem = ""
        try:
            connection = load_database(refs, catalogues.get(record["id"], {}))
            value = scalar(connection, sql)
            connection.close()
        except Exception as exc:  # noqa: BLE001 — a wrong query is data, not a crash
            value = None
            problem = f"{type(exc).__name__}: {exc}"[:160]
        with lock:
            if value is None:
                counters["sql loi"] += 1
                # A query that will not run is the most useful thing this run produces
                # while the prompt is still being tuned, so it is kept, not counted.
                if failures is not None:
                    failures.write(json.dumps(
                        {"id": record["id"], "sql": sql, "loi": problem},
                        ensure_ascii=False) + chr(10))
                    failures.flush()
            else:
                counters["ok"] += 1
                handle.write(json.dumps(
                    {"id": record["id"], "answer": value, "sql": sql,
                     "refs": refs}, ensure_ascii=False) + "\n")
                handle.flush()
            total = sum(counters.values())
            if total % 25 == 0:
                print(f"  {total}/{len(rows)}  {time.time() - started:.0f}s", flush=True)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, rows))
    handle.close()
    if failures is not None:
        failures.close()
    print()
    for name, count in counters.items():
        print(f"  {name}: {count}")
    print(f"-> {target}")


if __name__ == "__main__":
    main()
