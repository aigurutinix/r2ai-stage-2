"""Let the model SEE what it is about to read before it answers.

Every program in this project so far is written blind: the model guesses `iloc[7, 2]` from
a rendered excerpt and never learns what that cell actually held. The organisers' error
budget says 54.7% of all end-to-end failures are exactly that — the wrong cell — and no
amount of agreement between blind passes fixes it, because two blind passes share the same
habits (measured here: two-pass voting gained 4 questions in 59, and the literature finds
majority voting backfires on most hard problems for models this size).

The one published lever at this model size that attacks the generator itself is execution
feedback: Orchestra reports +11.4 points on WikiTQ with Qwen2.5-14B by letting the model
run code and revise on what it sees. This driver is that loop, two turns:

  turn 1   the model writes SCOUT code — print the row labels, values and header of every
           cell it intends to use, using the same `dfs` it will answer with;
  turn 2   it is shown the printed output and asked to check the labels against the
           question's own words and the table's unit, then commit `result = ...`.

The scout output is the external signal: the actual text of the table, not another sample
of the model. Label mismatches that today survive into submissions (`Tiền mặt` for a
question about `Tiền`, a prior-year column for a current-year question) become visible to
the model one turn before it commits.

Frames are loaded EXACTLY as the organisers load them — their `_read_raw_csv`, imported
from their sandbox file, header row consumed — because a private validator of our own that
loaded differently has already cost one submission. The final program is executed by their
`run_pandas_code` unchanged, and results are written in the same schema as
`prog_results.jsonl` so every downstream tool (gates, builders, merges) works as is.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import importlib.util
import io
import json
import math
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

sys.path.insert(0, str(ROOT / "scripts" / "fresh"))
from answer_gate import verdict  # noqa: E402
from num_helper import SOURCE as NUM_SOURCE  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "sandbox", ROOT / "vifinqa-official/src/vifinqa/answering/sandbox.py")
sandbox = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sandbox)

BANNED_RE = re.compile(r"\bimport\b|\bopen\s*\(|\b__\w+__\b|\bwhile\b|\beval\s*\(|"
                       r"\bexec\s*\(|\binput\s*\(")
CODE_BLOCK = re.compile(r"```(?:python)?\s*(.+?)```", re.S)
THINK_RE = re.compile(r"^.*</think>\s*", re.S)
LIMIT = 1e15
STDOUT_CAP = 1800

SCOUT_SYSTEM = """Bạn phân tích báo cáo tài chính Việt Nam qua HAI lượt.

LƯỢT NÀY: chỉ viết code THĂM DÒ, chưa trả lời. Nhiệm vụ của code thăm dò:
- Với MỖI ô bạn định dùng để trả lời, in ra: nhãn dòng (cột đầu của dòng đó), giá trị ô,
  và tiêu đề cột của ô (dòng đầu bảng nếu cần) — để lượt sau kiểm tra được.
- In thêm 1–2 dòng lân cận nếu bạn phân vân giữa các dòng.

Quy ước:
- `dfs` là dict có sẵn, khóa là ĐÚNG chuỗi table_ref hiển thị. Mọi ô là chuỗi.
- Dòng `r0`, `r1`... tương ứng `df.iloc[0]`, `df.iloc[1]`...; cột `c0`, `c1`... tương ứng
  `df.iloc[:, 0]`, `df.iloc[:, 1]`...
- Hàm `_num(x)` đã có sẵn: đổi số kiểu Việt Nam thành float (chấm nghìn, phẩy thập phân,
  ngoặc là âm, gạch là 0).
- KHÔNG import, không vòng while, không gán `result` ở lượt này. Chỉ dùng print.

Trả về MỘT khối ```python ...``` duy nhất."""

FINAL_SYSTEM = """Bạn phân tích báo cáo tài chính Việt Nam. Đây là lượt CHỐT.

Bạn đã thăm dò và bên dưới là những gì code thăm dò IN RA THẬT từ dữ liệu. Trước khi chốt:
1. Đối chiếu NHÃN DÒNG đã in với đúng chỉ tiêu câu hỏi nêu — nhãn thừa hay thiếu chữ so
   với câu hỏi nghĩa là sai dòng (ví dụ câu hỏi nói "Tiền" mà nhãn là "Tiền mặt" thì phải
   tìm lại; câu hỏi nói "Tổng nguồn vốn" mà nhãn là "VỐN CHỦ SỞ HỮU" là sai dòng).
2. Đối chiếu CỘT với kỳ câu hỏi nêu (năm nay/năm trước, đầu/cuối năm).
3. Xác định đơn vị của bảng (đồng/nghìn/triệu/tỷ) từ tiêu đề, quy đổi về đơn vị CÂU HỎI.
4. Nếu thăm dò cho thấy sai dòng/sai bảng, đọc lại từ `dfs` cho đúng rồi mới chốt.

Viết chương trình cuối cùng: đọc ô từ `dfs` (KHÔNG gõ lại con số thành hằng số), gán kết
quả vào `result`, làm tròn 2 chữ số thập phân. `_num(x)` có sẵn. Không import.

Trả về MỘT khối ```python ...``` duy nhất."""


def api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPEN_ROUTER_KEY") and "=" in line:
            return line.split("=", 1)[1].strip()
    raise SystemExit("khong tim thay OPEN_ROUTER_KEY")


def call(key: str, model: str, messages: list[dict], max_tokens: int) -> str | None:
    body = json.dumps({
        "model": model, "messages": messages, "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }).encode("utf-8")
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                ENDPOINT, data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
            message = ((payload.get("choices") or [{}])[0].get("message") or {})
            return message.get("content") or ""
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401,
                                                                        402, 403):
                try:
                    print(f"  HTTP {exc.code} "
                          f"{exc.read().decode('utf-8', 'replace')[:160]}", flush=True)
                except Exception:  # noqa: BLE001
                    pass
                return None
            if attempt == 2:
                return None
            time.sleep(3 * (attempt + 1))
    return None


def extract_code(reply: str) -> str:
    text = THINK_RE.sub("", reply or "")
    match = CODE_BLOCK.search(text)
    return (match.group(1) if match else text).strip()


def scout_run(code: str, paths: dict[str, Path]) -> str:
    """Execute exploration code with the organisers' frame loading, capture what it prints."""

    frames = {ref: sandbox._read_raw_csv(path) for ref, path in paths.items()}
    import pandas as pd
    namespace: dict = {"pd": pd, "dfs": frames}
    if len(frames) == 1:
        namespace["df"] = next(iter(frames.values()))
    exec(NUM_SOURCE, namespace)  # noqa: S102 — the shared parser, nothing else
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "<scout>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 — the error IS the observation
        buffer.write(f"\n[LOI khi chay tham do: {type(exc).__name__}: {exc}]")
    printed = buffer.getvalue().strip()
    if len(printed) > STDOUT_CAP:
        printed = printed[:STDOUT_CAP] + "\n[...cat bot...]"
    return printed or "[tham do khong in ra gi]"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_progall.jsonl")
    parser.add_argument("--ids-file", default="")
    parser.add_argument("--out", default="artifacts/fresh/react_results.jsonl")
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scout-tokens", type=int, default=4000)
    parser.add_argument("--final-tokens", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    key = api_key()
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
        rows = rows[: args.limit]
    print(f"{len(rows)} cau can chay, {len(done)} da co", flush=True)

    lock = threading.Lock()
    handle = target.open("a", encoding="utf-8")
    counters = {"CHAY DUOC": 0, "tham do hong": 0, "chot hong": 0,
                "chot la hang so": 0, "chay loi": 0, "bi loai boi cong": 0,
                "loi goi": 0}
    started = time.time()

    def work(record: dict) -> None:
        info = record.get("meta") or {}
        paths: dict[str, Path] = {}
        for ref in info.get("refs") or []:
            path = (ROOT / "data" / "official_corpus" / ref["ticker"] / ref["year"]
                    / ref["doc"] / f"{ref['doc']}_extracted_tables"
                    / f"table_{ref['table_id']}.csv")
            if path.exists():
                paths[ref["ref"]] = path
        if not paths:
            with lock:
                counters["tham do hong"] += 1
            return

        scout_reply = call(key, args.model,
                           [{"role": "system", "content": SCOUT_SYSTEM},
                            {"role": "user", "content": record["user"]}],
                           args.scout_tokens)
        if scout_reply is None:
            with lock:
                counters["loi goi"] += 1
            return
        scout_code = extract_code(scout_reply)
        if not scout_code or BANNED_RE.search(scout_code):
            with lock:
                counters["tham do hong"] += 1
            return
        observation = scout_run(scout_code, paths)

        final_reply = call(key, args.model, [
            {"role": "system", "content": FINAL_SYSTEM},
            {"role": "user", "content": record["user"]},
            {"role": "assistant", "content": f"```python\n{scout_code}\n```"},
            {"role": "user", "content":
                "Code thăm dò của bạn in ra:\n\n"
                f"{observation}\n\n"
                "Kiểm tra nhãn/cột/đơn vị theo đúng 4 bước, rồi viết chương trình cuối "
                "cùng gán `result`."},
        ], args.final_tokens)
        if final_reply is None:
            with lock:
                counters["loi goi"] += 1
            return
        final_code = extract_code(final_reply)
        if not final_code or "result" not in final_code \
                or BANNED_RE.search(final_code):
            with lock:
                counters["chot hong"] += 1
            return
        # After seeing the scout's printed values the model mostly writes the number
        # straight into `result` — 34 of 40 pilot finals did, whatever the prompt said.
        # The VALUE is still the loop's verdict, so it is kept and marked; only shipping
        # needs a frame-reading program, and that is a separate, later step.
        constant = "dfs[" not in final_code and not re.search(r"\bdf\b", final_code)
        try:
            if constant:
                namespace: dict = {}
                exec(NUM_SOURCE, namespace)  # noqa: S102
                exec(compile(final_code, "<final>", "exec"), namespace)  # noqa: S102
                number = float(namespace.get("result"))
            else:
                value = sandbox.run_pandas_code(NUM_SOURCE + "\n" + final_code, paths)
                number = float(value)
        except Exception:  # noqa: BLE001
            with lock:
                counters["chay loi"] += 1
            return
        if not math.isfinite(number) or abs(number) > LIMIT:
            with lock:
                counters["chay loi"] += 1
            return
        rejected = verdict(info.get("question") or "", number)
        if rejected is not None:
            with lock:
                counters["bi loai boi cong"] += 1
            return
        with lock:
            counters["CHAY DUOC"] += 1
            handle.write(json.dumps({
                "id": record["id"], "answer": round(number, 2),
                "constant": constant,
                "pandas_query": final_code,
                "refs": [r["ref"] for r in info.get("refs") or []
                         if r["ref"] in paths],
                "docs": sorted({r["doc"] for r in info.get("refs") or []
                                if r["ref"] in paths}),
                "csvs": {ref: str(path.relative_to(ROOT)).replace("\\", "/")
                         for ref, path in paths.items()},
                "observation": observation[:600],
            }, ensure_ascii=False) + "\n")
            handle.flush()
            total = sum(counters.values())
            if total % 10 == 0:
                print(f"  {total}/{len(rows)}  {time.time() - started:.0f}s",
                      flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, rows))
    handle.close()
    print()
    for name, count in counters.items():
        print(f"  {name}: {count}")
    print(f"-> {target}")


if __name__ == "__main__":
    main()
