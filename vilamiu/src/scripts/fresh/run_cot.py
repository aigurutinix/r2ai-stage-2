"""Ask for the number instead of a program, and run it through OpenRouter.

The organisers' own error analysis over 430 end-to-end failures puts 54.7% on reading the
wrong cell and 34.7% on the retriever missing the table, against 9.3% on code crashes and
0.9% on arithmetic. A day spent on parse helpers, constant bans and averaging divisors was
aimed at the 10% band.

Their comparison of the two answering strategies is just as direct: with retrieved
evidence, chain-of-thought reaches 64.0% against 62.0% for program-of-thought, and for
models under 10B the program path collapses — syntax error rates up to 99%. So the
question is asked in words and answered with a number.

The tables are the SAME rendered candidates the program pass used, read straight out of
`prompts_progall.jsonl`, so a difference in score is attributable to the strategy and not
to retrieval.

Reasoning is requested explicitly: it moved the program pass from 48% to 78% on the
offline set, and through OpenRouter it arrives in a separate `reasoning` field rather than
inside the content — so `max_tokens` has to cover both or the answer comes back empty,
which is exactly what the first probe returned.

Usage:
  python scripts/fresh/run_cot.py --ids-file artifacts/fresh/cot_ids.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = """Bạn là trợ lý phân tích báo cáo tài chính Việt Nam. Dựa trên các bảng CSV \
được cung cấp, trả lời câu hỏi bằng ĐÚNG MỘT giá trị số.

Quy ước dữ liệu:
- Số kiểu Việt Nam: dấu chấm phân cách hàng nghìn, dấu phẩy là thập phân, ngoặc là số âm.
- Mỗi bảng có đơn vị riêng (đồng / nghìn / triệu / tỷ). Xác định đơn vị của từng bảng,
  rồi quy đổi kết quả cuối về đơn vị mà CÂU HỎI yêu cầu.
- `r12` và `c3` là nhãn dòng/cột in ra để bạn định vị, không phải tên cột trong dữ liệu.

Quy ước ngữ nghĩa:
- Chênh lệch không nêu chiều thì lấy trị tuyệt đối (không âm); có nêu chiều thì giữ dấu.
- Không làm tròn ở bước trung gian; chỉ làm tròn kết quả cuối 2 chữ số thập phân.

Định dạng đầu ra: dòng cuối cùng của câu trả lời phải là

ĐÁP ÁN: <một con số>

không kèm đơn vị, ký hiệu, hay chữ nào khác sau con số."""

ANSWER_RE = re.compile(r"ĐÁP\s*ÁN\s*[:：]\s*(-?[\d.,]+)", re.I)
FALLBACK_RE = re.compile(r"(-?\d[\d.,]*)")


def api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPEN_ROUTER_KEY") and "=" in line:
            return line.split("=", 1)[1].strip()
    raise SystemExit("khong tim thay OPEN_ROUTER_KEY")


def parse_number(text: str) -> float | None:
    """The number on the ĐÁP ÁN line, or the last number in the reply."""

    match = ANSWER_RE.search(text)
    raw = match.group(1) if match else None
    if raw is None:
        found = FALLBACK_RE.findall(text.strip().splitlines()[-1]
                                    if text.strip() else "")
        raw = found[-1] if found else None
    if raw is None:
        return None
    raw = raw.rstrip(".,")
    # A reply may print 1.234,56 or 1234.56; a comma always means the decimal mark
    # here, and a dot is a thousands separator only when it groups three digits.
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif re.search(r"\.\d{3}\b", raw) and raw.count(".") >= 1 and not re.search(
            r"\.\d{1,2}$", raw):
        raw = raw.replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="artifacts/fresh/prompts_progall.jsonl")
    parser.add_argument("--ids-file", default="")
    parser.add_argument("--out", default="artifacts/fresh/cot_results.jsonl")
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--max-tokens", type=int, default=2600)
    parser.add_argument("--limit", type=int, default=0)
    # Only Qwen3 takes `enable_thinking`. Sending it to Gemma, Phi or Ministral is at
    # best ignored and at worst a 400 on every request, which would look like the model
    # failing rather than the client sending a field it does not accept.
    parser.add_argument("--no-thinking", action="store_true")
    # Ignore whatever system message the prompt file carries. `prompts_progall.jsonl`
    # carries the PROGRAM-writing instructions, so a chain-of-thought run over it comes
    # back as pandas code instead of numbers — which is exactly the hazard recorded in
    # VOTE3.md, and it bit again the moment a second model family was pointed at that
    # file.
    parser.add_argument("--force-system", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    # A self-hosted vLLM speaks the same OpenAI-compatible dialect, so pointing this at a
    # tunnelled port is the whole of what it takes to run a model no provider serves.
    parser.add_argument("--endpoint", default=ENDPOINT)
    # A self-hosted card generating 18k tokens takes far longer than a provider's fleet:
    # 76 of 116 requests died on the 240-second default before anything came back.
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    key = api_key() if args.endpoint == ENDPOINT else "local"
    rows = [json.loads(line) for line
            in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.ids_file:
        wanted = set(json.loads(
            (ROOT / args.ids_file).read_text(encoding="utf-8")))
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
    counters = {"ok": 0, "khong parse duoc so": 0, "loi": 0}
    started = time.time()

    def work(record: dict) -> None:
        body = json.dumps({
            "model": args.model,
            # A prompt file may carry its own system message — the cell audit does, and
            # ignoring it told the auditor to answer with a number instead of the JSON
            # verdict, so all seventy replies were unparseable.
            "messages": [{"role": "system",
                          "content": SYSTEM if args.force_system
                          else (record.get("system") or SYSTEM)},
                         {"role": "user", "content": record["user"]}],
            # A second pass has to be able to reason differently, or two runs agreeing
            # says only that the decoder is deterministic. Nothing else about the request
            # changes between passes.
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            **({} if args.no_thinking
               else {"chat_template_kwargs": {"enable_thinking": True}}),
        }).encode("utf-8")
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    args.endpoint, data=body,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    method="POST")
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except Exception as exc:  # noqa: BLE001 — the reason has to survive
                # Print what the server said. Swallowing it cost an hour: 576 requests
                # failed and the count alone looked like rate limiting, so the fix
                # attempted was fewer workers. The body said HTTP 402, out of credits —
                # a diagnosis no retry could reach and no counter could show.
                detail = ""
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        detail = exc.read().decode("utf-8", "replace")[:300]
                    except Exception:  # noqa: BLE001
                        detail = ""
                if attempt == 3:
                    with lock:
                        counters["loi"] += 1
                        if counters["loi"] <= 3:
                            print(f"  loi id={record['id']}: "
                                  f"{type(exc).__name__} {exc} {detail}", flush=True)
                    return
                # A refusal the server will repeat — credits, a bad request, a model that
                # does not exist — is not worth three more attempts.
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401,
                                                                           402, 403):
                    with lock:
                        counters["loi"] += 1
                        if counters["loi"] <= 3:
                            print(f"  loi id={record['id']}: HTTP {exc.code} {detail}",
                                  flush=True)
                    return
                time.sleep(3 * (attempt + 1))
        message = ((payload.get("choices") or [{}])[0].get("message") or {})
        content = message.get("content") or ""
        value = parse_number(content)
        with lock:
            if value is None:
                counters["khong parse duoc so"] += 1
            else:
                counters["ok"] += 1
            handle.write(json.dumps(
                {"id": record["id"], "answer": value,
                 "reply": content[-400:]}, ensure_ascii=False) + "\n")
            handle.flush()
            total = sum(counters.values())
            if total % 25 == 0:
                print(f"  {total}/{len(rows)}  {time.time() - started:.0f}s",
                      flush=True)

    threads: list[threading.Thread] = []
    queue = list(rows)
    index = 0

    def runner() -> None:
        nonlocal index
        while True:
            with lock:
                if index >= len(queue):
                    return
                item = queue[index]
                index += 1
            work(item)

    for _ in range(min(args.workers, max(1, len(queue)))):
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    handle.close()

    print()
    for name, count in counters.items():
        print(f"  {name}: {count}")
    print(f"-> {target}")


if __name__ == "__main__":
    main()
