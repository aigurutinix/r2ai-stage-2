"""Call OpenRouter for table-reading prompts; save raw JSON replies.

Qwen3 on OpenRouter puts the answer in `content` only when thinking is enabled
and max_tokens is large enough. With thinking off, `content` is often empty while
`reasoning` still consumes tokens — those rows must not be saved as success.

Resumes by id; skips rows whose reply is empty or unparseable (so --retry-empty
can re-run only the wasted calls).

Usage:
  python scripts/fresh/run_tab.py --prompts artifacts/fresh/prompts_tab_rev.jsonl \\
      --out artifacts/fresh/replies_tab_rev.jsonl
  python scripts/fresh/run_tab.py ... --retry-empty   # re-call ids with bad replies
  python scripts/fresh/run_tab.py ... --smoke 5       # probe before a full batch
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_model import parse  # noqa: E402

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPEN_ROUTER_KEY") and "=" in line:
            return line.split("=", 1)[1].strip()
    raise SystemExit("khong tim thay OPEN_ROUTER_KEY")


def message_text(message: dict) -> str:
    content = (message.get("content") or "").strip()
    if content:
        return content
    return (message.get("reasoning") or "").strip()


def good_reply(text: str) -> bool:
    return parse(text) is not None


def load_existing(path: Path, *, retry_empty: bool) -> dict[int, str]:
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        qid = record["id"]
        reply = str(record.get("reply") or "")
        if retry_empty or not good_reply(reply):
            continue
        out[qid] = reply
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="qwen/qwen3-14b")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--retry-empty", action="store_true",
                        help="re-call ids whose saved reply is empty or bad JSON")
    parser.add_argument("--smoke", type=int, default=0,
                        help="run only N prompts first; abort if none parse")
    args = parser.parse_args()

    key = api_key()
    rows = [json.loads(line) for line in
            (ROOT / args.prompts).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    target = ROOT / args.out
    kept = load_existing(target, retry_empty=args.retry_empty)
    todo = [r for r in rows if r["id"] not in kept]
    if args.smoke:
        todo = todo[: args.smoke]
        print(f"SMOKE: {len(todo)} prompt", flush=True)
    else:
        print(f"{len(rows)} prompt, {len(kept)} da co hop le, {len(todo)} can chay",
              flush=True)
    if not todo:
        return

    lock = threading.Lock()
    counters = {"ok": 0, "rong": 0, "loi": 0}
    results: dict[int, str] = dict(kept)
    started = time.time()

    def work(record: dict) -> tuple[int, str | None]:
        body = json.dumps({
            "model": args.model,
            "messages": [
                {"role": "system", "content": record["system"]},
                {"role": "user", "content": record["user"]},
            ],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        }).encode("utf-8")
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    ENDPOINT, data=body,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=240) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                message = ((payload.get("choices") or [{}])[0].get("message") or {})
                text = message_text(message)
                if good_reply(text):
                    return record["id"], text
                if not text.strip():
                    time.sleep(2 * (attempt + 1))
                    continue
                # Has text but no JSON — one more try, then give up.
                time.sleep(2)
            except Exception:
                if attempt == 3:
                    return record["id"], None
                time.sleep(3 * (attempt + 1))
        return record["id"], None

    threads: list[threading.Thread] = []
    queue = list(todo)
    index = 0

    def runner() -> None:
        nonlocal index
        while True:
            with lock:
                if index >= len(queue):
                    return
                item = queue[index]
                index += 1
            qid, reply = work(item)
            with lock:
                if reply is not None:
                    results[qid] = reply
                    counters["ok"] += 1
                elif reply is None:
                    counters["loi"] += 1
                else:
                    counters["rong"] += 1
                done_now = counters["ok"] + counters["loi"] + counters["rong"]
                if counters["ok"] and counters["ok"] % 25 == 0:
                    print(f"  ok={counters['ok']}/{len(todo)}  "
                          f"{time.time() - started:.0f}s", flush=True)

    for _ in range(min(args.workers, max(1, len(queue)))):
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    if args.smoke and counters["ok"] == 0:
        raise SystemExit(
            f"SMOKE FAIL: 0/{len(todo)} parse duoc JSON — dung batch, sua config truoc")

    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = {r["id"]: r for r in rows}
    with target.open("w", encoding="utf-8") as handle:
        for qid in sorted(ordered):
            if qid in results:
                handle.write(json.dumps({"id": qid, "reply": results[qid]},
                                        ensure_ascii=False) + "\n")

    print(f"xong: ok={counters['ok']} rong={counters['rong']} loi={counters['loi']} "
          f"ghi {len(results)} dong -> {target}")


if __name__ == "__main__":
    main()
