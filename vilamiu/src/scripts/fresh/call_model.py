"""Call the local vLLM server on prepared prompts. Nothing else runs here.

The prompts arrive finished from the workstation, so this file has no knowledge of the
corpus, the index or the scoring. That separation is deliberate: a rented box running
its own copy of the rendering code is how a measurement ends up describing stale code
rather than the current one.

Resumes by id, so a killed run continues rather than repeating, and prints transport
failures instead of swallowing them into a summary count.

Usage:
  python call_model.py --prompts prompts.jsonl --out replies.jsonl \
      --url http://127.0.0.1:18000/v1 --model Qwen/Qwen3-14B --workers 24
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def call(url: str, model: str, system: str, user: str, timeout: int) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 160,
        # Qwen3 emits a reasoning block unless told not to; the answer here is one
        # JSON object and the thinking only costs tokens and truncation risk.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    prompts = [json.loads(line) for line in
               Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(prompts)} prompt, {len(done)} da co, {len(todo)} can chay",
          flush=True)
    if not todo:
        return

    handle = out_path.open("a", encoding="utf-8")
    lock = threading.Lock()
    counters = {"ok": 0, "loi": 0}
    started = time.time()

    def work(prompt: dict) -> None:
        try:
            reply = call(args.url, args.model, prompt["system"], prompt["user"],
                         args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                KeyError, json.JSONDecodeError) as error:
            with lock:
                counters["loi"] += 1
                if counters["loi"] <= 5:
                    print(f"  loi id={prompt['id']}: {type(error).__name__}: "
                          f"{str(error)[:160]}", flush=True)
            return
        with lock:
            counters["ok"] += 1
            handle.write(json.dumps({"id": prompt["id"], "reply": reply},
                                    ensure_ascii=False) + "\n")
            handle.flush()
            if counters["ok"] % 25 == 0:
                print(f"  {counters['ok']}/{len(todo)}  "
                      f"{time.time() - started:.0f}s", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    handle.close()
    print(f"xong: ok={counters['ok']} loi={counters['loi']} "
          f"{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
