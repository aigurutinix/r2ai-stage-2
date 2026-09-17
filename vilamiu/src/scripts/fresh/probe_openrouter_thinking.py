"""Can reasoning be turned on for qwen/qwen3-14b through OpenRouter?

Enabling reasoning moved this task from 48% to 78% on the offline set, so it is the whole
reason to prefer one runner over another. Self-hosting has it for certain — the flag goes
straight into the chat template. Through OpenRouter it depends on what the provider
exposes, and there are two candidate spellings: OpenRouter's own unified `reasoning`
field, and passing `chat_template_kwargs` through to the backend.

This sends one short request per spelling and reports whether reasoning came back, either
as a `reasoning` field on the message or as a `<think>` block inside the content. One
request each, a few hundred tokens — cheap enough to answer a question that decides where
the rest of the budget goes.

The key is read from .env and never printed.

Usage:
  python scripts/fresh/probe_openrouter_thinking.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3-14b"

QUESTION = ("Một bảng có ô A = 1.234.567 đồng và ô B = 2.345.678 đồng. "
            "Tổng A + B là bao nhiêu triệu đồng? Trả lời ngắn.")


def api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPEN_ROUTER_KEY") and "=" in line:
            return line.split("=", 1)[1].strip()
    raise SystemExit("khong tim thay OPEN_ROUTER_KEY trong .env")


def ask(key: str, label: str, extra: dict) -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": QUESTION}],
        "temperature": 0.0,
        "max_tokens": 600,
    }
    body.update(extra)
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:200]
        print(f"  {label}: HTTP {error.code} — {detail}")
        return
    except Exception as error:  # network, timeout
        print(f"  {label}: loi — {type(error).__name__}")
        return

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    print(f"  {label}:")
    print(f"    truong reasoning : {len(reasoning)} ky tu")
    print(f"    the <think> trong content: {'<think>' in content}")
    print(f"    reasoning_tokens : {details.get('reasoning_tokens')}")
    print(f"    completion_tokens: {usage.get('completion_tokens')}")
    print(f"    dap an           : {content.strip()[:90]!r}")
    print(f"    nha cung cap     : {payload.get('provider')}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = api_key()
    print(f"model: {MODEL}\n")
    ask(key, "khong bat gi (mac dinh)", {})
    ask(key, "reasoning: {enabled: true}", {"reasoning": {"enabled": True}})
    ask(key, "chat_template_kwargs enable_thinking",
        {"chat_template_kwargs": {"enable_thinking": True}})


if __name__ == "__main__":
    main()
