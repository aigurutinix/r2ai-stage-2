"""Opt-in, bounded Qwen reasoning client; never use thoughts as an answer.

The legacy fast client is unchanged. This client is for source audits and
requires a llama-server started with a finite --reasoning-budget.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReasoningCompletion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    finish_reason: str
    reasoning_characters: int


class IncompleteCompletion(ValueError):
    """A truncated or reasoning-only response is not a usable prediction."""


def chat(
    *,
    system: str,
    user: str,
    model: str,
    base_url: str,
    max_tokens: int = 4096,
    seed: int = 20260827,
    timeout: float = 900,
) -> ReasoningCompletion:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system.rstrip()},
            {"role": "user", "content": user.rstrip()},
        ],
        "chat_template_kwargs": {"enable_thinking": True},
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    choice = result["choices"][0]
    message = choice["message"]
    finish = str(choice.get("finish_reason", ""))
    content = str(message.get("content") or "").strip()
    reasoning = str(message.get("reasoning_content") or "")
    if finish != "stop":
        raise IncompleteCompletion(f"Unfinished generation: finish_reason={finish!r}")
    # Some server parsers keep thought tags in content. Only accept text after
    # a closed reasoning block, and never extract JSON from inside that block.
    if "<think>" in content:
        if "</think>" not in content:
            raise IncompleteCompletion("Reasoning block was not closed")
        thoughts, content = content.split("</think>", 1)
        reasoning += thoughts
        content = content.strip()
    if not content:
        raise IncompleteCompletion("No final answer; reasoning is not a substitute")
    usage = result.get("usage", {})
    return ReasoningCompletion(
        content=content,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        elapsed_seconds=time.perf_counter() - started,
        finish_reason=finish,
        reasoning_characters=len(reasoning),
    )
