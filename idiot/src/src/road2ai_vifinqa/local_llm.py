"""Reproducible client for local LLM via Ollama (OpenAI-compatible)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .paths import ARTIFACT_ROOT, PROJECT_ROOT

# ====== CẤU HÌNH OLLAMA ======
DEFAULT_URL = os.environ.get("VIFINQA_BASE_URL", "http://127.0.0.1:11434")
MODEL_SOURCE = os.environ.get("VIFINQA_MODEL_SOURCE", "qwen3:8b")
# =============================


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float


def server_ready(base_url: str = DEFAULT_URL) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_server(*, base_url: str = DEFAULT_URL, timeout: float = 180.0):
    """Ollama đã chạy sẵn → không cần start process."""
    if server_ready(base_url):
        return None
    raise RuntimeError(
        f"Ollama chưa sẵn sàng tại {base_url}. "
        "Hãy chạy 'ollama serve' hoặc mở Ollama app trước."
    )


def chat(
    *,
    system: str,
    user: str,
    base_url: str = DEFAULT_URL,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Completion:
    # Dùng API native của Ollama (hỗ trợ format json tốt hơn)
    payload = json.dumps(
        {
            "model": MODEL_SOURCE,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url}/api/chat",          # ← native endpoint
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.time()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))

    content = result.get("message", {}).get("content", "")
    return Completion(
        content=content,
        prompt_tokens=int(result.get("prompt_eval_count", 0)),
        completion_tokens=int(result.get("eval_count", 0)),
        elapsed_seconds=time.time() - started,
    )



def extract_json(text: str) -> dict[str, object]:
    """Extract the first balanced JSON object from a model response."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in response: {text[:200]!r}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("Unbalanced JSON object in model response")