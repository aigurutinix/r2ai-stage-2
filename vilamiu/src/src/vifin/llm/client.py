"""Minimal OpenAI-compatible chat client (OpenRouter).

Only open-weight models are permitted by the rules — the constraint is on the
model, not on who hosts it, so serving Qwen3-8B through OpenRouter is fine while
GPT/Gemini are not, wherever they run. `ALLOWED_MODELS` keeps that explicit so a
convenient default cannot quietly disqualify the submission.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Open-weight, <= 14B, published before 2026-06-01.
# Verified present in OpenRouter's catalogue. Every Qwen2.5-Coder build hosted
# there is 32B, over the 14B cap, so a coder-tuned model has to be self-hosted.
ALLOWED_MODELS = {
    "qwen/qwen3-14b": "Qwen3-14B, Apr 2025, 14B",
    "qwen/qwen3-8b": "Qwen3-8B, Apr 2025, 8B",
    "qwen/qwen-2.5-7b-instruct": "Qwen2.5-7B, Sep 2024, 7B",
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B, Jul 2024, 8B",
    # Non-Qwen families, added because every model measured on this task so far
    # has been a Qwen, and the step that is costing us — turning a table we
    # already hold into the right number — is the one where a different
    # pre-training mix is most likely to behave differently. Both are open-weight
    # and inside the size and date limits; confirm against the rules page before
    # a submission depends on either.
    "microsoft/phi-4": "Phi-4, Dec 2024, 14B",
    "google/gemma-3-12b-it": "Gemma-3-12B, Mar 2025, 12B",
    "mistralai/ministral-14b-2512": "Ministral-14B, Dec 2025, 14B",
    "ibm-granite/granite-4.1-8b": "Granite-4.1, 8B",
}

# Self-hosted only. OpenRouter carries no Qwen2.5-Coder build at or below the
# 14B cap (its smallest is 32B), so the strongest permitted code model for this
# task is reachable only by serving it ourselves.
SELF_HOSTED_MODELS = {
    "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ": "Qwen2.5-Coder-14B, Nov 2024, 14B, AWQ",
    # Unquantised weights of the same model. Same release, same size, so the same
    # ruling applies; it is listed separately because QLoRA quantises at load
    # time and therefore trains from the fp16 checkpoint, not the AWQ one.
    "Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen2.5-Coder-14B, Nov 2024, 14B",
    # Half the weights of the 14B, so more of a 24 GB card is left for KV cache
    # and more sequences run at once. Used as the dataset *generator*, where the
    # execution gate filters what it proposes; whether its question-judging is
    # good enough is measured, not assumed — see `_probe_synth_quality.py`.
    "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ": "Qwen2.5-Coder-7B, Nov 2024, 7B, AWQ",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder-7B, Nov 2024, 7B",
    "Qwen/Qwen3-14B-AWQ": "Qwen3-14B, Apr 2025, 14B, AWQ",
    # Unquantised weights of the same release. The cache behind our best
    # submission was generated from these through OpenRouter (`qwen/qwen3-14b`),
    # and every regeneration from the AWQ build came back worse: 89.2% executable
    # against 83.7%, and EXEC 0.3854 against 0.3320. 4-bit quantisation costs
    # code-writing accuracy, so the fp16 checkpoint is listed in its own right.
    # Needs a 48 GB card; 14.8B parameters at fp16 is 29.6 GB of weights alone.
    "Qwen/Qwen3-14B": "Qwen3-14B, Apr 2025, 14B, fp16",
    # Feb/Mar 2026 — inside the 2026-06-01 cutoff, 9B is inside the 14B cap.
    # Release date confirmed by the team lead; it is past this assistant's
    # knowledge cutoff, so absence from this list was an artefact of how the list
    # was compiled, not a rules finding. The vast.ai vLLM template serves it by
    # default, which also saves downloading 9 GB of weights.
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B, Feb/Mar 2026, 9B",
    # Our own QLoRA over those same weights, served by vLLM as `--lora-modules
    # vifin-easy-lora=/workspace/lora_easy`. Named here so the A/B can address it
    # without loosening the allow-list: same permitted base, adapter trained only
    # on pairs we generated ourselves.
    "vifin-easy-lora": "Qwen3.5-9B + our QLoRA on synthetic easy pairs, 9B",
    # The successor adapter. Different base (the code model, which scored 61% to
    # Qwen3.5-9B's 9.4% at emitting structured output) and a different target: it
    # writes the whole program, where `vifin-easy-lora` only wrote a cell
    # address. Trained on `artifacts/sft_mixed.jsonl`, whose class mix is drawn
    # from the exam rather than from what was easy to generate.
    "vifin-mixed": "Qwen2.5-Coder-14B + our QLoRA on exam-matched program pairs, 14B",
    # Trained for question answering over tables with generated Python, which
    # is the shape of this task rather than a general capability we hope
    # transfers. Both are smaller than what we run now (7.6B and 8.2B against
    # 14B) and neither is on OpenRouter, so they can only be measured by
    # serving them. Sizes and dates read from the Hugging Face API.
    "tablegpt/TableGPT2-7B": "TableGPT2, Nov 2024, 7.6B, base Qwen2.5-7B",
    "tablegpt/TableGPT-R1": "TableGPT-R1, Dec 2025, 8.2B, base Qwen3-8B",
}
ALLOWED_MODELS = {**ALLOWED_MODELS, **SELF_HOSTED_MODELS}

DEFAULT_MODEL = "qwen/qwen3-8b"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


@dataclass(slots=True)
class ChatClient:
    api_key: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    # Qwen3 spends part of the budget on <think> before emitting code; at 1200
    # the tail of a multi-table program was being cut mid-string, which surfaced
    # as "unterminated string literal" rather than as a truncation.
    max_tokens: int = 3500
    timeout: int = 180
    disable_reasoning: bool = True
    max_retries: int = 4
    endpoint: str = ENDPOINT

    @classmethod
    def local(cls, model: str, base_url: str = "http://localhost:18000/v1", **kwargs) -> "ChatClient":
        """Talk to a self-hosted OpenAI-compatible server (vLLM).

        Same prompts, same pipeline, same client code — only the model changes,
        so a score difference is attributable to the model and nothing else.
        """

        if model not in ALLOWED_MODELS:
            raise ValueError(f"{model!r} is not on the allow-list: {sorted(ALLOWED_MODELS)}")
        return cls(api_key="local", model=model,
                   endpoint=base_url.rstrip("/") + "/chat/completions", **kwargs)

    @classmethod
    def from_env(cls, root: Path, model: str = DEFAULT_MODEL, **kwargs) -> "ChatClient":
        if model not in ALLOWED_MODELS:
            raise ValueError(
                f"{model!r} is not on the allow-list; the rules require an open-weight "
                f"model <=14B released before 2026-06-01. Allowed: {sorted(ALLOWED_MODELS)}"
            )
        env = load_env(root / ".env")
        key = env.get("OPEN_ROUTER_KEY") or os.environ.get("OPEN_ROUTER_KEY", "")
        if not key:
            raise RuntimeError("OPEN_ROUTER_KEY not found in .env or environment")
        return cls(api_key=key, model=model, **kwargs)

    def complete(self, system: str, user: str, extra: dict | None = None) -> str:
        body_dict = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Per-call sampling overrides. Greedy decoding on a long prompt drops into
        # repetition loops: one program came back as 62,114 characters ending in a
        # wall of spaces. The caller needs a way to ask for a penalised retry
        # without changing sampling for the attempts that are working.
        if extra:
            body_dict.update(extra)
        if self.disable_reasoning and "openrouter" in self.endpoint:
            # Spend the token budget on the program, not on visible reasoning.
            body_dict["reasoning"] = {"enabled": False}
        if self.disable_reasoning and "openrouter" not in self.endpoint:
            # vLLM's Qwen3/3.5 chat template honors this; without it the model
            # fills half of `max_tokens` with `<think>` and truncates the program.
            body_dict["chat_template_kwargs"] = {"enable_thinking": False}
        payload = json.dumps(body_dict).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        # Shared free-tier capacity returns 429 in bursts; without backoff a
        # dozen workers turn a transient limit into most of the batch failing.
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8", "replace")
                # OpenRouter pads a slow provider's reply with keepalive blank
                # lines. When that provider then times out the body is padding
                # and nothing else, so a bare json.loads raises JSONDecodeError
                # outside the HTTP error path and kills the worker thread.
                stripped = raw.strip()
                if not stripped:
                    raise ValueError("empty body (provider keepalive only)")
                body = json.loads(stripped)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"malformed response: {exc}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:200]
                retryable = exc.code in (429, 500, 502, 503, 504)
                if not retryable or attempt == self.max_retries:
                    raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 30.0)
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"network error: {exc}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"no choices in response: {str(body)[:300]}")
        message = choices[0].get("message") or {}
        # A server configured with a reasoning parser routes everything into
        # `reasoning` and leaves `content` null — vLLM's Qwen3 parser did exactly
        # that to Qwen2.5-Coder output, so every program came back empty.
        return (message.get("content") or message.get("reasoning") or "").strip()
