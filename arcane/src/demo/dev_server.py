"""Small-model stand-in for the two demo endpoints, for a machine without 42 GB of VRAM.

Serves the same interfaces the demo pipeline calls in production — an
OpenAI-compatible chat completion and a rerank endpoint — backed by the 0.6B
development tier instead of Qwen3.5-9B and Qwen3-Reranker-8B. The answers are
worse; the contract is identical, which is the point.

    PYTHONPATH=src:. python -m uvicorn demo.dev_server:app --port 8001
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import torch
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


CHAT_MODEL = os.getenv("DEV_CHAT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
RERANK_MODEL = os.getenv("DEV_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
DEVICE = os.getenv("DEV_DEVICE", "cpu")
MAX_NEW_TOKENS = int(os.getenv("DEV_MAX_NEW_TOKENS", "256"))
# float32 is the safe CPU default; a 4B reranker only fits in RAM at bfloat16.
DTYPE = getattr(torch, os.getenv("DEV_DTYPE", "float32"))

_loaded: dict[str, Any] = {}


def _chat_model():
    if "chat" not in _loaded:
        tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=DTYPE).to(DEVICE).eval()
        _loaded["chat"] = (tokenizer, model)
    return _loaded["chat"]


def _rerank_model():
    if "rerank" not in _loaded:
        tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(RERANK_MODEL, dtype=DTYPE).to(DEVICE).eval()
        _loaded["rerank"] = (tokenizer, model, tokenizer.convert_tokens_to_ids("yes"),
                             tokenizer.convert_tokens_to_ids("no"))
    return _loaded["rerank"]


def _first_json_object(text: str) -> dict:
    """Recover the first balanced JSON object a small model emitted around its prose."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for position in range(start, len(text)):
            character = text[position]
            if in_string:
                in_string = not (character == '"' and not escaped)
                escaped = character == "\\" and not escaped
                continue
            if character == '"':
                in_string, escaped = True, False
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:position + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object in model output: {text[:200]!r}")


app = FastAPI(title="ViFinQA dev models")


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": CHAT_MODEL, "object": "model"},
                                       {"id": RERANK_MODEL, "object": "model"}]}


@app.post("/v1/chat/completions")
async def completions(payload: dict) -> dict:
    tokenizer, model = _chat_model()
    prompt = tokenizer.apply_chat_template(payload["messages"], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    content = json.dumps(_first_json_object(text), ensure_ascii=False)
    return {
        "id": "dev-completion", "object": "chat.completion", "created": int(time.time()),
        "model": payload.get("model", CHAT_MODEL),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
    }


@app.post("/rerank")
async def rerank(payload: dict, request: Request) -> dict:
    """Score every document, giving up early if the caller has gone away.

    Inference runs synchronously in this handler, so a cancelled request would
    otherwise keep a CPU busy for minutes and make the next health probe look
    like an outage. Checking between batches is the only interruption point.
    """
    tokenizer, model, yes_id, no_id = _rerank_model()
    query, documents = payload["query"], payload["documents"]
    prompts = [
        f"Judge whether the table answers the query. Answer only yes or no.\n"
        f"<Query>: {query}\n<Document>: {document}\n<Answer>:"
        for document in documents
    ]
    scores = []
    for start in range(0, len(prompts), 8):
        if await request.is_disconnected():
            return {"results": [], "cancelled": True}
        batch = tokenizer(prompts[start:start + 8], return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to(DEVICE)
        with torch.inference_mode():
            logits = model(**batch).logits[:, -1]
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
        scores.extend(torch.softmax(pair, dim=1)[:, 1].tolist())
    return {"results": [{"index": index, "relevance_score": score}
                        for index, score in enumerate(scores)]}
