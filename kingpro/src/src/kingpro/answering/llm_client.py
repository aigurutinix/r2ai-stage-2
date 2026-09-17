"""Client gọi endpoint tương thích OpenAI (vLLM/RunPod/FPT) — không cần thư viện ngoài.

Đổi model = đổi biến môi trường, KHÔNG sửa code:
  KINGPRO_LLM_BASE_URL = https://....runpod.../v1   (hoặc FPT)
  KINGPRO_LLM_API_KEY  = ...
  KINGPRO_LLM_MODEL    = Qwen/Qwen2.5-Coder-14B-Instruct
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import time
import urllib.request
import uuid
from contextlib import contextmanager
from urllib.parse import urlsplit

from kingpro.governance.audit import get_audit_ledger
from kingpro.governance.secure_config import resolve_secret
from kingpro.governance.egress import sanitize_egress


_AUDIT_CONTEXT: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "kingpro_llm_audit_context", default={}
)


@contextmanager
def llm_audit_context(**metadata):
    """Bind request metadata to every nested LLM call without exposing prompts."""
    token = _AUDIT_CONTEXT.set({str(key): value for key, value in metadata.items()})
    try:
        yield
    finally:
        _AUDIT_CONTEXT.reset(token)


def _text_hash(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def endpoint_compliance(base_url=None) -> dict:
    """Return a credential-free endpoint policy report for the live runtime.

    Remote endpoints must use HTTPS and an explicitly allowlisted hostname.
    Loopback vLLM endpoints may use HTTP. A passing host check does not prove
    which weights a remote provider loaded; keep deployment evidence separately.
    """
    base_url = (base_url or os.environ.get("KINGPRO_LLM_BASE_URL", "")).strip()
    configured_hosts = os.environ.get(
        "KINGPRO_ALLOWED_ENDPOINT_HOSTS",
        "127.0.0.1,localhost,::1,api.runpod.ai",
    )
    allowed_hosts = {
        item.strip().lower() for item in configured_hosts.split(",") if item.strip()
    }
    try:
        parsed = urlsplit(base_url)
        host = (parsed.hostname or "").lower()
        scheme = parsed.scheme.lower()
        port = parsed.port
    except (TypeError, ValueError):
        host, scheme, port = "", "", None
    local = host in {"127.0.0.1", "localhost", "::1"}
    secure_transport = scheme == "https" or (local and scheme == "http")
    host_allowed = bool(host and host in allowed_hosts)
    if local:
        provider = "local-vllm"
    elif host == "api.runpod.ai":
        provider = "runpod"
    elif host:
        provider = "allowlisted-remote"
    else:
        provider = "unconfigured"
    return {
        "configured": bool(base_url),
        "scheme": scheme,
        "host": host,
        "port": port,
        "provider": provider,
        "local": local,
        "secure_transport": secure_transport,
        "host_allowed": host_allowed,
        "allowed": bool(base_url and host_allowed and secure_transport),
        "allowed_hosts": sorted(allowed_hosts),
    }


def chat(system: str, user: str, *, base_url=None, api_key=None, model=None,
         temperature: float = 0.0, max_tokens: int = 1024, timeout: float = 150) -> str:
    started = time.perf_counter()
    call_id = uuid.uuid4().hex
    base_url = (base_url or os.environ.get("KINGPRO_LLM_BASE_URL", "")).rstrip("/")
    api_key = api_key or resolve_secret("KINGPRO_LLM_API_KEY", "EMPTY")
    model = model or os.environ.get("KINGPRO_LLM_MODEL", "")
    if not base_url or not model:
        raise RuntimeError("Chưa cấu hình endpoint. Đặt KINGPRO_LLM_BASE_URL + KINGPRO_LLM_MODEL.")
    endpoint = endpoint_compliance(base_url)
    if not endpoint["allowed"]:
        raise RuntimeError(
            "Endpoint model khong dat policy host/transport; "
            "cap nhat KINGPRO_ALLOWED_ENDPOINT_HOSTS sau khi kiem tra nha cung cap."
        )
    safe_system, system_egress = sanitize_egress(system)
    safe_user, user_egress = sanitize_egress(user)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": safe_system},
            {"role": "user", "content": safe_user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    response_text = ""
    error: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        response_text = data["choices"][0]["message"]["content"]
        return response_text
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:180]}"
        raise
    finally:
        context = dict(_AUDIT_CONTEXT.get())
        get_audit_ledger().append(
            "llm.call",
            {
                **context,
                "call_id": call_id,
                "model": model,
                "endpoint_host": endpoint.get("host"),
                "endpoint_provider": endpoint.get("provider"),
                "secure_transport": endpoint.get("secure_transport"),
                "system_sha256": _text_hash(safe_system),
                "user_sha256": _text_hash(safe_user),
                "system_chars": len(safe_system),
                "user_chars": len(safe_user),
                "egress_guard": {
                    "system": system_egress,
                    "user": user_egress,
                },
                "response_chars": len(response_text or ""),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "success": error is None,
                "error": error,
            },
        )


def make_llm_from_env():
    """Trả callable llm_fn(system, user)->text dùng cấu hình môi trường."""
    # Serverless vLLM workers can require more than 45 seconds to cold-start.
    # Keep this below the frontend proxy's 180-second ceiling so a refusal can
    # still reach the browser instead of being turned into a transport error.
    timeout = float(os.environ.get("KINGPRO_LLM_TIMEOUT", "150"))
    return lambda system, user: chat(system, user, timeout=timeout)
