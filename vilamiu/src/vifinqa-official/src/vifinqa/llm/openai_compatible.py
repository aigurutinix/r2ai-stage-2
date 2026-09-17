
from __future__ import annotations

from json import JSONDecodeError

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
)

from vifinqa.llm.base import LLMError

_MAX_COMPLETION_TOKENS = 8192


def _log_usage(model: str, resp: object) -> None:
    """Append one line of token accounting per call. Local addition; see below."""
    import os

    path = os.environ.get("VIFIN_USAGE_LOG")
    if not path:
        return
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    details = getattr(usage, "completion_tokens_details", None)
    record = {
        "model": model,
        "prompt": getattr(usage, "prompt_tokens", None),
        "completion": getattr(usage, "completion_tokens", None),
        # The number that decides whether to turn thinking off: reasoning tokens
        # are billed as completion but never appear in the returned text.
        "reasoning": getattr(details, "reasoning_tokens", None) if details else None,
    }
    try:
        import json

        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 0,
        max_completion_tokens: int = _MAX_COMPLETION_TOKENS,
        reasoning_effort: str | None = None,
        temperature: float = 0,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._model = model
        self._max_completion_tokens = max_completion_tokens
        self._reasoning_effort = reasoning_effort
        self._temperature = temperature

    def complete(self, *, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            request: dict[str, object] = {
                "model": self._model,
                "messages": messages,
                "max_completion_tokens": self._max_completion_tokens,
                "temperature": self._temperature,
            }
            if self._reasoning_effort is not None:
                request["extra_body"] = {
                    "reasoning": {"effort": self._reasoning_effort, "exclude": True}
                }
            resp = self._client.chat.completions.create(**request)
            # LOCAL ADDITION (not the organisers'): record token usage when
            # VIFIN_USAGE_LOG is set. The response carries `usage` and this class
            # discards it, which left the run's real cost unknowable — the
            # per-record spend came out 7x the estimate and there was no way to
            # tell whether that was reasoning tokens, retries, or prompt size.
            # No effect on behaviour; the variable is unset by default.
            _log_usage(self._model, resp)
        except (AuthenticationError, PermissionDeniedError, NotFoundError) as exc:
            # Invalid credentials, model, or provider affect the whole run; per-item retries waste calls.
            raise LLMError(f"OpenAI-compatible configuration error: {exc}", fatal=True) from exc
        except (APITimeoutError, APIConnectionError, APIError, JSONDecodeError) as exc:
            # The SDK already retried according to max_retries; mark this item and continue.
            raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc
        return resp.choices[0].message.content or ""
