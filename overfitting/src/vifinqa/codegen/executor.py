"""Sandboxed-ish execution of generated pandas code.

- expression code -> eval; script code -> exec, must define `answer`
- coerces numpy/pandas scalars & len-1 Series to float
- POSIX (Kaggle/Linux): hard timeout via fork; Windows: inline with try/except
  (rule-generated code is short & safe there)
"""
from __future__ import annotations

import os
import re
import signal
import traceback

import numpy as np
import pandas as pd

BANNED = ("import os", "import sys", "import subprocess", "importlib", "open(",
          "__", "eval(", "exec(", "system(", "popen", "shutil", "pathlib",
          "requests", "urllib", "socket", "pickle", "to_csv", "to_pickle",
          "read_csv", "read_pickle", "input(", "while True")


def check_safe(code: str) -> str | None:
    low = code.lower()
    for b in BANNED:
        if b in low:
            return b
    return None


def _coerce_scalar(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return f if f == f and abs(f) != float("inf") else None
    if isinstance(v, pd.Series):
        return _coerce_scalar(v.iloc[0]) if len(v) == 1 else None
    if isinstance(v, pd.DataFrame):
        return _coerce_scalar(v.iloc[0, 0]) if v.size == 1 else None
    if isinstance(v, str):
        try:
            return float(v.replace(",", ""))
        except ValueError:
            return None
    if isinstance(v, np.ndarray):
        return _coerce_scalar(v.item()) if v.size == 1 else None
    return None


def _run_inline(code: str, dfs: dict[str, pd.DataFrame]) -> dict:
    ns = {"pd": pd, "np": np, "round": round, "abs": abs, "float": float,
          "int": int, "str": str, "len": len, "min": min, "max": max,
          "sum": sum, "sorted": sorted, "__builtins__": {}}
    ns.update({k: v.copy() for k, v in dfs.items()})
    try:
        try:
            compiled = compile(code, "<gen>", "eval")
            val = eval(compiled, ns)  # noqa: S307 - namespace is restricted
        except SyntaxError:
            exec(compile(code, "<gen>", "exec"), ns)  # noqa: S102
            val = ns.get("answer")
        f = _coerce_scalar(val)
        if f is None:
            return {"status": "not_scalar", "value": None,
                    "error": f"result is not a numeric scalar: {type(val).__name__}"}
        return {"status": "ok", "value": f, "error": None}
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()
        return {"status": "error", "value": None,
                "error": f"{type(e).__name__}: {e} | {tb[-1] if tb else ''}"[:500]}


def run_code(code: str, dfs: dict[str, pd.DataFrame], timeout: int = 10) -> dict:
    code = code.strip()
    if not code:
        return {"status": "error", "value": None, "error": "empty code"}
    bad = check_safe(code)
    if bad:
        return {"status": "unsafe", "value": None, "error": f"banned token: {bad}"}
    if os.name == "posix" and timeout:
        return _run_with_alarm(code, dfs, timeout)
    return _run_inline(code, dfs)


class _Timeout(Exception):
    pass


def _run_with_alarm(code, dfs, timeout) -> dict:
    def handler(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        return _run_inline(code, dfs)
    except _Timeout:
        return {"status": "timeout", "value": None, "error": f"timeout>{timeout}s"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(llm_text: str) -> str:
    m = CODE_BLOCK_RE.search(llm_text)
    code = (m.group(1) if m else llm_text).strip()
    # drop chatter lines before the first assignment/df usage if no block found
    return code
