"""Chạy pandas_query an toàn: subprocess riêng + TIMEOUT (baseline không có timeout).

Bê logic đọc CSV của baseline: đọc TOÀN CHUỖI (dtype=str, keep_default_na=False, utf-8-sig),
1 bảng -> biến `df`; nhiều bảng -> dict `dfs`. Code phải gán `result` = 1 số vô hướng.

`python_exe`: trỏ tới env py3.7+pandas1.1.5 để giống MÁY CHẤM (mặc định python hiện tại).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve()

_BLOCKED_NAMES = {
    "__import__", "breakpoint", "compile", "eval", "exec", "globals",
    "input", "locals", "open", "vars",
}
_BLOCKED_IO_ATTRIBUTES = {
    "ExcelFile", "ExcelWriter", "HDFStore", "read_clipboard", "read_csv",
    "read_excel", "read_feather", "read_fwf", "read_gbq", "read_hdf",
    "read_html", "read_json", "read_orc", "read_parquet", "read_pickle",
    "read_sas", "read_spss", "read_sql", "read_sql_query", "read_sql_table",
    "read_stata", "read_table", "read_xml", "to_clipboard", "to_csv",
    "to_excel", "to_feather", "to_gbq", "to_hdf", "to_html", "to_json",
    "to_latex", "to_markdown", "to_orc", "to_parquet", "to_pickle",
    "to_sql", "to_stata", "to_xml",
}


def validate_code(code: str, max_chars: int = 50_000) -> tuple[bool, str | None]:
    """Static safety gate before the isolated child process executes code.

    Pandas remains powerful enough to read/write files even when Python builtins
    are restricted, so the guard explicitly blocks imports, dunder traversal and
    Pandas I/O entry points. ``while`` is rejected; ordinary finite ``for`` loops
    remain available for financial calculations and are bounded by the process
    timeout.
    """
    if len(code) > max_chars:
        return False, f"code vượt giới hạn {max_chars} ký tự"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (dòng {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "không cho phép import trong pandas_query"
        if isinstance(node, (ast.While, ast.AsyncFor, ast.AsyncWith, ast.Await)):
            return False, f"không cho phép cấu trúc {type(node).__name__}"
        if isinstance(node, ast.Name):
            if node.id in _BLOCKED_NAMES or node.id.startswith("__"):
                return False, f"không cho phép tên {node.id!r}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, f"không cho phép thuộc tính {node.attr!r}"
            if node.attr in _BLOCKED_IO_ATTRIBUTES:
                return False, f"không cho phép I/O qua {node.attr!r}"
    return True, None


def run_pandas_code(code: str, csv_paths: dict[str, str], timeout: float = 5.0, python_exe: str | None = None) -> dict:
    """Run one query in an isolated Python child and enforce a wall-clock timeout.

    The parent sends UTF-8 bytes instead of relying on the Windows console code
    page. This keeps Vietnamese code/data stable on Python 3.9 through 3.14 and
    restores the timeout contract that an in-process ``exec`` cannot provide.
    """
    src = _sanitize(code)
    valid, validation_error = validate_code(src)
    if not valid:
        return {"ok": False, "result": None, "error": f"SafetyError: {validation_error}"}

    payload = json.dumps(
        {"code": src, "csv_paths": csv_paths}, ensure_ascii=False
    ).encode("utf-8")
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            [python_exe or sys.executable, str(RUNNER), "--run"],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(0.1, float(timeout)),
            env=child_env,
            creationflags=creationflags,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "result": None,
            "error": f"TimeoutError: pandas_query vượt quá {float(timeout):g} giây",
        }
    except Exception as exc:
        return {"ok": False, "result": None, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"child exit code {completed.returncode}"
        return {"ok": False, "result": None, "error": f"ChildProcessError: {detail[:300]}"}
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "result": None, "error": "ChildProcessError: không có kết quả"}
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"ok": False, "result": None, "error": f"ChildProcessError: JSON không hợp lệ: {stdout[-300:]}"}
    if not isinstance(value, dict) or "ok" not in value:
        return {"ok": False, "result": None, "error": "ChildProcessError: sai hợp đồng kết quả"}
    return value


def _sanitize(code: str) -> str:
    # Qwen3 "thinking": bỏ khối <think>...</think> (và mọi thứ trước </think>)
    if "</think>" in code:
        code = code.split("</think>")[-1]
    lines = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("```"):
            continue
        if s.startswith("import pandas") or s == "import pandas as pd":
            continue
        lines.append(ln)
    return "\n".join(lines)


def _child_main() -> None:
    import builtins as _b  # noqa
    import pandas as pd  # noqa

    # GIỐNG MÁY CHẤM: builtins giới hạn đúng whitelist prompt gốc + CHỈ df (1 bảng)/dfs.
    _WL = ("abs round len min max sum sorted float int str bool list dict set "
           "range enumerate zip all any isinstance").split()
    safe_builtins = {n: getattr(_b, n) for n in _WL}

    spec = json.loads(sys.stdin.read())
    code = _sanitize(spec["code"])
    csv_paths = spec["csv_paths"]
    try:
        # csv_paths: {tên_biến -> đường dẫn}. dfs theo thứ tự evidence; df khi đúng 1 bảng (KHÔNG df1/df2 trần).
        loaded = {var: pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig") for var, p in csv_paths.items()}
        ns = {"pd": pd, "dfs": loaded, "__builtins__": safe_builtins}
        if len(loaded) == 1:
            ns["df"] = next(iter(loaded.values()))
        exec(compile(code, "<pandas_query>", "exec"), ns)
        if "result" not in ns:
            print(json.dumps({"ok": False, "result": None, "error": "Code không gán biến `result`"}))
            return
        res = ns["result"]
        # ép numpy scalar -> python
        if hasattr(res, "item"):
            try:
                res = res.item()
            except Exception:
                pass
        try:
            res = json.loads(json.dumps(res, default=str))
        except Exception:
            res = str(res)
        print(json.dumps({"ok": True, "result": res, "error": None}, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"ok": False, "result": None, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        _child_main()
    else:
        # tự test nhanh
        import tempfile

        d = tempfile.mkdtemp()
        p = Path(d) / "t.csv"
        p.write_text("chi_tieu,gia_tri\nDoanh thu,1.234\nLoi nhuan,567\n", encoding="utf-8-sig")
        code = "result = df[df['chi_tieu']=='Loi nhuan']['gia_tri'].values[0]"
        print("test1 (đọc ô):", run_pandas_code(code, {"t": str(p)}))
        print("test2 (thiếu result):", run_pandas_code("x = 1", {"t": str(p)}))
        print("test3 (timeout):", run_pandas_code("while True:\n  pass", {"t": str(p)}, timeout=2))
        print("test4 (lỗi cột):", run_pandas_code("result = df['khong_co'].values[0]", {"t": str(p)}))
