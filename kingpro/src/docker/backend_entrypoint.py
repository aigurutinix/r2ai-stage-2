"""Fail-fast container preflight before the KINGPRO API accepts traffic."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("KINGPRO_ROOT", "/app")).resolve()
DEFAULT_REPLAY_SHA256 = "E19E4748DDAFFAD28112292EAE17E9296F85BBC99265C9D52EEB27E8F8539C85"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main(argv: list[str]) -> int:
    required = (
        ROOT / "build" / "catalog.jsonl",
        ROOT / "build" / "bm25" / "params.index.json",
        ROOT / "build" / "bm25" / "vocab.index.json",
        ROOT / "build" / "tables",
        ROOT / "data" / "code_stock.csv",
        ROOT / "sub_v297_scope2" / "submission.json",
        ROOT / "outputs",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(json.dumps({"status": "error", "stage": "container_preflight", "missing": missing}), flush=True)
        return 78

    replay = ROOT / "sub_v297_scope2" / "submission.json"
    actual = sha256(replay)
    expected = os.environ.get("KINGPRO_EXPECTED_REPLAY_SHA256", DEFAULT_REPLAY_SHA256).strip().upper()
    if expected and actual != expected:
        print(
            json.dumps(
                {
                    "status": "error",
                    "stage": "container_preflight",
                    "reason": "replay_sha256_mismatch",
                    "expected": expected,
                    "actual": actual,
                }
            ),
            flush=True,
        )
        return 78

    probe = ROOT / "outputs" / ".container-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "stage": "container_preflight",
                    "reason": "outputs_not_writable",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ),
            flush=True,
        )
        return 78

    if not argv:
        print("container preflight passed; no command supplied", flush=True)
        return 0

    print(
        json.dumps(
            {
                "status": "ok",
                "stage": "container_preflight",
                "replay_sha256": actual,
                "command": argv,
            }
        ),
        flush=True,
    )
    os.chdir(ROOT)
    os.execvp(argv[0], argv)
    return 127  # pragma: no cover - os.execvp replaces this process


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
