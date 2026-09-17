"""Download one pinned, pre-cutoff Qwen3.5 checkpoint with SHA256 verification.

The E: workspace is nearly full; the dedicated D: model cache must be approved
before running this script. No existing checkpoint is overwritten or deleted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3885219b6810b007914f3a7950a8d1b469d598a5"
REPO = "unsloth/Qwen3.5-9B-GGUF"
FILENAME = "Qwen3.5-9B-Q4_K_M.gguf"
SIZE = 5_680_522_464
SHA256 = "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
DESTINATION = Path("D:/Road2AI-models") / FILENAME


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    partial = DESTINATION.with_suffix(".gguf.partial")
    if DESTINATION.exists():
        if DESTINATION.stat().st_size != SIZE or digest(DESTINATION) != SHA256:
            raise RuntimeError("Existing destination differs; refusing to overwrite")
    else:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > SIZE:
            raise RuntimeError("Partial file has unexpected size")
        if shutil.disk_usage(DESTINATION.anchor).free < SIZE - offset + 2_000_000_000:
            raise RuntimeError("Insufficient free space for checkpoint plus 2 GB safety margin")
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"
        if offset < SIZE:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise RuntimeError("Server did not honor resume range; partial file retained")
                started = last_print = time.monotonic()
                with partial.open("ab" if offset else "xb") as stream:
                    total = offset
                    for block in response.iter_content(4 * 1024 * 1024):
                        if not block:
                            continue
                        stream.write(block)
                        total += len(block)
                        if total > SIZE:
                            raise RuntimeError("Download exceeds pinned size")
                        now = time.monotonic()
                        if now - last_print >= 20:
                            print(f"Downloaded {total / SIZE:.1%}; {(total-offset)/(now-started)/1e6:.1f} MB/s", flush=True)
                            last_print = now
        if partial.stat().st_size != SIZE or digest(partial) != SHA256:
            raise RuntimeError("Checkpoint verification failed; partial file retained for diagnosis")
        partial.rename(DESTINATION)
    manifest = {
        "base_model": "Qwen/Qwen3.5-9B", "base_release_date": "2026-03-02",
        "quantization_repo": REPO, "revision": REVISION,
        "quantization_commit_date": "2026-03-02T14:08:36Z",
        "filename": FILENAME, "path": str(DESTINATION), "size": SIZE,
        "sha256": SHA256, "verified": True,
        "release_source": "https://github.com/QwenLM/Qwen3.5#news",
    }
    output = ROOT / "artifacts/models/qwen35_9b_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
