"""What this machine can and cannot do for a fine-tuning run.

Splitting the work correctly matters more than starting it fast. Dataset
generation is API-bound and CPU-only, so it can run here for hours while nothing
else is blocked. Training a 9-14B model is VRAM-bound and cannot.

Usage:  python scripts/_env_check.py
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys

WANTED = [
    ("torch", "training and local inference"),
    ("transformers", "model loading"),
    ("peft", "LoRA adapters"),
    ("bitsandbytes", "4-bit quantised training"),
    ("datasets", "SFT data pipeline"),
    ("trl", "SFT trainer"),
    ("accelerate", "training launcher"),
    ("sentence_transformers", "bge-m3 embeddings, needed by the organisers' generator"),
    ("typer", "organisers' CLI"),
    ("pydantic", "organisers' schemas"),
    ("pydantic_settings", "organisers' config"),
    ("bm25s", "organisers' retrieval"),
    ("underthesea", "Vietnamese tokenisation"),
    ("openai", "organisers' LLM client"),
    ("pandas", "everything"),
    ("yaml", "organisers' config files"),
    ("tenacity", "organisers' retry logic"),
]


def main() -> None:
    print(f"python {sys.version.split()[0]}")

    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
        print(f"gpu: {out}")
    else:
        print("gpu: nvidia-smi not found")

    print("\npackages:")
    missing = []
    for name, why in WANTED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            print(f"  {name:24s} {version:12s} {why}")
        except Exception:
            missing.append(name)
            print(f"  {name:24s} {'MISSING':12s} {why}")

    if missing:
        print(f"\ninstall: pip install {' '.join(missing)}")

    try:
        import torch  # noqa: PLC0415
        print(f"\ntorch cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  vram free/total: {free / 1e9:.1f} / {total / 1e9:.1f} GB")
    except Exception as exc:
        print(f"\ntorch unavailable: {exc}")


if __name__ == "__main__":
    main()
