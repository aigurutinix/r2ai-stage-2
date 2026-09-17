"""Ask one free-form Vietnamese financial question through the product pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from _env import load_local_env  # noqa: E402

load_local_env(ROOT)

from kingpro.product import ProductService  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="Câu hỏi tài chính tiếng Việt")
    args = parser.parse_args()
    response = ProductService(root=ROOT).ask(" ".join(args.question))
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
