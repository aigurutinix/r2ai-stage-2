"""Run the organisers' question generator over our re-rendered corpus.

Their generator emits `QARecord`, which carries `question`, `answer`,
`relevant_docs`, `relevant_tables`, `pandas_query` and `csv_path` — complete
supervision in exactly the submission format, and the training data a fine-tuned
model needs. It is execution-validated: a program that does not run and reproduce
its own answer never reaches the output, so the generator proposes and the
validator decides.

This wires it to our environment without putting the API key on a command line
or into a log: the key is read from `.env` and passed through the process
environment only.

Usage:
  PYTHONPATH=src python scripts/run_official_generate.py --count 10
  PYTHONPATH=src python scripts/run_official_generate.py --config configs/gen_easy_or.yaml --count 400 --workers 6
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SRC = ROOT / "vifinqa-official" / "src"


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gen_easy_or.yaml")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default="")
    parser.add_argument("--data-root", default="data/official_corpus")
    parser.add_argument("--verbose", action="store_true",
                        help="DEBUG logs with the per-stage reason each attempt "
                             "was discarded")
    parser.add_argument("--usage-log", default="artifacts/llm_usage.jsonl",
                        help="append per-call token counts here; set empty to "
                             "disable")
    parser.add_argument("--base-url", default="",
                        help="OpenAI-compatible endpoint; set to "
                             "http://127.0.0.1:18000/v1 to use a tunnelled vLLM "
                             "instead of OpenRouter")
    args = parser.parse_args()

    env = dict(os.environ)
    if args.base_url:
        # A tunnelled vLLM needs no credential; send a placeholder so the client
        # library does not refuse to construct.
        key, base_url = "local", args.base_url
    else:
        secrets = read_env(ROOT / ".env")
        key = secrets.get("OPEN_ROUTER_KEY") or secrets.get("OPENROUTER_API_KEY", "")
        if not key:
            raise SystemExit("no OpenRouter key found in .env "
                             "(expected OPEN_ROUTER_KEY)")
        base_url = "https://openrouter.ai/api/v1"

    # Their `generation.manual.queue` imports `fcntl`, so the package is
    # Linux-only as shipped. `vendor/win_shims` supplies a no-op advisory lock —
    # correct for a single generation process, and documented in the shim itself.
    path_entries = [str(OFFICIAL_SRC)]
    if os.name == "nt":
        path_entries.append(str(ROOT / "vendor" / "win_shims"))

    env.update({
        "PYTHONPATH": os.pathsep.join(path_entries),
        "PYTHONIOENCODING": "utf-8",
        "OPENAI_URL": base_url,
        "OPENAI_API_KEY": key,
        "DATA_ROOT": args.data_root,
        "COMPANY_META_PATH": "data/code_stock.csv",
        "QUESTIONS_DIR": "data/questions",
        "CACHE_DIR": ".cache/vifinqa",
        "RUNS_DIR": "runs",
        # Token accounting per call, read by scripts/_probe_usage.py. Without it
        # the only visible cost signal is the wallet balance, which cannot say
        # whether the money went to reasoning tokens, retries or prompt size.
        "VIFIN_USAGE_LOG": args.usage_log,
    })

    # Their generator logs the reason each attempt is discarded — bad JSON, wrong
    # report scope, a query that will not execute, a malformed answer — but all of
    # it sits at DEBUG, and `cli.py` exposes no way to turn that on. It matters
    # because the measured rate is 12 LLM calls per kept record against a floor of
    # 2, so five sixths of the spend is on attempts that are thrown away, and the
    # four causes have four different fixes.
    #
    # Configuring the root logger before `main()` runs is enough: their
    # `logging.basicConfig` becomes a no-op once handlers exist, while the lines
    # that pin openai/httpx to WARNING still run, so full request bodies stay out
    # of the log.
    bootstrap = (
        "import logging, sys; "
        "logging.basicConfig(level=logging.DEBUG if %r else logging.INFO, "
        "format='%%(asctime)s %%(levelname)s %%(message)s'); "
        "from vifinqa.cli import main; sys.exit(main() or 0)" % bool(args.verbose)
    )
    command = [
        sys.executable, "-c", bootstrap,
        "generate",
        "--config", args.config,
        "--count", str(args.count),
        "--workers", str(args.workers),
        "--data-root", args.data_root,
    ]
    if args.output:
        command += ["--output", args.output]

    printable = [c for c in command if c != key]
    print("running:", " ".join(printable[3:]))
    print(f"  model from {args.config}, corpus {args.data_root}")
    completed = subprocess.run(command, cwd=ROOT, env=env)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
