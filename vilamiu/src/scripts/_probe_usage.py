"""What the generation run is actually spending, per call and per kept record.

The wallet says money is leaving; it does not say why. The first cost estimate
for this run came out 7x low ($0.00088/record predicted, $0.0062 measured), and
the wallet alone cannot distinguish the three possible causes — reasoning tokens
billed as completion, retries multiplying the calls, or prompts larger than
sampled. Each has a different fix, so the three have to be told apart.

Reads the per-call log written by the local addition in
`vifinqa-official/src/vifinqa/llm/openai_compatible.py`.

Usage:  python scripts/_probe_usage.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USAGE = ROOT / "artifacts" / "llm_usage.jsonl"
RECORDS = ROOT / "artifacts" / "easy_full.jsonl"
PRICE_IN, PRICE_OUT = 0.12, 0.24  # $/M, qwen3-14b list, fetched 2026-08-12


def main() -> None:
    if not USAGE.exists():
        raise SystemExit(f"no usage log at {USAGE} — is VIFIN_USAGE_LOG set?")
    calls = [
        json.loads(line)
        for line in USAGE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not calls:
        raise SystemExit("usage log is empty — no completed calls yet")

    kept = 0
    if RECORDS.exists():
        kept = sum(1 for line in RECORDS.read_text(encoding="utf-8").splitlines()
                   if line.strip())

    prompt = [c["prompt"] or 0 for c in calls]
    completion = [c["completion"] or 0 for c in calls]
    reasoning = [c["reasoning"] or 0 for c in calls]

    spend = (sum(prompt) * PRICE_IN + sum(completion) * PRICE_OUT) / 1e6

    print(f"{len(calls)} calls, {kept} kept records\n")
    print(f"  prompt      total {sum(prompt):9,d}   median {statistics.median(prompt):7,.0f}")
    print(f"  completion  total {sum(completion):9,d}   median {statistics.median(completion):7,.0f}")
    print(f"  reasoning   total {sum(reasoning):9,d}   median {statistics.median(reasoning):7,.0f}")

    share = sum(reasoning) / sum(completion) if sum(completion) else 0.0
    print(f"\n  reasoning is {share:.0%} of all completion tokens "
          f"(= {sum(reasoning) * PRICE_OUT / 1e6:.3f} $ so far)")

    if kept:
        print(f"\n  calls per kept record : {len(calls) / kept:5.2f}   "
              f"(2.00 = one fact call + one question call, no retries)")
        print(f"  cost per kept record  : ${spend / kept:7.5f}")
        print(f"  2,000 records would cost ${spend / kept * 2000:6.2f}")

    print(f"\n  spend so far: ${spend:.4f}")
    print("\n  If reasoning dominates completion, the lever is turning thinking")
    print("  off. If calls-per-record is well above 2, the lever is validation")
    print("  failures. If the prompt median is far above the 3,096 sampled, the")
    print("  lever is page context. They are different fixes.")


if __name__ == "__main__":
    main()
