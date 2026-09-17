"""Measure what generating one easy pair actually costs in tokens.

The choice between buying OpenRouter credit and renting a GPU turns on a number
neither option advertises: how many tokens one training pair consumes. Guessing
it has already gone wrong twice this week — the medium tier was estimated at
50-60 records/hour and delivered 2.7 — so this reads the real prompts off the
real corpus instead.

The easy tier issues **two** calls per record. The first hands over the table's
raw CSV plus the surrounding page text and asks the model to lock a fact; the
second is a short rewrite with no table in it at all. Only the first matters for
cost, and it is charged twice over: once for the attempt that fails validation
and once for the one that survives.

Tokenised with the locally cached Qwen2.5 tokenizer, which shares its BPE vocab
with the Qwen3 generator — close enough to price a decision, not a billing
statement.

Usage:  python scripts/_probe_gen_cost.py [sample_size]
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "official_corpus"
SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 40

# The two system prompts are fixed overhead on every call; measured from the
# source rather than re-pasted, so they cannot drift.
PROMPTS = ROOT / "vifinqa-official/src/vifinqa/generation/prompts/easy.py"


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    source = PROMPTS.read_text(encoding="utf-8")
    fact_system = source.split('_FACT_SYSTEM = """\\', 1)[1].split('"""', 1)[0]
    question_system = source.split('_QUESTION_SYSTEM = """\\', 1)[1].split('"""', 1)[0]
    fact_sys_tokens = count(fact_system)
    question_sys_tokens = count(question_system)

    csv_files = []
    for doc_dir in CORPUS.rglob("*_extracted_tables"):
        found = list(doc_dir.glob("*.csv"))
        if found:
            csv_files.append((doc_dir, found))
        if len(csv_files) >= 400:
            break
    if not csv_files:
        raise SystemExit(f"no corpus under {CORPUS} — run build_official_corpus.py")

    rng = random.Random(20260812)
    rng.shuffle(csv_files)

    csv_tokens: list[int] = []
    context_tokens: list[int] = []
    for doc_dir, found in csv_files[:SAMPLE]:
        table = rng.choice(found)
        csv_tokens.append(count(table.read_text(encoding="utf-8", errors="replace")))
        # The document text sitting beside the tables is what `table_context`
        # slices for `surrounding_pages`; one page before and one after.
        siblings = list(doc_dir.parent.glob("*.txt"))
        if siblings:
            text = siblings[0].read_text(encoding="utf-8", errors="replace")
            pages = text.split("\f") if "\f" in text else [text]
            middle = pages[len(pages) // 2:len(pages) // 2 + 3]
            context_tokens.append(count(" ".join(middle)[:6000]))

    csv_median = statistics.median(csv_tokens)
    context_median = statistics.median(context_tokens) if context_tokens else 0.0

    print(f"sampled {len(csv_tokens)} real tables\n")
    print(f"  fact system prompt      {fact_sys_tokens:7,d} tokens")
    print(f"  question system prompt  {question_sys_tokens:7,d} tokens")
    print(f"  table CSV      median   {csv_median:7,.0f}   "
          f"p90 {statistics.quantiles(csv_tokens, n=10)[8]:,.0f}")
    print(f"  page context   median   {context_median:7,.0f}")

    fact_in = fact_sys_tokens + csv_median + context_median + 60
    fact_out = 220        # the JSON fact, measured shape: query + six short fields
    question_in = question_sys_tokens + 90
    question_out = 60

    per_attempt_in = fact_in + question_in
    per_attempt_out = fact_out + question_out
    print(f"\n  one attempt: {per_attempt_in:,.0f} in + {per_attempt_out:,.0f} out")

    # Not every attempt yields a usable pair. The audited local run kept 78 of
    # 91 records, and the executable-validation gate upstream of that rejects
    # more, so ~2 attempts per shipped pair is the conservative figure.
    ATTEMPTS = 2.0
    print(f"  at {ATTEMPTS:.0f} attempts per kept pair: "
          f"{per_attempt_in * ATTEMPTS:,.0f} in + {per_attempt_out * ATTEMPTS:,.0f} out")

    # Live prices from https://openrouter.ai/api/v1/models on 2026-08-12, not
    # from memory: the first draft of this script guessed $0.06/M input and the
    # real figure is double that, which matters because input is 92% of the bill.
    PRICE_IN, PRICE_OUT = 0.12, 0.24
    print(f"\n  cost for a training set, qwen3-14b at OpenRouter list "
          f"(${PRICE_IN}/M in, ${PRICE_OUT}/M out):")
    for pairs in (500, 1000, 2000, 4000):
        cost = (per_attempt_in * ATTEMPTS * pairs / 1e6 * PRICE_IN
                + per_attempt_out * ATTEMPTS * pairs / 1e6 * PRICE_OUT)
        print(f"    {pairs:5,d} pairs   ${cost:6.2f}")

    # The same job on a rented box, so the two sit side by side. Generation is
    # prefill-heavy, so the GPU has to chew the same tokens the API would bill.
    total_in = per_attempt_in * ATTEMPTS
    total_out = per_attempt_out * ATTEMPTS
    PREFILL, DECODE, RENT = 5000.0, 800.0, 0.40  # tok/s, tok/s, $/hour for a 4090
    print(f"\n  same job on a rented 4090 (AWQ 14B, vLLM batched, ${RENT}/hour):")
    for pairs in (500, 1000, 2000, 4000):
        hours = (total_in * pairs / PREFILL + total_out * pairs / DECODE) / 3600
        print(f"    {pairs:5,d} pairs   {hours:4.1f}h compute   "
              f"${hours * RENT:5.2f}   (+ ~0.7h setup = ${(hours + 0.7) * RENT:5.2f})")

    print("\n  The two land in the same order of magnitude, so price does not")
    print("  decide this. What decides it is that the setup half-hour is the")
    print("  part that has actually gone wrong: torch silently upgrading and")
    print("  breaking vLLM twice, an orphan engine holding 20GB, a BOM turning")
    print("  a bash script into sh. Each of those cost more than the whole bill.")

    print("\n  The output side is trivial. Almost the whole bill is the table")
    print("  being re-sent on every attempt, which is also exactly what a rented")
    print("  GPU would have to prefill — so the comparison is prefill throughput")
    print("  against list price, not 'API vs local' in the abstract.")


if __name__ == "__main__":
    main()
