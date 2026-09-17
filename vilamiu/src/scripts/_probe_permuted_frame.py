"""Does the adapter read the tables, or has it learned to always answer `df1`?

`build_sft.py` assembles each prompt as `keys = list(gold_keys)` followed by the
retriever's other candidates, so the gold table is always first and its variable
is always `df1` — 1,562 of 1,562 training pairs. The evaluation inherited that,
which is why `frame` came back at exactly 100%: a constant, not a skill.

At inference the gold table is wherever retrieval puts it. If the adapter learned
"answer from df1" rather than "find the table that has the figure", every number
measured on the held-out set is an overestimate, and by an unknown amount.

The test renames the variables in the already-rendered prompt: `df1` becomes
`df4` and `df4` becomes `df1`, everywhere they appear — the `<table variable=...>`
attributes, the `<schema>` blocks, the variable hint. The tables and the question
are untouched; only the label on the gold table moves. A model that reads the
tables should now answer `df4` with the same cell. A model that memorised the
position will keep saying `df1` and be wrong.

Usage (on the box):
  python scripts/_probe_permuted_frame.py --data artifacts/sft_locate.jsonl \
      --adapter /workspace/lora_locate --limit 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_locate import close, norm, number, parse_target  # noqa: E402

SWAP_WITH = 4


def swap_variables(text: str, a: str = "df1", b: str = f"df{SWAP_WITH}") -> str:
    """Exchange two frame names everywhere, in one pass so neither overwrites."""

    placeholder = "\x00SWAP\x00"
    pattern_a = re.compile(rf"\b{re.escape(a)}\b")
    pattern_b = re.compile(rf"\b{re.escape(b)}\b")
    text = pattern_a.sub(placeholder, text)
    text = pattern_b.sub(a, text)
    return text.replace(placeholder, b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-new", type=int, default=96)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    # Only prompts that actually contain the swap partner can be permuted.
    rows = [r for r in rows if f"df{SWAP_WITH}" in r["messages"][1]["content"]]
    print(f"{len(rows)} held-out pairs that have a df{SWAP_WITH} to swap with")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def complete(system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        with torch.no_grad():
            out = model.generate(
                torch.tensor([ids], device=model.device),
                max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][len(ids):], skip_special_tokens=True)

    stats: Counter[str] = Counter()
    started = time.time()

    for position, row in enumerate(rows, start=1):
        system = row["messages"][0]["content"]
        user = row["messages"][1]["content"]
        gold = parse_target(row["messages"][2]["content"])
        if gold is None:
            continue

        for arm, prompt, want_frame in (
                ("original", user, gold["frame"]),
                ("permuted", swap_variables(user), f"df{SWAP_WITH}")):
            got = parse_target(complete(system, prompt))
            stats[f"{arm}:n"] += 1
            if got is None:
                stats[f"{arm}:unparseable"] += 1
                continue
            if norm(got["frame"]) == norm(want_frame):
                stats[f"{arm}:frame"] += 1
            if norm(got["row"]) == norm(gold["row"]):
                stats[f"{arm}:row"] += 1
            wanted, have = number(gold["value"]), number(got["value"])
            if wanted is not None and have is not None and close(have, wanted):
                stats[f"{arm}:value"] += 1
            # The failure this probe exists to detect: still naming df1 after the
            # gold table has been relabelled.
            if arm == "permuted" and norm(got["frame"]) == "df1":
                stats["permuted:said_df1_anyway"] += 1

        if position % 10 == 0:
            print(f"  {position}/{len(rows)}  {time.time() - started:.0f}s")

    print(f"\n  {'arm':10} {'frame':>7} {'row':>7} {'VALUE':>7} {'unparsed':>9}")
    for arm in ("original", "permuted"):
        n = stats[f"{arm}:n"] or 1
        print(f"  {arm:10} {stats[f'{arm}:frame'] / n:6.1%} "
              f"{stats[f'{arm}:row'] / n:6.1%} {stats[f'{arm}:value'] / n:6.1%} "
              f"{stats[f'{arm}:unparseable'] / n:8.1%}")
    n = stats["permuted:n"] or 1
    print(f"\n  still answered df1 after the swap: "
          f"{stats['permuted:said_df1_anyway']}/{n} = "
          f"{stats['permuted:said_df1_anyway'] / n:.1%}")
    print("\n  A high figure there means the adapter learned the position, not the")
    print("  table, and every held-out number is an overestimate. `row` holding up")
    print("  while `frame` collapses would mean it reads cells fine but cannot")
    print("  choose a table — which is what the training data never asked it to do.")


if __name__ == "__main__":
    main()
