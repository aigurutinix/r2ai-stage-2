"""How much of the SFT loss lands on the program, and how much on the tables.

`train_qlora.py` renders each pair with `apply_chat_template` into one `text` field
and hands it to `SFTConfig(dataset_text_field="text")` with no completion-only
collator and no `assistant_only_loss`. So the loss — and the reported
`eval_mean_token_accuracy` — cover every token, prompt included.

That matters because of what the prompt is: eight rendered financial tables. If the
prompt dominates the token count then both the gradient and the eval number are
mostly about predicting the next cell of a balance sheet, which is repetitive,
easy, and not the task. This measures the ratio so the training log can be read for
what it is.

Characters stand in for tokens. The bias is mild and known: numerals and Vietnamese
diacritics tokenize denser than ASCII code, so a character-based estimate slightly
*understates* how much of the token budget the tables take.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = ROOT / "artifacts" / (sys.argv[1] if len(sys.argv) > 1 else "sft_easy.jsonl")

    shares: list[float] = []
    prompts: list[int] = []
    completions: list[int] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        prompt = sum(
            len(m["content"]) for m in messages if m["role"] in ("system", "user")
        )
        completion = sum(
            len(m["content"]) for m in messages if m["role"] == "assistant"
        )
        if not completion:
            continue
        prompts.append(prompt)
        completions.append(completion)
        shares.append(completion / (prompt + completion))

    n = len(shares)
    print(f"{path.name}: {n} pairs")
    print(f"  prompt     median {statistics.median(prompts):,.0f} chars")
    print(f"  completion median {statistics.median(completions):,.0f} chars")
    print(f"  completion share of the text: median {statistics.median(shares):.2%}, "
          f"max {max(shares):.2%}")
    print(f"\n  => roughly {1 - statistics.median(shares):.0%} of the loss and of "
          f"`eval_mean_token_accuracy` is spent predicting table text,")
    print(f"     not the program. Masking the prompt would point the same compute "
          f"at ~{1 / statistics.median(shares):.0f}x more of the task.")


if __name__ == "__main__":
    main()
