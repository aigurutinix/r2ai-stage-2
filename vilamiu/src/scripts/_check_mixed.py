"""Last checks on the blended set before it costs GPU hours.

Two things can still ruin it after the mix is right:

* **Length.** Training capped at 12,288 tokens because Qwen3.5-9B's linear
  attention path forces float32 activations and 16k did not fit 24 GB. In SFT the
  text is prompt+completion, so a pair over the cap is not shortened, it is
  *truncated* — and the tail is the program being taught. Measured in characters
  against the file whose token distribution is already known.
* **The target itself.** A blended pair is only as good as its assistant turn.
  One multi-cell pair is printed whole so the program can be read against the
  tables it is supposed to read.
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> list[dict]:
    path = ROOT / "artifacts" / name
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def lengths(pairs: list[dict]) -> list[int]:
    return [sum(len(m["content"]) for m in p["messages"]) for p in pairs]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"{'file':<24}{'pairs':>7}{'median':>9}{'p90':>9}{'max':>9}")
    reference = None
    for name in ("sft_prog2.jsonl", "sft_shapes.jsonl", "sft_mixed.jsonl"):
        pairs = load(name)
        sizes = sorted(lengths(pairs))
        if name == "sft_prog2.jsonl":
            reference = statistics.median(sizes)
        print(f"{name:<24}{len(pairs):>7}{statistics.median(sizes):>9.0f}"
              f"{sizes[int(len(sizes) * 0.9)]:>9}{sizes[-1]:>9}")

    mixed = load("sft_mixed.jsonl")
    sizes = sorted(lengths(mixed))
    # `sft_prog2.jsonl` had a median of 8,179 tokens at this character median, so
    # the ratio carries over as long as the text is the same kind of text.
    ratio = 8179 / reference if reference else 0
    print(f"\nchars-per-token implied by the earlier run: {1 / ratio:.2f}")
    print(f"blended median ≈ {statistics.median(sizes) * ratio:,.0f} tokens, "
          f"p90 ≈ {sizes[int(len(sizes) * 0.9)] * ratio:,.0f}, "
          f"max ≈ {sizes[-1] * ratio:,.0f}")
    over = sum(1 for s in sizes if s * ratio > 12288)
    print(f"pairs over the 12,288 cap: {over}/{len(sizes)} = {over / len(sizes):.1%}")

    print("\ncompletion length by shape (the part that gets a gradient):")
    by_shape: dict[str, list[int]] = collections.defaultdict(list)
    for pair in mixed:
        shape = pair["meta"].get("shape") or "one-cell"
        by_shape[shape].append(len(pair["messages"][2]["content"]))
    for shape, values in sorted(by_shape.items()):
        print(f"  {shape:<18}{len(values):>6} pairs, median {statistics.median(values):>5.0f} chars")

    multi = next(p for p in mixed
                 if p["meta"].get("shape") in ("extreme", "argmax_year"))
    print("\n" + "=" * 70)
    print(f"one {multi['meta']['shape']} pair, target in full:")
    print(f"Q: {multi['meta']['question']}")
    print(f"answer: {multi['meta']['answer']}   frames in prompt: "
          f"{multi['meta']['frames']}")
    print("assistant turn:")
    for line in multi["messages"][2]["content"].splitlines():
        print(f"    {line}")


if __name__ == "__main__":
    main()
