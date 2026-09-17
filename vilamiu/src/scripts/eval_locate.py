"""A/B the locate-form adapter against its own base, field by field.

The target is `<frame> | <row label> | <column> | <number>`, and the four fields
fail for different reasons and cost different things:

  * **frame**  — wrong table. Nothing downstream can recover it.
  * **row**    — the skill the fine-tune exists to teach. The organisers' own
                 error analysis puts 54.7% of end-to-end failures here, against
                 0.9% on arithmetic: "LLMs không dốt toán, vấn đề là chọn sai ô".
  * **column** — usually a period confusion (this year vs last year).
  * **value**  — what ANSWER scores. It can be right with a wrong label when the
                 figure repeats, and wrong with a right label when the unit
                 conversion slips, so it is reported separately rather than as
                 the summary of the other three.

Both arms come from one model load and one `disable_adapter()` toggle, so the
comparison cannot drift on quantisation, dtype, chat template or sampling.
Greedy decoding throughout: the question is what the model believes, not what it
can be made to say across samples.

Held-out is the first `--limit` rows of the file, which is exactly what
`train_qlora.py` holds out (`rows[:split]`) — evaluating on rows it trained on
would measure memorisation.

Usage (on the box):
  python scripts/eval_locate.py --data artifacts/sft_locate.jsonl \
      --adapter /workspace/lora_locate --limit 78 --out /workspace/evalgen.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

FIELDS = ("frame", "row", "column", "value")


def parse_target(text: str) -> dict[str, str] | None:
    """Read `<frame> | <row> | <column> | <number>` out of a completion.

    The model is trained to emit the line alone, but a base model with no
    fine-tuning will wrap it in prose, so the last line that has three pipes is
    taken rather than the whole reply. Being lenient here is deliberate: the
    comparison is between what the two arms *know*, and refusing to parse the
    base model's format would score formatting, not knowledge.
    """

    body = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    candidates = [ln.strip() for ln in body.splitlines() if ln.count("|") >= 3]
    if not candidates:
        return None
    parts = [p.strip() for p in candidates[-1].split("|")]
    if len(parts) < 4:
        return None
    return {"frame": parts[0], "row": parts[1],
            "column": parts[2], "value": parts[-1]}


def number(text: str) -> float | None:
    cleaned = str(text).strip().replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", cleaned or "x"):
        return None
    cleaned = (cleaned.replace(".", "").replace(",", ".")
               if "," in cleaned else cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return None


def close(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) / scale <= 2e-3) or abs(a - b) <= 0.01


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--limit", type=int, default=78)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--out", default="evalgen_locate.jsonl")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    print(f"{len(rows)} held-out pairs")

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

    # An adapter whose weights fail to attach computes exactly the base model,
    # because LoRA initialises B to zero. That produced a full A/B where every
    # tuned answer equalled its base answer, read as "SFT taught nothing" when it
    # meant "no adapter was applied".
    nonzero = sum(1 for name, tensor in model.named_parameters()
                  if "lora_B" in name and bool(tensor.abs().sum() > 0))
    total = sum(1 for name, _ in model.named_parameters() if "lora_B" in name)
    print(f"lora_B non-zero: {nonzero}/{total}")
    if total == 0 or nonzero == 0:
        raise SystemExit("adapter contributes nothing — both arms would be identical")

    def complete(messages: list[dict]) -> str:
        ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        with torch.no_grad():
            out = model.generate(
                torch.tensor([ids], device=model.device),
                max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0][len(ids):], skip_special_tokens=True)

    tally: dict[str, Counter[str]] = {arm: Counter() for arm in ("base", "tuned")}
    # Scored again over only the rows where *both* arms produced a parseable
    # line. The headline base-vs-tuned gap is mostly formatting — an untuned
    # model does not know this output shape exists — and formatting is the
    # cheapest thing a fine-tune can teach and the least worth 2.7 GPU-hours.
    # Restricting to rows both arms could answer isolates what the adapter knows
    # from what it merely learned to type. This project has been fooled by the
    # other kind of number before: a change that lifted runnable programs from
    # 52.4% to 76.9% moved the score by two questions.
    both: dict[str, Counter[str]] = {arm: Counter() for arm in ("base", "tuned")}
    started = time.time()

    with Path(args.out).open("w", encoding="utf-8") as handle:
        for position, row in enumerate(rows, start=1):
            gold = parse_target(row["messages"][2]["content"])
            tuned_text = complete(row["messages"])
            with model.disable_adapter():
                base_text = complete(row["messages"])

            parsed = {arm: parse_target(text)
                      for arm, text in (("tuned", tuned_text), ("base", base_text))}
            comparable = all(parsed[arm] is not None for arm in parsed) and gold is not None

            for arm in ("tuned", "base"):
                got = parsed[arm]
                counter = tally[arm]
                if got is None:
                    counter["unparseable"] += 1
                    continue
                counter["parsed"] += 1
                if gold is None:
                    continue
                hits = {f: norm(got[f]) == norm(gold[f])
                        for f in ("frame", "row", "column")}
                want, have = number(gold["value"]), number(got["value"])
                hits["value"] = (want is not None and have is not None
                                 and close(have, want))
                hits["cell_exact"] = all(hits[f] for f in ("frame", "row", "column"))
                for key, hit in hits.items():
                    if hit:
                        counter[key] += 1
                        if comparable:
                            both[arm][key] += 1
            if comparable:
                both["tuned"]["n"] += 1
                both["base"]["n"] += 1

            handle.write(json.dumps({
                "index": position - 1, "meta": row["meta"],
                "gold": row["messages"][2]["content"],
                "tuned": tuned_text, "base": base_text,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 10 == 0:
                print(f"  {position}/{len(rows)}  {time.time() - started:.0f}s")

    n = len(rows) or 1
    print(f"\n  {'arm':8} {'parsed':>7} {'frame':>7} {'row':>7} {'column':>7} "
          f"{'cell':>7} {'VALUE':>7}")
    for arm in ("base", "tuned"):
        t = tally[arm]
        print(f"  {arm:8} {t['parsed'] / n:6.1%} {t['frame'] / n:6.1%} "
              f"{t['row'] / n:6.1%} {t['column'] / n:6.1%} "
              f"{t['cell_exact'] / n:6.1%} {t['value'] / n:6.1%}")

    shared = both["tuned"]["n"] or 0
    if shared:
        print(f"\n  on the {shared} rows BOTH arms parsed — formatting removed:")
        print(f"  {'arm':8} {'frame':>7} {'row':>7} {'column':>7} {'cell':>7} "
              f"{'VALUE':>7}")
        for arm in ("base", "tuned"):
            t = both[arm]
            print(f"  {arm:8} {t['frame'] / shared:6.1%} {t['row'] / shared:6.1%} "
                  f"{t['column'] / shared:6.1%} {t['cell_exact'] / shared:6.1%} "
                  f"{t['value'] / shared:6.1%}")
        print("\n  This block, not the one above, says whether the adapter learned")
        print("  to read. If `row` is flat here while the headline gap is large,")
        print("  2.7 GPU-hours bought an output format.")

    print("\n  `row` is the number to read: it is the skill the target was chosen")
    print("  to teach, and the one the organisers measured as 54.7% of all")
    print("  end-to-end errors. `VALUE` is what ANSWER scores. A large gap")
    print("  between tuned and base on `parsed` alone would mean the fine-tune")
    print("  taught formatting rather than reading — worth knowing, worth little.")


if __name__ == "__main__":
    main()
