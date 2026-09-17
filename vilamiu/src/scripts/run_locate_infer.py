"""Answer the 1,012 exam questions with the tuned adapter.

The prompts were built on the laptop by `build_infer_prompts.py` through the same
renderer, table budget and system prompt as training, so the adapter is being
asked the question it was taught. This script only generates.

Batched deliberately. One-at-a-time generation measured 8.2 s per prompt during
evaluation, which is 2.3 hours for 1,012 questions and $0.80 of rented GPU spent
waiting on a mostly idle card. The prompts run 12k tokens, so the batch is small
and set from the command line rather than guessed — an OOM three quarters of the
way through costs more than the batching saves.

Left-padding, because these are decoder-only: pad on the right and the model
continues from padding tokens instead of from the question.

Writes one line per question as it completes, so an interrupted run keeps what it
finished. That is not hypothetical — a finished training run was lost this week
when the machine went away before its output was copied off.

Usage:
  python scripts/run_locate_infer.py --prompts artifacts/infer_prompts.jsonl \
      --adapter /workspace/lora_locate --out /workspace/locate_answers.jsonl \
      --batch 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [
        json.loads(line)
        for line in Path(args.prompts).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    # Resume: skip anything already written, so a killed run continues instead of
    # starting over.
    done: set[int] = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} prompts, {len(done)} already done, {len(todo)} to go")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # An adapter whose weights fail to attach computes the base model exactly,
    # because LoRA initialises B to zero — a full run once came back identical to
    # its baseline for that reason and was read as "the training taught nothing".
    nonzero = sum(1 for name, tensor in model.named_parameters()
                  if "lora_B" in name and bool(tensor.abs().sum() > 0))
    print(f"lora_B non-zero: {nonzero}")
    if nonzero == 0:
        raise SystemExit("adapter contributes nothing — refusing to spend a GPU hour")

    # Longest first: a batch costs what its longest member costs, so grouping
    # similar lengths wastes less, and any OOM happens in the first minute rather
    # than an hour in.
    todo.sort(key=lambda r: -len(r["messages"][1]["content"]))

    started = time.time()
    written = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for index in range(0, len(todo), args.batch):
            chunk = todo[index:index + args.batch]
            texts = [
                tokenizer.apply_chat_template(
                    r["messages"], tokenize=False, add_generation_prompt=True)
                for r in chunk
            ]
            batch = tokenizer(texts, return_tensors="pt", padding=True,
                              truncation=False).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **batch, max_new_tokens=args.max_new, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id)
            prompt_len = batch["input_ids"].shape[1]
            for row, sequence in zip(chunk, out):
                reply = tokenizer.decode(sequence[prompt_len:],
                                         skip_special_tokens=True)
                handle.write(json.dumps({
                    "id": row["id"], "meta": row["meta"], "reply": reply,
                }, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            elapsed = time.time() - started
            rate = written / elapsed if elapsed else 0
            remaining = (len(todo) - written) / rate if rate else 0
            print(f"  {written}/{len(todo)}  {elapsed:.0f}s  "
                  f"eta {remaining / 60:.0f}m", flush=True)

    print(f"\nwrote {written} answers -> {out_path}")


if __name__ == "__main__":
    main()
