"""Generate completions for the held-out pairs, with and without the adapter.

Both arms come out of one model load and one `disable_adapter()` toggle, so the
comparison cannot drift on quantisation, dtype, chat template or sampling. The
prompt is rebuilt exactly as training built it — `enable_thinking=False`, the same
generation prefix — because a base-vs-adapter A/B run under a different prompt
measures the prompt.

Greedy decoding: the question is what the model believes, not what it can be
made to say across samples.

Nothing is scored here. This writes raw completions; `score_eval.py` executes them
against the real tables, because string equality is the wrong metric — a
differently-written program reading the same cell is a success, and an
identical-looking one reading the neighbouring cell is a failure.

Usage (on the box that has the adapter):
  python gen_eval.py --data sft_easy2.jsonl --adapter /workspace/lora_easy_masked \
      --limit 57 --out evalgen.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--limit", type=int, default=57)
    # 96 cut off 20 of 57 generations mid-program, which scores as a crash and
    # reads as a wrong answer. The label-based lookups the model actually writes
    # run to several lines, so the budget has to clear the longest of them.
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--out", default="evalgen.jsonl")
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

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # An adapter whose weights silently fail to attach is indistinguishable from
    # a working one: PEFT initialises B to zero, so an unloaded adapter computes
    # exactly the base model. That is not a hypothetical — it is what produced a
    # full A/B run where `tuned == base` on all 57 questions, which was read as
    # "SFT taught nothing" when it meant "no adapter was ever applied". A loaded,
    # trained adapter has non-zero B somewhere; verify before spending an hour.
    nonzero_b = sum(1 for name, tensor in model.named_parameters()
                    if "lora_B" in name and bool(tensor.abs().sum() > 0))
    total_b = sum(1 for name, _ in model.named_parameters() if "lora_B" in name)
    print(f"base + adapter loaded — lora_B tensors: {nonzero_b}/{total_b} non-zero")
    if total_b == 0 or nonzero_b == 0:
        raise SystemExit(
            f"{args.adapter} contributes nothing: every lora_B is zero. Both arms "
            f"would produce identical output and the A/B would measure nothing.")

    def complete(messages: list[dict]) -> str:
        ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        tensor = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            out = model.generate(
                tensor,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0][len(ids):], skip_special_tokens=True)

    started = time.time()
    with Path(args.out).open("w", encoding="utf-8") as handle:
        for position, row in enumerate(rows, start=1):
            tuned = complete(row["messages"])
            # Same weights, adapter switched off — the only difference between
            # the two arms.
            with model.disable_adapter():
                base = complete(row["messages"])
            handle.write(json.dumps({
                "index": position - 1,
                "meta": row["meta"],
                "tuned": tuned,
                "base": base,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 5 == 0:
                print(f"  {position}/{len(rows)}  {time.time() - started:.0f}s")

    print(f"\nwrote {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
