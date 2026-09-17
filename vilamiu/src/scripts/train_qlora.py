"""QLoRA the student model on the generated program pairs.

What this is teaching, in the order it matters:

1. **Emit a program and nothing else.** The base model reasons in prose with no
   `<think>` tags — "Thinking Process: 1. Analyze the Request..." — which is why
   it timed out on every call when used as a generator, and why our own earlier
   measurement put it at 9.4% usable output. That is a formatting failure, and
   formatting is what SFT removes most reliably.
2. **Return a float.** The organisers' own reference ships ANSWER 1.0 against
   EXECUTION 0.3577 because its gold programs end on a raw cell string. Every
   target here ends on a numeric value.
3. **Choose among candidates.** Each prompt carries the retrieval shortlist, not
   just the answer's table, because that is what inference looks like.

Sized for a single 24 GB card. The student is 9B in 4-bit; sequences are long
because the prompt carries several tables, and sequence length is what actually
decides whether this fits.

Usage (on the rented box, with vLLM stopped so the card is free):
  python scripts/train_qlora.py --data artifacts/sft.jsonl --out /workspace/lora
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="/workspace/lora")
    # Was Qwen3.5-9B. Changed on the strength of the closest measurement we have:
    # on the plan/locate task, which asks for structured output exactly as the
    # `sft_locate.jsonl` target does, Qwen2.5-Coder-14B produced 61% usable plans
    # against Qwen3.5-9B's 9.4%. The cause is not a token budget — raising
    # max-tokens from 200 to 1600 moved the first 20 questions from 0% to 45% and
    # the full set not at all. Qwen3.5-9B reasons in bare prose with no `<think>`
    # tags, so nothing can separate its answer from its thinking; `/no_think`, a
    # system instruction and an assistant prefill were all measured and all failed.
    #
    # Its architecture is also what broke the previous run: linear attention on
    # three of every four layers plus an MoE MLP, so Llama-style `target_modules`
    # matched nothing and the adapter was a no-op. Qwen2.5-Coder is a standard
    # stack — the same family the local smoke test verified module discovery on.
    #
    # Both are inside the contest's <=14B open-weight rule.
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--epochs", type=float, default=2.0)
    # ~30 minutes of work at the measured 107 s/step. The number that matters is
    # not how often a checkpoint is written but how much is unrecoverable when
    # the machine disappears, which it does.
    parser.add_argument("--save-steps", type=int, default=16)
    # Measured on the real pairs: median 8179 tokens, max 12319. A prompt carries
    # eight rendered tables, so 8192 truncates half the set — and in SFT the text
    # is prompt+completion, so what gets cut is the *program being taught*. The
    # run would report a healthy loss while learning from targetless prompts.
    # Inference serves at --max-model-len 20000; training has to match it.
    # 16384 OOMs on a 24 GB card: Qwen3.5's `torch_chunk_gated_delta_rule`
    # upcasts its activations to float32, so the linear-attention path costs more
    # than ordinary attention at the same length. Measured on the real pairs the
    # longest is 12319 tokens, so 12288 keeps all but a handful and fits.
    parser.add_argument("--max-len", type=int, default=12288)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--eval-frac", type=float, default=0.05)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorForSeq2Seq)
    from trl import SFTConfig, SFTTrainer

    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"{len(rows)} pairs from {args.data}")
    if len(rows) < 100:
        print("  WARNING: under 100 pairs. LoRA will fit this set rather than "
              "learn the task; treat any gain as unmeasured.")

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # A held-out slice is the only thing that will say whether the run learned
    # the task or memorised the set. At 380 pairs a 5% slice is 19 samples, too
    # few to separate the two, which is why the real run holds out 15%.
    split = max(1, int(len(rows) * args.eval_frac))
    train_rows, eval_rows = rows[split:], rows[:split]
    print(f"  train {len(train_rows)}  eval {len(eval_rows)}")

    # Mask the prompt by building labels ourselves. Trusting trl's
    # prompt/completion split failed twice on Qwen3.5:
    #   1. message lists + default thinking → generation prompt `<think>\n`
    #      is not a prefix of the full chat's `<think>\n\n</think>\n\n…`
    #   2. string split with matching text → BPE retokenises the boundary,
    #      trl logs "Mismatch between tokenized prompt…" for every row and
    #      falls back to training on the whole sequence
    # Tokenising once with `enable_thinking=False` as a *direct* kwarg (not
    # `chat_template_kwargs`, which this tokenizer ignores) gives a real
    # prefix; labels = -100 on that prefix is the mask we actually need.
    def ids_of(messages: list[dict], *, add_generation_prompt: bool = False) -> list[int]:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def render(row: dict) -> dict:
        messages = row["messages"]
        prompt_ids = ids_of(messages[:-1], add_generation_prompt=True)
        full_ids = ids_of(messages)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "chat template broke the prompt prefix; refusal to train unmasked"
            )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    train_ds = Dataset.from_list([render(r) for r in train_rows])
    eval_ds = Dataset.from_list([render(r) for r in eval_rows])

    # skip_prepare_dataset also skips trl's truncator, so enforce the cap here.
    # Drop, don't right-truncate: the tail is the program being taught.
    def under_cap(ds: Dataset) -> Dataset:
        keep = [i for i, ids in enumerate(ds["input_ids"]) if len(ids) <= args.max_len]
        dropped = len(ds) - len(keep)
        if dropped:
            print(f"  dropped {dropped} pairs over the {args.max_len} cap")
        return ds.select(keep)

    train_ds = under_cap(train_ds)
    eval_ds = under_cap(eval_ds)
    print(f"  kept train {len(train_ds)}  eval {len(eval_ds)}")

    # Sanity: how many tokens actually get a gradient. If this is ~0 the mask
    # is wrong and we should stop before burning three hours.
    n_supervised = [sum(1 for x in row["labels"] if x != -100) for row in train_ds]
    print(f"  supervised tokens/pair: median {sorted(n_supervised)[len(n_supervised)//2]}, "
          f"min {min(n_supervised)}, max {max(n_supervised)}")
    if min(n_supervised) < 4:
        raise SystemExit(
            "supervised span is near-empty — the mask ate the program; aborting"
        )

    lengths = [len(row["input_ids"]) for row in train_ds.select(range(min(64, len(train_ds))))]
    lengths.sort()
    over = sum(1 for n in lengths if n > args.max_len)
    print(f"  token length: median {lengths[len(lengths) // 2]}, "
          f"max {lengths[-1]} (cap {args.max_len})")
    print(f"  pairs over the cap: {over}/{len(lengths)}")
    if over:
        # Loud, because the failure is silent: truncation removes the tail, and
        # the tail is the program being taught. Loss keeps falling on prompts
        # whose target is gone.
        print("  WARNING: those pairs lose their completion to truncation and "
              "teach nothing. Raise --max-len or render fewer tables.")

    # trl 1.9 computes the loss with a chunked LM head, and the chunk is where a
    # 24 GB card runs out: `logits = h.float() @ w.float().t()` upcasts to float32
    # over a 152k vocabulary, so one chunk of 256 positions is already ~150 MB and
    # the peak sat at 22.66 of 23.52 GiB before asking for 1.45 GiB more. The
    # model itself is only 10.3 GB — this is the loss, not the weights, which is
    # why watching VRAM during a step looked comfortable right up to the failure.
    #
    # The chunk size is a module constant with no config knob, so it is lowered
    # here. Smaller chunks trade a little throughput for a much lower peak, and
    # the arithmetic is linear: 64 is a quarter of the allocation.
    import trl.trainer.sft_trainer as _sft

    _sft._CHUNKED_LM_HEAD_CHUNK_SIZE = int(os.environ.get("VIFIN_CE_CHUNK", "64"))
    print(f"chunked LM-head chunk size: {_sft._CHUNKED_LM_HEAD_CHUNK_SIZE}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # The model is loaded here rather than left to SFTTrainer's `model_init_kwargs`
    # because `target_modules` has to be read off the real architecture. The
    # previous run hard-coded the Llama names below:
    #
    #     ["q_proj", "k_proj", "v_proj", "o_proj",
    #      "gate_proj", "up_proj", "down_proj"]
    #
    # Qwen3.5-9B runs `torch_chunk_gated_delta_rule` linear attention on three of
    # every four layers and an MoE-style MLP, so most of those names match nothing.
    # PEFT does not treat an empty match as an error, LoRA initialises B to zero,
    # and a zero-B adapter is a perfect no-op — the run trained, saved, and scored
    # `tuned == base` on 57 of 57 questions. Nothing in the loss curve said so.
    # A checkpoint that is already stored in bitsandbytes NF4 carries its own
    # `quantization_config`, and handing it a second one is either rejected or
    # silently re-quantises already-quantised weights. Using the pre-quantised
    # build is not a shortcut here — it is what makes this fit: the bf16 weights
    # are 29.5 GB against a 32 GB disk, the NF4 build is 9.9 GB, and both are the
    # same Qwen2.5-Coder-14B.
    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
    already_quantised = getattr(base_config, "quantization_config", None) is not None
    load_kwargs: dict = {"dtype": torch.bfloat16, "device_map": {"": 0},
                         "trust_remote_code": True}
    if already_quantised:
        print(f"{args.base} ships pre-quantised — using its own config")
    else:
        load_kwargs["quantization_config"] = quant
    model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)

    # Every adaptable leaf, grouped by the suffix PEFT matches on.
    suffixes: dict[str, int] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ in ("Linear", "Linear4bit", "Linear8bitLt"):
            suffixes[name.rsplit(".", 1)[-1]] = suffixes.get(name.rsplit(".", 1)[-1], 0) + 1
    print("\nadaptable linear modules found in", args.base)
    for suffix, count in sorted(suffixes.items(), key=lambda kv: -kv[1]):
        print(f"  {suffix:24s} {count:5d}")

    # Everything except the output head. Naming it explicitly beats "all-linear"
    # because the list is printed above, so what got adapted is on the record
    # instead of being inferred from a parameter count.
    targets = sorted(s for s in suffixes if s != "lm_head")
    if not targets:
        raise SystemExit("no adaptable linear modules — check the base model id")
    print(f"\ntargeting {len(targets)} suffixes: {', '.join(targets)}")

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        # `per_device_eval_batch_size` defaults to 8, so the first eval pass ran
        # eight 12k sequences at once and OOM'd immediately after a training step
        # that had fit at batch 1. `prediction_loss_only` drops the logits the
        # Trainer would otherwise keep: [1, 12288, ~152k] upcast to float32 is
        # several GB per sample and nothing here reads them.
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # trl 1.9's SFTConfig dropped `warmup_ratio`; `warmup_steps` is the
        # surviving knob.
        warmup_steps=5,
        logging_steps=5,
        # Was `save_strategy="epoch"`. A 5.5-hour run then had exactly two moments
        # where its output existed on disk, and the rented box was reclaimed
        # before the adapter could be copied off — the whole run was lost with
        # nothing recoverable. Saving every `--save-steps` optimiser steps caps
        # that exposure at roughly half an hour, and `save_total_limit` keeps the
        # disk from filling with 275 MB adapters.
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="epoch" if eval_rows else "no",
        bf16=True,
        max_length=args.max_len,
        # Activation memory at 8k is what decides whether a 9B fits in 24 GB;
        # without checkpointing it does not.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to=[],
        # Labels are already baked in; ask trl not to retokenise (which would
        # reintroduce the BPE-boundary mismatch and undo the mask).
        dataset_kwargs={"skip_prepare_dataset": True},
        # Without this, Trainer drops `labels` because the model forward signature
        # does not list it, and the run learns nothing (or worse, segfaults in the
        # empty-label path of some flash kernels).
        remove_unused_columns=False,
        # `model_init_kwargs` is gone: it only applies when `model` is a string,
        # and the model is now constructed above so its module names can be read.
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) else None,
        peft_config=lora,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100,
        ),
    )
    # Refuse to spend GPU hours on an adapter that cannot change the output. This
    # is the check whose absence cost the last run: `target_modules` matched
    # nothing, so there were no LoRA layers at all, and the only symptom was a
    # scoring pass where every tuned answer equalled its base answer.
    adapted = sum(1 for name, _ in trainer.model.named_modules()
                  if name.endswith("lora_A.default"))
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\nLoRA layers attached: {adapted}   trainable params: {trainable:,}")
    if adapted == 0 or trainable == 0:
        raise SystemExit(
            "adapter is a no-op — LoRA matched no modules. Training would run to "
            "completion, save cleanly and change nothing.")

    trainer.train()
    trainer.save_model(args.out)

    # After training, B must have moved off zero somewhere. If it has not, the
    # gradients never reached the adapter and the saved weights are still the
    # identity — the same failure as above, arriving one step later.
    moved = sum(1 for name, tensor in trainer.model.named_parameters()
                if "lora_B" in name and bool(tensor.abs().sum() > 0))
    print(f"lora_B tensors that moved off zero: {moved}")
    if moved == 0:
        raise SystemExit("every lora_B is still zero — the adapter learned nothing")
    tokenizer.save_pretrained(args.out)
    print(f"\nadapter written to {args.out}")
    print("serve it with:")
    print(f"  vllm serve {args.base} --enable-lora --lora-modules "
          f"vifin={args.out} --max-model-len 20000 --host 127.0.0.1 --port 18000")


if __name__ == "__main__":
    main()
