"""SFT LoRA — R2AI Stage 2 (KINGPRO). HuggingFace THUẦN (transformers.Trainer + peft).
Bản này thay bản unsloth (train_lora_unsloth.py) vì driver container = CUDA 12.4 -> torch cap 2.6+cu124,
mà unsloth/unsloth_zoo bleeding-edge lệch version. HF thuần = coherent + chắc ăn.
Warm-start: Qwen/Qwen2.5-Coder-14B-Instruct. Mask assistant-only (chỉ học phần code).
GIỮ format ĐÚNG contract (SYSTEM + build_user + pandas) — data từ export_chatml.py.

  python scripts/train_lora.py --data build/sft_chatml.jsonl --out outputs/kingpro-lora --epochs 2 --r 32 --bs 2 --ga 8
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    ap.add_argument("--data", default="build/sft_chatml.jsonl")
    ap.add_argument("--out", default="outputs/kingpro-lora")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--ga", type=int, default=8)          # effective batch = bs*ga = 16
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    MAX = a.max_seq
    ds = load_dataset("json", data_files=a.data, split="train")

    def build(ex):
        ids_b, lab_b = [], []
        for conv in ex["conversations"]:
            # prompt = system+user (+ <|im_start|>assistant\n); full = +code+<|im_end|>
            prompt = tok.apply_chat_template(conv[:-1], tokenize=False, add_generation_prompt=True)
            full = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            f = tok(full, add_special_tokens=False)["input_ids"][:MAX]
            plen = min(len(p), len(f))
            lab = [-100] * plen + f[plen:]     # mask prompt -> chỉ học phần code (assistant)
            ids_b.append(f)
            lab_b.append(lab)
        return {"input_ids": ids_b, "labels": lab_b}

    ds = ds.map(build, batched=True, remove_columns=ds.column_names)
    print(f"train: {len(ds)} mẫu | model: {a.model}")

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    peft_cfg = LoraConfig(
        r=a.r, lora_alpha=a.r * 2, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.ga,
        warmup_ratio=0.05, num_train_epochs=a.epochs, learning_rate=a.lr,
        optim="adamw_torch", weight_decay=0.01, lr_scheduler_type="cosine",
        logging_steps=10, save_strategy="epoch", bf16=True, seed=3407, report_to="none")
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"LoRA adapter -> {a.out}. TẢI VỀ MÁY giữ cho private test.")
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(a.out + "-merged", safe_serialization=True)
        tok.save_pretrained(a.out + "-merged")
        print(f"MERGED_OK {a.out}-merged (vLLM serve thẳng)")
    except Exception as e:
        print(f"(bỏ qua merge: {e})")
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
