"""SFT LoRA — R2AI Stage 2 (KINGPRO). Unsloth. Chạy trên FPT Cloud GPU VM (>=40GB VRAM).
Warm-start mặc định: unsloth/Qwen2.5-Coder-14B-Instruct. Thử OmniSQL: --model seeklhy/OmniSQL-14B.
Data: build/sft_chatml.jsonl (từ export_chatml.py). GIỮ format ĐÚNG contract (SYSTEM + build_user + pandas).

  python scripts/train_lora.py --data build/sft_chatml.jsonl --out outputs/kingpro-lora --epochs 2
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-Coder-14B-Instruct",
                    help="warm-start. Thử: seeklhy/OmniSQL-14B (đã biết suy luận SQL)")
    ap.add_argument("--data", default="build/sft_chatml.jsonl")
    ap.add_argument("--out", default="outputs/kingpro-lora")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--r", type=int, default=32)          # FinStat2SQL dùng 64; 32 an toàn
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ga", type=int, default=4)          # effective batch = bs*ga = 16
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA nếu VRAM < 40GB")
    a = ap.parse_args()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import DataCollatorForSeq2Seq

    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model, max_seq_length=a.max_seq, dtype=None, load_in_4bit=a.load_4bit)
    model = FastLanguageModel.get_peft_model(
        model, r=a.r, lora_alpha=a.r * 2, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)
    tok = get_chat_template(tok, chat_template="qwen-2.5")

    ds = load_dataset("json", data_files=a.data, split="train")   # data đã {role,content} chuẩn -> khỏi standardize_sharegpt
    def fmt(ex):
        return {"text": [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                         for c in ex["conversations"]]}
    ds = ds.map(fmt, batched=True)
    print(f"train: {len(ds)} mẫu | model: {a.model}")

    # max_seq_length/dataset_text_field/packing NẰM TRONG SFTConfig (TRL mới); tokenizer có fallback processing_class
    cfg = SFTConfig(
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.ga,
        warmup_ratio=0.05, num_train_epochs=a.epochs, learning_rate=a.lr,
        optim="paged_adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine",
        logging_steps=10, seed=3407, output_dir=a.out, report_to="none", save_strategy="epoch",
        max_seq_length=a.max_seq, dataset_text_field="text", packing=False, dataset_num_proc=2)
    _kw = dict(model=model, train_dataset=ds,
               data_collator=DataCollatorForSeq2Seq(tokenizer=tok), args=cfg)
    try:
        trainer = SFTTrainer(processing_class=tok, **_kw)   # TRL mới (>=0.12)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tok, **_kw)          # TRL/Unsloth cũ
    # Qwen ChatML markers — BẮT BUỘC: thiếu thì train_on_responses_only dùng marker Llama-3 -> mask SAI toàn bộ
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n")
    trainer.train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    # xuất bản merged để vLLM serve (tuỳ chọn)
    try:
        model.save_pretrained_merged(a.out + "-merged", tok, save_method="merged_16bit")
        print(f"merged 16bit -> {a.out}-merged (vLLM serve thẳng)")
    except Exception as e:
        print(f"(bỏ qua merge: {e})")
    print(f"LoRA adapter -> {a.out}. TẢI VỀ MÁY để giữ (dùng lại cho private test).")


if __name__ == "__main__":
    main()
