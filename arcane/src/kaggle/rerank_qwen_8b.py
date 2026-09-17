"""Kaggle GPU notebook: score candidate pairs with a Qwen3 reranker.

Paste into a notebook with a GPU accelerator and the exported pairs attached as
a dataset, edit the settings block, run. It writes scores.jsonl for
`scripts/apply_rerank_scores.py` to fuse locally.

To A/B two settings, paste it twice in one session and change the one line that
differs plus SCORES_PATH. Two runs in different sessions are not comparable.

Retrieval is untouched — only the order of already-retrieved candidates changes,
so discarding a run means deleting one file.

README.md has the settings and the run order. Read it before changing one.
"""

import json
import sys
import os
import time
import hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---- edit these, then run ----------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Reranker-8B")
MODEL_REVISIONS = {
    "Qwen/Qwen3-Reranker-8B": "77d193c791ed757ca307ee72715aa132723da912",
    "Qwen/Qwen3-Reranker-4B": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
}
PAIRS_PATH = "/kaggle/input/vifinqa-rerank-pairs/pairs_bench_v4.jsonl"
SCORES_PATH = "/kaggle/working/scores.jsonl"
QUANTIZATION = os.environ.get("QUANTIZATION", "fp16")
RESUME_PATHS = []        # prior score shards: skips whole questions
SKIP_PATH = ""          # a finished scores.jsonl: skips individual candidates
ADAPTER_PATH = ""       # a LoRA adapter from train_reranker.py
# ------------------------------------------------------------------------------

if MODEL_NAME not in MODEL_REVISIONS:
    raise ValueError(f"pin a revision for {MODEL_NAME!r} in MODEL_REVISIONS")
MODEL_REVISION = MODEL_REVISIONS[MODEL_NAME]

if len(sys.argv) > 2 and not sys.argv[1].startswith("-"):
    PAIRS_PATH, SCORES_PATH = sys.argv[1], sys.argv[2]
    ADAPTER_PATH = sys.argv[3] if len(sys.argv) > 3 else ""
    SKIP_PATH = sys.argv[4] if len(sys.argv) > 4 else ""

MAX_LENGTH = 1024
RERANK_DEPTH = int(os.environ.get("RERANK_DEPTH", "130"))
MAX_BATCH = 16
PROGRESS_INTERVAL = 1_000
PROGRESS_WIDTH = 30

# Byte-identical to train_reranker.py. The adapter is tuned against this exact
# string, so a change here that is not made there scores the model on a prompt it
# was never trained on — a test pins the two copies together.
INSTRUCTION = (
    "The Query asks about a Vietnamese listed company's financial statements. "
    "The Document is one table from a filing whose first line identifies its "
    "ticker, reporting period, and scope (hợp nhất or công ty mẹ). The candidate "
    "pool is already narrowed to compatible filings, but reject a Document when "
    "that filing line conflicts with the Query. Answer yes only when the table "
    "reports an exact requested accounting item with a value for the requested "
    "period. For a multi-step question, a table is yes when it supplies any one "
    "required input. Do not accept a merely related label, a reference to an "
    "item, a related-party or subsidiary listing, a movement or allocation "
    "schedule, or a column header. Prefer the statement or note that reports the "
    "figure; a note or segment breakdown that validly restates it is yes."
)

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha256():
    text = json.dumps(
        {"instruction": INSTRUCTION, "prefix": PREFIX, "suffix": SUFFIX},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_pairs(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def already_scored(path):
    """Question IDs a previous run finished, so a stopped session resumes."""
    if not path or not os.path.exists(path):
        return set()
    return {record["id"] for record in load_pairs(path)}


def already_judged(path):
    """Table IDs per question a previous run scored, so a deeper export resumes."""
    if not path or not os.path.exists(path):
        return {}
    judged = {}
    for record in load_pairs(path):
        judged.setdefault(record["id"], set()).update(record["scores"])
    return judged


def to_judge(record, skip=frozenset()):
    """Candidate positions this run should score."""
    judged = []
    for index, candidate in enumerate(record["candidates"]):
        rank = candidate.get("sparse_rank") or candidate.get("dense_rank") or index + 1
        if rank <= RERANK_DEPTH and candidate["table_id"] not in skip:
            judged.append(index)
    return judged


def build_queries(record):
    """The query a candidate is scored against: the question, as it was asked.

    A list of one, because the scoring loop maxes over queries. That existed to
    let a lexicon of our own split a question into its named line items and
    score each separately. The lexicon is gone — it was a rule of ours, and two
    rounds of checking found rules of ours wrong more often than not — so there
    is one query, and the reranker is asked the question the organizers asked.
    """
    return [record["question"]]


def pair_count(record, skip=frozenset()):
    """Forward passes this record costs: one per (candidate, query)."""
    return len(to_judge(record, skip)) * len(build_queries(record))


def progress_bar(completed, total, started):
    """Return a compact progress bar with measured rate and remaining time."""
    fraction = completed / total if total else 1.0
    filled = round(PROGRESS_WIDTH * fraction)
    elapsed = max(time.time() - started, 1e-9)
    rate = completed / elapsed
    remaining_minutes = (total - completed) / rate / 60 if rate else 0.0
    return (
        f"[{'#' * filled}{'-' * (PROGRESS_WIDTH - filled)}] "
        f"{completed}/{total} ({fraction:.1%}) | {rate:.1f} pairs/s | "
        f"ETA {remaining_minutes:.0f} min"
    )


def packed_batches(lengths, order):
    """Length-sorted rows grouped so each batch costs about the same work.

    Padding makes a batch cost its widest row times its width, so packing to a
    token budget lets the short candidates run many at a time.
    """
    budget = 8 * MAX_LENGTH
    batch = []
    for index in order:
        width = max([lengths[index]] + [lengths[i] for i in batch])
        if batch and (width * (len(batch) + 1) > budget or len(batch) >= MAX_BATCH):
            yield batch
            batch = []
        batch.append(index)
    if batch:
        yield batch


def last_position_logits(model, padded):
    """Vocabulary logits for the final position only.

    Projecting 4096 -> 151,936 at every position builds a 7.5 GB tensor to read
    48 numbers, which is what put a 16 GB T4 out of memory. Applying the head to
    the last hidden state alone gives the identical numbers.
    """
    # A PeftModel's ``.model`` is another causal-LM wrapper, not the decoder
    # body.  Calling it here returns logits rather than ``last_hidden_state``.
    # ``get_base_model`` keeps the LoRA-injected modules but exposes the Qwen
    # causal LM, whose ``.model`` is the decoder body.  The bare-model branch
    # deliberately follows the same path.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.inference_mode():
        hidden = base.model(**padded, use_cache=False).last_hidden_state[:, -1, :]
        return base.lm_head(hidden).float()


def score_question(record, tokenizer, model, prompt_ids, yes_id, no_id, budget, skip=frozenset()):
    queries = build_queries(record)
    candidates = record["candidates"]
    scores = [0.0] * len(candidates)
    judged = to_judge(record, skip)
    if not judged:
        return scores, judged
    # Keyed by query position, not query text, so two identical items stay two rows.
    matrix = {index: [0.0] * len(queries) for index in judged}
    rows = [(index, position) for index in judged for position in range(len(queries))]
    prefix_ids, suffix_ids = prompt_ids
    encoded = {
        row: prefix_ids + ids + suffix_ids
        for row, ids in zip(rows, tokenizer(
            [
                f"<Instruct>: {INSTRUCTION}\n<Query>: {queries[position]}"
                f"\n<Document>: {candidates[index]['text']}"
                for index, position in rows
            ],
            truncation=True, max_length=budget, add_special_tokens=False,
        )["input_ids"])
    }
    lengths = {row: len(ids) for row, ids in encoded.items()}
    pending = list(packed_batches(lengths, sorted(rows, key=lambda row: lengths[row])))
    while pending:
        chunk = pending.pop()
        padded = tokenizer.pad(
            {"input_ids": [encoded[row] for row in chunk]}, padding=True, return_tensors="pt",
        ).to(model.device)
        try:
            logits = last_position_logits(model, padded)
        except torch.cuda.OutOfMemoryError:
            # Memory depends on the widest row, so one unlucky batch should not
            # end a ten hour session. This is the only handler here worth having.
            if len(chunk) == 1:
                raise
            del padded
            torch.cuda.empty_cache()
            middle = len(chunk) // 2
            pending.extend([chunk[:middle], chunk[middle:]])
            continue
        # The score is the probability of "yes" at the final position, which is
        # why padding must be on the left. Max over queries because a table
        # counts if it supplies any one named item, not all of them.
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
        for (index, position), value in zip(chunk, torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()):
            matrix[index][position] = value
            scores[index] = max(scores[index], value)
    return scores, judged


def score_payload(record, values, judged):
    """The scores.jsonl line for one question.

    Only the candidates this run judged are written. A zero is a score, and would
    rank a candidate below everything the model disliked rather than leaving it
    to the sparse order or to the file it was already scored in.
    """
    kept = [(index, record["candidates"][index]) for index in judged]
    payload = {
        "id": record["id"],
        "scores": {candidate["table_id"]: round(values[index], 6) for index, candidate in kept},
    }
    return payload


def load_model():
    if QUANTIZATION == "fp16":
        config, placement = None, "auto"          # 16.4 GB: needs T4 x2
    elif QUANTIZATION == "nf4":
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        placement = {"": 0}
    else:
        config, placement = BitsAndBytesConfig(load_in_8bit=True), {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=config, torch_dtype=torch.float16,
        device_map=placement, attn_implementation="sdpa", revision=MODEL_REVISION,
    ).eval()
    if ADAPTER_PATH:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH).eval()
    return model


def main():
    if not torch.cuda.is_available():
        raise SystemExit("select a GPU accelerator in the notebook settings")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, padding_side="left", revision=MODEL_REVISION,
    )
    model = load_model()
    print(f"{MODEL_NAME} in {QUANTIZATION}, depth={RERANK_DEPTH}"
          + (f", adapter={ADAPTER_PATH}" if ADAPTER_PATH else ", zero-shot"), flush=True)

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    prompt_ids = (
        tokenizer.encode(PREFIX, add_special_tokens=False),
        tokenizer.encode(SUFFIX, add_special_tokens=False),
    )
    budget = MAX_LENGTH - sum(len(ids) for ids in prompt_ids)

    records = load_pairs(PAIRS_PATH)
    done = already_scored(SCORES_PATH)
    for path in RESUME_PATHS:
        done |= already_scored(path)
    skip = already_judged(SKIP_PATH)
    skipped = lambda record: skip.get(record["id"], frozenset())
    pending = [record for record in records if record["id"] not in done]
    total = sum(pair_count(record, skipped(record)) for record in pending)
    print(f"{len(pending)} questions, {total} pairs"
          + (f", resuming past {len(done)}" if done else "")
          + (f", skipping {sum(len(s) for s in skip.values())} judged pairs" if skip else ""), flush=True)

    started, scored = time.time(), 0
    next_progress = PROGRESS_INTERVAL
    with open(SCORES_PATH, "a", encoding="utf-8") as out:
        for record in pending:
            values, judged = score_question(
                record, tokenizer, model, prompt_ids, yes_id, no_id, budget, skipped(record),
            )
            out.write(json.dumps(
                score_payload(record, values, judged), ensure_ascii=False,
            ) + "\n")
            out.flush()
            scored += pair_count(record, skipped(record))
            if scored >= next_progress or scored == total:
                print(progress_bar(scored, total, started), flush=True)
                next_progress += PROGRESS_INTERVAL

    elapsed = time.time() - started
    manifest = {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "quantization": QUANTIZATION,
        "adapter": ADAPTER_PATH or None,
        "max_length": MAX_LENGTH,
        "rerank_depth": RERANK_DEPTH,
        "max_batch": MAX_BATCH,
        "prompt_sha256": prompt_sha256(),
        "pairs_sha256": sha256(PAIRS_PATH),
        "scores_sha256": sha256(SCORES_PATH),
        "resume_sha256": [sha256(path) for path in RESUME_PATHS],
        "skip_sha256": sha256(SKIP_PATH)
        if SKIP_PATH and os.path.exists(SKIP_PATH) else None,
        "questions_in_pairs": len(records),
        "questions_scored_this_run": len(pending),
        "pairs_scored_this_run": scored,
        "elapsed_seconds": round(elapsed, 3),
        "gpu": [torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())],
        "torch": torch.__version__,
    }
    manifest_path = SCORES_PATH + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"wrote {SCORES_PATH} and {manifest_path} in {elapsed / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
