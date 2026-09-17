"""Embed the table corpus and the questions with Qwen3-Embedding, on a free GPU.

The sparse ranker matches a row's wording. A question that names an item the
filing words differently never reaches the table, and no reranker can recover a
table that was never a candidate. This builds the other half of the first stage:
one vector per table, one per question, cosine between them.

The organizers published where this lands (their Table 4, Recall@10 over the
same corpus): BM25 47.41, BGE-M3 53.05, Qwen3-Embedding-4B 63.90, 8B 67.48; with
a reranker on top, 4B 80.19 and 8B 80.80. The 8B embedder is worth +3.58 alone
and +0.61 once a reranker follows it, so 4B is the size to index — the reranker
absorbs the difference and the index costs half as much to build and to hold.

Their BM25 is not ours, though. Theirs ranks flat over all 146k tables; ours
ranks inside reports a gate has already narrowed by ticker, year and scope. Most
of what dense retrieval buys them, the gate has already bought us, so expect the
margin here to be smaller than their table suggests. What it should still buy is
the gate's own misses, which nothing downstream can currently fix.

## Steps

1. `python scripts/export_table_texts.py` locally, then upload
   `output/dense/tables.jsonl` and `data/raw/vifinqa/questions/questions.jsonl`
   as a Kaggle Dataset.
2. New notebook, any 16 GB GPU, attach the dataset, paste this file in, run.
3. Run it once as it stands, with `LIMIT = 2000`. Five minutes, and the
   self-check at the end says whether this is an index or noise before three
   hours go into it.
4. Then set `LIMIT = 0` and run it again: 130,556 passages of 256 tokens through
   a 4B model is 2.7e17 FLOPs, near three hours at the 25 TFLOP/s a T4 sustains,
   inside a 12-hour session with room to spare.
5. Download `/kaggle/working/dense/` to `output/dense/`, then run the repository
   validators before scoring pairs. The exact commands are in the root
   `README.md`.

Nothing here reads a label. The passages come from the corpus and the queries
from the released question file, so the whole stage is reproducible by anyone
holding the public data.
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
TABLES_PATH = "/kaggle/input/vifinqa-dense/tables.jsonl"
QUESTIONS_PATH = "/kaggle/input/vifinqa-dense/questions.jsonl"
# Set TABLES_PATH to "" to embed a question file alone. That is the two-minute
# second run: the held-out evaluation questions do not exist until the generator
# has written them, and re-embedding 130,556 tables to reach them would be absurd.
OUTPUT_DIR = "/kaggle/working/dense"

MAX_LENGTH = 256      # p90 of a passage; the tail is period and unit boilerplate
BATCH_SIZE = 32
DIMS = 1024           # Matryoshka truncation; the full 2560 would be 750 MB
LIMIT = 2000          # 0 for the real run, anything else is a smoke test

# Qwen3-Embedding is instruction-tuned on the query side only. Documents are
# embedded bare; the asymmetry is the model's, not a choice of ours.
INSTRUCT = ("Given a Vietnamese financial question, retrieve the financial "
            "statement table that contains the answer")


def tokenize(texts, tokenizer):
    """Encode a batch, guaranteeing the token that last-token pooling reads.

    Qwen3-Embedding pools the final position, and that position is meant to hold
    the end-of-sequence token the model was trained with there. Whether the
    tokenizer appends one is a release detail, and getting it wrong yields an
    index quietly worse in a way nothing downstream can detect — there is no
    offline scorer on this task to catch it. So it is checked, not assumed.
    """
    end = tokenizer.eos_token_id
    encoded = [tokenizer(text, truncation=True, max_length=MAX_LENGTH - 1,
                         add_special_tokens=True)["input_ids"] for text in texts]
    encoded = [ids if ids and ids[-1] == end else ids + [end] for ids in encoded]
    return tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt")


def encode(texts, model, tokenizer, label):
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = tokenize(texts[start:start + BATCH_SIZE], tokenizer).to(model.device)
        with torch.inference_mode():
            # Left padding, so the final column is every row's last real token
            # and pooling is a plain slice.
            pooled = model(**batch).last_hidden_state[:, -1].float()
        # Truncate before normalising: a Matryoshka prefix is only a unit vector
        # once it has been renormalised.
        pooled = torch.nn.functional.normalize(pooled[:, :DIMS], p=2, dim=1)
        vectors.append(pooled.half().cpu().numpy())
        done = min(start + BATCH_SIZE, len(texts))
        if (start // BATCH_SIZE) % 50 == 0 or done == len(texts):
            print(f"{label} {done}/{len(texts)}", flush=True)
    return np.concatenate(vectors)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def self_check(tables, questions):
    """Is this an index, or is it noise? The one question answerable on the box.

    A working retriever's best match scores far above a table picked at random.
    If those two are close, something upstream is wrong — the pooling, the
    instruction asymmetry, an fp16 overflow — and seeing it here costs minutes
    where seeing it downstream costs a scoring run.
    """
    sample = tables[:20000].astype(np.float32)
    scores = sample @ questions[:50].astype(np.float32).T
    best, average = scores.max(axis=0).mean(), scores.mean()
    print(f"self-check: best cosine {best:.3f}, random pair {average:.3f}, "
          f"spread {best - average:.3f}", flush=True)
    if best - average < 0.05:
        print("WARNING: the best match is no better than an average one. "
              "Do not build on this index.", flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, padding_side="left", revision=MODEL_REVISION,
    )
    # One card. The 4B is 8 GB in fp16 and fits whole; "auto" would spread its
    # layers across a two-GPU box and run slower, because the layers are
    # sequential and every boundary becomes a transfer.
    model = AutoModel.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map={"": 0},
        revision=MODEL_REVISION,
    ).eval()

    table_vectors = None
    if TABLES_PATH:
        tables = read_jsonl(TABLES_PATH)[: LIMIT or None]
        print(f"{len(tables)} passages", flush=True)
        table_vectors = encode([row["text"] for row in tables], model, tokenizer, "tables")
        np.save(f"{OUTPUT_DIR}/table_vectors.npy", table_vectors)
        with open(f"{OUTPUT_DIR}/table_ids.json", "w", encoding="utf-8") as handle:
            json.dump([[row["report_id"], row["table_id"]] for row in tables], handle)

    questions = read_jsonl(QUESTIONS_PATH)
    queries = [f"Instruct: {INSTRUCT}\nQuery: {row['question']}" for row in questions]
    question_vectors = encode(queries, model, tokenizer, "questions")
    np.save(f"{OUTPUT_DIR}/question_vectors.npy", question_vectors)
    with open(f"{OUTPUT_DIR}/question_ids.json", "w", encoding="utf-8") as handle:
        json.dump([row["id"] for row in questions], handle)

    if table_vectors is not None:
        self_check(table_vectors, question_vectors)
    print("done", OUTPUT_DIR, flush=True)


main()
