# Rehearsal on a rented Vast.ai box

Goal: prove the demo works end to end against real serving endpoints. This is a
rehearsal profile, not the graded run. Quantisation shifts reranker scores, so
no number produced here should be quoted as a submission result.

## One command

Rent the instance, then paste the SSH line from its card:

```bash
demo/vast_up.sh 'ssh -p 12345 root@1.2.3.4'
```

That installs vLLM on the box if it is missing, starts the planner and the
reranker with their memory split already set, tunnels both ports back, starts
the local UI, and asks one real Vietnamese question to prove the whole path
works. It prints the URL when every check passes.

To take it all down again, including the model servers on the box:

```bash
demo/vast_down.sh
```

Destroying the instance in the Vast console is still your call; the script only
frees the GPU.

It works the same on a 4090 or a 3090. The script reads the card on the box and
picks matching checkpoints itself.

The rest of this document explains what that script does and why, for when a
step needs to be run by hand.

## Card choice

The plan asks for int8 on both models. Off-the-shelf int8 exists for the
planner but **not** for the reranker: a Hub search for a W8A8 build of
`Qwen3-Reranker-4B` returns nothing. FP8 builds exist for both, so FP8 is the
route that needs no calibration step on the box.

FP8 needs Ada or Hopper silicon. An RTX 3090 is Ampere and has no FP8 tensor
cores, so rent an Ada card instead.

| card | VRAM | FP8 | verdict |
|---|---|---|---|
| RTX 4090 | 24 GB | yes | recommended, both models fit |
| L40S / L4 | 48 / 24 GB | yes | works, usually pricier per hour |
| RTX 3090 | 24 GB | no | only via the AWQ fallback below |

## Footprint, measured from the Hub

These are the published file sizes, not estimates from parameter counts. They
matter because Qwen3.5-9B is a conditional-generation model with a vision tower
and multi-token-prediction weights, and the quantised builds leave large parts
of it alone.

| checkpoint | on disk | quantised |
|---|---:|---|
| `lovedheart/Qwen3.5-9B-FP8` | 11.4 GiB | FP8 dynamic, blockwise |
| `DCC-BS/Qwen3-Reranker-4B-FP8-Dynamic` | 4.8 GiB | FP8 dynamic |
| `QuantTrio/Qwen3.5-9B-AWQ` | 11.6 GiB | 4-bit, but skips vision, linear attention, all self-attention, layer 0 and MTP |
| `datapizza-ai-lab/Qwen3-Reranker-4B-AWQ-4bit` | 3.2 GiB | 4-bit |

Note the third row. The AWQ planner is no smaller than the FP8 one, because its
`modules_to_not_convert` list leaves most of the attention stack in bf16. Do not
expect AWQ to buy planner VRAM on this model.

The FP8 pair is about 16.2 GiB of weights. On a 24 GB card that leaves roughly
6 GB for KV cache, activations and two CUDA contexts, which is workable but not
generous. The memory fractions below are set accordingly. If the second server
fails to allocate, swap the reranker to its AWQ build to recover 1.6 GiB.

### Disk

| item | size |
|---|---:|
| vLLM template image, unpacked | 25-30 GB |
| FP8 planner and reranker weights | 16.2 GiB |
| torch and vLLM compile caches | 1-2 GB |

Ask for **60 GB on a 24 GB card** and **80 GB on a 48 GB card**. The larger
card runs the bf16 8B reranker, which is 16 GiB on disk rather than 3, and a
session that tries a second checkpoint fills 60 GB quickly: on one run the disk
hit 85% and a download died half way. The image is the part that varies between
templates, so check the figure Vast shows for the one you pick.

The corpus stays on your laptop. Only the two model servers run on the box, so
none of the OCR'd reports or the index needs disk there.

## Checkpoints

| role | repo | note |
|---|---|---|
| planner, FP8 | `lovedheart/Qwen3.5-9B-FP8` | 4 shards |
| reranker, FP8 | `DCC-BS/Qwen3-Reranker-4B-FP8-Dynamic` | single shard, dynamic activation scales |
| planner, int8 | `RedHatAI/Qwen3.5-9B-quantized.w8a8` | Ampere-compatible, published by the llm-compressor team |
| planner, AWQ 4-bit | `QuantTrio/Qwen3.5-9B-AWQ` | Ampere fallback |
| reranker, AWQ 4-bit | `datapizza-ai-lab/Qwen3-Reranker-4B-AWQ-4bit` | Ampere fallback, low download count so verify it loads |

Pin a revision for whichever you pick, the way `demo/vllm.env.example` pins the
bf16 planner. These are third-party requantisations and their main branches can
move.

## Bring the box up

vLLM 0.28.0, matching the notebook pin. Two servers share one GPU, so each one
needs an explicit memory fraction or the first to start will claim almost
everything.

Planner on port 8001:

```bash
VLLM_MODEL=lovedheart/Qwen3.5-9B-FP8 \
VLLM_SERVED_MODEL_NAME=Qwen3.5-9B \
VLLM_PORT=8001 \
demo/launch_vllm.sh --quantization fp8 --gpu-memory-utilization 0.55 --max-model-len 8192
```

Reranker on port 8002:

```bash
VLLM_MODEL=DCC-BS/Qwen3-Reranker-4B-FP8-Dynamic \
VLLM_SERVED_MODEL_NAME=Qwen3-Reranker-4B \
VLLM_PORT=8002 \
demo/launch_vllm.sh --runner pooling --gpu-memory-utilization 0.30 --max-model-len 4096
```

The reranker must come up as a scoring model so it exposes the Cohere-style
`/rerank` route the pipeline calls. If `--runner pooling` is rejected by this
vLLM build, use `--task score`.

## Point the demo at it

From the presenter laptop, tunnel both ports and run the UI locally:

```bash
ssh -N -L 8001:127.0.0.1:8001 -L 8002:127.0.0.1:8002 GPU_USER@GPU_HOST
export DEMO_MODE=remote
export PLANNER_URL=http://127.0.0.1:8001
export RERANK_URL=http://127.0.0.1:8002
export RERANKER_MODEL=Qwen3-Reranker-4B
demo/launch_demo.sh
demo/check_health.sh
```

`check_health.sh` should report both endpoints up and a non-zero table count
before you ask a question.

## What counts as a pass

1. Health reports `ready` with both endpoints up.
2. A Vietnamese question streams every stage: gate, plan, retrieve, rerank,
   select, answer, execute.
3. The reranker stage returns real scores, not a uniform fallback.
4. The answer carries a cited cell, and clicking it opens the source table.
5. A second question in the same session reuses the servers without a reload.

## Cost and timing

A 4090 on Vast is typically well under a dollar an hour. Model download is the
long pole: about 13 GB of weights, a few minutes on a well-connected host.
Budget one hour for a first rehearsal including setup.

## Running on a 3090 instead

Nothing changes in what you type. `demo/vast_up.sh` reads the card's compute
capability on the box and picks the checkpoints to match, so an Ampere instance
gets the AWQ pair and `awq_marlin` kernels without being told.

| | 4090 and newer | 3090 |
|---|---|---|
| compute capability | 8.9 or above | 8.6 |
| planner | `lovedheart/Qwen3.5-9B-FP8` | `QuantTrio/Qwen3.5-9B-AWQ` |
| reranker | `DCC-BS/Qwen3-Reranker-4B-FP8-Dynamic` | `datapizza-ai-lab/Qwen3-Reranker-4B-AWQ-4bit` |
| weights on the card | 16.2 GiB | 14.8 GiB |
| memory fractions | 0.55 and 0.30 | 0.60 and 0.25 |

The 3090 is the roomier of the two, because the AWQ reranker is genuinely
4-bit at 3.2 GiB while its FP8 build is 4.8 GiB. Roughly 9 GB stays free for
KV cache and the two CUDA contexts, against about 6 GB on the 4090.

Disk is the same 60 GB either way.

What the 3090 costs is speed. AWQ is W4A16, so every matmul dequantises back to
fp16 and runs on the fp16 tensor cores, where a 3090 has less than half a
4090's throughput. Reranking is 130 prefill passes per question and dominates
the wall clock, so expect it to be several times slower. Fine for a rehearsal,
noticeable on stage.

Fidelity also drifts further. AWQ 4-bit perturbs the reranker's scores more
than FP8 does, and the arity gate thresholds on a logit gap of 3. Neither
profile reproduces the frozen submission's table selection, so treat both as
proof the wiring works rather than as a source of numbers.

## Forcing a specific pair

Detection only sets the defaults. `PLANNER_CHECKPOINT` and `RERANK_CHECKPOINT`
override them. They are deliberately not called `PLANNER_MODEL`, which the
pipeline itself reads as the model name to request:

```bash
PLANNER_CHECKPOINT=RedHatAI/Qwen3.5-9B-quantized.w8a8 \
  demo/vast_up.sh 'ssh -p 12345 root@1.2.3.4'
```

## Measured on a real 3090

A full run against a rented Vast 3090 (compute 8.6, 24 GB, 60 GB disk, vLLM
0.28.0 preinstalled) completed end to end. What the box actually showed:

| | measured |
|---|---|
| both servers resident | 20.7 GB of 24 GB |
| question latency, gate to executed answer | 11-13 s |
| reports loaded locally | 1,973 |
| disk used by the image plus both checkpoints | well inside 60 GB |

Latency is better than a roofline estimate suggests, because the reranker is a
4B model and the pass is prefill-only.

Several things only surfaced against real hardware, and the script now handles
all of them:

- The image boots its own bf16 server for whatever `VLLM_MODEL` says and holds
  most of the card. It is stopped, and the script waits for the memory to
  actually come back rather than assuming the process exit released it.
- Two servers started together both size their KV cache against the same free
  memory and overcommit. They start one at a time.
- Qwen3.5 is a hybrid Mamba model. Every concurrent sequence claims a Mamba
  cache block, and the default limit of 256 cannot be satisfied on 24 GB, so
  the planner runs with `--max-num-seqs 16`.
- The reranker checkpoint is compressed-tensors, not AWQ. Passing an explicit
  `--quantization` flag makes vLLM reject it, so no such flag is passed.
- A server backgrounded over a one-shot SSH command dies with the session.
  They are started under `setsid` with stdin detached.
- `PLANNER_MODEL` is read by the pipeline as the model name to request, so the
  script's checkpoint overrides are named `PLANNER_CHECKPOINT` and
  `RERANK_CHECKPOINT` instead, and the planner is served under the full
  Hugging Face id the pipeline asks for.

### Answer quality is not wiring

On one test question the pipeline ran every stage and then returned no answer,
because the tables the AWQ reranker selected did not contain the requested line
item and the planner honestly emitted `result = None`. That is a retrieval
fidelity cost of quantisation, not a broken path. Treat a rehearsal answer as
proof the plumbing works, never as a number to quote.
