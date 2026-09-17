"""Kaggle GPU entry point: Text-to-Pandas codegen.

Two backends (same pipeline, same output file):

  --backend hf    transformers batched generate. Slower but WORKS ON T4.
                  Recommended on Kaggle: recent vLLM V1 engines fail to start
                  on Turing/SM75 GPUs ("Engine core initialization failed").

  --backend vllm  vLLM offline batching (much faster where supported; on T4 you
                  must pin an older release, e.g. pip install vllm==0.7.3).

Examples:
  !python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD \
      --backend hf --out /kaggle/working/codegen_smoke.jsonl --limit 20 --n 2
  !python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD \
      --backend hf --out /kaggle/working/codegen_results.jsonl --n 2
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

PAYLOAD_SCHEMA_VERSION = 2
MANIFEST_NAME = "payload-manifest.json"
RUNNER_BUILD = "private-n5-safe-v4"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_payload(payload: Path, runtime_code_dir: Path | None = None) -> tuple[dict, str]:
    """Fail fast when Kaggle is attached to stale/partial payload code."""
    manifest_path = payload / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(
            f"missing {manifest_path}; rebuild and re-upload with "
            "python scripts/04_make_kaggle_payload.py"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"invalid payload manifest: {e}") from e
    if manifest.get("schema_version") != PAYLOAD_SCHEMA_VERSION:
        raise SystemExit(
            f"payload schema={manifest.get('schema_version')!r}, runner requires "
            f"{PAYLOAD_SCHEMA_VERSION}; rebuild and re-upload the payload"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit("payload manifest has no file hashes")
    required = {
        "retrieval.jsonl",
        "store/reports.parquet",
        "code/kaggle_codegen.py",
        "code/vifinqa/codegen/generate.py",
        "code/vifinqa/codegen/llm_client.py",
        "code/vifinqa/codegen/prompts.py",
        "code/vifinqa/codegen/executor.py",
    }
    missing_manifest = sorted(required - set(files))
    if missing_manifest:
        raise SystemExit(f"payload manifest misses required files: {missing_manifest}")

    for rel, expected in files.items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise SystemExit(f"invalid path in payload manifest: {rel}")
        path = payload / rel_path
        if not path.is_file():
            raise SystemExit(f"payload file missing: {rel}")
        actual = _sha256(path)
        if actual != expected:
            raise SystemExit(f"payload hash mismatch: {rel}")

    packaged_code = (payload / "code").resolve()
    if runtime_code_dir is not None and runtime_code_dir.resolve() != packaged_code:
        # The notebook copies code to /kaggle/working. Verify that it did not
        # silently hot-patch a different runner after the manifest check.
        for rel, expected in files.items():
            if not rel.startswith("code/"):
                continue
            runtime_path = runtime_code_dir / Path(rel).relative_to("code")
            if not runtime_path.is_file() or _sha256(runtime_path) != expected:
                raise SystemExit(f"runtime code differs from payload: {rel}")
    stable_manifest = json.dumps(
        {"schema_version": manifest["schema_version"], "files": files},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return manifest, hashlib.sha256(stable_manifest).hexdigest()


def run_signature(args, manifest_hash: str) -> str:
    """Hash semantic generation settings; performance-only knobs are omitted."""
    semantic = {
        "payload_manifest": manifest_hash,
        "backend": args.backend,
        "model": args.model,
        "n": args.n,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "debug_rounds": args.debug_rounds,
        "k": args.k,
        "rule_first": args.rule_first,
        "llm_target": args.llm_target,
        "llm_mode": args.llm_mode,
        "llm_ids_sha256": (
            _sha256(Path(args.llm_ids_file)) if args.llm_ids_file else ""
        ),
        "no_rule_fallback": args.no_rule_fallback,
        # dense matching changes the prompt shortlist -> a different semantic run
        "use_dense": args.use_dense,
        "dense_model": args.dense_model if args.use_dense else "",
        "seed": args.seed,
        "load_4bit": args.load_4bit,
        "max_input_tokens": args.max_input_tokens,
        "quantization": args.quantization,
        "tp": args.tp,
        "max_model_len": args.max_model_len,
        "smoke_first": args.smoke_first,
    }
    raw = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    print(f"runner build: {RUNNER_BUILD}", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, help="/kaggle/input/<...>/vifinqa-payload")
    ap.add_argument("--out", default="/kaggle/working/codegen_results.jsonl")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ")
    # shared generation knobs
    ap.add_argument("--n", type=int, default=2, help="self-consistency samples")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--debug-rounds", type=int, default=1)
    ap.add_argument("--k", type=int, default=6, help="tables shown to the LLM")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-dense", action="store_true",
                    help="BGE-M3 row matching (needs store/label_index + sentence-transformers)")
    ap.add_argument("--dense-model", default="BAAI/bge-m3")
    ap.add_argument("--llm-mode", choices=["code", "select"], default="code",
                    help="select = model picks shortlist rows, we write the pandas "
                         "(removes the unit/column/regex error classes)")
    ap.add_argument("--llm-target", choices=["all", "empty", "weak"], default="all",
                    help="which questions the LLM should spend GPU time on")
    ap.add_argument("--rule-first", action="store_true",
                    help="skip the LLM for high-confidence rule matches (faster)")
    ap.add_argument("--no-rule-fallback", action="store_true",
                    help="emit raw LLM challengers for targeted audit; non-target "
                         "rows remain failed placeholders")
    ap.add_argument("--llm-ids-file", default="",
                    help="optional text/JSON file of question ids to send to the LLM")
    ap.add_argument("--checkpoint-every", type=int, default=32,
                    help="flush results to --out every N questions (crash safety)")
    ap.add_argument("--time-budget-min", type=float, default=420,
                    help="stop cleanly after N minutes (Kaggle GPU session ~9h)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore LLM answers already present in --out")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--smoke-first", type=int, default=0,
                    help="run and verify this many targeted ids before the full job")
    ap.add_argument("--skip-payload-verification", action="store_true",
                    help="emergency/debug only: allow an unverified payload")
    # hf backend
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--load-4bit", action="store_true",
                    help="bitsandbytes NF4 (for non-AWQ models like the 7B fp16)")
    ap.add_argument("--max-input-tokens", type=int, default=5000)
    # vllm backend
    ap.add_argument("--quantization", default="awq")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    llm_ids = None
    if args.llm_ids_file:
        ids_path = Path(args.llm_ids_file)
        if not ids_path.is_file():
            raise SystemExit(f"LLM ids file not found: {ids_path}")
        raw_ids = ids_path.read_text(encoding="utf-8").strip()
        try:
            parsed = json.loads(raw_ids)
        except json.JSONDecodeError:
            parsed = [value for value in raw_ids.replace(",", " ").split() if value]
        if isinstance(parsed, dict):
            parsed = parsed.get("ids", [])
        if not isinstance(parsed, list):
            raise SystemExit("--llm-ids-file must contain a JSON list/object or text ids")
        llm_ids = {int(value) for value in parsed}
        if not llm_ids:
            raise SystemExit("--llm-ids-file contains no ids")
        print(f"targeted LLM ids: {len(llm_ids)}", flush=True)

    payload = Path(args.payload)
    # import the vifinqa package sitting NEXT TO this script
    runtime_code_dir = Path(__file__).resolve().parent
    if args.skip_payload_verification:
        print("[WARN] payload verification explicitly disabled", flush=True)
        manifest_hash = "unverified"
    else:
        manifest, manifest_hash = verify_payload(payload, runtime_code_dir)
        print(f"payload verified: schema={manifest['schema_version']} "
              f"files={len(manifest['files'])}", flush=True)
    signature = run_signature(args, manifest_hash)
    print(f"run signature: {signature[:16]}", flush=True)

    sys.path.insert(0, str(runtime_code_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    # T4-friendly vLLM hints (harmless / ignored on versions that dropped them)
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    os.environ.setdefault("VLLM_DISABLE_FLASHINFER", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from vifinqa.codegen.generate import run_codegen

    if args.backend == "hf":
        from vifinqa.codegen.llm_client import HfBatchClient
        client = HfBatchClient(model=args.model, load_4bit=args.load_4bit,
                               batch_size=args.batch_size,
                               max_input_tokens=args.max_input_tokens)
    else:
        from vifinqa.codegen.llm_client import VllmBatchClient
        client = VllmBatchClient(model=args.model, tensor_parallel=args.tp,
                                 max_model_len=args.max_model_len,
                                 gpu_mem=args.gpu_mem, dtype="half",
                                 quantization=args.quantization or None,
                                 enforce_eager=args.enforce_eager,
                                 seed=args.seed)

    common = dict(
        retrieval_path=payload / "retrieval.jsonl",
        store_dir=payload / "store",
        client=client, k=args.k, n_samples=args.n,
        temperature=args.temperature, debug_rounds=args.debug_rounds,
        limit=args.limit, use_rule_fallback=not args.no_rule_fallback,
        rule_first=args.rule_first,
        max_tokens=args.max_tokens, time_budget_s=args.time_budget_min * 60,
        use_dense=args.use_dense, dense_model=args.dense_model,
        llm_target=args.llm_target,
        llm_mode=args.llm_mode,
        run_signature=signature,
    )

    if args.smoke_first:
        if llm_ids is None:
            raise SystemExit("--smoke-first requires --llm-ids-file")
        smoke_ids = set(sorted(llm_ids)[:args.smoke_first])
        out_path = Path(args.out)
        smoke_out = out_path.with_name(f"{out_path.stem}.smoke{out_path.suffix}")
        print(f"smoke test: {len(smoke_ids)} ids -> {smoke_out}", flush=True)
        smoke_common = {
            **common,
            # Smoke checks prompt preservation and parsing, not consensus.
            # Using one deterministic sample avoids spending ~25 minutes before
            # the real N=5 run starts on T4.
            "n_samples": 1,
            "temperature": 0.0,
            "max_tokens": min(args.max_tokens, 256),
        }
        run_codegen(
            out_path=smoke_out, llm_ids=smoke_ids, resume=False,
            checkpoint_every=max(1, len(smoke_ids)), **smoke_common,
        )
        smoke_rows = [json.loads(line) for line in smoke_out.read_text().splitlines()]
        smoke_llm = [
            row for row in smoke_rows
            if row.get("id") in smoke_ids
            and str(row.get("source", "")).startswith("llm")
        ]
        if not smoke_llm:
            failed = [row for row in smoke_rows if row.get("id") in smoke_ids]
            raise RuntimeError(
                "smoke test produced no executable LLM candidate; aborting full run. "
                f"Diagnostics: {[row.get('llm_diagnostics', {}) for row in failed]}"
            )
        print(f"smoke test passed: {len(smoke_llm)}/{len(smoke_ids)} executable",
              flush=True)

    run_codegen(
        out_path=Path(args.out), llm_ids=llm_ids,
        resume=not args.no_resume, checkpoint_every=args.checkpoint_every,
        **common,
    )
    print(f"OK -> {args.out}")


if __name__ == "__main__":
    main()
