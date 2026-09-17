#!/usr/bin/env python3
"""Fail unless a reranker score file covers every exported candidate exactly once."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifests(pairs_path: Path, score_paths: list[Path],
                       manifest_paths: list[Path]) -> dict:
    if len(score_paths) != len(manifest_paths):
        raise ValueError("pass one --manifest for each --scores shard, in the same order")
    pair_hash = sha256(pairs_path)
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    invariant_keys = (
        "model", "revision", "quantization", "adapter", "max_length",
        "rerank_depth", "prompt_sha256", "pairs_sha256",
    )
    reference = {key: manifests[0].get(key) for key in invariant_keys}
    for score_path, manifest_path, manifest in zip(score_paths, manifest_paths, manifests):
        if manifest.get("pairs_sha256") != pair_hash:
            raise ValueError(f"{manifest_path}: pair hash does not match {pairs_path}")
        if manifest.get("scores_sha256") != sha256(score_path):
            raise ValueError(f"{manifest_path}: score hash does not match {score_path}")
        current = {key: manifest.get(key) for key in invariant_keys}
        if current != reference:
            raise ValueError(f"{manifest_path}: scorer configuration differs between shards")
    return reference


def validate(pairs: list[dict], scores: list[dict]) -> dict:
    expected: dict[str, set[str]] = {}
    for record in pairs:
        identifier = str(record["id"])
        if identifier in expected:
            raise ValueError(f"duplicate pair record for question {identifier}")
        table_ids = [candidate["table_id"] for candidate in record["candidates"]]
        if len(table_ids) != len(set(table_ids)):
            raise ValueError(f"duplicate candidate for question {identifier}")
        expected[identifier] = set(table_ids)

    seen: dict[str, set[str]] = {}
    for record in scores:
        identifier = str(record["id"])
        if identifier in seen:
            raise ValueError(f"duplicate score record for question {identifier}")
        values = record["scores"]
        if not isinstance(values, dict):
            raise ValueError(f"scores for question {identifier} must be an object")
        for table_id, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"score for {identifier}/{table_id} is not numeric")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"score for {identifier}/{table_id} is outside [0, 1]")
        seen[identifier] = set(values)

    missing_questions = sorted(set(expected) - set(seen))
    extra_questions = sorted(set(seen) - set(expected))
    if missing_questions or extra_questions:
        raise ValueError(
            f"question mismatch: missing={missing_questions[:10]} extra={extra_questions[:10]}"
        )

    incomplete = {
        identifier: sorted(tables - seen[identifier])[:10]
        for identifier, tables in expected.items()
        if tables != seen[identifier]
    }
    if incomplete:
        identifier = next(iter(incomplete))
        extra = sorted(seen[identifier] - expected[identifier])[:10]
        raise ValueError(
            f"candidate mismatch for question {identifier}: "
            f"missing={incomplete[identifier]} extra={extra}"
        )

    return {
        "questions": len(expected),
        "candidates": sum(len(tables) for tables in expected.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--scores", type=Path, action="append", required=True,
        help="score shard; repeat for resumable runs",
    )
    parser.add_argument(
        "--manifest", type=Path, action="append", required=True,
        help="manifest beside the corresponding score shard; repeat in the same order",
    )
    args = parser.parse_args()
    score_records = [record for path in args.scores for record in load_jsonl(path)]
    result = validate(load_jsonl(args.pairs), score_records)
    provenance = validate_manifests(args.pairs, args.scores, args.manifest)
    print(json.dumps({"valid": True, **result, "provenance": provenance}, indent=2))


if __name__ == "__main__":
    main()
