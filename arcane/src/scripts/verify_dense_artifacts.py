#!/usr/bin/env python3
"""Check the Qwen3 dense index and write the manifest that pins its provenance.

The index was built on a Kaggle GPU and downloaded. Nothing downstream can tell
a healthy index from a damaged one: a truncated download, a Matryoshka prefix
that was never renormalised, or an id list that drifted out of step with its
vector rows all produce candidates that look ordinary and rank badly. There is
no offline scorer on this task to catch it either, so the index is checked
before a scoring run is spent on it.

The manifest ties the four artifact hashes to the model revision, the query
instruction, and the embedding configuration that produced them. Recording the
hashes without that context would assert reproducibility we could not
demonstrate, so the configuration is read from `kaggle/embed_tables.py` rather
than restated here — a drift between the two is itself a failure.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIMS = 1024
NORM_RANGE = (0.99, 1.01)
NORM_SAMPLE = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_config(path: Path) -> dict:
    """Read the pinned model and embedding settings out of the builder itself."""
    wanted = {"MODEL_NAME", "MODEL_REVISION", "MAX_LENGTH", "DIMS", "INSTRUCT"}
    namespace: dict = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            try:
                namespace[target.id] = ast.literal_eval(node.value)
            except (TypeError, ValueError) as error:
                raise SystemExit(f"{path}: {target.id} must be a literal") from error
    missing = wanted - namespace.keys()
    if missing:
        raise SystemExit(f"{path}: could not read {sorted(missing)}")
    return namespace


def validate_table_ids(entries: list, failures: list[str]) -> list[str]:
    """Validate the report/table pairing used to materialise dense candidates."""
    table_ids = []
    for index, entry in enumerate(entries):
        if (not isinstance(entry, list) or len(entry) != 2
                or not all(isinstance(value, str) and value for value in entry)):
            failures.append(f"table_ids: row {index} is not [report_id, table_id]")
            continue
        report_id, table_id = entry
        if table_id.rpartition("|")[0] != report_id:
            failures.append(
                f"table_ids: row {index} pairs {table_id!r} with report {report_id!r}"
            )
        table_ids.append(table_id)
    return table_ids


def check_vectors(vectors: np.ndarray, identifiers: list, label: str,
                  failures: list[str]) -> None:
    if vectors.dtype != np.float16:
        failures.append(f"{label}: dtype {vectors.dtype}, expected float16")
    if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_DIMS:
        failures.append(f"{label}: shape {vectors.shape}, expected (n, {EXPECTED_DIMS})")
        return
    if vectors.shape[0] != len(identifiers):
        failures.append(
            f"{label}: {vectors.shape[0]} vectors against {len(identifiers)} ids"
        )
    if len(set(identifiers)) != len(identifiers):
        failures.append(f"{label}: ids are not unique")
    as_float = vectors.astype(np.float32)
    if not np.isfinite(as_float).all():
        failures.append(f"{label}: contains NaN or Inf")
        return
    # A Matryoshka prefix is only a unit vector once renormalised, and skipping
    # that step is the failure this catches. Sampling is enough: the fault is
    # systematic, never a handful of rows.
    step = max(1, vectors.shape[0] // NORM_SAMPLE)
    norms = np.linalg.norm(as_float[::step], axis=1)
    low, high = float(norms.min()), float(norms.max())
    if low < NORM_RANGE[0] or high > NORM_RANGE[1]:
        failures.append(
            f"{label}: sampled L2 norms span [{low:.4f}, {high:.4f}], "
            f"expected [{NORM_RANGE[0]}, {NORM_RANGE[1]}]"
        )


def load_jsonl_ids(path: Path, key: str) -> list[str]:
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line)[key])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, default=ROOT / "output" / "dense")
    parser.add_argument("--tables", type=Path,
                        help="passage source; defaults to <dense>/tables.jsonl")
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "data" / "raw" / "vifinqa" / "questions" / "questions.jsonl")
    parser.add_argument("--builder", type=Path, default=ROOT / "kaggle" / "embed_tables.py")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    tables_path = args.tables or args.dense / "tables.jsonl"
    paths = {
        "table_vectors": args.dense / "table_vectors.npy",
        "question_vectors": args.dense / "question_vectors.npy",
        "table_ids": args.dense / "table_ids.json",
        "question_ids": args.dense / "question_ids.json",
    }
    for name, path in list(paths.items()) + [("tables", tables_path),
                                             ("questions", args.questions)]:
        if not path.exists():
            raise SystemExit(f"missing {name}: {path}")

    failures: list[str] = []

    table_ids_raw = json.loads(paths["table_ids"].read_text(encoding="utf-8"))
    table_ids = validate_table_ids(table_ids_raw, failures)
    question_ids = json.loads(paths["question_ids"].read_text(encoding="utf-8"))

    table_vectors = np.load(paths["table_vectors"], mmap_mode="r")
    question_vectors = np.load(paths["question_vectors"], mmap_mode="r")
    check_vectors(table_vectors, table_ids, "table_vectors", failures)
    check_vectors(question_vectors, question_ids, "question_vectors", failures)

    passages = load_jsonl_ids(tables_path, "table_id")
    if passages != table_ids:
        failures.append(
            f"table_ids does not match {tables_path.name} row order "
            f"({len(passage_ids)} passages, {len(table_ids)} ids)"
        )

    asked = [json.loads(line)["id"] for line in
             args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [str(value) for value in question_ids] != [str(value) for value in asked]:
        failures.append(
            f"question_ids does not match {args.questions.name} "
            f"({len(question_ids)} embedded, {len(asked)} asked)"
        )

    config = embedding_config(args.builder)
    if config["DIMS"] != EXPECTED_DIMS:
        failures.append(f"{args.builder.name}: DIMS is {config['DIMS']}, index is {EXPECTED_DIMS}")

    if failures:
        print(f"dense index rejected ({len(failures)} problems):")
        for line in failures:
            print(f"  {line}")
        return 1

    manifest = {
        "dense_dir": str(args.dense),
        "tables": len(table_ids),
        "questions": len(question_ids),
        "dimensions": EXPECTED_DIMS,
        "dtype": "float16",
        "model": {"name": config["MODEL_NAME"], "revision": config["MODEL_REVISION"]},
        "embedding": {
            "max_length": config["MAX_LENGTH"],
            "pooling": "last-token with an explicit EOS",
            "matryoshka_dims": config["DIMS"],
            "normalised": "L2 after truncation",
            "query_instruction": config["INSTRUCT"],
            "documents": "embedded bare; the instruction is query-side only",
        },
        "sha256": {
            **{name: sha256(path) for name, path in paths.items()},
            "tables_jsonl": sha256(tables_path),
            "questions_jsonl": sha256(args.questions),
            "builder": sha256(args.builder),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
