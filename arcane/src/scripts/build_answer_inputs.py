from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


RETRIEVAL_FIELDS = ("id", "question", "relevant_docs", "relevant_tables", "evidence")


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "retrieval_manifest.jsonl"

    with ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise ValueError("source ZIP failed integrity check")
        rows = json.loads(archive.read("submission.json"))
        if len(rows) != 1012 or len({int(row["id"]) for row in rows}) != 1012:
            raise ValueError("expected 1,012 unique questions")

        retrieval = [{key: row[key] for key in RETRIEVAL_FIELDS} for row in rows]
        evidence = sorted({item["csv_path"] for row in retrieval for item in row["evidence"]})
        missing = [name for name in evidence if name not in archive.namelist()]
        if missing:
            raise FileNotFoundError(missing[0])

        manifest.write_text(jsonl(retrieval), encoding="utf-8")
        for name in evidence:
            target = output_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))

    if any(not (output_dir / name).is_file() for name in evidence):
        raise ValueError("retrieval input validation failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a validated submission into answer-stage inputs.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source, args.output_dir)
    print(json.dumps({
        "retrieval_manifest": str(manifest),
        "retrieval_manifest_sha256": sha256(manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
