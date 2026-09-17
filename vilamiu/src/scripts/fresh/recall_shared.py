"""The two pieces every recall measurement here needs, defined once.

Both were got wrong once and the errors were expensive, so they live in one place rather
than being retyped per script.

`known_answers` is the only ground truth available without a submission: the questions
where the incumbent and an independent reader landed on the same value. Two mechanisms
that fail differently do not produce the same twelve-digit figure by accident. It is not
proof — the set carries an unknown error rate and must never be called gold — but it is
unbiased with respect to which retriever is being tested, which is what a recall
comparison needs.

`cells_of` returns raw magnitudes rather than a set of pre-scaled keys, because the
comparison cannot be exact: a known answer was rounded to two decimals in the question's
unit, so 145.731.366.146 đồng is stored as 145.73 tỷ. Comparing in đồng differs by 1.37
million and never matches.
"""

from __future__ import annotations

import csv as csv_mod
import json
import zipfile
from pathlib import Path


def _answers(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not path.exists():
        return out
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            records = json.loads(archive.read("submission.json"))
    else:
        records = [json.loads(line) for line
                   in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        value = record.get("answer")
        if value in (None, "", 0.0):
            continue
        try:
            out[record["id"]] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def known_answers(root: Path, neutral: bool = False) -> dict[int, float]:
    """Questions two independent mechanisms agree on.

    `neutral` restricts the confirming reader to the raw-text one, which scanned every
    table of the document rather than a shortlist. Every other reader here was fed the
    keyword shortlist, so a set confirmed by them contains mostly questions that
    shortlist got right — and comparing retrievers on it hands the keyword retriever a
    89% against the dense retriever's 70% for reasons that have nothing to do with
    either. This flag is the difference between a comparison and a tautology.
    """

    incumbent = _answers(root / "submissions" / "aimed.zip")
    if neutral:
        readers = [_answers(root / "submissions" / "fresh_raw.zip")]
    else:
        readers = [_answers(root / "submissions" / "fresh_tab.zip"),
                   _answers(root / "submissions" / "fresh_raw.zip"),
                   _answers(root / "artifacts" / "fresh" / "cot_results.jsonl")]
    out: dict[int, float] = {}
    for qid, value in incumbent.items():
        for reader in readers:
            other = reader.get(qid)
            if other is not None and abs(other - value) <= 0.01:
                out[qid] = value
                break
    return out


def cells_of(csv_path: Path) -> list[float]:
    """Every numeric cell of one table, as an absolute magnitude."""

    found: list[float] = []
    import parse_statements as ps

    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv_mod.reader(handle):
                for cell in row:
                    raw = str(cell).strip()
                    if not raw:
                        continue
                    value = ps.parse_vn_number(raw)
                    if value is not None:
                        found.append(abs(value))
    except OSError:
        pass
    return found
