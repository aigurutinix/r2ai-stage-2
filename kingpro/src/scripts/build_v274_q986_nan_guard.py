"""Build a rollbackable q986 NaN-normalization hardening from local V272."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import mkdtemp


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_v272_q749_q780"
OUTPUT = ROOT / "sub_v274_q986_nan"
OLD = """    if not isinstance(x, str):
        return float(x) * float(typed_factor)
"""
NEW = """    if not isinstance(x, str):
        value = float(x)
        if value != value:
            return 0.0
        return value * float(typed_factor)
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build() -> dict:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {OUTPUT}")
    temp = Path(mkdtemp(prefix="v274_q986_", dir=str(ROOT)))
    try:
        staged = temp / OUTPUT.name
        shutil.copytree(SOURCE, staged)
        submission_path = staged / "submission.json"
        rows = json.loads(submission_path.read_text(encoding="utf-8"))
        hits = [row for row in rows if int(row.get("id", -1)) == 986]
        if len(hits) != 1:
            raise AssertionError(f"expected one q986 row, got {len(hits)}")
        row = hits[0]
        before = {
            key: row.get(key)
            for key in ("answer", "relevant_docs", "relevant_tables", "evidence")
        }
        query = str(row.get("pandas_query", ""))
        if query.count(OLD) != 1:
            raise AssertionError("q986 numeric parser marker is not unique")
        row["pandas_query"] = query.replace(OLD, NEW, 1)
        after = {key: row.get(key) for key in before}
        if before != after:
            raise AssertionError("q986 hardening changed scored fields")
        submission_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        audit = {
            "candidate": OUTPUT.name,
            "baseline": SOURCE.name,
            "purpose": "q986 typed-pandas blank/NaN normalization",
            "changed_code_ids": [986],
            "answer_changes": [],
            "retrieval_changes": [],
            "baseline_submission_sha256": sha256(SOURCE / "submission.json"),
            "candidate_submission_sha256": sha256(submission_path),
            "source_values": {
                "2016": 8142646.0,
                "2017": 912141.0,
                "2024": 37670.0,
                "2025": 0.0,
            },
            "expected_answer": 2016.0,
            "invariants": {
                "only_q986_submission_row_changed": True,
                "answer_unchanged": True,
                "docs_tables_evidence_unchanged": True,
                "nan_normalized_to_zero": True,
                "automatic_promotion": False,
            },
        }
        (staged / "v274_q986_nan_guard_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(staged), str(OUTPUT))
        return audit
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))

