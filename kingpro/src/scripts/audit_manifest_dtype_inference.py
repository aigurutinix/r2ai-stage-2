"""Audit compact source manifests against Pandas dtype inference.

Source-coordinate gates prove that ``raw`` matches a physical CSV token, but
they do not prove that ``raw`` plus ``typed_factor`` reconstructs the same
accounting number after ``pd.read_csv`` coerces a column (or a single token) to
numeric dtype.  This read-only audit checks both the current whole-manifest
inference and the latent single-token inference contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLAIN_NUMERIC = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def accounting_number(value: Any, typed_factor: float = 1.0) -> float:
    """Mirror the compact-query number contract."""

    if not isinstance(value, str):
        return float(value) * float(typed_factor)
    text = value.strip().replace("\u00a0", " ")
    if text in {"", "-"}:
        return 0.0
    if " " in text:
        text = text.split()[0]
    if ")(" in text:
        text = text.split(")(")[0] + ")"
    negative = text.startswith("(") and text.endswith(")")
    text = text.replace("(", "").replace(")", "").replace("%", "").replace("$", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        tail = text.split(",")[-1]
        text = text.replace(",", "." if len(tail) <= 2 else "")
    elif "." in text:
        tail = text.split(".")[-1]
        if len(tail) == 3:
            text = text.replace(".", "")
    result = float(text)
    return -abs(result) if negative else result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)


def audit(submission_dir: Path, ids: set[int] | None = None) -> dict[str, Any]:
    submission_dir = submission_dir.resolve()
    data_dir = submission_dir / "data"
    manifests: list[tuple[int, Path]] = []
    for path in sorted(data_dir.glob("q*_source_cells.csv")):
        match = re.fullmatch(r"q(\d+)_source_cells\.csv", path.name)
        if match is None:
            continue
        qid = int(match.group(1))
        if ids and qid not in ids:
            continue
        manifests.append((qid, path))

    findings: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    cells = latent_checks = 0
    manifest_hashes: dict[str, str] = {}
    for qid, path in manifests:
        manifest_hashes[path.name] = sha256(path)
        string_frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            index_col=None,
        )
        typed_frame = pd.read_csv(path, encoding="utf-8-sig", index_col=None)
        if len(string_frame) != len(typed_frame):
            findings.append({"id": qid, "kind": "row-count-mismatch", "manifest": path.name})
            continue
        required = {"raw", "typed_factor", "scale"}
        if not required.issubset(string_frame.columns):
            unsupported.append(
                {
                    "id": qid,
                    "kind": "legacy-schema-without-typed-factor",
                    "manifest": path.name,
                    "columns": sorted(required - set(string_frame.columns)),
                }
            )
            continue
        for row_index in range(len(string_frame)):
            cells += 1
            string_row = string_frame.iloc[row_index]
            raw = str(string_row["raw"])
            factor = float(string_row["typed_factor"] or 1.0)
            scale = float(string_row["scale"] or 1.0)
            expected = accounting_number(raw) * scale
            try:
                actual = accounting_number(typed_frame.iloc[row_index]["raw"], factor) * scale
            except Exception as exc:
                findings.append(
                    {
                        "id": qid,
                        "row": row_index,
                        "kind": "current-inference-error",
                        "manifest": path.name,
                        "raw": raw,
                        "typed_factor": factor,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                if not _same(expected, actual):
                    findings.append(
                        {
                            "id": qid,
                            "row": row_index,
                            "kind": "current-dtype-inference-mismatch",
                            "manifest": path.name,
                            "raw": raw,
                            "typed_factor": factor,
                            "expected": expected,
                            "actual": actual,
                        }
                    )

            # A currently mixed string column can become numeric after a
            # future manifest cleanup.  Check each plain token independently
            # so the factor contract is stable under that harmless mutation.
            if PLAIN_NUMERIC.fullmatch(raw):
                latent_checks += 1
                latent = accounting_number(float(raw), factor) * scale
                if not _same(expected, latent):
                    findings.append(
                        {
                            "id": qid,
                            "row": row_index,
                            "kind": "single-token-numeric-inference-mismatch",
                            "manifest": path.name,
                            "raw": raw,
                            "typed_factor": factor,
                            "expected": expected,
                            "actual": latent,
                        }
                    )

    return {
        "schema_version": 1,
        "audit": "compact_manifest_dtype_inference",
        "submission": submission_dir.name,
        "submission_sha256": sha256(submission_dir / "submission.json"),
        "requested_ids": sorted(ids or set()),
        "manifests_checked": len(manifests),
        "cells_checked": cells,
        "latent_single_token_checks": latent_checks,
        "finding_count": len(findings),
        "finding_ids": sorted({int(item["id"]) for item in findings}),
        "findings": findings,
        "unsupported_count": len(unsupported),
        "unsupported_ids": sorted({int(item["id"]) for item in unsupported}),
        "unsupported": unsupported,
        "manifest_hashes": manifest_hashes,
        "passed_for_supported_contract": bool(manifests) and not findings,
        "complete_coverage": not unsupported,
        "passed": bool(manifests) and not findings and not unsupported,
        "claim_limit": "Checks compact-manifest numeric reconstruction only; it does not prove source semantics or answer correctness.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--ids", default="")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()
    ids = {int(value) for value in args.ids.split(",") if value.strip()}
    report = audit(args.submission_dir, ids or None)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(rendered, end="")
    return 1 if args.fail_on_findings and report["finding_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
