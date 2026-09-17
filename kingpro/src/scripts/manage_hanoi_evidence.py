"""Attach and validate private Hanoi Demo Day evidence without leaking it.

Evidence files are copied under ``private_evidence/hanoi``. That directory is
ignored and is deliberately excluded from the public evidence bundle. An
attachment is only marked confirmed when the operator passes ``--confirm``;
the tool never infers legal or human confirmation from the presence of a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "demo_manual_evidence.json"
PRIVATE_ROOT = ROOT / "private_evidence" / "hanoi"
sys.path.insert(0, str(ROOT / "scripts"))

from audit_demo_readiness import MANUAL_EVIDENCE_KEYS, manual_evidence_summary  # noqa: E402


def _load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manual evidence manifest must be a JSON object")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        payload["evidence"] = {}
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_filename(name: str) -> str:
    cleaned = "".join(
        character
        for character in Path(name).name
        if character.isalnum() or character in {"-", "_", "."}
    )
    if cleaned in {"", ".", ".."}:
        raise ValueError("source filename has no safe characters")
    return cleaned


def attach(
    *,
    manifest_path: Path,
    key: str,
    source: Path,
    note: str,
    confirmed: bool,
    root: Path = ROOT,
    private_root: Path = PRIVATE_ROOT,
) -> dict[str, object]:
    if key not in MANUAL_EVIDENCE_KEYS:
        raise ValueError("unknown evidence key: {0}".format(key))
    source = source.resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError("source evidence must be an existing non-empty file")

    source_sha256 = _sha256(source)
    destination_dir = private_root / key
    destination_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(_safe_filename(source.name))
    destination = destination_dir / (
        safe_name.stem + "-" + source_sha256[:12] + safe_name.suffix
    )
    if destination.exists() and _sha256(destination) != source_sha256:
        raise FileExistsError("refusing to overwrite a different evidence file")
    if not destination.exists():
        shutil.copy2(source, destination)

    relative = destination.resolve().relative_to(root.resolve()).as_posix()
    manifest = _load_manifest(manifest_path)
    evidence = manifest.setdefault("evidence", {})
    assert isinstance(evidence, dict)
    previous = evidence.get(key)
    previous_note = previous.get("note", "") if isinstance(previous, dict) else ""
    evidence[key] = {
        "confirmed": confirmed,
        "evidence_path": relative,
        "sha256": source_sha256,
        "bytes": destination.stat().st_size,
        "attached_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(
            timespec="seconds"
        ),
        "note": note.strip() or previous_note,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence[key]


def status(manifest_path: Path, *, root: Path = ROOT) -> dict[str, object]:
    return manual_evidence_summary(_load_manifest(manifest_path), root=root)


def render_checklist(summary: dict[str, object]) -> str:
    entries = summary.get("entries", {})
    assert isinstance(entries, dict)
    lines = [
        "# Hanoi manual evidence status",
        "",
        "This file is generated from the private-evidence manifest. Attached does not mean confirmed.",
        "",
        "| Key | Attached | Confirmed | Hash valid | Next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in MANUAL_EVIDENCE_KEYS:
        item = entries.get(key, {})
        assert isinstance(item, dict)
        attached = bool(item.get("evidence_path"))
        confirmed = item.get("confirmed") is True
        hash_valid = bool(attached and item.get("actual_sha256") == item.get("sha256"))
        if confirmed and hash_valid:
            action = "Done"
        elif attached and hash_valid:
            action = "Team review, redact if needed, then re-attach with `--confirm`"
        else:
            action = f"Attach real evidence: `manage_hanoi_evidence.py attach --key {key} --file <PATH>`"
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                key,
                "yes" if attached else "no",
                "yes" if confirmed else "no",
                "yes" if hash_valid else "no",
                action,
            )
        )
    lines.extend(
        [
            "",
            "Never confirm screenshots/video before manually checking that credentials and unnecessary PII are redacted.",
            "",
            "Overall gate: **{}**".format("PASS" if summary.get("ok") else "PENDING"),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST.relative_to(ROOT)))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status")
    checklist_parser = subparsers.add_parser("checklist")
    checklist_parser.add_argument(
        "--out", default="docs/HANOI_MANUAL_EVIDENCE_STATUS.md"
    )
    attach_parser = subparsers.add_parser("attach")
    attach_parser.add_argument("--key", required=True, choices=MANUAL_EVIDENCE_KEYS)
    attach_parser.add_argument("--file", required=True)
    attach_parser.add_argument("--note", default="")
    attach_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicitly affirm that the attachment proves this requirement.",
    )
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest).resolve()
    try:
        manifest_path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SystemExit("manifest must remain inside the project") from exc
    if not manifest_path.is_file():
        raise SystemExit("missing manifest: {0}".format(manifest_path))

    if args.command == "attach":
        result = attach(
            manifest_path=manifest_path,
            key=args.key,
            source=Path(args.file),
            note=args.note,
            confirmed=args.confirm,
        )
    elif args.command == "checklist":
        result = status(manifest_path)
        output = (ROOT / args.out).resolve()
        try:
            output.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise SystemExit("checklist output must remain inside the project") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_checklist(result), encoding="utf-8")
        result = {"output": str(output.relative_to(ROOT)), **result}
    else:
        result = status(manifest_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
