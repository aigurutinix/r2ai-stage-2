"""Plan or recoverably apply TTL policy to KINGPRO-owned output folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.governance.audit import get_audit_ledger  # noqa: E402
from kingpro.governance.retention import RetentionManager, default_retention_roots  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manager = RetentionManager(args.outputs)
    plan = manager.plan(default_retention_roots())
    moved = manager.execute(plan) if args.execute else []
    report = {
        "mode": "execute" if args.execute else "dry-run",
        "planned": len(plan),
        "moved": len(moved),
        "actions": moved if args.execute else plan,
        "recoverable": True,
    }
    get_audit_ledger().append(
        "retention.executed" if args.execute else "retention.planned",
        {"planned": len(plan), "moved": len(moved)},
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

