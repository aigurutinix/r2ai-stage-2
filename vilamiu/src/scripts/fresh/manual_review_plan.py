"""Review address_plan + tab_prog OK splices."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    aud = [
        json.loads(line)
        for line in (ROOT / "artifacts/fresh/pipeline_audit.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    man = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / "artifacts/fresh/pipeline_manifest.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    for layer in ("tab_prog", "address_plan"):
        ok = [a for a in aud if a["verdict"] == "OK" and a["layer"] == layer]
        print(f"=== {layer} ({len(ok)}) ===")
        for a in sorted(ok, key=lambda x: (-x["mag"], x["id"])):
            m = man[a["id"]]
            print(
                f"Q{a['id']:4d} mag={a['mag']:4.2f} block={m.get('block','?'):20s} "
                f"{a['old']:.4g} -> {a['new']:.4g}")
            print(f"       {qs[a['id']][:100]}")
        print()


if __name__ == "__main__":
    main()
