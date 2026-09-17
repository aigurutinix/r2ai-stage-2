"""Evidence-docs missing from relevant_tables; CRLF start_line risk."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.store import TableKey, TableStore  # noqa: E402


def main() -> None:
    ts = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    with zipfile.ZipFile(ROOT / "submissions" / "evidence_pad4.zip") as z:
        recs = json.loads(z.read("submission.json"))

    # rebuild
    sl = {(r.doc_name, int(r.start_line)): TableKey(r.doc_name, int(r.table_id)) for r in ts.frame.itertuples(index=False)}

    missing = []
    for rec in recs:
        ref_docs = {t.split("|", 1)[0] for t in rec["relevant_tables"]}
        evid_keys = []
        for e in rec.get("evidence") or []:
            name = e["csv_path"].split("/")[-1]
            if "_table_" not in name:
                continue  # inline panel
            base = name[: -len(".csv")]
            doc, tid = base.rsplit("_table_", 1)
            evid_keys.append(TableKey(doc, int(tid)))
        for k in evid_keys:
            if k.doc_name not in ref_docs:
                missing.append((rec["id"], k.doc_name, k.table_id, rec["relevant_tables"][:3]))

    print(f"pad4 evidence docs missing from relevant_tables: {len(missing)}")
    for row in missing[:15]:
        print(" ", row)

    # CRLF risk: if OCR has \\r\\n, counting \\n only still gives right visual lines
    # in most editors (\\r doesn't start a new line alone). But if files were
    # normalized differently than gold...
    # Check a sample OCR file newline style vs our start_line.
    sample_docs = list({m[1] for m in missing[:5]}) or []
    # Also pick first declared ref's doc
    sample_ref = recs[0]["relevant_tables"][0]
    doc0 = sample_ref.split("|", 1)[0]
    sample_docs = [doc0] + sample_docs

    text_root = ROOT / "data"
    # find ocr path pattern
    hits = list(text_root.rglob(f"{doc0}*.txt"))[:3]
    print("sample ocr paths for", doc0, ":", hits[:3])
    # Broader search
    if not hits:
        hits = list(text_root.rglob("*.txt"))[:0]
        # walk financial statements layout
        for p in text_root.rglob(doc0 + ".txt"):
            hits.append(p)
            break
    print("found:", hits)

    # Check whether start_line points at <table>
    checked = 0
    bad = 0
    for rec in recs[:50]:
        for ref in rec["relevant_tables"][:1]:
            doc, line = ref.split("|", 1)
            line = int(line)
            key = sl.get((doc, line))
            if key is None:
                continue
            # find txt
            paths = list((ROOT / "data").rglob(doc + ".txt"))
            if not paths:
                continue
            text = paths[0].read_text(encoding="utf-8")
            # get line content at start_line (1-based)
            lines = text.splitlines()
            if line - 1 >= len(lines):
                bad += 1
                print("OOB", ref, "nlines", len(lines))
                continue
            content = lines[line - 1]
            ok = "<table" in content.lower() or content.strip().startswith("<table")
            # table tag might be mid-line
            if not ok:
                # check nearby
                window = "\n".join(lines[max(0, line - 2) : line + 1])
                ok = "<table" in window.lower()
            checked += 1
            if not ok:
                bad += 1
                if bad <= 5:
                    print("MISALIGN", ref, "line:", repr(content[:80]))
    print(f"start_line points at/near <table>: checked={checked} bad={bad}")

    # newline style of one file
    if hits:
        raw = hits[0].read_bytes()[:5000]
        print(
            f"newline bytes in sample: CR={raw.count(b'\\r')} LF={raw.count(b'\\n')} CRLF={raw.count(b'\\r\\n')}"
        )


if __name__ == "__main__":
    main()
