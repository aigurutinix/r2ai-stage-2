"""Render one company-year as a clean, canonical block a model can choose from.

The reason to build this rather than hand a model a table: the earlier attempt gave a
model a 60-row OCR grid of one possibly-wrong table and asked it for a (row, column).
That cost 0.38 questions per row changed, and it could not be checked — a hallucinated
coordinate looks exactly like a real one.

The address book turns that into a bounded choice. Everything below is already
verified: the labels are the consensus rendering across hundreds of reports rather
than one document's OCR, the values satisfy the printed Thông tư 200 identities at
97–99.6%, and the note headings come from tables whose column total ties to a
statement line in both periods.

So the model's job is to name a `Mã số`, or a note, from a list. Its answer is
checkable by construction — a code outside the list is rejected — and every step
around it stays deterministic: the read, the sign rule, the unit conversion and the
emitted program.

Two renderings:

  block   all three statements as (Mã số, label, this period, prior period) plus the
          headings of the notes that tie, at roughly 3-5k tokens
  note    one note's rows, for the second step when the line item is a note detail

Usage:
  python scripts/fresh/render_block.py --ticker HPG --year 2023 --scope consolidated
  python scripts/fresh/render_block.py --question 6
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_identities import IDENTITIES, REL_TOL  # noqa: E402
from refine_codes import RELIABLE_LEN  # noqa: E402

KIND_NAMES = {
    "cdkt": "BẢNG CÂN ĐỐI KẾ TOÁN",
    "kqkd": "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH",
    "lctt": "BÁO CÁO LƯU CHUYỂN TIỀN TỆ",
}
# Codes whose printed sign is unreliable: 17% of these cells are bracketed. Shown as
# magnitudes so the model is not asked to reason about a convention the corpus does
# not keep.
MAGNITUDE_CODES = {"kqkd": {"02", "11", "22", "24", "25", "26", "51", "52"}}


def verified(record: dict) -> bool:
    for kind, target, plus, minus in IDENTITIES:
        if record["kind"] != kind:
            continue
        for period in ("current", "prior"):
            cells = record[period]
            if not all(c in cells for c in (target,) + plus + minus):
                continue
            expected = cells[target][0]
            total = sum(cells[c][0] for c in plus) - sum(cells[c][0] for c in minus)
            if abs(expected - total) <= REL_TOL * max(abs(expected), abs(total), 1.0):
                return True
    return False


class Corpus:
    def __init__(self, index: Path, notes: Path) -> None:
        self.rows: dict[tuple, dict] = defaultdict(dict)
        self.notes: dict[tuple, list[dict]] = defaultdict(list)
        consensus: dict[tuple, Counter] = defaultdict(Counter)

        for line in index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            scope = self.scope_of(record["scope"])
            key = (record["ticker"], record["year"], scope)
            for period in ("current", "prior"):
                for code, cell in record[period].items():
                    slot = self.rows[key].setdefault((record["kind"], code), {})
                    slot.setdefault(period, {
                        "value": cell[0], "label": cell[1], "row": cell[2],
                        "col": cell[3], "csv": record["csv"], "doc": record["doc"],
                        "table_id": record["table_id"],
                        "table_ref": record["table_ref"], "scale": record["scale"],
                    })
            if verified(record):
                want = RELIABLE_LEN[record["kind"]]
                for code, cell in record["current"].items():
                    if len(code) == want and str(cell[1]).strip():
                        consensus[(record["kind"], code)][
                            " ".join(str(cell[1]).split())] += 1
        self.consensus = {k: c.most_common(1)[0][0] for k, c in consensus.items()}

        for line in notes.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            note = json.loads(line)
            if note.get("paired"):
                self.notes[(note["ticker"], note["year"],
                            note["scope"])].append(note)

    @staticmethod
    def scope_of(raw: str) -> str:
        if "separate" in raw:
            return "separate"
        if "consolidated" in raw:
            return "consolidated"
        return raw

    def block(self, ticker: str, year: str, scope: str, *,
              max_notes: int = 40) -> str:
        rows = self.rows.get((ticker, year, scope))
        if not rows:
            return ""
        lines = [f"CÔNG TY {ticker} — NĂM {year} — "
                 f"{'BÁO CÁO RIÊNG (công ty mẹ)' if scope == 'separate' else 'BÁO CÁO HỢP NHẤT'}",
                 "Mọi giá trị đã quy về ĐỒNG.", ""]
        for kind in ("cdkt", "kqkd", "lctt"):
            want = RELIABLE_LEN[kind]
            present = sorted((code for (k, code) in rows if k == kind
                              and len(code) == want),
                             key=lambda c: (len(c), c))
            if not present:
                continue
            lines.append(f"=== {KIND_NAMES[kind]} ===")
            for code in present:
                slot = rows[(kind, code)]
                label = self.consensus.get(
                    (kind, code),
                    (slot.get("current") or slot.get("prior", {})).get("label", ""))
                label = " ".join(str(label).split())[:78]
                magnitude = code in MAGNITUDE_CODES.get(kind, set())

                def show(period: str) -> str:
                    cell = slot.get(period)
                    if not cell:
                        return "—"
                    value = abs(cell["value"]) if magnitude else cell["value"]
                    return f"{value:,.0f}".replace(",", ".")

                lines.append(f"  {code:>4s}  {label:<78s}  "
                             f"{show('current'):>22s}  {show('prior'):>22s}")
            lines.append("")

        notes = self.notes.get((ticker, year, scope), [])
        if notes:
            lines.append("=== THUYẾT MINH (tổng của bảng khớp một dòng ở trên) ===")
            seen = set()
            for note in notes:
                if note["table_ref"] in seen:
                    continue
                seen.add(note["table_ref"])
                if len(seen) > max_notes:
                    lines.append(f"  … còn {len(notes) - max_notes} thuyết minh nữa")
                    break
                heading = " ".join(str(note["heading"]).split())[:96]
                lines.append(f"  [TM{note['table_id']:<4d}] nối {','.join(note['paired'][:2]):<18s} "
                             f"{heading}")
        return "\n".join(lines)

    def note_rows(self, ticker: str, year: str, scope: str, table_id: int) -> str:
        for note in self.notes.get((ticker, year, scope), []):
            if note["table_id"] != table_id:
                continue
            try:
                with (ROOT / note["csv"]).open(encoding="utf-8-sig",
                                               newline="") as file:
                    grid = [row for row in csv.reader(file)]
            except OSError:
                return ""
            lines = [f"THUYẾT MINH TM{table_id} — "
                     f"{' '.join(str(note['heading']).split())[:96]}",
                     f"(nối với {', '.join(note['paired'])})", ""]
            for index, row in enumerate(grid[1:]):
                cells = [str(c).strip() for c in row]
                lines.append(f"  r{index:<3d} " + " | ".join(c[:44] for c in cells))
            return "\n".join(lines)
        return ""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--ticker")
    parser.add_argument("--year")
    parser.add_argument("--scope", default="consolidated")
    parser.add_argument("--note", type=int, default=0)
    parser.add_argument("--question", type=int, default=0)
    args = parser.parse_args()

    corpus = Corpus(ROOT / args.index, ROOT / args.notes)

    ticker, year, scope = args.ticker, args.year, args.scope
    if args.question:
        from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
        from resolve_ticker import TickerResolver  # noqa: E402

        resolver = TickerResolver()
        text = ""
        for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["id"] == args.question:
                text = json.loads(line)["question"]
                break
        found = sorted(resolver.resolve(text))
        years = YEAR_RE.findall(text)
        if not found or not years:
            print("khong nhan ra ma hoac nam")
            return
        ticker, year = found[0], max(years)
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        print(f"CÂU HỎI {args.question}: {text}\n")

    if not ticker or not year:
        print("can --ticker va --year, hoac --question")
        return

    text = (corpus.note_rows(ticker, year, scope, args.note) if args.note
            else corpus.block(ticker, year, scope))
    if not text:
        print(f"khong co du lieu cho {ticker} {year} {scope}")
        return
    print(text)
    print(f"\n--- {len(text)} ky tu, uoc {len(text) // 3.5:.0f} token")


if __name__ == "__main__":
    main()
