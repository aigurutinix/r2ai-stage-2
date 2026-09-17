"""Answer a note-table question in two steps, the hard one done by arithmetic.

A note question names a line item with a qualifier that the statutory chart does not
contain — "trả trước cho người bán ngắn hạn TRONG NƯỚC", "cho vay khách hàng NGÀNH
THƯƠNG MẠI", "thù lao HĐQT CHU THỊ BÌNH". 250 of the questions the Mã số book cannot
reach look like that.

The decomposition that makes them tractable:

  which note?   the note's column total equals the statement line it details, and
                that is arithmetic over the address book, not a wording match. 21% of
                non-statement tables tie on BOTH periods to the same code, which is
                two independent matches in two different columns.
  which row?    inside one identified note there are about ten rows, so matching the
                question's qualifier against them is a small closed problem rather
                than a search over 143,000 tables.

The first step also recovers the note's unit scale, because only one of 1, 1e3, 1e6,
1e9 makes its total land on a statement line.

The question's own head noun picks the note: "trả trước cho người bán ngắn hạn" is the
statement line, so the note tied to the code whose consensus label matches it is the
one to open. The qualifier is then what distinguishes the row inside it.

Usage:  python scripts/fresh/plan_notes.py --show 12
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

import parse_statements as ps  # noqa: E402
from check_identities import IDENTITIES, REL_TOL  # noqa: E402
from map_metric import contradicts  # noqa: E402
from plan_answers import OPENING_RE, PARENT_RE, STRIP_CHUNKS, YEAR_RE  # noqa: E402
from refine_codes import RELIABLE_LEN, tokens_exact  # noqa: E402

PERIOD_LABEL_RE = re.compile(
    r"^\s*(?:năm|số|tại ngày|cuối|đầu|kỳ)\s*(?:nay|trước|này|cuối năm|đầu năm)?\s*"
    r"[\d/.\s-]*$", re.I)


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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/note_plan.jsonl")
    parser.add_argument("--show", type=int, default=10)
    # A question the Mã số book already addresses must not be routed through a note.
    # Without this, "lợi nhuận trước thuế" — which is kqkd/50 — was answered from a
    # tax-reconciliation note with the row "Lỗ năm trước chuyển sang".
    parser.add_argument("--skip-plan", default="artifacts/fresh/answer_plan.jsonl")
    # The qualifier left after removing the statement line's own name is what picks
    # the row. When only one token survives, its Jaccard against a row label is
    # noise: the wrong samples all scored 0.12 to 0.29.
    parser.add_argument("--min-residual", type=int, default=2)
    parser.add_argument("--min-row-score", type=float, default=0.4)
    parser.add_argument("--route", choices=("parent", "direct"), default="direct")
    args = parser.parse_args()

    already = set()
    skip_path = ROOT / args.skip_plan
    if skip_path.exists():
        for line in skip_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                already.add(json.loads(line)["id"])
    print(f"{len(already)} cau da co dia chi Ma so, bo qua")

    # Consensus label per code, from statements the identities verified.
    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not verified(record):
            continue
        want = RELIABLE_LEN[record["kind"]]
        for code, cell in record["current"].items():
            if len(code) == want:
                label = tokens_exact(cell[1])
                if label:
                    votes[(record["kind"], code)][label] += 1
    labels = {k: v.most_common(1)[0][0] for k, v in votes.items()}
    print(f"tu dien ma: {len(labels)}")

    # (ticker, year, scope) -> code -> [note]
    by_code: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    total_notes = 0
    for line in (ROOT / args.notes).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        note = json.loads(line)
        if not note.get("paired"):
            continue
        total_notes += 1
        key = (note["ticker"], note["year"], note["scope"])
        for ref in note["paired"]:
            kind, code = ref.split("/")
            by_code[key][f"{kind}/{code}"].append(note)
    print(f"thuyet minh noi ca hai ky: {total_notes}")

    tickers = {}
    for line in (ROOT / "data" / "code_stock.csv").read_text(
            encoding="utf-8").splitlines()[1:]:
        if "," in line:
            code, name = line.split(",", 1)
            tickers[code.strip()] = name.strip().strip('"')
    by_length = sorted(tickers.items(), key=lambda item: -len(item[1]))

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    counters: Counter[str] = Counter()
    plan, samples = [], []
    for question in questions:
        if question["id"] in already:
            counters["da co dia chi Ma so"] += 1
            continue
        text = question["question"]
        found = {c for c in tickers if re.search(rf"\b{re.escape(c)}\b", text)}
        for code, name in by_length:
            if name and name.casefold() in text.casefold():
                found.add(code)
        years = YEAR_RE.findall(text)
        if len(found) != 1 or not years:
            counters["ngoai pham vi"] += 1
            continue
        ticker = next(iter(found))
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        notes_here = by_code.get((ticker, max(years), scope))
        if not notes_here:
            counters["khong co thuyet minh noi duoc cho ma-nam-pham vi"] += 1
            continue

        probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", text)
        probe = re.sub(r"\([^)]*\)", " ", probe)
        probe = YEAR_RE.sub(" ", probe)
        for chunk in (tickers[ticker],) + STRIP_CHUNKS:
            probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
        probe_tokens = tokens_exact(probe)
        if not probe_tokens:
            counters["khong con tu de doi chieu"] += 1
            continue

        # MEASURED AND ABANDONED: routing through the parent line. The design assumed
        # the question names the statement line and a qualifier picks the row inside
        # its note. It does not — a note question names ONLY the sub-item ("Lãi tiền
        # gửi", "Doanh thu cho thuê khô tàu bay") and never the parent
        # ("Doanh thu bán hàng và cung cấp dịch vụ"). 187 questions failed to match
        # any parent line, and the whole path yielded 3 answers, all of them derived
        # questions a single cell cannot answer anyway.
        #
        # So the tie becomes a FILTER rather than a routing key: a table whose total
        # lands on a statement line is a real financial breakdown, and the question's
        # own words are matched against the rows of every such table in that report —
        # about twenty, not 143,000.
        best_ref = None
        best_score = 0.0
        remaining = probe_tokens
        if args.route == "parent":
            for ref in notes_here:
                kind, code = ref.split("/")
                label = labels.get((kind, code))
                if not label:
                    continue
                shared = probe_tokens & label
                if not shared or contradicts(probe, " ".join(label)):
                    continue
                score = len(shared) / len(label)
                if score > best_score:
                    best_ref, best_score = ref, score
            if best_ref is None or best_score < 0.6:
                counters["khong khop dong bao cao nao co thuyet minh"] += 1
                continue
            remaining = probe_tokens - labels[tuple(best_ref.split("/"))]
            if len(remaining) < args.min_residual:
                counters["phan du qua mong de chon dong"] += 1
                continue

        candidates = (notes_here[best_ref] if best_ref is not None else
                      [n for group in notes_here.values() for n in group])
        seen_tables = set()
        rows_scored = []
        for note in candidates:
            if note["table_ref"] in seen_tables:
                continue
            seen_tables.add(note["table_ref"])
            with (ROOT / note["csv"]).open(encoding="utf-8-sig", newline="") as file:
                grid = [row for row in csv.reader(file)]
            for index, row in enumerate(grid[1:]):
                if not row:
                    continue
                label = max((str(c).strip() for c in row
                             if not any(ch.isdigit() for ch in str(c))),
                            key=len, default="")
                row_tokens = tokens_exact(label)
                if not row_tokens:
                    continue
                # A period header is not a line item. "Năm trước" and "Số cuối năm"
                # were both selected as answers before this check existed.
                if PERIOD_LABEL_RE.match(label.strip()):
                    continue
                hit = len(remaining & row_tokens) / len(remaining | row_tokens)
                if hit >= args.min_row_score:
                    rows_scored.append((hit, note, index, label))
        if not rows_scored:
            counters["mo duoc thuyet minh nhung khong khop dong nao"] += 1
            continue
        rows_scored.sort(key=lambda item: -item[0])
        hit, note, row_index, label = rows_scored[0]
        runner = rows_scored[1][0] if len(rows_scored) > 1 else 0.0
        if hit - runner < 0.05:
            counters["hoa giua cac dong trong thuyet minh"] += 1
            continue

        # The cell, not just the row. Same positional convention the statement parser
        # uses: the row's first value cell is the current period and the second is the
        # prior one. And the note's own scale is whatever its ties agreed on — only
        # one of 1, 1e3, 1e6, 1e9 makes a total land on a statement line, so the tie
        # recovered it even where the page declares no unit.
        with (ROOT / note["csv"]).open(encoding="utf-8-sig", newline="") as file:
            note_grid = [row for row in csv.reader(file)]
        row_cells = note_grid[row_index + 1] if row_index + 1 < len(note_grid) else []
        value_cols = [i for i, cell in enumerate(row_cells)
                      if str(cell).strip() and not ps.BARE_INT_RE.match(str(cell).strip())
                      and ps.parse_vn_number(str(cell)) is not None]
        wanted = 1 if OPENING_RE.search(text) else 0
        if len(value_cols) <= wanted:
            counters["dong khop nhung khong co o gia tri cho ky do"] += 1
            continue
        column = value_cols[wanted]
        scales = Counter(tie["scale"] for tie in note["ties"])
        scale = scales.most_common(1)[0][0]

        counters["CO DIA CHI THUYET MINH"] += 1
        plan.append({"id": question["id"], "ref": best_ref,
                     "col": column, "scale": scale,
                     "period": "prior" if wanted else "current",
                     "line_score": round(best_score, 3), "row_score": round(hit, 3),
                     "doc": note["doc"], "table_id": note["table_id"],
                     "table_ref": note["table_ref"], "csv": note["csv"],
                     "row": row_index, "row_label": label,
                     "heading": note["heading"][:120]})
        if len(samples) < args.show:
            samples.append(
                f"  id={question['id']:<5d} {best_ref} dong={row_index} "
                f"diem_dong={best_score:.2f} diem_muc={hit:.2f}\n"
                f"     hoi  : {text[:100]}\n"
                f"     muc  : {label[:76]}\n"
                f"     tm   : {note['heading'][:76]}")

    (ROOT / args.out).write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in plan),
        encoding="utf-8")
    print()
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print("\nvi du:")
    for line in samples:
        print(line)
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
