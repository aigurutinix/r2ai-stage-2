"""Check shipped answers against the statements themselves.

Cross-branch agreement says two mechanisms concur; it does not say the figure is
in the report. This goes to the source: for each question it finds the company and
year, opens every eligible table of that report, and lists the rows whose label
matches the line item the question names, with their values converted to the unit
the question asks for.

What it prints is evidence, not a verdict — the judgement is the reader's. That is
the point: an automated check would repeat whatever assumption produced the answer
in the first place.

Usage:
  PYTHONPATH=src python scripts/verify_against_reports.py \
      --base guarded3_clean.zip --n 6 --seed 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

UNIT_SCALE = (("nghìn tỷ", 1e12), ("tỷ", 1e9), ("triệu", 1e6), ("nghìn", 1e3))
# Words that appear in every question and so cannot identify a line item.
STOP = {
    "cua", "la", "bao", "nhieu", "nam", "cuoi", "dau", "vao", "tai", "cong",
    "ty", "ctcp", "cp", "va", "cac", "trong", "den", "ngay", "thang", "co",
    "phan", "tap", "doan", "viet", "dong", "trieu", "nghin", "so", "muc",
    "mot", "hay", "theo", "tinh", "gia", "tri", "khoan", "bang", "voi", "cho",
    "tong", "hop", "nhat", "rieng", "me", "ma", "chi", "tieu", "hoi",
}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def metric_phrase(question: str) -> str:
    """The line item, which in Vietnamese sits before the first "của".

    Matching on the whole question pulled in the company name and then matched
    subsidiary rows: a question about SAB's cash returned four rows named "Công ty
    Cổ phần Bia Nước giải khát Sài Gòn - Tây Đô". The metric is the part the
    reader would look up in the index.
    """

    head = re.split(r"của", question, maxsplit=1)[0]
    head = re.sub(r"^(tính|cho biết|hỏi|xác định)\s+", "", head.strip(), flags=re.I)
    return head if len(head) >= 8 else question


def key_words(question: str) -> set:
    return {w for w in fold(question).split() if len(w) > 2 and w not in STOP}


def asked_scale(question: str):
    lowered = question.lower()
    for name, scale in UNIT_SCALE:
        if re.search(rf"bao nhiêu[^?]*\b{re.escape(name)}\b", lowered) or \
           re.search(rf"\({re.escape(name)}\s*đồng\)", lowered):
            return name, scale
    return "đồng", 1.0


def cell_number(text):
    raw = str(text).strip()
    if not raw or raw in ("-", "--"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    cleaned = raw.replace(".", "").replace(",", ".") if "," in raw else raw.replace(".", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--rows", type=int, default=4, help="candidate rows per question")
    # Attribution matters more than volume: a defect in `lookup` is worth six
    # times one in `screen`, because lookup answers 300 questions and screen 7.
    parser.add_argument("--branch", default="",
                        help="only questions answered by this branch "
                             "(artifacts/branches.json)")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    frame = store.frame
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
    rows = payload if isinstance(payload, list) else (
        payload.get("predictions") or list(payload.values())[0])

    # Only questions naming one company and one year can be checked this way; the
    # rest need several reports and the comparison stops being readable.
    wanted_branch = args.branch.strip()
    branches = {}
    if wanted_branch:
        branches = {int(k): v for k, v in json.loads(
            (ROOT / "artifacts" / "branches.json").read_text(encoding="utf-8")).items()}

    simple = []
    for row in rows:
        if wanted_branch and branches.get(row.get("id")) != wanted_branch:
            continue
        parsed = parse_question(0, row["question"], roster)
        if len(parsed.tickers) == 1 and len(parsed.years) == 1:
            simple.append((row, parsed))
    random.Random(args.seed).shuffle(simple)
    print(f"{len(simple)} câu một công ty một năm; đọc {args.n} câu\n")

    for row, parsed in simple[: args.n]:
        ticker = parsed.tickers[0]
        year = str(parsed.years[0])
        unit_name, scale = asked_scale(row["question"])
        wanted = key_words(metric_phrase(row["question"]))

        candidates = frame[(frame.ticker == ticker) & (frame.year == year) &
                           frame.eligible.astype(bool)]
        best = []
        for record in candidates.itertuples():
            grid = store.rows(TableKey(str(record.doc_name), int(record.table_id)))
            if not grid:
                continue
            context = f"{record.unit_page} {record.unit_doc} {record.caption}"
            for index, line in enumerate(grid[1:]):
                label = str(line[0]).strip()
                if not label:
                    continue
                label_words = key_words(label)
                overlap = len(wanted & label_words)
                # Require most of the metric phrase, not two stray words, and
                # skip rows that are plainly a company name.
                if overlap < 2 or overlap < max(2, len(wanted) // 2):
                    continue
                if re.match(r"(công ty|tổng công ty|ngân hàng|ctcp)",
                            label.strip(), re.I):
                    continue
                for column, cell in enumerate(line[1:], start=1):
                    value = cell_number(cell)
                    if value is None or value == 0:
                        continue
                    converted = value * lookup_mod.column_scale(grid, column, context) / scale
                    best.append((overlap, label[:52], str(grid[0][column])[:22],
                                 cell, converted, str(record.doc_name)[:44]))
        best.sort(key=lambda item: -item[0])

        print("=" * 104)
        print(f"CÂU  : {row['question'][:150]}")
        print(f"NỘP  : {row['answer']}   (đơn vị hỏi: {unit_name})")
        if not best:
            print("       không tìm thấy dòng nào khớp từ khoá trong báo cáo")
            continue
        print(f"BÁO CÁO — các dòng khớp nhất, giá trị đã quy về {unit_name}:")
        seen = set()
        shown = 0
        for overlap, label, header, raw, converted, doc in best:
            signature = (label, header)
            if signature in seen:
                continue
            seen.add(signature)
            print(f"   [{overlap} từ chung] {label!r} | cột {header!r} | ô {raw!r}"
                  f" -> {converted:,.2f}")
            shown += 1
            if shown >= args.rows:
                break


if __name__ == "__main__":
    main()
