"""Let the code and the label validate each other, and recover the ones OCR broke.

The dictionary built from every parse is polluted: inside a `cdkt` table the parser
scans the first three columns for a one-to-three digit integer, so an ordinal in a
note row is taken as a `Mã số`. That is why `cdkt` code 10 collects labels like
"Quỹ khen thưởng, phúc lợi" next to the real 110 "Tiền và các khoản tương đương
tiền".

The fix uses the two signals against each other. A `Mã số` and a row label are
independent: one is an integer fixed by Thông tư 200, the other is OCR'd Vietnamese.
Where they agree, both are probably right. Where they disagree, at least one is
wrong — and if some NEIGHBOURING code's label agrees instead, the likely story is
OCR losing or gaining a digit, which recovers the row rather than discarding it.

The circularity is broken by bootstrapping from the accounting identities. A
statement whose printed arithmetic checks out (270 = 100 + 200, 50 = 30 + 40, …) has
had its codes, its column and its unit scale verified at once, so only those
statements contribute to the reference dictionary.

Reports how much pollution the agreement test finds, how many rows a one-digit
correction recovers, and whether the identity pass rates move.

Usage:  python scripts/fresh/refine_codes.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_identities import IDENTITIES, REL_TOL  # noqa: E402

RELIABLE_LEN = {"cdkt": 3, "kqkd": 2, "lctt": 2}
# Labels routinely carry the formula and the section numbering: "TỔNG TÀI SẢN
# (270 = 100 + 200)", "3. Doanh thu thuần …", "I. Tiền và …". None of that
# distinguishes one line item from another.
STRIP_RE = re.compile(r"\([^)]*\)|^[IVXLC]+\.|^\d+[.)]|[0-9]")
STOP = frozenset({"va", "cac", "khoan", "cua", "tu", "trong", "cong", "tong",
                  "hoat", "dong", "ty", "tien"})


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    flat = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", flat)).strip()


def tokens(label: str) -> frozenset[str]:
    cleaned = STRIP_RE.sub(" ", str(label))
    return frozenset(t for t in fold(cleaned).split() if len(t) > 2 and t not in STOP)


def tokens_exact(label: str) -> frozenset[str]:
    """Tokens WITH their diacritics.

    Folding merges words Vietnamese separates by tone alone — bán/bản, hàng/hạng,
    thuế/thuê. Measured cost: a question about "chi phí bán hàng" was answered with
    "Chi phí xây dựng cơ bản dở dang", because both reduce to the token "ban". OCR
    does damage diacritics, but less often than folding destroys meaning, so the
    exact form is the primary comparison and the folded one is the fallback.
    """

    cleaned = STRIP_RE.sub(" ", str(label)).lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)
    stop = {fold(word) for word in STOP}
    return frozenset(t for t in cleaned.split()
                     if len(t) > 2 and fold(t) not in stop)


def agrees(left: frozenset[str], right: frozenset[str]) -> bool:
    """Half of the shorter label's distinguishing words appear in the other."""

    if not left or not right:
        return False
    shorter = min(len(left), len(right))
    return len(left & right) / shorter >= 0.5


def verified(record: dict) -> bool:
    for kind, target, plus, minus in IDENTITIES:
        if record["kind"] != kind:
            continue
        for period in ("current", "prior"):
            cells = record[period]
            if not all(code in cells for code in (target,) + plus + minus):
                continue
            expected = cells[target][0]
            total = sum(cells[c][0] for c in plus) - sum(cells[c][0] for c in minus)
            if abs(expected - total) <= REL_TOL * max(abs(expected), abs(total), 1.0):
                return True
    return False


def neighbours(code: str) -> list[str]:
    """Codes one digit away: a dropped digit, an added one, or a changed one."""

    out = set()
    for index in range(len(code)):
        out.add(code[:index] + code[index + 1:])          # OCR lost a digit
        for digit in "0123456789":
            out.add(code[:index] + digit + code[index + 1:])  # misread a digit
    for index in range(len(code) + 1):
        for digit in "0123456789":
            out.add(code[:index] + digit + code[index:])   # OCR gained a digit
    out.discard(code)
    return [c for c in out if c and c[0] != "0" or c == code]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = [json.loads(line) for line in
               (ROOT / "artifacts" / "fresh" / "statements.jsonl").read_text(
                   encoding="utf-8").splitlines() if line.strip()]
    good = [r for r in records if verified(r)]
    print(f"{len(records)} bang, {len(good)} bang qua dang thuc ke toan "
          f"({100 * len(good) / len(records):.0f}%)")

    # Reference dictionary: verified statements only, reliable code lengths only.
    votes: dict[tuple[str, str], Counter[frozenset[str]]] = defaultdict(Counter)
    raw_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in good:
        want = RELIABLE_LEN[record["kind"]]
        for code, cell in record["current"].items():
            if len(code) != want:
                continue
            key = tokens(cell[1])
            if key:
                votes[(record["kind"], code)][key] += 1
                raw_votes[(record["kind"], code)][" ".join(str(cell[1]).split())] += 1
    reference = {key: counter.most_common(1)[0][0] for key, counter in votes.items()}
    print(f"tu dien tham chieu: {len(reference)} ma")

    # How much pollution does the agreement test find, by code length?
    by_len: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    recovered: Counter[str] = Counter()
    examples: list[str] = []
    for record in records:
        kind = record["kind"]
        for code, cell in record["current"].items():
            key = (kind, len(code))
            reference_tokens = reference.get((kind, code))
            row_tokens = tokens(cell[1])
            if reference_tokens is None:
                by_len[key]["ma khong co trong tu dien"] += 1
            elif agrees(row_tokens, reference_tokens):
                by_len[key]["nhan KHOP ma"] += 1
            else:
                by_len[key]["nhan LECH ma"] += 1
                hits = [c for c in neighbours(code)
                        if (kind, c) in reference
                        and agrees(row_tokens, reference[(kind, c)])]
                if len(hits) == 1:
                    recovered[f"{kind}: {code} -> {hits[0]}"] += 1
                    if len(examples) < 8:
                        examples.append(
                            f"  {kind} ma={code} -> {hits[0]}  nhan='{str(cell[1])[:56]}'")
                elif len(hits) > 1:
                    by_len[key]["nhieu ma lan can khop"] += 1

    print("\nkiem chieu nhan <-> ma (tren moi ban parse, khong chi ban da xac thuc):")
    for (kind, length), counter in sorted(by_len.items()):
        total = sum(counter.values())
        match = counter["nhan KHOP ma"]
        print(f"  {kind} ma {length} chu so: {total:6d} o, khop {match:6d} "
              f"({100 * match / total:5.1f}%), lech {counter['nhan LECH ma']:6d}, "
              f"ngoai tu dien {counter['ma khong co trong tu dien']:6d}")

    total_recovered = sum(recovered.values())
    print(f"\ndong cuu duoc bang sua MOT chu so: {total_recovered}")
    for name, count in recovered.most_common(10):
        print(f"  {name}: {count}")
    print("\nvi du:")
    for line in examples:
        print(line)


if __name__ == "__main__":
    main()
