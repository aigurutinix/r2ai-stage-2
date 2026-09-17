"""Resolve which company a question is about, without inventing extra ones.

The coverage breakdown put 252 questions in a "more than one ticker" bucket that
neither answering path handles. Reading them shows the bucket is contaminated:

  "Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023"
  "Giá trị ghi sổ tiền gửi ... của Công ty Cổ phần Viễn thông FPT (FOX) cuối năm ..."

The first is FTS and the second is FOX. Both were flagged as two companies because the
letters FPT inside the company's NAME also match the ticker FPT. Nothing about those
questions is a cohort question.

The order that fixes it:

  1. a code in parentheses is the author stating the ticker outright — take it and
     stop guessing
  2. otherwise match company names, longest first, and REMOVE the matched span before
     looking any further, so letters inside a name cannot be read as a code
  3. only then look for bare codes in what is left

Returns the set of companies actually named. A genuine cohort question still returns
several; a single-company question no longer returns two.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PAREN_CODE_RE = re.compile(r"\(\s*(?:mã\s*(?:ck|cp|chứng khoán)?\s*:?\s*)?"
                           r"([A-Z]{3}[0-9]?)\s*\)")

# `code_stock.csv` writes "CTCP Thép Nam Kim" and the questions write "Công ty Cổ phần
# Thép Nam Kim". Matching the names verbatim missed 51 questions on that difference
# alone. Both sides are expanded to the same long form before comparing.
ABBREVIATIONS = (
    (r"\bctcp\b", "cong ty co phan"),
    (r"\btcp\b", "cong ty co phan"),
    (r"\btong ctcp\b", "tong cong ty co phan"),
    (r"\bcty\b", "cong ty"),
    # `Công ty CP Nông nghiệp Quốc tế Hoàng Anh Gia Lai` — the register writes `CTCP`,
    # the question writes `Công ty CP`; both have to land on `cong ty co phan`.
    (r"\bcong ty cp\b", "cong ty co phan"),
    (r"\bnh\s*tmcp\b", "ngan hang thuong mai co phan"),
    (r"\btmcp\b", "thuong mai co phan"),
    (r"\btnhh\b", "trach nhiem huu han"),
    (r"\bmtv\b", "mot thanh vien"),
    (r"\bbctc\b", "bao cao tai chinh"),
)


def normalise(text: str) -> str:
    """Diacritic-free, abbreviation-expanded, whitespace-collapsed."""

    flat = str(text).replace("đ", "d").replace("Đ", "D")
    flat = "".join(c for c in unicodedata.normalize("NFD", flat)
                   if unicodedata.category(c) != "Mn").casefold()
    flat = re.sub(r"[^a-z0-9\s]", " ", flat)
    for pattern, expansion in ABBREVIATIONS:
        flat = re.sub(pattern, expansion, flat)
    return re.sub(r"\s+", " ", flat).strip()


def load_tickers() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ROOT / "data" / "code_stock.csv").read_text(
            encoding="utf-8").splitlines()[1:]:
        if "," in line:
            code, name = line.split(",", 1)
            out[code.strip()] = name.strip().strip('"')
    return out


class TickerResolver:
    def __init__(self, tickers: dict[str, str] | None = None) -> None:
        self.tickers = tickers or load_tickers()
        # Longest name first so "CTCP Tập đoàn Hòa Phát" is not consumed by a shorter
        # name that happens to be a prefix of it.
        pairs = [(code, name) for code, name in self.tickers.items() if name]
        # Everyday names beside the registered ones. Thirty-five exam questions never got a
        # prompt because they say `Hoà Phát`, `Vinamilk`, `Eximbank`, `Đô thị Kinh Bắc` —
        # and the registered name is `CTCP Tập đoàn Hòa Phát`, `CTCP Sữa Việt Nam`, and so
        # on. Aliases join the same longest-first matching, so `Masan MeatLife` is consumed
        # before `Masan` can claim it.
        aliases = ROOT / "data" / "ticker_aliases.csv"
        if aliases.exists():
            for line in aliases.read_text(encoding="utf-8").splitlines()[1:]:
                if "," not in line:
                    continue
                alias, code = line.rsplit(",", 1)
                alias, code = alias.strip().strip('"'), code.strip()
                if alias and code in self.tickers:
                    pairs.append((code, alias))
        self.by_length = sorted(pairs, key=lambda item: -len(item[1]))

    def resolve(self, text: str) -> set[str]:
        found: set[str] = set()

        # A parenthesised code is the author naming the ticker outright.
        for match in PAREN_CODE_RE.finditer(text):
            code = match.group(1)
            if code in self.tickers:
                found.add(code)

        # Company names next, on the normalised form so "Công ty Cổ phần Thép Nam
        # Kim" matches "CTCP Thép Nam Kim". Each match is removed so letters inside a
        # name cannot be read as a bare code afterwards.
        remaining = normalise(text)
        for code, name in self.by_length:
            flat = normalise(name)
            if not flat:
                continue
            # Whole words only. Substring search read `No Va` inside `nợ vay`, `An Bình`
            # inside `toàn bình quân`, and `Dầu khí Việt Nam` inside `Bao bì Dầu khí Việt
            # Nam` — 50 questions gained a company that is not in them.
            match = re.search(rf"(?<![a-z0-9]){re.escape(flat)}(?![a-z0-9])", remaining)
            if match:
                found.add(code)
                remaining = remaining[:match.start()] + " " + remaining[match.end():]

        # Whatever codes are left standing on their own. The normalised text is
        # lower-cased, so the codes are matched against it in that form.
        for code in self.tickers:
            if re.search(rf"\b{re.escape(code.casefold())}\b", remaining):
                found.add(code)
        return found


def main() -> None:
    import json
    import sys
    from collections import Counter

    sys.stdout.reconfigure(encoding="invalid".replace("invalid", "utf-8"),
                           errors="replace")
    resolver = TickerResolver()
    old_by_length = resolver.by_length

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    before: Counter[int] = Counter()
    after: Counter[int] = Counter()
    fixed = []
    for question in questions:
        text = question["question"]
        naive = {c for c in resolver.tickers
                 if re.search(rf"\b{re.escape(c)}\b", text)}
        for code, name in old_by_length:
            if name.casefold() in text.casefold():
                naive.add(code)
        clean = resolver.resolve(text)
        before[len(naive)] += 1
        after[len(clean)] += 1
        if len(naive) > 1 and len(clean) == 1:
            fixed.append((question["id"], sorted(naive), next(iter(clean)), text))

    print("so ma nhan ra moi cau:")
    print(f"  {'so ma':>6s} {'truoc':>7s} {'sau':>7s}")
    for count in sorted(set(before) | set(after)):
        print(f"  {count:6d} {before.get(count, 0):7d} {after.get(count, 0):7d}")
    print(f"\ncau tu 'nhieu ma' thanh MOT ma: {len(fixed)}")
    for qid, naive, clean, text in fixed[:10]:
        print(f"  id={qid:<5d} {','.join(naive)} -> {clean}")
        print(f"     {text[:104]}")


if __name__ == "__main__":
    main()
