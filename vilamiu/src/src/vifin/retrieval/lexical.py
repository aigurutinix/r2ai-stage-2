"""Baseline table retrieval: hard metadata filter, then BM25 over the survivors.

No embeddings and no GPU. The point is not accuracy but a submittable end-to-end
run: P9 needs real ranked table refs to read the leaderboard's F2 signal, and a
lexical baseline also becomes the floor every later retriever has to beat.

The metadata filter does most of the work. Scoring a few hundred candidates per
question makes a per-question BM25 cheaper than maintaining a corpus-wide index.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from vifin.query.companies import strip_tones
from vifin.query.parse import ParsedQuestion
# MEASURED WORSE ON THE LEADERBOARD, so off by default. Adding the recovered note
# headings and the page prose to the index measured better on `easy_full_units.jsonl`
# — gold table at rank 1 rose 33.8% -> 43.5%, presence in the top eight 72.2% ->
# 76.2%, table selection 38.8% -> 52.9% — and then lost on the real exam:
# `notes_final.zip` scored EXEC 0.3498 against 0.3933, with TABLES_F2 falling
# 0.5603 -> 0.4884.
#
# The local gold set cannot measure retrieval and this project already knew it: 90% of
# its generated questions name the ticker in brackets against 22.6% of the exam, and
# the ticker is the strongest retrieval signal there is. With the ticker handed over,
# an index full of note headings looks sharper; without it, those headings are near
# identical across hundreds of companies ("5.1 Tiền và các khoản tương đương tiền" is
# in almost every report) and make the index less discriminating, not more.
#
# `VIFIN_PROSE=1` re-enables it. The heading extraction itself
# (`scripts/build_note_titles.py`) stays: the defect it documents is real — `caption`
# is a note heading on only 22.5% of tables — and the headings are useful anywhere
# that is not a bag-of-words index over the whole corpus.
USE_PROSE = os.environ.get("VIFIN_PROSE", "0") == "1"

from vifin.store import TableKey

_TOKEN_RE = re.compile(r"[0-9a-zà-ỹăâêôơưđ]+", re.I)

K1 = 1.5
B = 0.75

# Vietnamese words are mostly two syllables, so bigrams carry the real signal:
# "lợi nhuận", "doanh thu", "hàng tồn kho".
USE_BIGRAMS = True


def tokenize(text: str) -> list[str]:
    units = _TOKEN_RE.findall(strip_tones(text).casefold())
    if not USE_BIGRAMS:
        return units
    return units + [f"{a}_{b}" for a, b in zip(units, units[1:])]


@dataclass(frozen=True, slots=True)
class Candidate:
    key: TableKey
    score: float


class LexicalRetriever:
    def __init__(self, frame, eligible_only: bool = True) -> None:
        self.frame = frame[frame.eligible] if eligible_only else frame
        self.frame = self.frame.reset_index(drop=True)

        self._docs_by_ticker: dict[str, list[str]] = defaultdict(list)
        self._doc_meta: dict[str, tuple[str, str, str]] = {}
        for doc_name, ticker, year, scope in (
            self.frame[["doc_name", "ticker", "year", "scope"]].drop_duplicates().itertuples(index=False)
        ):
            self._docs_by_ticker[ticker].append(doc_name)
            self._doc_meta[doc_name] = (ticker, year, scope)

        self._prose = self._load_prose()

        self._rows_by_doc: dict[str, list[int]] = defaultdict(list)
        for position, doc_name in enumerate(self.frame.doc_name):
            self._rows_by_doc[doc_name].append(position)

        self._tokens: list[list[str] | None] = [None] * len(self.frame)

    def _load_prose(self) -> dict:
        """The note text introducing each table, if it has been cached.

        Table selection is the weakest measured step — the right table is in the
        eight-table shortlist 72.5% of the time and chosen 37.9% of those — and the
        caption alone is often the page footer rather than the note title. Built by
        `scripts/build_prose_index.py`; absent, retrieval falls back to the caption.
        """

        root = Path(__file__).resolve().parents[3] / "artifacts"
        if not USE_PROSE:
            return {}
        # `caption` is the note heading on only 22.5% of tables; the rest are form
        # codes, company names, addresses and date lines, so the index has been
        # built partly on page furniture. `build_note_titles.py` recovers the real
        # heading for 95.5% of tables and lifts table selection from 38.8% to 52.9%.
        # `table_prose.jsonl` is the earlier, cruder window and stays as a fallback.
        # Both files earn their place and were measured separately: the heading is
        # short and precise, which lifted rank-1 from 33.8% to 37.5%, while the
        # wider prose window is wordier and lifted presence in the top eight from
        # 72.2% to 76.5%. Indexing the two together keeps each effect.
        merged: dict[tuple[str, int], str] = {}
        # Which extra text goes into the index. The two were only ever submitted
        # together, and that lost; each on its own is still unmeasured on the
        # leaderboard, which is the only instrument that can settle it.
        sources = {
            "notes": (("table_notes.jsonl", "note"),),
            "prose": (("table_prose.jsonl", "prose"),),
            "both": (("table_notes.jsonl", "note"), ("table_prose.jsonl", "prose")),
        }[os.environ.get("VIFIN_PROSE_SRC", "both")]
        for name, field in sources:
            path = root / name
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (row["doc"], int(row["table_id"]))
                    text = str(row[field])
                    merged[key] = f"{merged[key]} {text}" if key in merged else text
        return merged

    def _table_tokens(self, position: int) -> list[str]:
        cached = self._tokens[position]
        if cached is not None:
            return cached
        row = self.frame.iloc[position]
        grid = json.loads(row.rows_json)
        # Caption carries the note number and section heading; the first column
        # carries the line-item labels a question actually names.
        parts = [str(row.caption), str(row.unit_line)]
        note = self._prose.get((str(row.doc_name), int(row.table_id)))
        if note:
            parts.append(note)
        if grid:
            parts.extend(str(cell) for cell in grid[0])
            parts.extend(str(line[0]) for line in grid if line)
        tokens = tokenize(" ".join(parts))
        self._tokens[position] = tokens
        return tokens

    def candidate_docs(self, parsed: ParsedQuestion) -> list[str]:
        """Documents matching ticker, year, and scope, with graceful widening."""

        tickers = parsed.tickers or list(self._docs_by_ticker)
        years = {str(y) for y in parsed.years}

        pool = [d for t in tickers for d in self._docs_by_ticker.get(t, [])]
        if years:
            narrowed = [d for d in pool if self._doc_meta[d][1] in years]
            # Prior-year comparatives live in the following year's report, so a
            # missing year is often still answerable from an adjacent one.
            if not narrowed:
                adjacent = {str(y + 1) for y in parsed.years} | {str(y - 1) for y in parsed.years}
                narrowed = [d for d in pool if self._doc_meta[d][1] in adjacent]
            pool = narrowed or pool

        preferred = [d for d in pool if self._doc_meta[d][2] == parsed.scope]
        return preferred or pool

    def search_balanced(self, parsed: ParsedQuestion, per_group: int = 2, cap: int = 8) -> list[Candidate]:
        """Ranked tables with a quota per (ticker, year).

        A comparison across five companies scored globally collapses onto
        whichever company's wording happens to match best, leaving the model
        without the other side of the comparison. Reserving slots per entity and
        period keeps every operand present.
        """

        groups: dict[tuple[str, str], list[Candidate]] = {}
        for candidate in self.search(parsed, top_k=cap * 4):
            ticker, year, _ = self._doc_meta[candidate.key.doc_name]
            groups.setdefault((ticker, year), []).append(candidate)

        picked: list[Candidate] = []
        for rank in range(per_group):
            for members in groups.values():
                if rank < len(members):
                    picked.append(members[rank])
        picked.sort(key=lambda c: -c.score)
        return picked[:cap]

    def search(self, parsed: ParsedQuestion, top_k: int = 5) -> list[Candidate]:
        docs = self.candidate_docs(parsed)
        positions = [p for d in docs for p in self._rows_by_doc.get(d, [])]
        if not positions:
            return []

        query = Counter(tokenize(parsed.question))
        doc_tokens = {p: self._table_tokens(p) for p in positions}
        lengths = {p: len(t) or 1 for p, t in doc_tokens.items()}
        avg_len = sum(lengths.values()) / len(lengths)

        frequency: Counter[str] = Counter()
        for tokens in doc_tokens.values():
            frequency.update(set(tokens) & query.keys())

        total = len(positions)
        idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }

        scored: list[Candidate] = []
        for position in positions:
            counts = Counter(doc_tokens[position])
            length = lengths[position]
            score = 0.0
            for term, weight in query.items():
                freq = counts.get(term, 0)
                if not freq:
                    continue
                denominator = freq + K1 * (1 - B + B * length / avg_len)
                score += idf.get(term, 0.0) * weight * freq * (K1 + 1) / denominator
            if score > 0:
                row = self.frame.iloc[position]
                scored.append(Candidate(TableKey(row.doc_name, int(row.table_id)), score))

        scored.sort(key=lambda c: -c.score)
        return scored[:top_k]
