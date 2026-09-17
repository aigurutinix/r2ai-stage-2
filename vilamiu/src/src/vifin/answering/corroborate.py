"""Pick between candidate tables using the next year's restated figures.

A lookup often matches a plausible label in several retrieved tables, and taking
the first one is close to a coin flip — cross-year checking put single-cell
extraction at 56.3%. But the corpus carries its own answer key: report(T, Y+1)
restates year Y beside year Y+1. A candidate whose value reappears there is
backed by two independently OCR'd documents.

Note on measurement: once this signal selects the answer, the cross-year check
can no longer score it — the two would be circular. What stays honest to report
is how many questions gain a corroborated candidate; the leaderboard decides the
rest.
"""

from __future__ import annotations

from dataclasses import dataclass

from vifin.answering import lookup as lookup_mod
from vifin.store import TableKey, TableStore

TOLERANCE = 2e-4


@dataclass(frozen=True, slots=True)
class Corroboration:
    key: TableKey
    value: float
    label: str
    confirmed: bool


class Corroborator:
    def __init__(self, store: TableStore) -> None:
        self.store = store
        self._docs: dict[tuple[str, str, str], list[str]] = {}
        for doc_name, ticker, year, scope in (
            store.frame[["doc_name", "ticker", "year", "scope"]]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            self._docs.setdefault((ticker, year, scope), []).append(doc_name)
        self._ids: dict[str, list[int]] = {}
        for doc_name, table_id in zip(store.frame.doc_name, store.frame.table_id):
            self._ids.setdefault(doc_name, []).append(int(table_id))
        self._cache: dict[tuple[str, int], dict[str, float]] = {}

    def _restated(self, doc_name: str, target_year: int) -> dict[str, float]:
        """Every label in `doc_name` mapped to its figure for `target_year`.

        Built once per document: a question's candidates all consult the same
        neighbour, and scanning 74 tables per candidate instead of per document
        made the whole pass unusably slow.
        """

        cached = self._cache.get((doc_name, target_year))
        if cached is not None:
            return cached

        values: dict[str, float] = {}
        for table_id in self._ids.get(doc_name, []):
            grid = self.store.rows(TableKey(doc_name, table_id))
            if len(grid) < 2:
                continue
            label_col = lookup_mod.label_column(grid)
            columns = lookup_mod.value_columns(grid, label_col)
            if not columns:
                continue
            chosen = None
            for column in columns:
                header = " ".join(str(r[column]) for r in grid[:2] if column < len(r))
                if str(target_year) in header:
                    chosen = column
                    break
            if chosen is None:
                # No year in the header: statements list the current period
                # first, so the restated prior year is the second figure column.
                if len(columns) < 2:
                    continue
                chosen = columns[1]
            scale = lookup_mod.column_scale(grid, chosen, "")
            for row in grid[1:]:
                if not row or chosen >= len(row):
                    continue
                if label_col >= len(row):
                    continue
                label = str(row[label_col]).strip()
                if not label or label in values:
                    continue
                parsed = lookup_mod._parse_cell(row[chosen])
                if parsed is not None:
                    values[label] = parsed * scale

        self._cache[(doc_name, target_year)] = values
        return values

    def neighbours(self, ticker: str, year: int, scope: str) -> list[str]:
        return self._docs.get((ticker, str(year + 1), scope), [])

    def choose(self, question, candidates: list[TableKey]) -> Corroboration | None:
        """Best candidate by label-match quality, corroboration breaking ties.

        Walking candidates in retrieval order and taking the first that clears
        MIN_LABEL_SCORE ignored better matches further down: 37 of 309 questions
        used a weaker label than one available elsewhere, and 24 of those skipped
        an exact 1.00 match. The figures differ, not just the wording — one took
        "Tổng nợ phải trả và vốn chủ sở hữu" over "Tổng nợ phải trả".

        Scope note, because this looks like a change that failed: ranking the
        *best-effort* pool by (corroborated, score) cost 8 answers. There, labels
        are matched with no floor, so the score is noise. Here every candidate has
        already cleared 0.75, so the score carries signal — and it leads, with
        cross-year agreement demoted to a tie-break rather than the primary key.
        """

        if not question.tickers or len(question.years) != 1:
            return self._first(question, candidates)

        year = question.years[0]
        restated: dict[str, float] = {}
        for doc_name in self.neighbours(question.tickers[0], year, question.scope):
            restated.update(self._restated(doc_name, year))

        scored: list[tuple[float, int, Corroboration]] = []
        for rank, key in enumerate(candidates):
            grid = self.store.rows(key)
            meta = self.store.meta(key)
            # The caption names the note this table belongs to, which is how an
            # unlabelled total row gets a name to be matched on.
            found = lookup_mod.find(grid, question, str(getattr(meta, "caption", "")))
            if found is None:
                continue
            scale = lookup_mod.column_scale(
                grid, found.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            )
            ours = found.value * scale
            theirs = restated.get(found.label.strip())
            confirmed = theirs is not None and abs(ours - theirs) <= TOLERANCE * max(
                abs(ours), abs(theirs), 1.0
            )
            scored.append(
                (found.score, int(confirmed), Corroboration(key, ours, found.label, confirmed))
            )
        if not scored:
            return None
        # Label score first, corroboration second, retrieval order last.
        best = max(range(len(scored)), key=lambda i: (scored[i][0], scored[i][1], -i))
        return scored[best][2]

    def choose_best_effort(self, question, candidates: list[TableKey]) -> Corroboration | None:
        """Best row across *all* candidates, with no confidence floor.

        The previous version stopped at the first table that yielded any number,
        which is arbitrary — and this pool turned out to answer 5.9% correctly,
        more than the two confident branches contribute together. Ranking every
        candidate and preferring one the next year's report confirms puts real
        signal where there was none.
        """

        restated: dict[str, float] = {}
        if question.tickers and len(question.years) == 1:
            year = question.years[0]
            for doc_name in self.neighbours(question.tickers[0], year, question.scope):
                restated.update(self._restated(doc_name, year))

        # MEASURED, do not "improve" again without measuring: re-ranking these
        # candidates by (corroborated, label score) instead of keeping retrieval
        # order cost 8 correct answers (EXECUTION 0.1818 -> 0.1680).
        #
        # Two reasons, both worth remembering. Cross-year agreement proves the
        # extraction is *consistent*, not that the row is *relevant* — and here
        # the label was matched with no confidence floor, so a consistently
        # extracted irrelevant figure is still wrong. And the candidate order is
        # BM25 over the whole table, a stronger relevance signal than the token
        # Jaccard `find_best_effort` falls back on. Corroboration is now reported
        # but never reorders.
        for key in candidates:
            grid = self.store.rows(key)
            found = lookup_mod.find_best_effort(grid, question)
            if found is None:
                continue
            meta = self.store.meta(key)
            scale = lookup_mod.column_scale(
                grid, found.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            )
            ours = found.value * scale
            theirs = restated.get(found.label.strip())
            confirmed = theirs is not None and abs(ours - theirs) <= TOLERANCE * max(
                abs(ours), abs(theirs), 1.0
            )
            return Corroboration(key, ours, found.label, confirmed)
        return None

    def _first(self, question, candidates: list[TableKey]) -> Corroboration | None:
        for key in candidates:
            grid = self.store.rows(key)
            meta = self.store.meta(key)
            # The caption names the note this table belongs to, which is how an
            # unlabelled total row gets a name to be matched on.
            found = lookup_mod.find(grid, question, str(getattr(meta, "caption", "")))
            if found is None:
                continue
            scale = lookup_mod.column_scale(
                grid, found.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            )
            return Corroboration(key, found.value * scale, found.label, False)
        return None
