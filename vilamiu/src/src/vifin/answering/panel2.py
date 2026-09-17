"""Build a per-question evidence panel from the question's own metric phrases.

330 questions land in the dead zone — cell plans (7.2%), best-effort fallback
(6%), and 46 that ship no program at all. Together they hold roughly 160 graded
answers and get about nine of them right. Every lexical mechanism that could
serve them has been built and measured; what is left needs multi-step reasoning
over several companies and years, which is the one thing the model does better
than a token match.

The first panel attempt failed for a reason that is not the idea's fault. It
resolved values through `corpus.metrics.METRICS`, a fixed whitelist of named
line items, and required *every* company-year and *every* referenced metric to be
present — which rejected 287 of 287 candidates and left 42 questions. The
leaderboard verdict "panel is neutral" was then measured on 34 of them, at a
sample size this project has already been burned by (63% on n=35 became 33% on
n=100).

This version drops the whitelist. The metric phrases come from the question
itself, resolved with the same label matcher that carries the strongest branch,
and a partial panel is allowed: a hole is a row the model can see is missing,
which is safer than a hole hidden behind a refusal to build anything.

The panel is shipped as an inline CSV, so the program the model writes runs
against exactly what it was shown.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from vifin.answering import compose, lookup as lookup_mod, ratio as ratio_mod
from vifin.query.parse import ParsedQuestion
from vifin.store import TableKey, TableStore

PANEL_HEADER = ["ma", "nam", "chi_tieu", "nhan_trong_bao_cao", "gia_tri_dong"]

# More rows than this and the prompt stops being a clean table; the model starts
# scanning instead of reasoning, which is the failure the panel exists to avoid.
MAX_ROWS = 60


@dataclass(frozen=True, slots=True)
class Panel:
    rows: list[list[str]]
    keys: list[TableKey]
    phrases: list[str]
    filled: int
    wanted: int


def metric_phrases(question: ParsedQuestion) -> list[str]:
    """Every figure the question refers to, not just the one it asks for.

    A screen names two ("số dư vay dài hạn ... trong năm có tổng vốn chủ sở hữu
    cao nhất") and a ratio names two more. Resolving only the asked metric leaves
    the model without the operand it needs to choose between years.
    """

    phrases: list[str] = []

    def offer(text: str | None) -> None:
        text = (text or "").strip(" ,.;:")
        if text and text not in phrases:
            phrases.append(text)

    shape = compose.screen_shape(question)
    if shape is not None:
        # The asked metric sits in front of the screen clause; the filter metric
        # is what the clause captured.
        pattern = (compose.SCREEN_YEAR_RE if shape[0] == "year"
                   else compose.SCREEN_TICKER_RE)
        match = pattern.search(question.question)
        if match:
            offer(lookup_mod.extract_metric(question.question[: match.start()]))
        offer(shape[1])
    else:
        offer(lookup_mod.extract_metric(question.question))

    quotient = ratio_mod.shape(question)
    if quotient is not None:
        offer(quotient[0])
        offer(quotient[1])
    return phrases


def expand_formulas(phrases: list[str]) -> list[str]:
    """Replace a named ratio with the line items it is computed from.

    "Biên lợi nhuận gộp" is not a row in any statement, so asking the label
    matcher for it returns nothing. Its operands are rows, and `ratio.FORMULAS`
    already holds the rewrite — so the panel carries those instead and
    `plan_json` divides them back at execution time.
    """

    out: list[str] = []
    for phrase in phrases:
        matched = False
        for pattern, numerator, denominator in ratio_mod.FORMULAS:
            if pattern.search(phrase):
                for part in (numerator, denominator):
                    if part not in out:
                        out.append(part)
                matched = True
                break
        if not matched and phrase not in out:
            out.append(phrase)
    return out


def _resolve(question, phrase, ticker, year, store, retriever, top_k):
    probe = dataclasses.replace(
        question, question=phrase, tickers=[ticker], years=[year]
    )
    best = None
    for variant in lookup_mod.metric_variants(phrase):
        for hit in retriever.search(dataclasses.replace(probe, question=variant),
                                    top_k=top_k):
            meta = store.meta(hit.key)
            if str(meta.ticker) != str(ticker) or str(meta.year) != str(year):
                continue
            found = lookup_mod.find(store.rows(hit.key), probe)
            if found is None:
                continue
            if best is None or found.score > best[1].score:
                scale = lookup_mod.column_scale(
                    store.rows(hit.key), found.column,
                    f"{meta.unit_page} {meta.unit_doc} {meta.caption}",
                )
                best = (hit.key, found, scale)
    return best


def build(
    question: ParsedQuestion,
    store: TableStore,
    retriever,
    top_k: int = 8,
    phrases: list[str] | None = None,
) -> Panel | None:
    """One row per (company, year, metric phrase) that resolves to a figure.

    `phrases` is normally derived from the question by regex, which is the weak
    step: on a screen ranked by a computed ratio, `screen_shape` declines and the
    fallback hands back the whole trailing clause. Letting the caller pass the
    phrases lets the model name them instead — it reads that structure reliably,
    which is the one thing it does better here than a pattern.
    """

    if phrases is None:
        phrases = metric_phrases(question)
    phrases = expand_formulas(phrases)
    tickers = question.tickers or []
    years = sorted(question.years) or []
    if not phrases or not tickers or not years:
        return None
    if len(tickers) * len(years) * len(phrases) > MAX_ROWS:
        years = years[-6:]
        if len(tickers) * len(years) * len(phrases) > MAX_ROWS:
            return None

    rows: list[list[str]] = [list(PANEL_HEADER)]
    keys: list[TableKey] = []
    wanted = 0
    for ticker in tickers:
        for year in years:
            for phrase in phrases:
                wanted += 1
                got = _resolve(question, phrase, ticker, year, store, retriever, top_k)
                if got is None:
                    continue
                key, found, scale = got
                if key not in keys:
                    keys.append(key)
                rows.append([
                    str(ticker), str(year), phrase, found.label.strip(),
                    repr(abs(found.value) * scale),
                ])
    filled = len(rows) - 1
    # One row is not a panel — the single-cell branches already had their turn at
    # anything that thin, and the model would only be re-deciding what they did.
    if filled < 2:
        return None
    return Panel(rows=rows, keys=keys, phrases=phrases, filled=filled, wanted=wanted)


def render(panel: Panel) -> str:
    lines = [",".join(PANEL_HEADER)]
    for row in panel.rows[1:]:
        lines.append(",".join(cell.replace(",", " ") for cell in row))
    return "\n".join(lines)
