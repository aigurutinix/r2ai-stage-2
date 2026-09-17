"""Two-hop answers from the Circular-200 panel: rank by X, read Y (or ratio).

`panel_det.extreme_*` used to answer "Y in the year when X is max" with max(X).
This module is the corrective shape: every year (or company) must resolve the
filter metric, the winner is chosen, then the asked figure is read only there.

Declines unless both sides map onto panel columns — note-line items stay with
`compose.resolve_screen` over OCR tables.
"""

from __future__ import annotations

import re

from vifin.answering.panel_det import (
    PANEL_HEADER,
    PanelAnswer,
    _cell,
    _emit_ratio,
    _pack,
    _scale,
    fold,
    metrics_mentioned,
    named_ratio,
)
from vifin.query.parse import ParsedQuestion

_SUPER = r"(cao nhất|lớn nhất|thấp nhất|nhỏ nhất)"
YEAR_SCREEN = re.compile(
    r"(?:tại|trong|vào|ở)\s+(?:cuối\s+)?năm\s+(?:mà\s+)?có\s+(.{4,90}?)\s+" + _SUPER,
    re.I | re.S,
)
# Trailing form without the preposition: "… năm có vốn chủ sở hữu cao nhất là …"
YEAR_SCREEN_LOOSE = re.compile(
    r"năm\s+có\s+(.{4,90}?)\s+" + _SUPER,
    re.I | re.S,
)
TICKER_SCREEN = re.compile(
    r"(?:doanh nghiệp|công ty|ngân hàng|đơn vị|mã)\s+(?:nào\s+|mà\s+)?có\s+"
    r"(.{4,90}?)\s+" + _SUPER,
    re.I | re.S,
)
MAX_WORDS = frozenset({"cao nhất", "lớn nhất"})
RATIOISH = re.compile(
    r"tỷ lệ|tỉ lệ|tỷ số|hệ số|biên |ROA|ROE|tốc độ tăng|tăng trưởng|trên|chia cho",
    re.I,
)


def _one_metric(phrase: str) -> str | None:
    mets = [m for m in metrics_mentioned(phrase) if not m.startswith("_")]
    if len(mets) != 1:
        return None
    return mets[0]


def _asked_figure(asked: str) -> tuple[str, tuple | None]:
    """Return (`metric`, None) or (`ratio`, (num, den, as_pct))."""

    ratio = named_ratio(asked)
    if ratio is not None:
        return "ratio", ratio
    metric = _one_metric(asked)
    if metric is None:
        return "none", None
    return "metric", (metric,)


def solve_year(question: ParsedQuestion, panel: dict) -> PanelAnswer | None:
    if len(question.tickers) != 1 or len(question.years) < 2:
        return None
    text = question.question
    # Nested cohort before the year-screen ("xét các năm doanh thu tăng, ROE của
    # năm có …") needs a prior filter this module does not apply.
    match = YEAR_SCREEN.search(text) or YEAR_SCREEN_LOOSE.search(text)
    if match is None:
        return None
    head = text[: match.start()]
    if re.search(r"\bxét\b|\btrong số\b|tăng so với|giảm so với", head, re.I):
        return None
    filter_phrase = match.group(1).strip(" ,.;:")
    want_max = match.group(2).lower() in MAX_WORDS
    if RATIOISH.search(filter_phrase):
        return None
    filt = _one_metric(filter_phrase)
    if filt is None:
        return None
    asked = text[: match.start()]
    kind, payload = _asked_figure(asked)
    if kind == "none":
        return None
    # Same metric on both sides → plain extreme, not this module.
    if kind == "metric" and payload[0] == filt:
        return None

    ticker = question.tickers[0]
    years = [str(y) for y in sorted(question.years)]
    scope = question.scope or "consolidated"
    scale = _scale(question)

    scores: dict[str, float] = {}
    body: list[list] = []
    for year in years:
        value = _cell(panel, ticker, year, scope, filt)
        if value is None:
            return None
        scores[year] = abs(value)
        body.append([ticker, year, filt, filt, value])
    winner = (max if want_max else min)(scores, key=scores.get)

    if kind == "ratio":
        num, den, as_pct = payload
        top = _cell(panel, ticker, winner, scope, num)
        bottom = _cell(panel, ticker, winner, scope, den)
        if top is None or bottom is None or bottom == 0:
            return None
        body.append([ticker, winner, num, num, top])
        body.append([ticker, winner, den, den, bottom])
        value = top / bottom * (100.0 if as_pct else 1.0)
        # Program: recompute argmax then ratio — py37-safe loops.
        code = (
            f"sub = df[df['chi_tieu'] == {filt!r}]\n"
            f"vals = {{}}\n"
            f"for i in range(len(sub)):\n"
            f"    y = str(sub.iloc[i]['nam'])\n"
            f"    if y in vals:\n"
            f"        continue\n"
            f"    vals[y] = abs(float(sub.iloc[i]['gia_tri_dong']))\n"
            f"keys = sorted(vals)\n"
            f"win = keys[0]\n"
            f"for y in keys:\n"
            f"    if vals[y] {'>' if want_max else '<'} vals[win]:\n"
            f"        win = y\n"
            f"num = float(df[(df['nam'].map(str) == win) & "
            f"(df['chi_tieu'] == {num!r})].iloc[0]['gia_tri_dong'])\n"
            f"den = float(df[(df['nam'].map(str) == win) & "
            f"(df['chi_tieu'] == {den!r})].iloc[0]['gia_tri_dong'])\n"
            f"result = round(num / den{' * 100.0' if as_pct else ''}, 2)"
        )
        return _pack(body, code, value, "twohop_year_ratio", [filt, num, den])

    metric = payload[0]
    amount = _cell(panel, ticker, winner, scope, metric)
    if amount is None:
        return None
    body.append([ticker, winner, metric, metric, amount])
    code = (
        f"sub = df[df['chi_tieu'] == {filt!r}]\n"
        f"vals = {{}}\n"
        f"for i in range(len(sub)):\n"
        f"    y = str(sub.iloc[i]['nam'])\n"
        f"    if y in vals:\n"
        f"        continue\n"
        f"    vals[y] = abs(float(sub.iloc[i]['gia_tri_dong']))\n"
        f"keys = sorted(vals)\n"
        f"win = keys[0]\n"
        f"for y in keys:\n"
        f"    if vals[y] {'>' if want_max else '<'} vals[win]:\n"
        f"        win = y\n"
        f"got = df[(df['nam'].map(str) == win) & (df['chi_tieu'] == {metric!r})]\n"
        f"result = round(abs(float(got.iloc[0]['gia_tri_dong'])) / {scale!r}, 2)"
    )
    return _pack(body, code, abs(amount) / scale, "twohop_year", [filt, metric])


def solve_ticker(question: ParsedQuestion, panel: dict) -> PanelAnswer | None:
    if len(question.tickers) < 2 or len(question.years) != 1:
        return None
    text = question.question
    match = TICKER_SCREEN.search(text)
    if match is None:
        return None
    filter_phrase = match.group(1).strip(" ,.;:")
    want_max = match.group(2).lower() in MAX_WORDS
    if RATIOISH.search(filter_phrase):
        return None
    filt = _one_metric(filter_phrase)
    if filt is None:
        return None
    asked = text[: match.start()] + " " + text[match.end():]
    kind, payload = _asked_figure(asked)
    if kind == "none":
        return None
    if kind == "metric" and payload[0] == filt:
        return None

    year = str(sorted(question.years)[0])
    scope = question.scope or "consolidated"
    scale = _scale(question)
    scores: dict[str, float] = {}
    body: list[list] = []
    for ticker in question.tickers:
        value = _cell(panel, ticker, year, scope, filt)
        if value is None:
            return None
        scores[ticker] = abs(value)
        body.append([ticker, year, filt, filt, value])
    winner = (max if want_max else min)(scores, key=scores.get)

    if kind == "ratio":
        num, den, as_pct = payload
        top = _cell(panel, winner, year, scope, num)
        bottom = _cell(panel, winner, year, scope, den)
        if top is None or bottom is None or bottom == 0:
            return None
        body.append([winner, year, num, num, top])
        body.append([winner, year, den, den, bottom])
        value = top / bottom * (100.0 if as_pct else 1.0)
        code = (
            f"tickers = []\n"
            f"for x in df['ma'].tolist():\n"
            f"    s = str(x)\n"
            f"    if s in tickers:\n"
            f"        continue\n"
            f"    tickers.append(s)\n"
            f"vals = {{}}\n"
            f"for t in tickers:\n"
            f"    sub = df[(df['ma'] == t) & (df['chi_tieu'] == {filt!r})]\n"
            f"    if len(sub) == 0:\n"
            f"        continue\n"
            f"    vals[t] = abs(float(sub.iloc[0]['gia_tri_dong']))\n"
            f"keys = sorted(vals)\n"
            f"if len(keys) == 0:\n"
            f"    result = 0.0\n"
            f"else:\n"
            f"    win = keys[0]\n"
            f"    for t in keys:\n"
            f"        if vals[t] {'>' if want_max else '<'} vals[win]:\n"
            f"            win = t\n"
            f"    num = float(df[(df['ma'] == win) & "
            f"(df['chi_tieu'] == {num!r})].iloc[0]['gia_tri_dong'])\n"
            f"    den = float(df[(df['ma'] == win) & "
            f"(df['chi_tieu'] == {den!r})].iloc[0]['gia_tri_dong'])\n"
            f"    result = round(num / den{' * 100.0' if as_pct else ''}, 2)"
        )
        return _pack(body, code, value, "twohop_ticker_ratio", [filt, num, den])

    metric = payload[0]
    amount = _cell(panel, winner, year, scope, metric)
    if amount is None:
        return None
    body.append([winner, year, metric, metric, amount])
    code = (
        f"tickers = []\n"
        f"for x in df['ma'].tolist():\n"
        f"    s = str(x)\n"
        f"    if s in tickers:\n"
        f"        continue\n"
        f"    tickers.append(s)\n"
        f"vals = {{}}\n"
        f"for t in tickers:\n"
        f"    sub = df[(df['ma'] == t) & (df['chi_tieu'] == {filt!r})]\n"
        f"    if len(sub) == 0:\n"
        f"        continue\n"
        f"    vals[t] = abs(float(sub.iloc[0]['gia_tri_dong']))\n"
        f"keys = sorted(vals)\n"
        f"win = keys[0]\n"
        f"for t in keys:\n"
        f"    if vals[t] {'>' if want_max else '<'} vals[win]:\n"
        f"        win = t\n"
        f"got = df[(df['ma'] == win) & (df['chi_tieu'] == {metric!r})]\n"
        f"result = round(abs(float(got.iloc[0]['gia_tri_dong'])) / {scale!r}, 2)"
    )
    return _pack(body, code, abs(amount) / scale, "twohop_ticker", [filt, metric])


def solve(question: ParsedQuestion, panel: dict) -> PanelAnswer | None:
    return solve_year(question, panel) or solve_ticker(question, panel)
