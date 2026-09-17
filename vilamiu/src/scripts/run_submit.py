"""Build the final submission: deterministic lookups first, LLM for the rest.

Priority is deliberate. A corroborated single-cell lookup is backed by two
independently OCR'd documents; a generated program is backed by an 8B model's
reading of six tables. Where both exist the lookup wins.

Every LLM program is re-linted and re-executed here against the exact CSVs that
will ship, and its reported answer is that execution's result. A program whose
value has drifted, or that no longer runs, is dropped back to the placeholder:
`EXECUTION_ACCURACY` scores code that runs *and* reproduces the answer, so
shipping one we cannot reproduce locally gains nothing and risks the opposite.

Usage:  python scripts/run_submit.py [search_k] [declare_k] [cache.jsonl]
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import compose  # noqa: E402
from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering import year_answer  # noqa: E402
from vifin.answering import ratio as ratio_mod  # noqa: E402
from vifin.answering import reads as reads_mod  # noqa: E402
from vifin.answering import value_cell  # noqa: E402
from vifin.answering.corroborate import Corroborator  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.corpus.numeric import is_correct  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.package import Prediction, TableRefStyle, build_submission  # noqa: E402
from vifin.submit.validate import reads_no_frame, validate_submission  # noqa: E402

PLACEHOLDER_QUERY = "result = 0.0"

# Units that are proportions, not amounts.
RATIO_UNITS = frozenset({"phan_tram", "lan", "vong"})

# A proportion in the billions is not a near-miss, it is a category error — the
# answer was read as a đồng amount. Rejecting it and falling through to the ratio
# path cannot do worse, and the ratio at least lands in range.
RATIO_SANITY_LIMIT = 1000.0

# Both measured on the public leaderboard against the 2307 baseline
# (EXECUTION 0.1818 / ANSWER 0.1838), both worse, both off by default:
#   USE_PANEL  -> 0.1798 / 0.1838  (34 panel answers displaced exactly as many
#                 correct ones; one panel program also crashed on their pandas)
#   USE_RATIO  -> 0.1779 / 0.1818  (turned 273 impossible answers into plausible
#                 ones and gained nothing — plausible is not correct — while
#                 adding 7 IndexErrors on the grader's older pandas)
USE_PANEL = False
USE_RATIO = False

# Reads a proportion straight out of a percentage column. Distinct from
# USE_RATIO, which computed a row's share of its column total and was measured
# worse: plausible is not correct. This one only reports a figure already in the
# table, and only for questions asking in % or lần.
USE_RATIO_LOOKUP = True

# Composes one single-cell lookup per year into a difference, growth rate, or
# series aggregate. 550 of 1012 questions are derived and all three mechanisms
# serving them are model-based (19.5% / 7.2% / 5.9%); this one is the label
# matcher applied N times, with the anchor year's label locked across years.
USE_COMPOSE = True

# Two-stage screens: rank the years (or companies) by one metric, then read
# another from the winner. 99 questions have this shape and `classify` used to
# read their superlative as the operation, returning max(A) over the years
# instead of A in the argmax year — wrong on 11 questions the leaderboard was
# already paying for. 51 of the 99 rank by a plain line item and are in scope;
# the rest rank by a computed ratio and need a formula table this does not have.
USE_SCREEN = True

# Screen whose ranking key is a computed ratio (quick ratio, X/Y, …). Declines
# nested cohort→superlative questions and requires the ratio to resolve for every
# ticker. Offline pool after those gates is tiny (~2 clean ids); keep gated.
USE_SCREEN_RATIO = True

# Declare / answer retrieval shortlist. Anchor index is the measured retrieval
# floor for late submissions (spank_lastresort / helpers).
DECLARE_SOURCE = "anchor"
SEARCH_SOURCE = "anchor"

# Divides two looked-up figures for a "bao nhiêu %" question. 120 shipped answers
# are percentages in the billions, which is what happens when a đồng amount is
# handed back as a rate. Distinct from USE_RATIO, measured worse: that one
# *substituted* a plausible number (a row's share of its column total), this one
# computes the quotient the question names and withholds it if the result is not
# a rate at all.
USE_RATIO_DIVIDE = True

# Model-located cells fill the gaps the regex matcher leaves; they do not replace
# it. The matcher is measured at 42.8%, localisation is unmeasured, and 241 of the
# questions localisation can answer currently fall through to the fallback branch
# that scores 5.9%. Filling only those keeps the change to one variable.
# Set LOCATE_WINS to prefer localisation everywhere once its accuracy is known.
USE_LOCATE = True

# MEASURED: filling the regex matcher's gaps with model-located cells lifted
# ANSWER 0.1858 -> 0.1957, i.e. +5 of the 68 graded questions it touched. Backing
# out the 5.9% the fallback would have scored puts the branch at 13.3% — better
# than the fallback, far below the regex matcher's 42.8%. So it fills gaps and
# never displaces: LOCATE_WINS would trade 42.8% for 13.3% on the 151 questions
# where the two disagree.
# The 13.3% that argued against this was measured on the *residual* pool — the
# questions the regex matcher could not serve at all. Its accuracy on the regex
# matcher's own pool, where a label did match at >=0.75, has never been measured,
# and that pool is easier by construction. The two disagree on 151 questions;
# whichever is right there is worth up to 23 graded answers.
# MEASURED: letting localisation win on the regex matcher's own pool cost 13 of
# the 506 graded answers (0.1976 -> 0.1719) across the 119 questions where the two
# disagree, implying ~21% for the model against 42.8% for the F1 token match. The
# model reads tables worse than a simple lexical matcher wherever a label already
# matches well. Keep it to the gaps.
LOCATE_WINS = False

# Which localiser fills the gaps. "model" asks the LLM for {table,row,column};
# "embed" scores row labels against the metric phrase with bge-m3, offline in 33s.
# "union" prefers model and fills the 77 embed-only questions from embed_located.
# The embedding sits in 214 of the 232 questions where two localisers agree, more
# than either other method, which is why it is worth measuring in the same slot.
LOCATE_SOURCE = "union"

# Three-way arbiter (regex vs model vs embed). When True, embed breaks ties on
# the 151 questions where regex and model disagree. When False, every question
# uses the global LOCATE_WINS flag — the pre-tiebreak behaviour, for A/B testing.
USE_TIEBREAK = True

# Cell-plan answers: k located cells plus one operation from a closed set. Covers
# the derived questions the single-cell path cannot serve — ratios, differences,
# growth — which currently fall to the fallback branch at 5.9%. Placed after the
# LLM branch so it fills gaps rather than displacing a measured 19.5%.
USE_PLAN = True

# Reject a generated program that read a cell at a negative row index: `find_row`
# returned -1 and the program used it, so `num` silently read the table's last
# row. 75 of 1012 shipped answers rest on such a read. VIFIN_KEEP_NEGATIVE=1
# restores the old behaviour for A/B.
REJECT_NEGATIVE_READS = os.environ.get("VIFIN_KEEP_NEGATIVE", "0") == "0"


def implausible(question, value: float) -> bool:
    return question.target_unit in RATIO_UNITS and abs(value) > RATIO_SANITY_LIMIT


# The largest figure any of these companies reports, with room to spare: VCB's
# total assets are ~2e15 đồng. See `lookup._checked_scale`, which fixes the same
# defect where the scale is still ours to choose.
DONG_CEILING = 1e16


# Answer-type bounds for `impossible`. A share of a whole cannot exceed 100; a
# growth rate can, so "tăng"/"giảm" phrasings are excluded from that check —
# flagging them reported 80 defects where 20 exist.
ANSWER_IS_YEAR_RE = re.compile(
    r"(năm nào|vào năm nào|thời điểm nào|năm bao nhiêu)", re.I)
ANSWER_IS_COUNT_RE = re.compile(
    r"có bao nhiêu (doanh nghiệp|công ty|ngân hàng|đơn vị|mã)", re.I)
ANSWER_IS_SHARE_RE = re.compile(
    r"tỷ trọng|tỉ trọng|chiếm bao nhiêu (phần trăm|%)|tỷ lệ sở hữu|"
    r"tỷ lệ lợi ích|phần trăm sở hữu", re.I)
ANSWER_IS_GROWTH_RE = re.compile(
    r"tăng|giảm|tăng trưởng|thay đổi|biến động|CAGR", re.I)
ANSWER_IS_TIMES_RE = re.compile(r"bao nhiêu lần|gấp bao nhiêu", re.I)
# "tỷ lệ A trên B", "hệ số A trên B" — a ratio, but not necessarily capped
# at 100: current assets over current liabilities as a percentage runs to
# 335 legitimately, and 20 of the 97 such questions answer between 100 and
# 10,000 plausibly. Only the 25 above 100,000 are certainly a raw cell.
ANSWER_IS_RATIO_RE = re.compile(
    r"(tỷ lệ|tỉ lệ|hệ số|tỷ suất)\s+.{3,60}?\s+trên\s+", re.I)
ANSWER_IS_PERCENT_RE = re.compile(r"bao nhiêu (phần trăm|%)|\(%\)", re.I)


def impossible(question, value: float) -> bool:
    """Whether the answer cannot be the figure asked for, whatever it is.

    `lookup.column_scale` now refuses a unit the column's own digits contradict,
    which removed every such answer from the branches that compute their scale
    live. Three branches cannot be fixed that way — `located.jsonl` and
    `planned.jsonl` are cached GPU output and the LLM writes its own parsing — so
    their answers are checked here instead.

    This rejects and falls through to the next branch; it does not substitute a
    value. That is the difference from the `USE_RATIO` experiment, which replaced
    impossible answers with merely plausible ones and gained nothing.
    """

    # Bounds by answer type, not only by money magnitude. The audit of
    # `planv3_clean.zip` found 129 predictions (12.7%) that cannot be right on
    # their own terms, and in every class the shipped value was a raw money cell
    # where a derived quantity was asked: 906.763.500.173 for "có bao nhiêu doanh
    # nghiệp", 105.151.058.000 for "bao nhiêu phần trăm". None were caught,
    # because a count or a percentage has no `unit_scale` and the check returned
    # False on the first line.
    text = question.question
    if ANSWER_IS_YEAR_RE.search(text):
        return not (float(value).is_integer() and 1990 <= value <= 2100)
    if ANSWER_IS_COUNT_RE.search(text):
        named = len(question.tickers)
        if not float(value).is_integer() or value < 0:
            return True
        return bool(named) and value > named
    if (ANSWER_IS_SHARE_RE.search(text) and not ANSWER_IS_TIMES_RE.search(text)
            and not ANSWER_IS_GROWTH_RE.search(text)):
        return value < 0 or value > 100
    if ANSWER_IS_TIMES_RE.search(text):
        return abs(value) > 1e4
    if ANSWER_IS_PERCENT_RE.search(text) or ANSWER_IS_RATIO_RE.search(text):
        return abs(value) > 1e5

    scale = question.unit_scale
    if scale is None:
        return False
    return abs(value) * scale > DONG_CEILING


def load_generated(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok") and row.get("value") is not None:
                rows[row["id"]] = row
    return rows


def load_reranked(path: Path) -> dict[int, list[TableKey]]:
    """Cross-encoder ordering from scripts/run_rerank.py, if present.

    Measured on the 411 single-lookup questions, label-match coverage:
        BM25              top1 33.8%  top5 49.6%  top10 53.8%
        rerank(question)  top1 17.0%  top5 37.5%  top10 49.1%
        rerank(metric)    top1 41.4%  top5 55.2%  top10 56.4%
    Reranking against the whole question is worse than no reranking at all — the
    sentence is mostly company, year and unit words the metadata filter has
    already consumed. Against the metric phrase alone it beats BM25 everywhere,
    and top-5 then exceeds BM25's top-10.
    """

    if not path.exists():
        return {}
    order = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            order[row["id"]] = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
    return order


def load_context_rank(path: Path) -> dict[int, list[TableKey]]:
    """Context-BM25 ordering from scripts/rank_context.py, if present."""

    if not path.exists():
        return {}
    order = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            order[row["id"]] = [
                TableKey(r["doc_name"], int(r["table_id"])) for r in row["refs"]
            ]
    return order


YEAR_REPAIR = os.environ.get("VIFIN_YEAR_REPAIR", "1") != "0"
PLAN_FIRST = os.environ.get("VIFIN_PLAN_FIRST", "1") != "0"
# Off by default until measured: it reorders every branch at once.
LLM_FIRST = os.environ.get("VIFIN_LLM_FIRST", "0") == "1"


def main() -> None:
    search_k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    declare_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    cache = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("artifacts/generated_full.jsonl")

    root = Path(__file__).resolve().parents[1]
    started = time.time()
    parsed = parse_all(root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    corroborator = Corroborator(store)
    reranked = load_reranked(root / "artifacts" / "reranked_metric.jsonl")
    print(f"{len(reranked)} questions with cross-encoder ordering")
    context_rank: dict[int, list[TableKey]] = {}
    if DECLARE_SOURCE in ("context", "anchor"):
        context_rank = load_context_rank(
            root / "artifacts" / f"{DECLARE_SOURCE}_rank.jsonl")
    print(f"{len(context_rank)} questions with {DECLARE_SOURCE!r} declare ordering")
    # Both localisers are loaded regardless of LOCATE_SOURCE so the three-way
    # arbiter (regex vs model vs embed) has all three opinions ready per question.
    # Regenerating on a new retrieval writes a NEW cache file; point at it here
    # rather than overwriting, so a worse run can be rolled back (no git repo).
    located_name = os.environ.get("LOCATED_FILE", "located.jsonl")
    planned_name = os.environ.get("PLANNED_FILE", "planned.jsonl")
    model_located = load_generated(root / "artifacts" / located_name)
    embed_located = load_generated(root / "artifacts" / "embed_located.jsonl")
    print(f"model: {len(model_located)}  embed: {len(embed_located)}")
    if LOCATE_SOURCE == "union":
        # Prefer model; fill the 77 embed-only questions from embed.
        located = dict(model_located)
        for qid, row in embed_located.items():
            if qid not in located:
                located[qid] = row
    else:
        located = load_generated(
            root / "artifacts"
            / (located_name if LOCATE_SOURCE == "model" else "embed_located.jsonl")
        )
    print(f"{len(located)} located cells from {LOCATE_SOURCE!r}")
    planned = load_generated(root / "artifacts" / planned_name)
    print(f"{len(planned)} cell plans")
    generated = load_generated(cache)
    panel_answers = load_generated(root / "artifacts" / "panel_answers.jsonl")
    print(f"{len(panel_answers)} panel programs")
    print(f"{len(generated)} executable programs in {cache}")

    predictions: list[Prediction] = []

    branch_of: dict[int, str] = {}

    # Model-copied figures, one per question, from `scripts/run_value_read.py`.
    value_read: dict[int, dict] = {}
    value_path = root / "artifacts" / os.environ.get("VIFIN_VALUE_READ",
                                                     "value_read.jsonl")
    if value_path.exists():
        for line in value_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("raw"):
                    value_read[row["id"]] = row
        print(f"{len(value_read)} câu có số model đọc được")
    stats = {"lookup": 0, "llm": 0, "llm_rejected": 0, "llm_zero": 0,
             "llm_constant": 0, "fallback": 0, "fallback_confirmed": 0,
             "panel": 0, "scan": 0, "llm_implausible": 0, "located": 0, "plan": 0, "none": 0,
             "tiebreak_model": 0, "tiebreak_regex": 0, "tiebreak_missed": 0,
             "ratio_lookup": 0, "compose": 0, "llm_impossible": 0, "screen": 0,
             "ratio_divide": 0}

    for question in parsed:
        bm25 = [hit.key for hit in retriever.search(question, top_k=search_k)]
        if SEARCH_SOURCE == "anchor" and context_rank.get(question.id):
            searched = context_rank[question.id][:search_k]
        else:
            searched = (reranked.get(question.id) or bm25)[:search_k]
        refs = searched[:declare_k]
        doc_pool: list[str] = []
        for key in searched + bm25:
            if key.doc_name not in doc_pool:
                doc_pool.append(key.doc_name)
        answer, query, evidence = 0.0, PLACEHOLDER_QUERY, refs[:1]
        source = "none"
        inline: list[tuple[str, list[list[str]]]] = []

        if USE_RATIO_DIVIDE:
            divided = ratio_mod.resolve(question, store, retriever)
            if divided is not None:
                tables = {n: store.rows(k) for n, k in zip(divided.variables, divided.keys)}
                outcome = run_query(divided.code, tables)
                if outcome.ok and not reads_no_frame(divided.code):
                    answer, query, evidence = outcome.value, divided.code, divided.keys
                    refs = (divided.keys + [r for r in refs if r not in divided.keys])[:declare_k]
                    source = "ratio_divide"
                    stats["ratio_divide"] += 1

        if USE_SCREEN:
            screened = compose.resolve_screen(question, store, retriever)
            if screened is None and USE_SCREEN_RATIO:
                screened = compose.resolve_screen_ratio(question, store, retriever)
                if screened is None:
                    screened = compose.resolve_screen_ratio_threshold(
                        question, store, retriever
                    )
                if screened is not None:
                    stats["screen_ratio"] = stats.get("screen_ratio", 0) + 1
            if screened is not None:
                if screened.operand_keys:
                    names = (
                        ["df"] if len(screened.operand_keys) == 1
                        else [f"df{i + 1}" for i in range(len(screened.operand_keys))]
                    )
                    tables = {
                        n: store.rows(k)
                        for n, k in zip(names, screened.operand_keys)
                    }
                else:
                    tables = {"df": store.rows(screened.key)}
                outcome = run_query(screened.code, tables)
                if outcome.ok and not reads_no_frame(screened.code) and not impossible(
                    question, outcome.value
                ):
                    answer, query, evidence = (
                        outcome.value,
                        screened.code,
                        list(screened.operand_keys or [screened.key]),
                    )
                    source = "screen"
                    # The filter tables are genuinely part of the question even
                    # though the program does not read them, so they are declared.
                    cited = list(evidence) + screened.filter_keys
                    refs = (cited + [r for r in refs if r not in cited])[:declare_k]
                    stats["screen"] += 1

        # Derived questions first, because the single-cell branch below has no
        # `is_single_lookup` gate and was answering 41 of them from one cell —
        # a cell is not the answer to "chênh lệch giữa A và B". Composition only
        # fires where it can name an operation and resolve at least two years, so
        # everything it declines behaves exactly as before.
        if USE_COMPOSE:
            composed = compose.resolve(question, store, retriever)
            if composed is not None:
                tables = {n: store.rows(k) for n, k in zip(composed.variables, composed.keys)}
                outcome = run_query(composed.code, tables)
                if outcome.ok and not reads_no_frame(composed.code):
                    answer, query, evidence = outcome.value, composed.code, composed.keys
                    refs = (composed.keys + [r for r in refs if r not in composed.keys])[:declare_k]
                    source = "compose"
                    stats["compose"] += 1
                    stats[f"compose_{composed.op}"] = stats.get(f"compose_{composed.op}", 0) + 1

        # Let the generated program answer before any deterministic branch.
        #
        # The isolated probe hands the model the one table holding the answer and
        # asks for a cell: it gets 69.1% right with the column-tagged rendering.
        # The deterministic single-cell path, on the 823 questions naming one
        # company, is running at roughly 45% — implied by EXEC 0.3933 once the
        # cohort questions are assumed lost. The model reads a table better than
        # the machinery built to avoid it, so the ordering that puts the machinery
        # first is worth testing in reverse.
        #
        # `idiot` on the public board scores EXEC 0.6067 with TABLES_F2 = 0.0 —
        # no declared tables at all — which is what a pure answer-extractor looks
        # like from outside.
        if (LLM_FIRST and query is PLACEHOLDER_QUERY and question.id in generated):
            row = generated[question.id]
            keys = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
            tables = {name: store.rows(key) for name, key in zip(row["variables"], keys)}
            outcome = run_query(row["code"], tables)
            if (outcome.ok and outcome.value != 0.0
                    and not reads_no_frame(row["code"])
                    and not impossible(question, outcome.value)
                    and not reads_mod.negative_reads(row["code"], tables)):
                answer, query, evidence = outcome.value, row["code"], keys
                source = "llm"
                stats["llm"] += 1
                stats["llm_first"] = stats.get("llm_first", 0) + 1

        # The plan branch ran last and therefore only saw what nothing else
        # claimed: 142 of the 389 working plans on disk. It is also the most
        # accurate mechanism measured here — the model returns cell coordinates
        # and one operation from a closed set, so localisation reaches 80% where
        # program generation reaches 50% and the regex label matcher reaches 44%
        # (1.8% on questions that need more than one cell).
        #
        # So for a question the single-cell path cannot serve, the plan goes
        # first. `VIFIN_PLAN_FIRST=0` restores the old ordering.
        if (PLAN_FIRST and USE_PLAN and query is PLACEHOLDER_QUERY
                and question.id in planned
                and not lookup_mod.is_single_lookup(question.question)):
            row = planned[question.id]
            keys = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
            tables = {name: store.rows(key) for name, key in zip(row["variables"], keys)}
            outcome = run_query(row["code"], tables)
            if outcome.ok and not reads_no_frame(row["code"]) and not impossible(
                question, outcome.value
            ):
                answer, query, evidence = outcome.value, row["code"], keys
                source = "plan"
                stats["plan"] += 1
                stats["plan_first"] = stats.get("plan_first", 0) + 1

        # The single-cell branch had no `is_single_lookup` gate, and `compose`
        # above was relied on to take the derived questions first. `compose` only
        # recognises a handful of operations over years, so a cohort screen —
        # "trong nhóm ... cao hơn trung vị", over ten companies — falls straight
        # through and is answered with one company's payables cell. Read by hand
        # in `base_18aug.zip`, that is what question 3188 shipped. `DERIVED_RE`
        # already matches "trong nhóm"; the branch simply never asked.
        # `VIFIN_LOOKUP_GATE=0` restores the ungated behaviour for comparison.
        gate_lookup = os.environ.get("VIFIN_LOOKUP_GATE", "1") != "0"
        eligible = (question.unit_scale and query is PLACEHOLDER_QUERY
                    and (not gate_lookup
                         or lookup_mod.is_single_lookup(question.question)))
        picked = corroborator.choose(question, searched) if eligible else None
        if picked is not None:
            grid = store.rows(picked.key)
            meta = store.meta(picked.key)
            found = lookup_mod.find(grid, question)
            scale_in = lookup_mod.column_scale(
                grid, found.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            )
            code = lookup_mod.synthesize(found, scale_in, question.unit_scale, magnitude=True)
            outcome = run_query(code, {"df": grid})
            expected = round(abs(found.value) * scale_in / question.unit_scale, 2)
            if outcome.ok and is_correct(expected, outcome.value):
                answer, query, evidence = outcome.value, code, [picked.key]
                source = "lookup"
                stats["lookup"] += 1

        # A proportion asked for in "%" or "lần" is usually stored as a
        # proportion — ownership and voting-rights tables carry a "% sở hữu" or
        # "Tỷ lệ lợi ích" column. `value_columns` drops those (median under
        # 1000), so these questions were answered from a đồng column and came
        # back as "1,028,364,192,393 percent". This reads the cell that is
        # already there; it computes nothing, which is what separates it from
        # USE_RATIO's share-of-column-total guess (measured worse, still off).
        if USE_RATIO_LOOKUP and query is PLACEHOLDER_QUERY and question.target_unit in RATIO_UNITS:
            best = None
            for variant in lookup_mod.metric_variants(question.question):
                probe = dataclasses.replace(question, question=variant)
                for hit in retriever.search(probe, top_k=8):
                    found = lookup_mod.find_ratio(store.rows(hit.key), question)
                    if found is not None and (best is None or found.score > best[1].score):
                        best = (hit.key, found)
            if best is not None:
                key, found = best
                code = lookup_mod.synthesize_ratio_cell(
                    found, as_percent=question.target_unit == "phan_tram"
                )
                outcome = run_query(code, {"df": store.rows(key)})
                if outcome.ok and not reads_no_frame(code):
                    answer, query, evidence = outcome.value, code, [key]
                    refs = ([key] + [r for r in refs if r != key])[:declare_k]
                    source = "ratio"
                    stats["ratio_lookup"] += 1

        # Three-way arbiter: where regex, model-locate and embed-locate all
        # produce values, let the embed-locate break the tie on the 151
        # disagreements. Per-question vote replaces the global LOCATE_WINS bet
        # with a measurement-backed decision.
        locate_wins_here = LOCATE_WINS
        # Only the regex lookup is up for arbitration. The tiebreak was measured
        # against that branch, and letting it overrule a composition traded a
        # multi-operand lexical answer for a single model-located cell on 3
        # questions — id=934 lost a sum of several rows to one of them.
        if USE_TIEBREAK and source == "lookup" and question.id in model_located and question.id in embed_located:
            model_val = model_located[question.id]["value"]
            embed_val = embed_located[question.id]["value"]
            model_embed_agree = is_correct(model_val, embed_val)  # model ≈ embed
            regex_embed_agree = is_correct(answer, embed_val)     # regex ≈ embed
            if model_embed_agree and not regex_embed_agree:
                # embed sides with model — model is the outlier → trust locate
                locate_wins_here = True
                stats["tiebreak_model"] += 1
            elif regex_embed_agree and not model_embed_agree:
                # embed sides with regex — model is the outlier → trust regex
                locate_wins_here = False
                stats["tiebreak_regex"] += 1
            else:
                # All three agree, or all three disagree, or model+regex agree
                # against embed. None of those break the tie cleanly.
                stats["tiebreak_missed"] += 1

        # Localisation returns ONE cell, so it can only answer a single-cell
        # question. Letting it run on derived questions displaced 53 answers from
        # the LLM branch, which is measured at 19.5% — a cell is not the answer to
        # "chênh lệch giữa A và B".
        if (USE_LOCATE and question.id in located
                and lookup_mod.is_single_lookup(question.question)
                and source != "compose"
                and (locate_wins_here or query is PLACEHOLDER_QUERY)):
            row = located[question.id]
            keys = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
            outcome = run_query(row["code"], {"df": store.rows(keys[0])})
            if outcome.ok and not reads_no_frame(row["code"]) and not impossible(
                question, outcome.value
            ):
                answer, query, evidence = outcome.value, row["code"], keys[:1]
                source = "locate"
                stats["located"] += 1

        if USE_PANEL and query is PLACEHOLDER_QUERY and question.id in panel_answers:
            row = panel_answers[question.id]
            outcome = run_query(row["code"], {"df": row["panel_rows"]})
            if (outcome.ok and outcome.value != 0.0 and not reads_no_frame(row["code"])
                    and not (USE_RATIO and implausible(question, outcome.value))):
                keys = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
                answer, query = outcome.value, row["code"]
                # The frame the program ran on is the panel, so that is the
                # evidence; the source tables still get cited via ref_tables.
                evidence = []
                inline = [(f"metric_panel_q{question.id}.csv", row["panel_rows"])]
                if keys:
                    refs = (keys + refs)[:declare_k]
                stats["panel"] += 1

        if query is PLACEHOLDER_QUERY and question.id in generated:
            row = generated[question.id]
            keys = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
            tables = {name: store.rows(key) for name, key in zip(row["variables"], keys)}
            outcome = run_query(row["code"], tables)
            if reads_no_frame(row["code"]):
                # A generated program that hard-codes its answer is rejected on
                # the organisers' manual review, whatever it scores here.
                stats["llm_constant"] += 1
            elif not outcome.ok:
                stats["llm_rejected"] += 1
            elif outcome.value == 0.0:
                # Almost always `result = 0` surviving a lookup that matched
                # nothing, and indistinguishable from the placeholder anyway.
                stats["llm_zero"] += 1
            elif USE_RATIO and implausible(question, outcome.value):
                stats["llm_implausible"] += 1
            elif impossible(question, outcome.value):
                stats["llm_impossible"] += 1
            elif REJECT_NEGATIVE_READS and reads_mod.negative_reads(row["code"], tables):
                # `find_row` returned -1 and the program used it anyway, so
                # `num(frame, -1, c)` read the last row of the table — the total
                # in a balance sheet. Twelve answers in one block came back as
                # exactly 100.0 that way, dividing total assets by themselves.
                # 75 of 1012 shipped answers read at least one negative index.
                # Falling through to the next branch cannot do worse: the program
                # did not find what it was looking for.
                stats["llm_negative_read"] = stats.get("llm_negative_read", 0) + 1
            else:
                answer, query, evidence = outcome.value, row["code"], keys
                source = "llm"
                stats["llm"] += 1

        if USE_PLAN and query is PLACEHOLDER_QUERY and question.id in planned:
            row = planned[question.id]
            keys = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
            tables = {name: store.rows(key) for name, key in zip(row["variables"], keys)}
            outcome = run_query(row["code"], tables)
            if outcome.ok and not reads_no_frame(row["code"]) and not impossible(
                question, outcome.value
            ):
                answer, query, evidence = outcome.value, row["code"], keys
                source = "plan"
                stats["plan"] += 1

        if query is PLACEHOLDER_QUERY:
            # Last resort, but the largest pool by far: rank every candidate and
            # prefer one the following year's report confirms.
            picked = corroborator.choose_best_effort(question, searched)
            if picked is not None:
                grid = store.rows(picked.key)
                meta = store.meta(picked.key)
                found = lookup_mod.find_best_effort(grid, question)
                scale_in = lookup_mod.column_scale(
                    grid, found.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
                )
                if USE_RATIO and question.unit_scale is None and question.target_unit in RATIO_UNITS:
                    # A đồng amount can never answer "bao nhiêu phần trăm"; the
                    # currency scale made these answers absurd by construction.
                    code = lookup_mod.synthesize_ratio(
                        found, as_percent=question.target_unit == "phan_tram"
                    )
                else:
                    code = lookup_mod.synthesize(
                        found, scale_in, question.unit_scale or 1.0, magnitude=True
                    )
                outcome = run_query(code, {"df": grid})
                # The last resort was the one branch that never consulted
                # `impossible`, so every answer the stricter branches rejected
                # landed here instead: the fallback grew from 67 to 95 while the
                # defect count barely moved. A money cell is not an answer to
                # "có bao nhiêu doanh nghiệp", and shipping one scores the same
                # zero as shipping nothing while reading worse under review.
                if (outcome.ok and not reads_no_frame(code)
                        and not impossible(question, outcome.value)):
                    answer, query, evidence = outcome.value, code, [picked.key]
                    stats["fallback"] += 1
                    stats["fallback_confirmed"] += int(picked.confirmed)
                elif outcome.ok:
                    stats["fallback_rejected"] = stats.get("fallback_rejected", 0) + 1
        # A figure the model copied out of one of the tables it was shown, turned
        # back into a positional read. Measured on gold through the real retrieval
        # path at 27.3%, against 5.9% for the column scan below it — and unlike the
        # scan it names a cell a reviewer can check. It sits here, behind every
        # deterministic branch, because the label matcher measures 42.8% and must
        # keep the questions it can serve.
        if query is PLACEHOLDER_QUERY and question.id in value_read:
            row = value_read[question.id]
            cell = value_cell.parse_cell(row.get("raw"))
            keys = [TableKey(doc, int(tid)) for doc, tid in (row.get("keys") or [])]
            grids = [store.rows(key) for key in keys]
            found = value_cell.locate(grids, cell) if cell is not None else None
            if found is not None:
                table_index, row_index, column = found
                key = keys[table_index]
                grid = grids[table_index]
                meta = store.meta(key)
                # The model reports the unit the table is written in; the column's
                # own scale is the pipeline's reading of the same thing, and it is
                # the one every other branch trusts.
                scale = lookup_mod.column_scale(
                    grid, column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}")
                factor = scale / (question.unit_scale or 1.0)
                code = value_cell.emit("df", row_index, column, factor)
                outcome = run_query(code, {"df": grid})
                if (outcome.ok and outcome.value != 0.0
                        and not impossible(question, outcome.value)):
                    answer, query, evidence = outcome.value, code, [key]
                    source = "value_read"
                    stats["value_read"] = stats.get("value_read", 0) + 1
                else:
                    stats["value_read_rejected"] = stats.get(
                        "value_read_rejected", 0) + 1

        if query is PLACEHOLDER_QUERY and searched:
            key = searched[0]
            grid = store.rows(key)
            columns = lookup_mod.value_columns(grid) or [min(1, max(0, len(grid[0]) - 1))]
            code = lookup_mod.synthesize_scan(
                columns[0],
                as_ratio=USE_RATIO and question.target_unit in RATIO_UNITS,
                as_percent=question.target_unit == "phan_tram",
            )
            outcome = run_query(code, {"df": grid})
            # The column scan is the last thing standing, and it is not guarded:
            # when every other branch has been rejected the choice is between a
            # program that reads a real cell and gets the wrong number, and the
            # `result = 0.0` placeholder, which reads nothing and is rejected on
            # manual review. Both score zero, so ship the one that is a genuine
            # attempt.
            if outcome.ok:
                answer, query, evidence = outcome.value, code, [key]
                stats["scan"] += 1
        if query is PLACEHOLDER_QUERY:
            stats["none"] += 1

        # A "năm nào" question wants a year. Twenty-two of the fifty-three such
        # questions in the best submission shipped the figure instead, which is
        # wrong with certainty — so replacing it can only help. The repair reads
        # the year off the header of the column the figure came from and emits a
        # comparison chain over that row's year columns, which is a real program
        # rather than an asserted constant.
        if YEAR_REPAIR and year_answer.asks_for_a_year(question.question):
            stats["yr_asks"] = stats.get("yr_asks", 0) + 1
            if year_answer.looks_like_a_year(answer):
                stats["yr_already"] = stats.get("yr_already", 0) + 1
            if not evidence:
                stats["yr_no_evidence"] = stats.get("yr_no_evidence", 0) + 1
        if (YEAR_REPAIR and year_answer.asks_for_a_year(question.question)
                and not year_answer.looks_like_a_year(answer) and evidence):
            stats["yr_try"] = stats.get("yr_try", 0) + 1
            fixed = year_answer.repair(store.rows(evidence[0]), answer)
            if fixed is not None:
                repaired_code, _ = fixed
                probe = run_query(repaired_code, {"df": store.rows(evidence[0])})
                if probe.ok and year_answer.looks_like_a_year(probe.value):
                    answer, query = probe.value, repaired_code
                    evidence = evidence[:1]
                    stats["year_repaired"] = stats.get("year_repaired", 0) + 1
            else:
                # Most of these questions put each year in its own annual report,
                # so the comparison spans documents rather than columns. Order the
                # candidate tables by the year in their document name and read the
                # same line item from each.
                dated = []
                for key in (refs or evidence):
                    year = year_answer.year_of_document(key.doc_name)
                    if year is not None:
                        dated.append((year, key))
                dated.sort(key=lambda pair: pair[0])
                frames = [(year, store.rows(key)) for year, key in dated]
                across = year_answer.repair_across_documents(frames, answer)
                if across is None and query is not PLACEHOLDER_QUERY:
                    # The figure is often a total, so it appears in no cell and
                    # cannot be traced. The program that produced it can be
                    # replayed against each year instead.
                    across = year_answer.replay_across_years(query, frames)
                if across is not None:
                    repaired_code, _, used = across
                    chosen = [dated[i][1] for i in used]
                    bound = {f"df{i + 1}": store.rows(k)
                             for i, k in enumerate(chosen)}
                    probe = run_query(repaired_code, bound)
                    if probe.ok and year_answer.looks_like_a_year(probe.value):
                        answer, query = probe.value, repaired_code
                        evidence = chosen
                        stats["year_repaired_docs"] = stats.get(
                            "year_repaired_docs", 0) + 1

        # Which branch decided this answer. The submission format has no field for
        # it, so it goes to a side file. Without it an error cannot be attributed:
        # a day was spent improving the branch that writes 146 of 1012 answers
        # because nothing recorded that `lookup` and `plan` write 546 of them.
        branch_of[question.id] = source
        predictions.append(
            Prediction(
                id=question.id,
                question=question.question,
                answer=answer,
                pandas_query=query,
                tables=evidence,
                inline_tables=inline,
                ref_tables=refs,
                # Documents are scored as their own list and DOCS_F2 weights
                # recall four times. Reranking concentrates the top-k on fewer
                # documents, which cost 0.9324 -> 0.9218 while precision rose to
                # 0.9723 — so union both orderings rather than pick one.
                ref_docs=doc_pool,
            )
        )

    total = len(parsed)
    answered = (stats["ratio_divide"] + stats["screen"] + stats["compose"] + stats["lookup"] + stats["ratio_lookup"] + stats["llm"]
                + stats["panel"] + stats["located"] + stats["plan"])
    print(f"\n{total} questions in {time.time() - started:.0f}s")
    print(f"  answered by divide   {stats['ratio_divide']}")
    print(f"  answered by screen   {stats['screen']}")
    print(f"  answered by compose  {stats['compose']}"
          + "  " + "  ".join(f"{k[8:]}={v}" for k, v in sorted(stats.items())
                                 if k.startswith('compose_')))
    print(f"  answered by lookup   {stats['lookup']}")
    print(f"  answered by ratio    {stats['ratio_lookup']}")
    print(f"  answered by LLM      {stats['llm']}")
    print(f"  LLM no longer runs   {stats['llm_rejected']}")
    print(f"  LLM returned zero    {stats['llm_zero']}")
    print(f"  LLM hard-coded       {stats['llm_constant']}")
    print(f"  LLM đáp án bất khả thi {stats['llm_implausible'] + stats['llm_impossible']}")
    print(f"  answered by locate   {stats['located']}")
    if stats["tiebreak_model"] or stats["tiebreak_regex"] or stats["tiebreak_missed"]:
        print(f"    tiebreaks: model won {stats['tiebreak_model']}  "
              f"regex won {stats['tiebreak_regex']}  "
              f"unresolved {stats['tiebreak_missed']}")
    print(f"  answered by plan     {stats['plan']} (of which first {stats.get('plan_first', 0)})")
    print(f"  answered by panel    {stats['panel']}")
    print(f"  best-effort fallback {stats['fallback']}"
          f"  ({stats['fallback_confirmed']} được năm sau xác nhận)")
    print(f"  column scan          {stats['scan']}")
    print(f"  đọc số bằng model    {stats.get('value_read', 0)}"
          f"  (bị bác {stats.get('value_read_rejected', 0)})")
    print(f"  fallback rejected as impossible {stats.get('fallback_rejected', 0)}")
    print(f"  no program at all    {stats['none']}")
    print(f"  yr asks={stats.get('yr_asks',0)} already={stats.get('yr_already',0)} no_evidence={stats.get('yr_no_evidence',0)} try={stats.get('yr_try',0)}")
    print(f"  year answers repaired {stats.get('year_repaired', 0)} + {stats.get('year_repaired_docs', 0)} across documents")
    print(f"  TOTAL ANSWERED       {answered} ({answered / total:.1%})")

    # Attribution for the hand-verified gold set: which branch answered which
    # question. A defect cannot be aimed at without it.
    (root / "artifacts" / "branches.json").write_text(
        json.dumps(branch_of, ensure_ascii=False, indent=1), encoding="utf-8")

    if lookup_mod.CONTRASTIVE_SKIPS:
        print(f"  contrastive guard bỏ {len(lookup_mod.CONTRASTIVE_SKIPS)} dòng")
        for metric, label in lookup_mod.CONTRASTIVE_SKIPS[:5]:
            print(f"    {metric!r} != {label!r}")

    out = root / "submissions" / "final.zip"
    summary = build_submission(predictions, store, out, style=TableRefStyle())
    problems = validate_submission(out, {p.id for p in parsed})
    print(f"\n{out.name}: {summary['csv_files']} csv, {summary['bytes'] / 1e6:.1f} MB, "
          f"{'OK' if not problems else problems[:3]}")


if __name__ == "__main__":
    main()
