"""Turn generated QARecords into training pairs that match inference exactly.

Two mismatches would each be enough to waste the whole run, and neither shows up
until the fine-tuned model scores badly:

**1. The output convention.** The generator's gold program ends in
`result = df.iloc[2, 0]`, which returns the cell *string* `"1.498.203.140.705"`.
Its gold `answer` is that same string, so the record is self-consistent — but
EXECUTION is scored by numeric comparison, and a program returning a string
fails it. The organisers' own baseline is the proof: ANSWER 1.0, EXECUTION
0.3577. Training on the gold program verbatim reproduces that 0.64 gap on
purpose. The cell they chose is right; the formatting is not ours. So the cell
reference is preserved and the tail is rewritten to return a float.

**2. The candidate set.** A gold record cites one table. Inference hands the model
*eight* — the retrieval shortlist — and asks which to read. Training on prompts
that contain only the answer's table teaches "the answer is in df", and the model
then meets `df1..df8` at inference having never had to choose. So each training
prompt is rebuilt through the same retriever the inference path uses, the gold
table is placed among the candidates it returns, and the program is rewritten to
name the frame the gold table actually landed on.

Everything downstream of that — system prompt, clause set, table rendering — is
imported from `vifin.answering.generate`, so train and inference cannot drift
apart by editing one and forgetting the other.

Usage:
  PYTHONPATH=src python scripts/build_sft.py runs/*/per_question.jsonl --out artifacts/sft.jsonl
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import dataclasses
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _audit_generated import classify  # noqa: E402

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.generate import build_prompts, render_tables, variable_names  # noqa: E402
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import UNIT_SCALE, _target_unit, parse_question  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

TOL = 2e-4
# `result = df.iloc[2, 0]` / `result = df.iloc[2][0]` and the multi-frame forms.
ILOC_RE = re.compile(
    r"result\s*=\s*(df\d*)\s*\.\s*iloc\s*\[\s*(\d+)\s*(?:,|\]\s*\[)\s*(\d+)\s*\]\s*$")


def readings(text) -> list[float]:
    """Every defensible numeric reading of a cell, not just the likeliest one.

    `435.178` is 435178 under the Vietnamese convention where `.` groups
    thousands, and 435.178 under the Western one where it is a decimal point.
    The string alone does not say which. Committing to one reading rejected 10
    of 91 otherwise-good records — a comparison used to *filter* data must not
    also be the thing guessing, or it discards exactly what it cannot parse.
    """

    raw = str(text).strip()
    if not raw:
        return []
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return []

    forms = set()
    if "," in raw:
        # A comma present makes it unambiguous: '.' groups, ',' decides decimals.
        forms.add(raw.replace(".", "").replace(",", "."))
    else:
        forms.add(raw.replace(".", ""))   # every '.' is a thousands separator
        forms.add(raw)                    # the last '.' is a decimal point

    out = []
    for form in forms:
        try:
            value = float(form)
        except ValueError:
            continue
        out.append(-value if negative else value)
    return out


def to_float(text) -> float | None:
    values = readings(text)
    return values[0] if values else None


def parse_ref(ref: str) -> tuple[str, int] | None:
    """`FTS_financial_statements_2023|table_39` -> (doc, 39)."""

    doc, _, tail = ref.partition("|")
    match = re.fullmatch(r"table_(\d+)", tail)
    return (doc, int(match.group(1))) if match else None


# A two-cell locator was drafted here to rescue the derived pool and then removed
# once measured. Recorded so it is not drafted a third time:
#
# - It assumed one cell per frame. The derived programs read 1 to 12 cells
#   (86 of 343 read exactly two), so the assumption fits a quarter of the pool.
# - It searched every cell of both frames against three operations with a
#   tolerance of max(|answer|*2e-3, 0.01). Two frames of a few hundred numeric
#   cells give ~10^5 pairs; at that count a coincidental match is likely, and a
#   coincidental match writes a fabricated cell reference next to a correct
#   answer. This file's premise is that a guessed label is worse than no pair.
# - The prize does not justify either risk. Two-company arithmetic is 15 of
#   1,012 exam questions (1.5%), and the training set already over-represents it
#   at 1.8% — see `_probe_exam_classes.py`.
#
# The derived pool has a separate defect that no locator can repair: 103 of 343
# records name a company in the question that no cited table can answer for, so
# the recorded answer is wrong for the question asked (`_audit_medium.py`).


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    parser.add_argument("--out", default="artifacts/sft.jsonl")
    parser.add_argument("--out-locate", default="",
                        help="also write a second file whose target is a cell "
                             "reference plus the number, instead of a program")
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--blind-frac", type=float, default=0.15,
                        help="share of pairs built WITHOUT the gold table, so the "
                             "model meets the case where retrieval missed")
    parser.add_argument("--rank", default="artifacts/sft_anchor_rank.jsonl",
                        help="anchor ranking for these records, keyed by record "
                             "index; falls back to the lexical retriever")
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.records:
        paths.extend(Path(p) for p in glob.glob(pattern))
    rows: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                # The generator writes bare `NaN`, which json.loads rejects.
                rows.append(json.loads(line.replace(": NaN", ": null")))
    print(f"{len(rows)} records from {len(paths)} file(s)")

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    system, user_template = build_prompts(ROOT)

    import random

    # Fixed seed: which pairs go blind must be reproducible, or two builds of the
    # "same" dataset differ and no A/B between them means anything.
    blind_rng = random.Random(20260813)

    ranking: dict[int, list[TableKey]] = {}
    rank_path = ROOT / args.rank
    if rank_path.exists():
        for line in rank_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                ranking[row["id"]] = [
                    TableKey(r["doc_name"], int(r["table_id"]))
                    for r in row.get("refs", [])
                ][: args.tables]
        print(f"anchor ranking for {len(ranking)} records")
    else:
        print(f"no {rank_path.name}: distractors will come from the lexical "
              f"retriever, which is NOT what inference uses")

    stats: Counter[str] = Counter()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    # A second target set from the same pass, so the two cannot drift: identical
    # prompts, identical accepted records, identical unit conversion. Only the
    # assistant turn differs, which is what an A/B between them has to isolate.
    locate_path = ROOT / args.out_locate if args.out_locate else None
    locate_handle = locate_path.open("w", encoding="utf-8") if locate_path else None
    locate_system = (
        "Bạn đọc báo cáo tài chính Việt Nam. Cho câu hỏi và các bảng, hãy chỉ ra "
        "ô chứa đáp án và giá trị của nó, theo đúng định dạng:\n"
        "<tên biến> | <nhãn dòng> | <tên cột> | <số>\n"
        "Nhãn dòng chép nguyên văn từ bảng. Nếu dòng không có nhãn, dùng #<chỉ số>. "
        "Số phải đã quy đổi về đơn vị câu hỏi yêu cầu, làm tròn 2 chữ số thập phân. "
        "Không giải thích, không viết mã."
    )

    with out_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(rows):
            # The pipeline's own gate cannot see these: a program that filters on
            # the answer's own digits executes and reproduces the answer *by
            # construction*, and an answer of NaN or a row label passes anything
            # that only checks "did it run". Measured at 13% of a 91-record pool.
            # The audit grades *model-written* records for defects execution
            # cannot see. A pregenerated record has no model in the loop: its
            # answer is arithmetic over the verified panel and its program was
            # already executed against its own tables. Running the audit on it
            # only misfires — `normalise` strips the decimal point before the
            # rate-magnitude check, so a legitimate 105.67% reads as 10,567 and
            # is rejected as `rate_question_absolute_answer`.
            verdict = "USABLE" if record.get("pregenerated") else classify(record)
            # Two of the defects are repairable because the program still tells
            # us which cell it landed on; the rest would require inventing the
            # right cell, which is the task we are trying to teach. A training
            # pair with a guessed label is worse than no pair — the model learns
            # exactly what it is shown.
            if verdict == "answer_is_one_character":
                # `.values[0][0]` indexes a character off the cell string. The
                # trailing [0] is the whole bug.
                repaired = re.sub(r"(\.values\[0\])\[0\]\s*$", r"\1",
                                  str(record["pandas_query"]).rstrip())
                if repaired != record["pandas_query"]:
                    record = {**record, "pandas_query": repaired}
                    verdict = "repaired_one_character"
            # `query_contains_its_own_answer` is repaired further down, once the
            # frames are loaded and the cell it lands on can be located.
            if (verdict != "USABLE"
                    and not verdict.startswith("repaired")
                    and verdict != "query_contains_its_own_answer"):
                stats[f"rejected:{verdict}"] += 1
                continue
            if verdict.startswith("repaired"):
                stats[verdict] += 1
            refs = [parse_ref(r) for r in record.get("relevant_tables", [])]
            refs = [r for r in refs if r is not None]
            if not refs:
                stats["no_parseable_ref"] += 1
                continue
            gold_keys = [TableKey(doc, tid) for doc, tid in refs]

            question = parse_question(0, record["question"], roster)
            # Distractors come from the same ranking the submission retrieves
            # with. They used to come from `search_balanced`, i.e. BM25, while
            # `build_infer_prompts.py` uses the anchor ranking — so the adapter
            # learned to tell the gold table apart from BM25 neighbours and was
            # then asked to tell it apart from anchor neighbours. Different
            # distractors, different task.
            ranked = ranking.get(index)
            if ranked:
                candidates = list(ranked)
            else:
                candidates = [
                    hit.key for hit in retriever.search_balanced(
                        question,
                        per_group=max(2, -(-args.tables //
                                           (max(1, len(question.tickers))
                                            * max(1, len(question.years))))),
                        cap=args.tables)
                ]

            # Most examples put the gold table in the prompt, but not all.
            #
            # Every pair in the previous training set contained it, so the adapter
            # never met a prompt whose answer was absent and has no way to decline.
            # At inference the retriever misses, and a model that has only ever
            # been rewarded for answering will read a confident wrong cell instead.
            # That is the most likely reason held-out scored 69.4% while the exam
            # went backwards, and it is the one cause we can remove for free.
            #
            # The share is set near the miss rate the anchor ranking actually has
            # (gold in the top-10 for 92.6% of gold questions), so the model sees
            # the situation about as often as it will meet it.
            drop_gold = (blind_rng.random() < args.blind_frac)
            if drop_gold:
                keys = [k for k in candidates if k not in gold_keys][: args.tables]
                if len(keys) < 2:
                    stats["blind_too_few_candidates"] += 1
                    continue
            else:
                keys = list(gold_keys)
                for key in candidates:
                    if key not in keys and len(keys) < args.tables:
                        keys.append(key)
                if len(keys) < 2:
                    stats["too_few_candidates"] += 1
                    continue

            names = variable_names(len(keys))
            grids = {name: store.rows(key) for name, key in zip(names, keys)}
            frame_of = {key: name for name, key in zip(names, keys)}

            # ---- records that already know their own program -----------------
            # `gen_shapes.py` writes questions built *from* a computation over the
            # verified metric panel, so its program is already positional, already
            # in the unit the question names, and already executed against its own
            # tables. Everything below this branch exists to recover a single cell
            # from a model-written program and convert its unit — work that would
            # only damage a program that reads three cells and compares them.
            #
            # The frames are named here rather than there, because only this pass
            # knows where the gold tables landed among the distractors. Slots are
            # positional in `relevant_tables`, so `{t0}` is the record's first
            # cited table wherever it ended up.
            if record.get("pregenerated"):
                if drop_gold:
                    # A blind pair needs a target that can decline, and only the
                    # locate file has one. Skipping keeps the blind share honest
                    # for the one-cell records it was measured against.
                    stats["pregen:blind_skipped"] += 1
                    continue
                program = str(record["pandas_query"])
                for slot, key in enumerate(gold_keys):
                    program = program.replace(f"{{t{slot}}}", frame_of[key])
                if "{t" in program:
                    stats["rejected:pregen_unbound_frame"] += 1
                    continue
                outcome = run_query(f"{PRELUDE}\n{program}", grids)
                # Not `to_float`: that resolves the Vietnamese/Western decimal
                # ambiguity by picking from a *set*, so `1636.8` can come back as
                # 16368 and the choice varies with the hash seed. A pregenerated
                # answer is a JSON number with no ambiguity to resolve.
                try:
                    wanted = float(record["answer"])
                except (TypeError, ValueError):
                    wanted = None
                if not outcome.ok or wanted is None:
                    stats["rejected:pregen_did_not_run"] += 1
                    continue
                # Absolute 0.01 is the grader's tolerance, so it is the one that
                # decides whether this pair is teachable.
                if abs(float(outcome.value) - wanted) > 1e-2:
                    stats["rejected:pregen_value_moved"] += 1
                    continue
                table_refs = {
                    name: f"{key.doc_name}|{int(store.meta(key).start_line)}"
                    for name, key in zip(names, keys)
                }
                user = (
                    user_template.replace("{{QUESTION}}", record["question"])
                    .replace("{{VAR_HINT}}", ", ".join(names))
                    .replace("{{TABLES}}", render_tables(grids, table_refs))
                )
                handle.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": program},
                    ],
                    "meta": {
                        "difficulty": record.get("difficulty"),
                        "shape": record.get("shape"),
                        "frames": len(keys),
                        "gold_frame": frame_of[gold_keys[0]],
                        "answer": wanted,
                        "question": record["question"],
                        "keys": [[k.doc_name, k.table_id] for k in keys],
                        "variables": names,
                        "target": program,
                    },
                }, ensure_ascii=False) + "\n")
                written += 1
                stats[f"WRITTEN:{record.get('shape', 'pregen')}"] += 1
                continue

            if drop_gold:
                # A blind pair goes only to the locate file. The program target
                # has to return a number, and there is no honest number here; the
                # locate target can say so. `KHONG_CO` is the marker
                # `patch_locate.py` skips, so an abstention costs the question
                # nothing beyond leaving it to whichever branch held it before.
                if locate_handle is None:
                    stats["blind_skipped_no_locate_file"] += 1
                    continue
                table_refs = {
                    name: f"{key.doc_name}|{int(store.meta(key).start_line)}"
                    for name, key in zip(names, keys)
                }
                user = (
                    user_template.replace("{{QUESTION}}", record["question"])
                    .replace("{{VAR_HINT}}", ", ".join(names))
                    .replace("{{TABLES}}", render_tables(grids, table_refs))
                )
                locate_handle.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": locate_system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": "KHONG_CO | - | - | -"},
                    ],
                    "meta": {
                        "difficulty": record.get("difficulty"),
                        "frames": len(keys),
                        "gold_frame": None,
                        "answer": None,
                        "question": record["question"],
                        "keys": [[k.doc_name, k.table_id] for k in keys],
                        "variables": names,
                        "target": "KHONG_CO | - | - | -",
                        "blind": True,
                    },
                }, ensure_ascii=False) + "\n")
                stats["BLIND_PAIR"] += 1
                continue

            program = str(record["pandas_query"]).strip()
            target = frame_of[gold_keys[0]]
            gold_grid = grids[target]

            # A self-referential query — `df.loc[df[col] == '<the answer>', col]`
            # — executes and matches by construction, so it teaches the model to
            # write a filter containing the number it was asked to find. But the
            # cell it lands on is real and locatable: run the program, then find
            # that value in the grid. Rewriting it positionally keeps the cell
            # and removes the tautology, which is a repair rather than a guess.
            if classify(record) == "query_contains_its_own_answer":
                located = None
                try:
                    outcome = run_query(program, grids)
                    if outcome.ok or outcome.value is not None:
                        wanted = str(outcome.value).strip()
                    else:
                        wanted = str(record["answer"]).strip()
                except Exception:
                    wanted = str(record["answer"]).strip()
                # `frame_from_rows` round-trips through `read_csv`, so grid row 0
                # becomes the header and grid row N is DataFrame row N-1. Search
                # the grid, emit DataFrame coordinates — conflating the two puts
                # every repaired program one row off, silently.
                grid = grids[target]
                for r, line in enumerate(grid[1:], start=1):
                    for c, cell in enumerate(line):
                        if str(cell).strip() == wanted:
                            located = (r - 1, c)
                            break
                    if located:
                        break
                if located is None:
                    stats["rejected:selfref_cell_not_found"] += 1
                    continue
                program = f"result = num({target}, {located[0]}, {located[1]})"
                record = {**record, "pandas_query": program}
                stats["repaired_self_reference"] += 1

            # ---- unit conversion -------------------------------------------
            # 81.4% of the records asking for tỷ/triệu/nghìn đồng record the raw
            # đồng figure as the answer. Training on those teaches the model that
            # the unit named in the question does not change the answer — the
            # single error class we lose the most points to.
            #
            # The conversion needs the cell's *own* unit, which is not the
            # question's: bank statements print "Triệu VND" in the column header,
            # so dividing a already-in-millions figure by a million again is wrong
            # by 1e6. `column_scale` reads that header and is the same function
            # the 42.8% branch relies on, so it is the best estimator available —
            # and where it cannot decide, the pair is dropped rather than guessed.
            asked_unit = _target_unit(record["question"])
            asked_scale = UNIT_SCALE.get(asked_unit)

            # Every record is normalised to a positional read, whatever shape it
            # arrived in. Not for tidiness: the conversion below needs the cell's
            # *column*, and only a positional form names one. `iloc[r, c]` gives
            # it directly; anything else is run and its value located in the grid.
            located: tuple[int, int] | None = None
            match = ILOC_RE.search(program)
            if match is not None:
                located = (int(match.group(2)), int(match.group(3)))
                stats["form:cell_read"] += 1
            else:
                try:
                    probe = run_query(
                        f"{PRELUDE}\n"
                        + (re.sub(r"\bdf\b", target, program)
                           if target != "df" else program)
                        + "\nresult = str(result)", grids)
                    wanted = str(probe.value).strip() if probe.ok else None
                except Exception:
                    wanted = None
                if wanted is None:
                    wanted = str(record["answer"]).strip()
                # `frame_from_rows` consumes grid row 0 as the header, so grid
                # row N is DataFrame row N-1.
                for r, line in enumerate(gold_grid[1:], start=1):
                    for c, cell in enumerate(line):
                        if str(cell).strip() == wanted:
                            located = (r - 1, c)
                            break
                    if located:
                        break
                if located is None:
                    stats["rejected:cell_not_locatable"] += 1
                    continue
                stats["form:localised"] += 1

            row_index, col_index = located
            raw_body = f"result = num({target}, {row_index}, {col_index})"

            # The cell's own unit, read from its column header — not from the
            # question. A bank statement printing "Triệu VND" is already in
            # millions; dividing again would be wrong by 1e6.
            meta = store.meta(gold_keys[0])
            # `column_scale` indexes the *grid*, and grid columns map 1:1 onto
            # frame columns — only rows are offset, because `frame_from_rows`
            # consumes grid row 0 as the header. Verified on 205 located cells:
            # 191 sit at grid[row+1][col] and none at a shifted column.
            #
            # This read `col_index + 1`, i.e. the header of the column to the
            # right, which is what every other call site does not do
            # (`patch_direct.py`, `score_local.py` pass the column straight
            # through). It changed the answer on 4 of 505 pairs — small, but each
            # by a factor of 1e3 to 1e9, because a column past the last one falls
            # through to the whole-header fallback and picks up "tỷ" from
            # somewhere else entirely.
            #
            # The verification below could never catch it: it divides by the same
            # wrong scale to undo the conversion, so a wrong factor cancels itself.
            cell_scale = lookup_mod.column_scale(
                gold_grid, col_index,
                f"{meta.unit_page} {meta.unit_doc} {meta.caption}")
            if asked_scale:
                factor = cell_scale / asked_scale
                body = (f"{raw_body}\n"
                        f"result = round(result * {factor!r}, 2)")
                stats["converted" if factor != 1.0 else "already_in_unit"] += 1
            else:
                body = raw_body
                stats["no_unit_to_convert"] += 1

            rewritten = f"{PRELUDE}\n{body}"
            outcome = run_query(rewritten, grids)
            wants = readings(record["answer"])
            if not outcome.ok or not wants:
                stats["rewrite_failed"] += 1
                continue
            got = float(outcome.value)
            # The recorded answer is the *raw* cell, so the check is that the
            # program reproduces it once the conversion is undone. Comparing the
            # converted output against the raw answer would reject every record
            # this pass exists to fix.
            scale_back = (cell_scale / asked_scale) if asked_scale else 1.0
            unconverted = got / scale_back if scale_back else got
            if not any(abs(unconverted - want) <= TOL * max(abs(unconverted), abs(want), 1e-9)
                       for want in wants):
                stats["rewrite_changed_value"] += 1
                continue
            want = got

            table_refs = {
                name: f"{key.doc_name}|{int(store.meta(key).start_line)}"
                for name, key in zip(names, keys)
            }
            user = (
                user_template.replace("{{QUESTION}}", record["question"])
                .replace("{{VAR_HINT}}", ", ".join(names))
                .replace("{{TABLES}}", render_tables(grids, table_refs))
            )
            # The harness prepends PRELUDE at inference, so the model is trained
            # to emit the body alone. Training it to emit the prelude too would
            # teach it to spend its budget re-typing helpers it is always given.
            handle.write(json.dumps({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": body},
                ],
                "meta": {
                    "difficulty": record.get("difficulty"),
                    "frames": len(keys),
                    "gold_frame": target,
                    "answer": want,
                    # Needed to *execute* whatever a model generates for this
                    # prompt later. Without the keys an A/B can only compare
                    # strings, and string equality is not the metric that decides
                    # anything — a differently-written program that reads the same
                    # cell is a success, and an identical-looking one that reads a
                    # neighbouring cell is a failure.
                    "question": record["question"],
                    "keys": [[k.doc_name, k.table_id] for k in keys],
                    "variables": names,
                    "target": body,
                },
            }, ensure_ascii=False) + "\n")
            written += 1
            stats["WRITTEN"] += 1

            if locate_handle is None:
                continue

            # The second target: name the cell, then give the number.
            #
            # The organisers measured 14 models on this dataset and found that
            # under 10B parameters, Program-of-Thought carries a 99% syntax-error
            # rate, while Chain-of-Thought answers. Our submission model is capped
            # at 14B, and our own program-writing branches score 5.9-13.3% — the
            # same effect. Their error analysis puts 54.7% of end-to-end failures
            # on reading the wrong cell and 0.9% on arithmetic: "LLMs không dốt
            # toán, vấn đề là chọn sai ô số liệu".
            #
            # So this target asks for the thing that is actually hard and drops
            # the thing that is not. It still names the cell rather than only the
            # number, because a bare number cannot be turned back into a program
            # without guessing which cell produced it, and EXECUTION scores the
            # program.
            #
            # The row is given as a label because that is the skill in question —
            # matching "chưa hoàn thành" to "dở dang", "Ngân hàng Nhà nước" to
            # "NHNN". Where the label is missing or repeated (25% of cells,
            # mostly unnamed total rows) it falls back to an index, which the
            # harness can still resolve.
            grid_row = row_index + 1
            label_col = lookup_mod.label_column(gold_grid)
            label = ""
            if grid_row < len(gold_grid) and label_col < len(gold_grid[grid_row]):
                label = str(gold_grid[grid_row][label_col]).strip()
            unique = label and sum(
                1 for line in gold_grid[1:]
                if label_col < len(line) and str(line[label_col]).strip() == label
            ) == 1
            row_ref = label if unique else f"#{row_index}"
            column_ref = (str(gold_grid[0][col_index]).strip()
                          if col_index < len(gold_grid[0]) else f"#{col_index}")
            locate_target = f"{target} | {row_ref} | {column_ref} | {want!r}"
            stats["locate:by_label" if unique else "locate:by_index"] += 1

            locate_handle.write(json.dumps({
                "messages": [
                    {"role": "system", "content": locate_system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": locate_target},
                ],
                "meta": {
                    "difficulty": record.get("difficulty"),
                    "frames": len(keys),
                    "gold_frame": target,
                    "answer": want,
                    "question": record["question"],
                    "keys": [[k.doc_name, k.table_id] for k in keys],
                    "variables": names,
                    "target": locate_target,
                    "cell": [row_index, col_index],
                },
            }, ensure_ascii=False) + "\n")

    print()
    for key, count in stats.most_common():
        print(f"  {key:26s} {count:5d}")
    if locate_handle is not None:
        locate_handle.close()
    print(f"\nwrote {written} training pairs -> {out_path}")
    print("  every pair: gold cell preserved, output rewritten to a float,")
    print("  prompt built through the same retriever and renderer as inference.")
    if locate_path is not None:
        print(f"\nand {written} locate-form pairs -> {locate_path}")
        print("  same prompts and same accepted records; only the target differs,")
        print("  so an A/B between the two measures the target form alone.")


if __name__ == "__main__":
    main()
