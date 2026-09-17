"""Turn the adapter's cell references into programs and splice them into a zip.

The model answers `<frame> | <row label> | <column> | <number>`. Only the first
three fields are used. Its own number is discarded and recomputed here, because
the smoke run showed the two failure modes are both arithmetic, not location:

    "Lãi tiền gửi ... triệu đồng?"  -> -208253201298.0   (raw VND, wrong unit)
    "Quỹ khen thưởng ... tỷ đồng?"  -> -87.54            (row reads "Trừ: ...")

In both the model had found the right cell. Converting the unit and taking the
magnitude are forty lines of code that have already been measured; asking a 14B
model to also do them is spending its attention on the part it is worst at. The
organisers' own error analysis says the same thing from the other side: 54.7% of
end-to-end failures are reading the wrong cell and 0.9% are arithmetic.

Rebuilding from `run_submit.py` is not an option — that file lost ~320 lines and
regenerating regresses by 90 answers — so this patches the best zip in place.

Replacement is confined to questions whose current program came from a branch
measured below the adapter, or is the `result = 0.0` placeholder.
`relevant_tables` is untouched, so TABLES_F2 is provably identical.

Usage:
  PYTHONPATH=src python scripts/patch_locate.py --base direct14b.zip \
      --answers artifacts/locate_answers.jsonl --out locate1.zip [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import compose, lookup as lookup_mod, ratio as ratio_mod  # noqa: E402
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.companies import strip_tones  # noqa: E402
from vifin.query.parse import UNIT_SCALE, parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402
from vifin.submit.package import _csv_name, _render_csv  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

WEAK_BRANCHES = {"locate_model", "locate_embed", "plan", "llm_8b", "llm_14b",
                 "placeholder"}
INDEX_ROW_RE = re.compile(r"^#(\d+)$")


def is_single_cell_question(question) -> bool:
    """Is this a question one cell can answer?

    The adapter was trained only on the easy tier — one company, one period, one
    figure read from one table — which the organisers' own labels put at 361 of
    1,012 questions. It answers everything in that shape, including questions
    that are not:

        "Trong nhóm Hoà Phát, Hoa Sen và Nam Kim ... xét các doanh nghiệp có
         doanh thu ..."          -> returned one company's raw revenue
        "Chênh lệch ... giữa năm 2022 so với năm 2021"
                                 -> returned a single cell, not a difference
        "... vào cuối năm nào"   -> returned a figure where a year was asked

    Four of six sampled patches were wrong this way, and each would have replaced
    a plausible derived answer with a number that cannot be the answer. Confining
    the patch to single-cell questions is the difference between using the model
    where it was taught and using it everywhere.
    """

    if len(question.tickers) > 1 or len(question.years) > 1:
        return False
    if ratio_mod.shape(question) is not None:
        return False
    if ratio_mod.compound_shape(question) is not None:
        return False
    if compose.screen_shape(question) is not None:
        return False
    if compose.SCREEN_RE.search(question.question):
        return False
    # "vào cuối năm nào", "công ty nào" — the answer is a label or a year, and no
    # amount read from a cell can be right.
    if re.search(r"\bnăm nào\b|\bcông ty nào\b|\bdoanh nghiệp nào\b|\bmã nào\b",
                 question.question, re.I):
        return False
    return lookup_mod.is_single_lookup(question.question)


def parse_reply(text: str) -> tuple[str, str, str] | None:
    """`<frame> | <row> | <column> | <number>` — the first three fields."""

    body = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    lines = [ln.strip() for ln in body.splitlines() if ln.count("|") >= 3]
    if not lines:
        return None
    parts = [p.strip() for p in lines[-1].split("|")]
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", strip_tones(str(text)).lower()).strip()


def find_row(grid: list[list[str]], label: str, label_col: int) -> int | None:
    """Locate the row the model named, exactly then loosely.

    `#N` is the form emitted when the row has no label — 25% of cells, mostly
    unnamed total rows. It indexes the DataFrame, so grid row N+1.
    """

    match = INDEX_ROW_RE.match(label)
    if match:
        index = int(match.group(1)) + 1
        return index if 0 < index < len(grid) else None

    wanted = norm(label)
    if not wanted:
        return None
    for index, line in enumerate(grid[1:], start=1):
        if label_col < len(line) and norm(line[label_col]) == wanted:
            return index
    # OCR mangles labels ("cơ bảnhoản thà"), so an exact miss is not a miss.
    # Containment either way is deliberately loose: the model was reading this
    # very grid, so a near match is far more likely to be the intended row than
    # a coincidence.
    for index, line in enumerate(grid[1:], start=1):
        if label_col >= len(line):
            continue
        have = norm(line[label_col])
        if have and (have in wanted or wanted in have):
            return index
    return None


def find_column(grid: list[list[str]], name: str) -> int | None:
    wanted = norm(name)
    if not wanted or not grid:
        return None
    header = grid[0]
    for index, cell in enumerate(header):
        if norm(cell) == wanted:
            return index
    for index, cell in enumerate(header):
        have = norm(cell)
        if have and (have in wanted or wanted in have):
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="direct14b.zip")
    parser.add_argument("--answers", default="artifacts/locate_answers.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--raw", action="store_true",
                        help="ship the cell as read: no unit conversion and "
                             "no magnitude, so the conversion layer can be "
                             "measured instead of assumed")
    parser.add_argument("--all-tiers", action="store_true",
                        help="apply the model to every question, not only the "
                             "single-cell ones it was trained on. Measured on a "
                             "sample: four of six such patches were wrong.")
    parser.add_argument("--everywhere", action="store_true",
                        help="replace regardless of which branch holds the "
                             "question now; default is weak branches only")
    args = parser.parse_args()

    parsed = {q.id: q for q in parse_all(
        ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")}
    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        records = json.loads(archive.read("submission.json"))
        payloads = {
            name.split("/", 1)[1]: archive.read(name).decode("utf-8")
            for name in archive.namelist() if name.startswith("data/")
        }
    answers = {}
    for line in Path(args.answers).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            answers[row["id"]] = row
    print(f"{args.base}: {len(records)} records | model answers: {len(answers)}")

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    caches: dict[str, dict[int, str]] = {}
    for label, name in (("locate_model", "located.jsonl"),
                        ("locate_embed", "embed_located.jsonl"),
                        ("plan", "planned.jsonl"),
                        ("llm_8b", "gen_helpers.jsonl"),
                        ("llm_14b", "gen14b_merged.jsonl")):
        path = ROOT / "artifacts" / name
        if not path.exists():
            continue
        found: dict[int, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("code"):
                    found[row["id"]] = row["code"].strip()
        caches[label] = found

    def branch_of(record: dict) -> str:
        code = (record.get("pandas_query") or "").strip()
        if code in ("", "result = 0.0"):
            return "placeholder"
        for label, cache in caches.items():
            if cache.get(record["id"]) == code:
                return label
        return "lexical"

    stats: Counter[str] = Counter()
    patched = 0

    for record in records:
        answer = answers.get(record["id"])
        question = parsed.get(record["id"])
        if answer is None or question is None:
            stats["no_model_answer"] += 1
            continue

        if not args.all_tiers and not is_single_cell_question(question):
            stats["not_a_single_cell_question"] += 1
            continue

        branch = branch_of(record)
        if not args.everywhere and branch not in WEAK_BRANCHES:
            stats[f"kept:{branch}"] += 1
            continue

        fields = parse_reply(answer["reply"])
        if fields is None:
            stats["unparseable_reply"] += 1
            continue
        frame, row_label, column_name = fields

        # `KHONG_CO` is the adapter declining: the training set now contains 17%
        # blind pairs, whose prompts genuinely do not hold the answer, so that the
        # model has a way to say the retriever missed instead of reading a
        # confident wrong cell. Counting abstentions separately is the point —
        # it is how we find out whether teaching restraint worked at all.
        if frame.upper().startswith("KHONG_CO"):
            stats["model_abstained"] += 1
            continue

        names = list(answer["meta"]["variables"])
        if frame not in names:
            stats["frame_not_in_prompt"] += 1
            continue
        doc_name, table_id = answer["meta"]["keys"][names.index(frame)]
        key = TableKey(doc_name, int(table_id))
        grid = store.rows(key)
        if not grid:
            stats["table_missing"] += 1
            continue

        label_col = lookup_mod.label_column(grid)
        row_index = find_row(grid, row_label, label_col)
        column_index = find_column(grid, column_name)
        if row_index is None:
            stats["row_label_not_found"] += 1
            continue
        if column_index is None:
            stats["column_not_found"] += 1
            continue

        meta = store.meta(key)
        unit_text = f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
        if args.raw:
            # `--raw` ships the cell as the model read it: no unit factor, no
            # magnitude. It exists to measure the conversion layer rather than
            # assume it. The smoke run says the layer is needed — the model
            # returned -208253201298 VND for a question asking triệu đồng, and
            # -87.54 for a fund whose row reads "Trừ: Quỹ khen thưởng" — but the
            # cost of finding out is one submission, and the cost of being wrong
            # about it is every submission after.
            scale_in = asked = 1.0
        else:
            scale_in = lookup_mod.column_scale(grid, column_index, unit_text)
            asked = UNIT_SCALE.get(question.target_unit) or 1.0

        # `Lookup` carries what `synthesize` needs; the score is not consulted
        # here because the model, not the label matcher, chose this cell.
        cell_text = (grid[row_index][column_index]
                     if column_index < len(grid[row_index]) else "")
        value = lookup_mod._parse_cell(cell_text)
        if value is None:
            stats["cell_not_numeric"] += 1
            continue
        found = lookup_mod.Lookup(
            row=row_index, column=column_index,
            label=str(grid[row_index][label_col]).strip()
            if label_col < len(grid[row_index]) else "",
            value=value, score=1.0, label_col=label_col)

        variable = "df"
        # magnitude=True: statements bracket deductions, so "Trừ: Quỹ khen
        # thưởng" reads negative while the question asks for its size. This is
        # the fault that produced -87.54 where 87.54 was wanted.
        body = lookup_mod.synthesize(found, scale_in, asked,
                                     magnitude=not args.raw)
        program = f"{PRELUDE}\n{body}"
        outcome = run_query(program, {variable: grid})
        if not outcome.ok or outcome.value is None or reads_no_frame(program):
            stats["program_failed"] += 1
            continue

        stats[f"patched:{branch}"] += 1
        patched += 1
        if args.dry_run:
            continue

        csv_name = _csv_name(key)
        if csv_name not in payloads:
            payloads[csv_name] = _render_csv(grid)
        record["answer"] = float(outcome.value)
        record["pandas_query"] = program
        record["evidence"] = [{"variable": variable, "csv_path": f"data/{csv_name}"}]
        # relevant_tables deliberately untouched — TABLES_F2 stays identical.

    print()
    for key, count in stats.most_common(16):
        print(f"  {key:26s} {count:5d}")
    print(f"\n  PATCHED {patched}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    referenced = {
        item["csv_path"].split("/", 1)[1]
        for record in records for item in record["evidence"]
    }
    missing = referenced - set(payloads)
    if missing:
        raise SystemExit(f"{len(missing)} csv files referenced but absent")

    out_path = ROOT / "submissions" / args.out
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json",
                         json.dumps(records, ensure_ascii=False, indent=1))
        for name in sorted(referenced):
            archive.writestr(f"data/{name}", payloads[name])
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(referenced)} csv)")


if __name__ == "__main__":
    main()
