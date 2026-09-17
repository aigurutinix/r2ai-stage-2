"""Answer only what the address book can address, to settle the unit convention.

Every part of the read is now measured except one, and that one multiplies into every
answer: the unit the gold is expressed in. The index stores đồng — the raw cell times
the table's declared scale. A question asks for "triệu đồng", "tỷ đồng", and in 149
cases "nghìn tỷ đồng" or "trăm tỷ đồng", two units no table is ever denominated in,
so at least those must be converted rather than reported raw. Nothing in the
organisers' code settles it; the generator's question text is written by a model and
its answer is locked to a `pandas_query` result, and the two are only reconciled by a
gate whose verdict is not in the released code.

So this build answers ONLY the questions whose metric maps to a `Mã số` with
confidence, converts to the unit the question names, and leaves the rest empty. The
score divided by the number answered says whether the convention is right: if it is,
roughly nine in ten of those should land, and if it is not, only the handful asking
for plain "đồng" can.

The emitted program reads the cell out of the organisers' own CSV, so it executes
against the same frame the scorer binds. Two rules from the measurements are applied:

  * a cost or deduction code is read as a MAGNITUDE. 17% of those cells are printed
    in brackets and parse negative, and `20 = 10 - 11` holds for 76.5% of statements
    signed against 90.4% on magnitudes.
  * every other code keeps its sign. Taking magnitudes there breaks `50 = 30 + 40`,
    which falls from 99.5% to 69.1%.

Usage:
  python scripts/fresh/build_submission.py --out submissions/fresh_maso.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Codes whose printed sign cannot be trusted: costs, deductions and provisions.
MAGNITUDE_CODES = {
    "kqkd": {"02", "11", "22", "24", "25", "26", "51", "52"},
    "lctt": set(),
    "cdkt": set(),
}

UNIT_SCALES = (
    ("nghìn tỷ đồng", 1e12),
    ("trăm tỷ đồng", 1e11),
    ("nghìn đồng", 1e3),
    ("triệu đồng", 1e6),
    ("tỷ đồng", 1e9),
    ("đồng", 1.0),
)

PROGRAM = '''import pandas as pd


def _num(raw):
    """Parse a Vietnamese figure: dot groups thousands, comma is the decimal mark,
    and brackets mark a negative."""
    text = str(raw).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = text.replace("%", "").replace(" ", "")
    value = float(text.replace(".", "").replace(",", "."))
    return -value if negative else value


# Row {row} of the frame carries Mã số {code} of the {kind} statement; the first
# value cell of a row is the current period and the second is the prior one.
_cell = _num(df.iloc[{row}, {col}])
result = round({wrap}(_cell) * {scale!r} / {unit!r}, 2)
'''


RATIO_PROGRAM = '''import pandas as pd


def _num(raw):
    """Parse a Vietnamese figure: dot groups thousands, comma is the decimal mark,
    and brackets mark a negative."""
    text = str(raw).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = text.replace("%", "").replace(" ", "")
    value = float(text.replace(".", "").replace(",", "."))
    return -value if negative else value


# {num_kind}/{num_code} over {den_kind}/{den_code}. Both operands are read as
# magnitudes: a rate asks how large one line is against another, and 17% of cost
# cells in this corpus are printed in brackets.
_top = abs(_num({num_var}.iloc[{num_row}, {num_col}])) * {num_scale!r}
_bottom = abs(_num({den_var}.iloc[{den_row}, {den_col}])) * {den_scale!r}
result = round(_top / _bottom * {factor!r}, 2)
'''


COHORT_PROGRAM = '''import pandas as pd


def _num(raw):
    """Parse a Vietnamese figure: dot groups thousands, comma is the decimal mark,
    and brackets mark a negative."""
    text = str(raw).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = text.replace("%", "").replace(" ", "")
    value = float(text.replace(".", "").replace(",", "."))
    return -value if negative else value


# {kind}/{code} read for each company named, then combined by {op}. Each company's
# table carries its own unit scale, so the scale is applied per read before combining.
_values = [
{reads}
]
result = round({expression} / {unit!r}, 2)
'''

OPERATIONS = {
    "tong": "sum(_values)",
    "hieu": "abs(_values[0] - _values[1])",
    "trung_binh": "sum(_values) / len(_values)",
    "lon_nhat": "max(_values)",
    "nho_nhat": "min(_values)",
}

# BTC registry expressions. Roles are already scaled to đồng in the plan.
FORMULA_OPS = {
    "roa_avg_pct": (
        "round(_roles['net_income'] / "
        "((_roles['assets_begin'] + _roles['assets_end']) / 2) * 100, 2)"),
    "roe_avg_pct": (
        "round(_roles['net_income'] / "
        "((_roles['equity_begin'] + _roles['equity_end']) / 2) * 100, 2)"),
    "quick_ratio": (
        "round((_roles['current_assets'] - _roles['inventory']) / "
        "_roles['current_liabilities'], 2)"),
    "current_ratio": (
        "round(_roles['current_assets'] / _roles['current_liabilities'], 2)"),
    "pct": "round(abs(_roles['num']) / abs(_roles['den']) * 100, 2)",
    "ratio": "round(abs(_roles['num']) / abs(_roles['den']), 2)",
}

FORMULA_PROGRAM = '''import pandas as pd


def _num(raw):
    text = str(raw).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = text.replace("%", "").replace(" ", "")
    value = float(text.replace(".", "").replace(",", "."))
    return -value if negative else value


_roles = {{
{role_reads}
}}
result = {expression}
'''


def unit_of(question: str) -> tuple[str, float]:
    lowered = question.casefold()
    for name, scale in UNIT_SCALES:
        if name in lowered:
            return name, scale
    return "", 0.0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="artifacts/fresh/answer_plan.jsonl")
    parser.add_argument("--note-plan", default="artifacts/fresh/note_plan.jsonl")
    parser.add_argument("--ratio-plan", default="artifacts/fresh/ratio_plan.jsonl")
    parser.add_argument("--ratio-greedy-plan",
                        default="artifacts/fresh/ratio_greedy_plan.jsonl")
    parser.add_argument("--btc-formula-plan",
                        default="artifacts/fresh/btc_formula_plan.jsonl")
    parser.add_argument("--greedy-plan", default="artifacts/fresh/greedy_plan.jsonl")
    parser.add_argument("--model-plan", default="artifacts/fresh/model_plan.jsonl")
    parser.add_argument("--ratio-model-plan",
                        default="artifacts/fresh/ratio_model_plan.jsonl")
    parser.add_argument("--cohort-plan", default="artifacts/fresh/cohort_plan.jsonl")
    parser.add_argument("--model-fill-only", action="store_true",
                        help="model picks only fill blanks, never override")
    parser.add_argument("--out", default="submissions/fresh_maso.zip")
    args = parser.parse_args()

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    plan = {}
    for line in (ROOT / args.plan).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            record["source"] = "maso"
            plan[record["id"]] = record

    # Note-table addresses. The Mã số plan wins where both exist: it is verified by
    # the printed identities and by the cross-year comparison, and a note row has
    # neither. `plan_notes` already skips those ids, so this is belt and braces.
    note_path = ROOT / args.note_plan
    if note_path.exists():
        for line in note_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["source"] = "note"
                plan.setdefault(record["id"], record)

    # Rates BEFORE any single-cell greedy fill. A % / lần question that gets a
    # money-cell plan is then skipped for missing unit — that was eating the
    # ratio_greedy coverage (168 blanks that already had a rate plan).
    ratio_path = ROOT / args.ratio_plan
    if ratio_path.exists():
        for line in ratio_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["source"] = "ratio"
                plan.setdefault(record["id"], record)

    ratio_greedy_path = ROOT / args.ratio_greedy_plan
    if ratio_greedy_path.exists():
        for line in ratio_greedy_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["source"] = "ratio"
                plan.setdefault(record["id"], record)

    btc_path = ROOT / args.btc_formula_plan
    if btc_path.exists():
        for line in btc_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                plan[record["id"]] = record

    # Model / cohort fill blanks after rates are locked.
    model_path = ROOT / args.model_plan
    if model_path.exists():
        for line in model_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if args.model_fill_only:
                    plan.setdefault(record["id"], record)
                else:
                    plan[record["id"]] = record

    for path, source in ((args.ratio_model_plan, "ratio"),
                         (args.cohort_plan, None)):
        full = ROOT / path
        if not full.exists():
            continue
        for line in full.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if source:
                    record["source"] = source
                if args.model_fill_only:
                    plan.setdefault(record["id"], record)
                else:
                    plan[record["id"]] = record

    greedy_path = ROOT / args.greedy_plan
    if greedy_path.exists():
        for line in greedy_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["source"] = "maso"
                record["greedy"] = True
                plan.setdefault(record["id"], record)

    payloads: dict[str, str] = {}
    rows = []
    answered = 0
    skipped_unit = 0
    for qid in sorted(questions):
        text = questions[qid]
        entry = plan.get(qid)
        record = {
            "id": qid,
            "question": text,
            "answer": 0.0,
            "relevant_docs": [],
            "relevant_tables": [],
            "evidence": [],
            "pandas_query": "result = 0.0",
        }
        if entry is not None and entry["source"] == "btc_formula":
            expression = FORMULA_OPS[entry["op"]]
            role_items = list(entry["roles"].items())
            names, evidence, reads = [], [], []
            frames: dict = {}
            import io

            import pandas as pd

            for position, (role, cell) in enumerate(role_items, start=1):
                name = f"{cell['doc']}_table_{cell['table_id']}.csv"
                if name not in payloads:
                    payloads[name] = (ROOT / cell["csv"]).read_text(
                        encoding="utf-8-sig")
                variable = "df" if len(role_items) == 1 else f"df{position}"
                # Reuse a frame name when the same CSV appears twice (begin/end).
                if name in names:
                    variable = evidence[names.index(name)]["variable"]
                else:
                    names.append(name)
                    evidence.append({"variable": variable,
                                     "csv_path": f"data/{name}"})
                    frames[variable] = pd.read_csv(
                        io.StringIO(payloads[name]), dtype=str,
                        keep_default_na=False)
                reads.append(
                    f"    {role!r}: _num({variable}.iloc"
                    f"[{cell['row']}, {cell['col']}]) * {cell['scale']!r},"
                )
            code = FORMULA_PROGRAM.format(
                role_reads="\n".join(reads), expression=expression)
            namespace: dict = {"pd": pd, **frames}
            exec(code, namespace, namespace)  # noqa: S102
            docs = list(dict.fromkeys(c["doc"] for c in entry["roles"].values()))
            refs = list(dict.fromkeys(
                c["table_ref"] for c in entry["roles"].values()))
            record.update({
                "answer": namespace["result"],
                "relevant_docs": docs,
                "relevant_tables": refs,
                "evidence": evidence,
                "pandas_query": code,
            })
            answered += 1
        elif entry is not None and entry["source"] == "cohort":
            _name, unit = unit_of(text)
            if not unit:
                skipped_unit += 1
            else:
                names, reads, evidence = [], [], []
                for position, cell in enumerate(entry["cells"], start=1):
                    name = f"{cell['doc']}_table_{cell['table_id']}.csv"
                    if name not in payloads:
                        payloads[name] = (ROOT / cell["csv"]).read_text(
                            encoding="utf-8-sig")
                    variable = f"df{position}"
                    names.append(name)
                    evidence.append({"variable": variable,
                                     "csv_path": f"data/{name}"})
                    reads.append(f"    abs(_num({variable}.iloc"
                                 f"[{cell['row']}, {cell['col']}])) "
                                 f"* {cell['scale']!r},  # {cell['ticker']}")
                code = COHORT_PROGRAM.format(
                    kind=entry["kind"], code=entry["code"], op=entry["op"],
                    reads="\n".join(reads),
                    expression=OPERATIONS[entry["op"]], unit=unit)
                import io

                import pandas as pd

                namespace: dict = {"pd": pd}
                for item, name in zip(evidence, names):
                    namespace[item["variable"]] = pd.read_csv(
                        io.StringIO(payloads[name]), dtype=str,
                        keep_default_na=False)
                exec(code, namespace, namespace)  # noqa: S102 - our own emitted code
                record.update({
                    "answer": namespace["result"],
                    "relevant_docs": list(dict.fromkeys(
                        c["doc"] for c in entry["cells"])),
                    "relevant_tables": list(dict.fromkeys(
                        c["table_ref"] for c in entry["cells"])),
                    "evidence": evidence,
                    "pandas_query": code,
                })
                answered += 1
        elif entry is not None and entry["source"] == "ratio":
            # A rate needs two cells, and they can sit in different tables, so the
            # program binds two frames. No money unit is involved: the question asks
            # for a % or a multiple, and the scales cancel except where the two
            # tables are denominated differently — which is why each operand carries
            # its own.
            names = []
            for side in ("num", "den"):
                part = entry[side]
                name = f"{part['doc']}_table_{part['table_id']}.csv"
                if name not in payloads:
                    payloads[name] = (ROOT / part["csv"]).read_text(
                        encoding="utf-8-sig")
                names.append(name)
            single = names[0] == names[1]
            num_var, den_var = ("df", "df") if single else ("df1", "df2")
            code = RATIO_PROGRAM.format(
                num_kind=entry["num"]["kind"], num_code=entry["num"]["code"],
                den_kind=entry["den"]["kind"], den_code=entry["den"]["code"],
                num_var=num_var, num_row=entry["num"]["row"],
                num_col=entry["num"]["col"], num_scale=entry["num"]["scale"],
                den_var=den_var, den_row=entry["den"]["row"],
                den_col=entry["den"]["col"], den_scale=entry["den"]["scale"],
                factor=100.0 if entry["op"] == "pct" else 1.0)
            import io

            import pandas as pd

            frames = {}
            if single:
                frames["df"] = pd.read_csv(io.StringIO(payloads[names[0]]),
                                           dtype=str, keep_default_na=False)
                evidence = [{"variable": "df", "csv_path": f"data/{names[0]}"}]
            else:
                frames["df1"] = pd.read_csv(io.StringIO(payloads[names[0]]),
                                            dtype=str, keep_default_na=False)
                frames["df2"] = pd.read_csv(io.StringIO(payloads[names[1]]),
                                            dtype=str, keep_default_na=False)
                evidence = [{"variable": "df1", "csv_path": f"data/{names[0]}"},
                            {"variable": "df2", "csv_path": f"data/{names[1]}"}]
            namespace: dict = {"pd": pd, **frames}
            exec(code, namespace, namespace)  # noqa: S102 - our own emitted code
            refs = list(dict.fromkeys([entry["num"]["table_ref"],
                                       entry["den"]["table_ref"]]))
            record.update({
                "answer": namespace["result"],
                "relevant_docs": list(dict.fromkeys([entry["num"]["doc"],
                                                     entry["den"]["doc"]])),
                "relevant_tables": refs,
                "evidence": evidence,
                "pandas_query": code,
            })
            answered += 1
        elif entry is not None:
            _name, unit = unit_of(text)
            if not unit:
                skipped_unit += 1
            else:
                source = ROOT / entry["csv"]
                name = f"{entry['doc']}_table_{entry['table_id']}.csv"
                if name not in payloads:
                    payloads[name] = source.read_text(encoding="utf-8-sig")
                if entry["source"] == "note":
                    # UNMEASURED CHOICE: note rows are read as magnitudes. There is
                    # no identity to check a note row against, so this rests on the
                    # question's phrasing — "chi phí X là bao nhiêu" asks for a size —
                    # and on the statement measurement, where 83% of cost cells are
                    # already positive so the magnitude changes only the bracketed
                    # 17%. Worth revisiting once these rows have a verifier.
                    wrap = "abs"
                    label = entry["ref"] or "thuyet minh"
                    kind = "thuyet minh"
                else:
                    wrap = "abs" if entry["code"] in MAGNITUDE_CODES.get(
                        entry["kind"], set()) else ""
                    label = entry["code"]
                    kind = entry["kind"]
                code = PROGRAM.format(row=entry["row"], col=entry["col"],
                                      code=label, kind=kind,
                                      wrap=wrap, scale=entry["scale"], unit=unit)
                # The answer is whatever the program produces, computed here the same
                # way so the two can never disagree. `validate_submission.py` runs
                # every program against the bundled csv and checks exactly that.
                import io

                import pandas as pd

                frame = pd.read_csv(io.StringIO(payloads[name]), dtype=str,
                                    keep_default_na=False)
                namespace: dict = {"df": frame, "pd": pd}
                exec(code, namespace, namespace)  # noqa: S102 - our own emitted code
                record.update({
                    "answer": namespace["result"],
                    "relevant_docs": [entry["doc"]],
                    "relevant_tables": [entry["table_ref"]],
                    "evidence": [{"variable": "df", "csv_path": f"data/{name}"}],
                    "pandas_query": code,
                })
                answered += 1
        rows.append(record)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in payloads.items():
            archive.writestr(f"data/{name}", text)
        archive.writestr("submission.json",
                         json.dumps(rows, ensure_ascii=False, indent=1))

    print(f"{len(rows)} dong, tra loi {answered} cau, {len(payloads)} csv")
    print(f"  bo qua vi khong nhan ra don vi: {skipped_unit}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
