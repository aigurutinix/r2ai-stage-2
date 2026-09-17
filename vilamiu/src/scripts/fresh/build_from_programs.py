"""Ship the model's programs in the format the shipped submissions actually use.

Two conventions have to be honoured, and neither is what the model was prompted with:

  the frames    a submission bundles its csv files and binds them to `df1`, `df2`, … via
                the `evidence` list. The model wrote `dfs["<table_ref>"]`, which is the
                organisers' answering convention, so a three-line prelude maps one onto
                the other. The model's code is shipped unchanged — rewriting it by hand
                would put my mistakes inside a program I am claiming the model wrote.
  the answer    `answer` is the value the program produced here, executed against the
                same csv files that ship in the zip. It is not copied from anywhere and
                not a constant: the private round rejects a constant assignment, and a
                program whose result was pasted in would not survive a re-run.

`relevant_tables` and `relevant_docs` are left empty on purpose. This build exists to be
spliced onto the best zip, which keeps its own declarations, so producing table
references here would only invite them to disagree with the ones that ship.

Usage:
  python scripts/fresh/build_from_programs.py --results artifacts/fresh/prog_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from num_helper import SOURCE as NUM_SOURCE  # noqa: E402

PRELUDE = '''# The frames are bundled with this submission and bound by `evidence`; the
# program below addresses them by the table reference it was shown, so the two
# names are joined here.
dfs = {{
{mapping}
}}
df = df1

'''


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="artifacts/fresh/prog_results.jsonl")
    parser.add_argument("--out", default="submissions/fresh_prog.zip")
    parser.add_argument("--ids-file", default="")
    args = parser.parse_args()

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    only = None
    if args.ids_file:
        only = set(json.loads((ROOT / args.ids_file).read_text(encoding="utf-8")))

    results = {}
    for line in (ROOT / args.results).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if only is None or record["id"] in only:
                results[record["id"]] = record

    rows, files, answered = [], {}, 0
    for qid, question in sorted(questions.items()):
        entry = results.get(qid)
        if entry is None:
            rows.append({"id": qid, "question": question, "answer": 0.0,
                         "pandas_query": "result = 0.0", "evidence": [],
                         "relevant_docs": [], "relevant_tables": []})
            continue
        # Bundle only what the program reads. Nine candidate tables were offered per
        # question and a program typically uses one or two, so shipping all nine would
        # multiply the zip and make `evidence` claim tables nothing looked at.
        used = {ref: rel for ref, rel in entry["csvs"].items()
                if ref in entry["pandas_query"]} or entry["csvs"]
        evidence, mapping = [], []
        for position, (ref, csv_rel) in enumerate(sorted(used.items()), start=1):
            name = f"data/{Path(csv_rel).parent.parent.name}_table_" \
                   f"{Path(csv_rel).stem.split('_')[-1]}.csv"
            source = ROOT / csv_rel
            if not source.exists():
                continue
            files[name] = source.read_bytes()
            evidence.append({"variable": f"df{position}", "csv_path": name})
            mapping.append(f'    {ref!r}: df{position},')
        if not evidence:
            rows.append({"id": qid, "question": question, "answer": 0.0,
                         "pandas_query": "result = 0.0", "evidence": [],
                         "relevant_docs": [], "relevant_tables": []})
            continue
        # The parser ships with the program: it was supplied at execution time, so a
        # submission that omitted it would contain code that cannot run.
        code = (NUM_SOURCE + PRELUDE.format(mapping="\n".join(mapping))
                + entry["pandas_query"])
        rows.append({"id": qid, "question": question,
                     "answer": entry["answer"], "pandas_query": code,
                     "evidence": evidence, "relevant_docs": [],
                     "relevant_tables": []})
        answered += 1

    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json",
                         json.dumps(rows, ensure_ascii=False, indent=1))
        for name, blob in sorted(files.items()):
            archive.writestr(name, blob)
    print(f"{len(rows)} dong, tra loi {answered} cau, {len(files)} csv")
    print(f"-> {target}")


if __name__ == "__main__":
    main()
