"""CPU answering pass: panel two-hop + hardened panel_det + compose screens.

No LLM. Builds `artifacts/cpu_fix.jsonl` for splicing into a submission.

Usage:
  PYTHONPATH=src python scripts/run_cpu_fix.py
  PYTHONPATH=src python scripts/run_cpu_fix.py --compose-only
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import compose  # noqa: E402
from vifin.answering import ratio as ratio_mod  # noqa: E402
from vifin.answering.panel_det import load_panel, solve as solve_panel  # noqa: E402
from vifin.answering.panel_twohop import solve as solve_twohop  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402
from vifin.submit.validate import reads_no_frame  # noqa: E402

spec = importlib.util.spec_from_file_location("aud", ROOT / "scripts" / "audit_submission.py")
aud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aud)


def impossible(question: str, answer) -> bool:
    value = aud.as_float(answer)
    if value is None:
        return True
    if aud.YEAR_Q.search(question) and not (
            float(value).is_integer() and 1990 <= value <= 2100):
        return True
    if aud.SHARE_Q.search(question) and not aud.TIMES_Q.search(question) and \
            not aud.GROWTH_Q.search(question) and (value > 100 or value < 0):
        return True
    if aud.PERCENT_Q.search(question) and not aud.TIMES_Q.search(question) and \
            abs(value) > 1e5:
        return True
    if aud.TIMES_Q.search(question) and (value > 500 or value < 0):
        return True
    if aud.COUNT_Q.search(question) and (
            not float(value).is_integer() or value < 0 or value > 60):
        return True
    return value == 0


def csv_text(rows: list[list[str]]) -> str:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def pack_panel(qid: int, answer, source: str) -> dict | None:
    outcome = run_query(answer.code, {"df": answer.panel_rows})
    if not outcome.ok or reads_no_frame(answer.code):
        return None
    if abs(outcome.value - answer.value) > 0.011:
        return None
    return {
        "id": qid,
        "ok": True,
        "value": outcome.value,
        "code": answer.code,
        "source": source,
        "shape": answer.shape,
        "mode": "panel",
        "panel_rows": answer.panel_rows,
        "keys": [],
        "variables": ["df"],
    }


def pack_tables(qid: int, code: str, keys, store: TableStore, source: str,
                shape: str) -> dict | None:
    if not keys:
        return None
    names = ["df"] if len(keys) == 1 else [f"df{i}" for i in range(1, len(keys) + 1)]
    tables = {n: store.rows(k) for n, k in zip(names, keys)}
    outcome = run_query(code, tables)
    if not outcome.ok or reads_no_frame(code):
        return None
    return {
        "id": qid,
        "ok": True,
        "value": outcome.value,
        "code": code,
        "source": source,
        "shape": shape,
        "mode": "tables",
        "panel_rows": None,
        "keys": [[k.doc_name, k.table_id] for k in keys],
        "variables": names,
    }


def try_compose_screen(question, store, retriever) -> dict | None:
    screened = compose.resolve_screen(question, store, retriever)
    source = "compose_screen"
    if screened is None:
        screened = compose.resolve_screen_ratio(question, store, retriever)
        source = "compose_screen_ratio"
    if screened is None:
        screened = compose.resolve_screen_ratio_threshold(question, store, retriever)
        source = "compose_screen_thresh"
    if screened is None:
        return None
    keys = list(screened.operand_keys or [screened.key])
    return pack_tables(question.id, screened.code, keys, store, source, "screen")


def try_compose_agg(question, store, retriever) -> dict | None:
    composed = compose.resolve(question, store, retriever)
    if composed is None:
        return None
    keys = list(composed.keys)
    return pack_tables(question.id, composed.code, keys, store, "compose", composed.op)


def try_ratio(question, store, retriever) -> dict | None:
    divided = ratio_mod.resolve(question, store, retriever)
    if divided is None:
        return None
    return pack_tables(
        question.id, divided.code, list(divided.keys), store, "ratio_divide", "ratio"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="artifacts/cpu_fix.jsonl")
    parser.add_argument("--base", default="submissions/aimed_fill.zip")
    parser.add_argument("--compose-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import zipfile

    with zipfile.ZipFile(ROOT / args.base) as archive:
        base_rows = {
            r["id"]: r
            for r in json.loads(archive.read("submission.json").decode("utf-8"))
        }

    parsed = parse_all(
        ROOT / "data" / "questions" / "questions.jsonl",
        ROOT / "data" / "code_stock.csv",
    )
    if args.limit:
        parsed = parsed[: args.limit]

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    panel = load_panel(ROOT / "artifacts" / "metrics.parquet")
    print(f"questions={len(parsed)}  panel_keys={len(panel)}")

    shapes: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    written = 0
    out = ROOT / args.cache
    with out.open("w", encoding="utf-8") as handle:
        for question in parsed:
            base = base_rows.get(question.id)
            if base is None:
                continue
            base_bad = impossible(base["question"], base.get("answer"))
            # Exact 100.0 on share questions is the known A/A bug.
            share_100 = (
                abs(float(aud.as_float(base.get("answer")) or -1) - 100.0) < 1e-9
                and bool(aud.SHARE_Q.search(base["question"]))
            )

            hit = None
            if not args.compose_only:
                two = solve_twohop(question, panel)
                if two is not None:
                    hit = pack_panel(question.id, two, "panel_twohop")
                if hit is None:
                    one = solve_panel(question, panel)
                    if one is not None:
                        hit = pack_panel(question.id, one, "panel_det")

            # Compose screens: always try (OCR path for note-line two-hops).
            if hit is None:
                hit = try_compose_screen(question, store, retriever)

            # On defects / 100.0 share: also try compose agg + ratio divide.
            if hit is None and (base_bad or share_100):
                hit = try_ratio(question, store, retriever)
            if hit is None and (base_bad or share_100):
                hit = try_compose_agg(question, store, retriever)

            if hit is None:
                continue
            if impossible(base["question"], hit["value"]):
                continue

            # Prefer not to displace a plausible base with a near-identical value
            # unless the base is defective.
            old = aud.as_float(base.get("answer"))
            if (old is not None and abs(old - hit["value"]) < 0.011
                    and not base_bad and not share_100):
                # Still ship when source is twohop — program may be healthier —
                # but skip pure no-ops to keep the zip diff small.
                if hit["source"] not in ("panel_twohop", "compose_screen",
                                         "compose_screen_ratio",
                                         "compose_screen_thresh"):
                    continue

            handle.write(json.dumps(hit, ensure_ascii=False) + "\n")
            shapes[hit["shape"]] += 1
            sources[hit["source"]] += 1
            written += 1
            if written % 10 == 0:
                print(f"  wrote {written} …")

    print(f"wrote {written} -> {args.cache}")
    print("sources:", dict(sources))
    print("shapes:", dict(shapes))


if __name__ == "__main__":
    main()
