"""Renderer v2: show the model the frame the grader will actually build.

Every earlier renderer showed a cleaned view while programs referenced iloc on the
 organisers' frame (`pd.read_csv(dtype=str)`, one header row consumed). Wherever the two
 disagreed — duplicate column suffixes, a second header row left inside the data, merged
 labels concatenated without a separator — the model aimed at coordinates that did not
 exist at grading time. Trap #15 in disguise, on the rendering side.

So this renderer is built FROM the grader frame, not towards it:

  * row indices `rN` are true `.iloc` positions of the post-read_csv frame;
  * column indices `cN` are true positions, and the legend prints the mangled names
    (`2016`, `2016.1`) exactly as `df.columns` will hold them;
  * a second header row is not hidden — it is shown marked `[header phụ]`, because the
    grader leaves it in the data;
  * composite header paths (from multi-level headers) are printed once as a LEGEND,
    which preserves parent-child structure of columns without moving any cell;
  * the anchor prose and the declared unit are printed above the table, because the
    unit lives outside the grid and the judge that could not see it endorsed a x10 error;
  * printed identities (`TỔNG ... (270 = 100 + 200)`) are verified against the frame and
    only then annotated `# r12 = r03+r05 ✓` — code owns the tree, the model reads it.

Usage:
  python scripts/fresh/render_v2.py --doc KHG_financial_statements_2024_separate --table-id 41
  python scripts/fresh/render_v2.py --gold-id 1          # table behind easy_416 id=1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from num_helper import SOURCE as NUM_SRC  # noqa: E402

CORPUS = ROOT / "data" / "official_corpus"
NUM_NS: dict = {}
exec(compile(NUM_SRC, "<num>", "exec"), NUM_NS)
_num = NUM_NS["_num"]

TOTAL_RE = re.compile(r"^\s*(TỔNG|TỔNG CỘNG|CỘNG)\b", re.I)
IDENT_RE = re.compile(r"\((\d+[a-z]?)\s*=\s*([^)]+)\)")


def corpus_csv(doc_name: str, table_id: int) -> Path | None:
    m = re.match(r"^(?P<t>.+?)_financial_statements_(?P<y>\d{4})_(?P<s>\w+)$", doc_name)
    if m is None:
        return None
    return (CORPUS / m.group("t") / m.group("y") / doc_name /
            f"{doc_name}_extracted_tables" / f"table_{table_id}.csv")


def grader_frame(path: Path) -> pd.DataFrame:
    """Exactly what the organisers' sandbox `_read_raw_csv` builds."""
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str,
                       keep_default_na=False, index_col=None)


def looks_like_header(row: list[str], value_cols: list[int]) -> bool:
    cells = [row[c] for c in value_cols if str(row[c]).strip()]
    if len(cells) < 2:
        return False
    numeric = sum(1 for c in cells if _num(c) != 0.0 or c.strip().isdigit())
    return numeric / len(cells) < 0.3


def find_code_column(frame: pd.DataFrame) -> int | None:
    for i, name in enumerate(frame.columns[:4]):
        if "mã số" in str(name).strip().lower():
            return i
    best, best_frac = None, 0.5
    for i in range(min(4, frame.shape[1])):
        vals = [str(v).strip() for v in frame.iloc[:, i]]
        vals = [v for v in vals if v]
        if not vals:
            continue
        frac = sum(1 for v in vals if re.fullmatch(r"\d{1,4}", v)) / len(vals)
        if frac > best_frac:
            best, best_frac = i, frac
    return best


def value_columns(frame: pd.DataFrame) -> list[int]:
    out = []
    for i in range(frame.shape[1]):
        vals = [str(v) for v in frame.iloc[:, i]]
        vals = [v for v in vals if v.strip() and v.strip() != "-"]
        if not vals:
            continue
        numeric = sum(1 for v in vals
                      if re.search(r"\d", v)) / len(vals)
        if numeric > 0.6:
            out.append(i)
    return out


def identity_annotations(frame: pd.DataFrame, code_col: int | None,
                         val_cols: list[int]) -> dict[int, str]:
    """Verify printed sums row-by-row; only verified ones get an annotation."""
    notes: dict[int, str] = {}
    if code_col is None or not val_cols:
        return notes
    codes: dict[str, int] = {}
    for r, row in enumerate(frame.itertuples(index=False)):
        code = str(row[code_col]).strip()
        if re.fullmatch(r"\d{1,4}[a-z]?", code):
            codes.setdefault(code, r)
    vc = val_cols[0]
    for r, row in enumerate(frame.itertuples(index=False)):
        label = " ".join(str(row[0]).split())
        for m in IDENT_RE.finditer(label):
            target = m.group(1)
            parts = [p.strip() for p in re.split(r"[+\-]", m.group(2))
                     if p.strip()]
            signs = []
            pos, buf = 0, m.group(2)
            for tok in re.finditer(r"[+\-]|[^+\-]+", buf):
                signs.append(tok.group(0).strip())
            if target not in codes or any(p not in codes for p in parts):
                continue
            got = _num(row[vc])
            want = _num(frame.iat[codes[target], vc])
            total = 0.0
            idx = 0
            for p in parts:
                sign = -1.0 if idx > 0 and signs[idx].startswith("-") else 1.0
                total += sign * _num(frame.iat[codes[p], vc])
                idx += 1
            scale = max(abs(want), 1.0)
            if abs(want) > 0 and abs(got - want) / scale < 1e-3 \
                    and abs(total - want) / scale < 5e-3:
                terms = [f"r{codes[p]}" for p in parts]
                notes[r] = f"# r{r} = {'+'.join(terms)} ✓"
                break
    return notes


def render(frame: pd.DataFrame, *, meta_rows: dict, max_rows: int = 60,
           width: int = 46) -> str:
    n_rows, n_cols = frame.shape
    value_cols = value_columns(frame)
    lead = []
    for r in range(min(2, n_rows)):
        row = [str(v) for v in frame.iloc[r]]
        if looks_like_header(row, value_cols):
            lead.append(r)

    lines: list[str] = [
        f"BẢNG {meta_rows['doc_name']}|table_{meta_rows['table_id']} · "
        f"trang {meta_rows.get('page_no','')} · dòng {meta_rows.get('start_line','')}",
    ]
    caption = " ".join(str(meta_rows.get("caption") or "").split())
    anchor = caption[:220]
    if anchor:
        lines.append(f"ANCHOR (văn bản ngay trên bảng): {anchor}")
    unit = " ".join(str(meta_rows.get("unit_line") or "").split()) \
        or " ".join(str(meta_rows.get("unit_page") or "").split())
    if not unit:
        ud = " ".join(str(meta_rows.get("unit_doc") or "").split())
        unit = f"(không có dòng đơn vị trong bảng; cả tài liệu ghi: {ud})" if ud \
            else "KHÔNG THẤY — không suy đoán"
    lines.append(f"ĐƠN VỊ: {unit}")

    legend = []
    for c, name in enumerate(frame.columns):
        parts = [str(name)]
        for r in lead:
            cell = " ".join(str(frame.iat[r, c]).split())
            if cell and cell not in parts:
                parts.append(cell)
        if len(parts) > 1 or "/" in str(name):
            legend.append(f"c{c} = " + "/".join(p[:34] for p in parts))
    if legend:
        lines.append("LEGEND HEADER ĐẦY ĐỦ:  " + "  ·  ".join(legend[:12]))

    cols = " ".join(f"c{i}='{str(n)[:28]}'" for i, n in enumerate(frame.columns)
                    if i < 8)
    lines.append(f"CỘT: {cols}" + (" …" if n_cols > 8 else ""))
    lines.append("---")

    code_col = find_code_column(frame)
    notes = identity_annotations(frame, code_col, value_cols) if value_cols else {}
    shown = 0
    for r in range(n_rows):
        if shown >= max_rows and r not in notes:
            lines.append(f"… còn {n_rows - shown} hàng")
            break
        row = [str(v) for v in frame.iloc[r]]
        tag = "[header phụ] " if r in lead else ""
        cells = " │ ".join(c.strip()[:width] for c in row if str(c).strip())
        lines.append(f"r{r:<3d}│ {tag}{cells}")
        if r in notes:
            lines.append(f"     {notes[r]}")
        shown += 1
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc")
    parser.add_argument("--table-id", type=int, default=0)
    parser.add_argument("--gold-id", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=60)
    args = parser.parse_args()

    store_meta: dict = {"doc_name": args.doc, "table_id": args.table_id}
    path = None
    if args.gold_id:
        rec = None
        for line in (ROOT / "artifacts" / "easy_416.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["id"] == args.gold_id:
                rec = json.loads(line)
                break
        if rec is None:
            sys.exit("không thấy id trong easy_416")
        print(f"CÂU HỎI {rec['id']}: {rec['question']}")
        print(f"GOLD ANSWER: {rec['answer']}\n")
        ref = rec["relevant_tables"][0]
        doc, tid = ref.rsplit("|table_", 1)
        store_meta = {"doc_name": doc, "table_id": int(tid)}
        path = corpus_csv(doc, int(tid))
    elif args.doc:
        path = corpus_csv(args.doc, args.table_id)

    import pandas as pd  # noqa: F401
    parquet = pd.read_parquet(ROOT / "artifacts" / "tables.parquet")
    key = (store_meta["doc_name"], int(store_meta["table_id"]))
    hit = parquet[(parquet.doc_name == key[0]) & (parquet.table_id == key[1])]
    if hit.empty:
        sys.exit(f"không thấy bảng {key} trong tables.parquet")
    meta = hit.iloc[0].to_dict()
    if path is None or not path.exists():
        sys.exit(f"không thấy CSV chính thức cho {key}: {path}")

    frame = grader_frame(path)
    print(render(frame, meta_rows=meta, max_rows=args.max_rows))


if __name__ == "__main__":
    main()
