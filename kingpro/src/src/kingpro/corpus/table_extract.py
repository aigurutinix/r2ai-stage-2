"""Bóc bảng HTML nội dòng từ file OCR .txt của ViFinQA → CSV + chỉ mục theo SỐ DÒNG.

Vì sao tự viết phần này (không có trong baseline): data công khai ViFinQA để bảng ở dạng
thẻ HTML <table>...</table> ngay trong .txt, KHÔNG phải CSV tách sẵn như baseline giả định.
Ta chỉ viết GLUE, còn phần parse HTML dùng thư viện sẵn `pandas.read_html` (bê về ráp vào).

Định danh khớp bài nộp:
  - report_id = TÊN THƯ MỤC báo cáo (vd 'VNM_financial_statements_2023_consolidated'),
    đúng như ví dụ trong Submission Instructions.
  - relevant_tables = f"{report_id}|{line}"  với `line` = SỐ DÒNG (1-based) chứa thẻ <table>.
  ⚠️ Quy ước dòng (1-based, tại thẻ <table>) và report_id (tên thư mục) CẦN xác nhận lại
     bằng một bài nộp thử khi tài khoản được BTC duyệt — đây là "Việc 1: khoá định dạng".
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RE_TABLE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
RE_PAGE = re.compile(r"=====\s*PAGE\s+(\d+)\s*=====", re.IGNORECASE)


@dataclass
class ExtractedTable:
    report_id: str
    ordinal: int            # bảng thứ mấy trong báo cáo (0-based)
    line: int               # SỐ DÒNG 1-based nơi thẻ <table> bắt đầu -> dùng cho relevant_tables
    page: int               # trang OCR chứa bảng (0 nếu không rõ)
    n_rows: int
    n_cols: int
    csv_path: str           # đường dẫn CSV đã ghi (tương đối theo out_dir)
    df: pd.DataFrame = field(repr=False, default=None)

    @property
    def table_ref(self) -> str:
        return f"{self.report_id}|{self.line}"


def _line_of_offset(text: str, offset: int) -> int:
    """Số dòng 1-based của vị trí ký tự `offset`."""
    return text.count("\n", 0, offset) + 1


def _page_of_offset(text: str, offset: int) -> int:
    """Trang OCR gần nhất phía trước offset (0 nếu chưa có mốc PAGE)."""
    page = 0
    for m in RE_PAGE.finditer(text):
        if m.start() > offset:
            break
        page = int(m.group(1))
    return page


def report_id_from_path(txt_path: Path) -> str:
    """report_id = tên thư mục báo cáo (khớp ví dụ Submission Instructions)."""
    return txt_path.parent.name


def extract_tables(txt_path: str | Path, out_dir: str | Path | None = None) -> list[ExtractedTable]:
    """Bóc mọi <table> trong 1 file OCR. Nếu out_dir != None thì ghi CSV ra đó."""
    txt_path = Path(txt_path)
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    report_id = report_id_from_path(txt_path)

    out: list[ExtractedTable] = []
    for ordinal, m in enumerate(RE_TABLE.finditer(text)):
        block = m.group(0)
        line = _line_of_offset(text, m.start())
        page = _page_of_offset(text, m.start())
        try:
            dfs = pd.read_html(io.StringIO(block), flavor="lxml")
            df = dfs[0] if dfs else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()  # bảng OCR hỏng: giữ vị trí, để trống, xử lý sau

        csv_rel = ""
        if out_dir is not None:
            out_dir = Path(out_dir)
            tdir = out_dir / report_id
            tdir.mkdir(parents=True, exist_ok=True)
            csv_file = tdir / f"table_{ordinal}_line{line}.csv"
            df.to_csv(csv_file, index=False, encoding="utf-8-sig")
            csv_rel = str(csv_file.relative_to(out_dir)).replace("\\", "/")

        out.append(
            ExtractedTable(
                report_id=report_id,
                ordinal=ordinal,
                line=line,
                page=page,
                n_rows=int(df.shape[0]),
                n_cols=int(df.shape[1]),
                csv_path=csv_rel,
                df=df,
            )
        )
    return out


if __name__ == "__main__":
    import sys

    p = sys.argv[1]
    tables = extract_tables(p)
    print(f"{report_id_from_path(Path(p))}: {len(tables)} bảng")
    for t in tables[:12]:
        print(f"  ordinal={t.ordinal:<3} line={t.line:<6} page={t.page:<3} shape=({t.n_rows}x{t.n_cols})  ref={t.table_ref}")
