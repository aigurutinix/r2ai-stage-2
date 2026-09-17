
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PAGE_RE = re.compile(r"===== PAGE (\d+) =====")
_TABLE_ANCHOR_RE = re.compile(r"\[table_(\d+)\]\([^)]*\)")

_UNIT_LINE_PREFIXES = ("Đơn vị tính", "Đơn vị tiền tệ", "Đơn vị", "ĐVT")


@dataclass(frozen=True, slots=True)
class TableAnchor:
    table_id: int
    page_no: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Document:
    pages: tuple[tuple[int, str], ...]
    table_page: dict[int, int]
    anchors: tuple[TableAnchor, ...] = ()

    def table_context(self, table_id: int, before: int = 1, after: int = 1) -> str:
        page_no = self.table_page.get(table_id)
        if page_no is None:
            return ""
        index = next((i for i, (no, _) in enumerate(self.pages) if no == page_no), None)
        if index is None:
            return ""
        lo = max(0, index - before)
        hi = min(len(self.pages), index + after + 1)
        chunks = [f"===== PAGE {no} =====\n{content}" for no, content in self.pages[lo:hi]]
        return "\n\n".join(chunks)

    def _page_content(self, table_id: int) -> str | None:
        page_no = self.table_page.get(table_id)
        if page_no is None:
            return None
        return next((content for no, content in self.pages if no == page_no), None)

    def table_anchor_context(self, table_id: int, *, lines_before: int = 6, lines_after: int = 2) -> str:
        content = self._page_content(table_id)
        if content is None:
            return ""
        lines = content.splitlines()
        anchor_idx = next((i for i, line in enumerate(lines) if f"[table_{table_id}]" in line), None)
        if anchor_idx is None:
            return content.strip()
        lo = max(0, anchor_idx - lines_before)
        hi = min(len(lines), anchor_idx + lines_after + 1)
        return "\n".join(lines[lo:hi]).strip()

    def table_unit_snippets(self, table_id: int) -> tuple[str, ...]:
        content = self._page_content(table_id)
        if content is None:
            return ()
        snippets: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if any(stripped.casefold().startswith(prefix.casefold()) for prefix in _UNIT_LINE_PREFIXES):
                snippets.append(stripped)
        return tuple(snippets)

    def table_context_before(self, table_id: int) -> str:
        current_idx = next((i for i, anchor in enumerate(self.anchors) if anchor.table_id == table_id), None)
        if current_idx is None:
            return ""
        current = self.anchors[current_idx]
        content = next((text for page_no, text in self.pages if page_no == current.page_no), None)
        if content is None:
            return ""

        previous = self.anchors[current_idx - 1] if current_idx > 0 else None
        start = previous.end if previous is not None and previous.page_no == current.page_no else 0
        line_end = content.find("\n", current.end)
        end = len(content) if line_end < 0 else line_end
        interval = _TABLE_ANCHOR_RE.sub("", content[start:end])
        return "\n".join(line.strip() for line in interval.splitlines() if line.strip()).strip()


def parse_document(text_path: Path) -> Document:
    text = text_path.read_text(encoding="utf-8")
    matches = list(_PAGE_RE.finditer(text))

    raw_pages: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_pages.append((int(m.group(1)), text[start:end].strip("\n")))
    raw_pages.sort(key=lambda item: item[0])

    table_page: dict[int, int] = {}
    anchors: list[TableAnchor] = []
    for page_no, content in raw_pages:
        for tm in _TABLE_ANCHOR_RE.finditer(content):
            table_id = int(tm.group(1))
            table_page[table_id] = page_no
            anchors.append(TableAnchor(table_id=table_id, page_no=page_no, start=tm.start(), end=tm.end()))

    return Document(pages=tuple(raw_pages), table_page=table_page, anchors=tuple(anchors))
