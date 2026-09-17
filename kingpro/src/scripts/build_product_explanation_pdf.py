"""Render the canonical KINGPRO V297 product explanation as a polished PDF."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
GREEN = colors.HexColor("#14845C")
NAVY = colors.HexColor("#102A43")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#627D98")
PALE = colors.HexColor("#EAF7F0")
GRID = colors.HexColor("#D9E2EC")


def register_fonts() -> None:
    candidates = [
        (
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path(r"C:\Windows\Fonts\consola.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        ),
    ]
    for regular, bold, mono in candidates:
        if all(path.is_file() for path in (regular, bold, mono)):
            pdfmetrics.registerFont(TTFont("KingproSans", str(regular)))
            pdfmetrics.registerFont(TTFont("KingproSans-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont("KingproMono", str(mono)))
            pdfmetrics.registerFontFamily(
                "KingproSans",
                normal="KingproSans",
                bold="KingproSans-Bold",
                italic="KingproSans",
                boldItalic="KingproSans-Bold",
            )
            return
    raise FileNotFoundError("No Unicode TrueType font available for PDF generation")


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        kwargs["invariant"] = 1
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        count = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(count)
            super().showPage()
        super().save()

    def _draw_footer(self, count: int) -> None:
        page = self._pageNumber
        self.saveState()
        width, _height = A4
        if page > 1:
            self.setStrokeColor(GRID)
            self.line(1.55 * cm, 1.35 * cm, width - 1.55 * cm, 1.35 * cm)
            self.setFont("KingproSans", 8)
            self.setFillColor(MUTED)
            self.drawString(1.55 * cm, 0.92 * cm, "KINGPRO · R2AI 2026 Stage 2 · V297")
            self.drawRightString(
                width - 1.55 * cm, 0.92 * cm, f"Trang {page}/{count}"
            )
        self.restoreState()


def inline_markup(text: str) -> str:
    escaped = html.escape(text.strip())
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#14845C">\1</link>',
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(
        r"`([^`]+)`", r'<font name="KingproMono" color="#0B5A3B">\1</font>', escaped
    )
    return escaped


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="KingproSans",
            fontSize=9.7,
            leading=14.2,
            textColor=INK,
            spaceAfter=7,
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName="KingproSans-Bold",
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName="KingproSans-Bold",
            fontSize=13.2,
            leading=17,
            textColor=GREEN,
            spaceBefore=9,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            fontName="KingproSans-Bold",
            fontSize=11.2,
            leading=14,
            textColor=NAVY,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName="KingproSans",
            fontSize=8.2,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "code": ParagraphStyle(
            "Code",
            fontName="KingproMono",
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#163B2D"),
            backColor=PALE,
            borderColor=colors.HexColor("#B7E2CE"),
            borderWidth=0.5,
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=8,
        ),
    }


def table_flow(lines: list[str], style_map: dict[str, ParagraphStyle], width: float):
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return Spacer(1, 1)
    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]
    rendered = [
        [Paragraph(inline_markup(cell), style_map["body"]) for cell in row]
        for row in rows
    ]
    col_widths = [width / columns] * columns
    table = Table(rendered, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "KingproSans-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.45, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return KeepTogether([table, Spacer(1, 7)])


def cover(story: list, style_map: dict[str, ParagraphStyle]) -> None:
    story.extend(
        [
            Spacer(1, 1.4 * cm),
            Table(
                [[Paragraph("KINGPRO", ParagraphStyle("Brand", fontName="KingproSans-Bold", fontSize=24, textColor=colors.white, alignment=TA_CENTER))]],
                colWidths=[5.2 * cm],
                rowHeights=[1.35 * cm],
                style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("BOX", (0, 0), (-1, -1), 0, NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
                hAlign="CENTER",
            ),
            Spacer(1, 1.5 * cm),
            Paragraph(
                "AI FINANCIAL DATA ASSISTANT",
                ParagraphStyle("CoverTitle", fontName="KingproSans-Bold", fontSize=25, leading=31, textColor=NAVY, alignment=TA_CENTER),
            ),
            Spacer(1, 0.35 * cm),
            Paragraph(
                "BẢN THUYẾT MINH SẢN PHẨM",
                ParagraphStyle("CoverSub", fontName="KingproSans-Bold", fontSize=15, leading=20, textColor=GREEN, alignment=TA_CENTER),
            ),
            Spacer(1, 0.8 * cm),
            Table(
                [
                    ["Cuộc thi", "Road to AI 2026 · Stage 2"],
                    ["Bản khóa", "V297 · Submission ID 3747"],
                    ["Trạng thái", "Public-final · Private-ready"],
                    ["Ngày hồ sơ", "28/08/2026"],
                ],
                colWidths=[4.2 * cm, 8.2 * cm],
                style=TableStyle(
                    [
                        ("FONTNAME", (0, 0), (0, -1), "KingproSans-Bold"),
                        ("FONTNAME", (1, 0), (1, -1), "KingproSans"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
                        ("TEXTCOLOR", (1, 0), (1, -1), INK),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
                        ("LEFTPADDING", (0, 0), (-1, -1), 9),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
                hAlign="CENTER",
            ),
            Spacer(1, 1.2 * cm),
            Paragraph(
                "Evidence-first · Deterministic execution · Cell-level lineage · Fail-closed",
                ParagraphStyle("Tagline", fontName="KingproSans", fontSize=10.5, leading=15, textColor=MUTED, alignment=TA_CENTER),
            ),
            Spacer(1, 2.2 * cm),
            Paragraph(
                "Canonical artifact SHA-256<br/><font name='KingproMono' size='7.5'>90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC</font>",
                ParagraphStyle("Hash", fontName="KingproSans", fontSize=8.5, leading=13, textColor=MUTED, alignment=TA_CENTER),
            ),
            PageBreak(),
        ]
    )


def parse_markdown(source: str, style_map: dict[str, ParagraphStyle], content_width: float):
    lines = source.splitlines()
    # The cover is rendered separately; start after the first horizontal rule.
    start = next((index + 1 for index, line in enumerate(lines) if line.strip() == "---"), 0)
    lines = lines[start:]
    story: list = []
    paragraph: list[str] = []
    bullets: list[str] = []
    code: list[str] = []
    in_code = False

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            text = " ".join(item.strip() for item in paragraph)
            story.append(Paragraph(inline_markup(text), style_map["body"]))
            paragraph = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            items = [
                ListItem(Paragraph(inline_markup(item), style_map["body"]), leftIndent=8)
                for item in bullets
            ]
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=18,
                    bulletColor=GREEN,
                    spaceAfter=6,
                )
            )
            bullets = []

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            flush_bullets()
            if in_code:
                story.append(Preformatted("\n".join(code), style_map["code"]))
                code = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code.append(line)
            index += 1
            continue
        image_match = re.fullmatch(r"<!--\s*IMAGE:([^|]+)\|(.+?)\s*-->", stripped)
        if image_match:
            flush_paragraph()
            flush_bullets()
            path = (ROOT / image_match.group(1).strip()).resolve()
            if path.is_file():
                img = Image(str(path))
                max_width, max_height = content_width, 12.4 * cm
                ratio = min(max_width / img.imageWidth, max_height / img.imageHeight)
                img.drawWidth = img.imageWidth * ratio
                img.drawHeight = img.imageHeight * ratio
                story.append(KeepTogether([img, Paragraph(inline_markup(image_match.group(2)), style_map["caption"])]))
            index += 1
            continue
        if stripped == "<!-- PAGEBREAK -->":
            flush_paragraph()
            flush_bullets()
            story.append(PageBreak())
            index += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            flush_bullets()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(table_flow(table_lines, style_map, content_width))
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            flush_bullets()
            level = len(heading.group(1))
            story.append(Paragraph(inline_markup(heading.group(2)), style_map[f"h{level}"]))
        elif re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            bullets.append(re.sub(r"^[-*]\s+", "", stripped))
        elif re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            bullets.append(stripped)
        elif not stripped:
            flush_paragraph()
            flush_bullets()
        elif stripped == "---":
            flush_paragraph()
            flush_bullets()
            story.append(Spacer(1, 6))
        else:
            paragraph.append(stripped.rstrip("  "))
        index += 1
    flush_paragraph()
    flush_bullets()
    if code:
        story.append(Preformatted("\n".join(code), style_map["code"]))
    return story


def build(source: Path, output: Path) -> None:
    register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = A4
    margin_x = 1.55 * cm
    top = 1.45 * cm
    bottom = 1.65 * cm
    content_width = page_width - 2 * margin_x
    doc = BaseDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=top,
        bottomMargin=bottom,
        title="KINGPRO V297 — Bản thuyết minh sản phẩm",
        author="Đội KINGPRO",
        subject="R2AI 2026 Stage 2 — AI Financial Data Assistant",
    )
    frame = Frame(margin_x, bottom, content_width, page_height - top - bottom, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame)])
    style_map = styles()
    story: list = []
    cover(story, style_map)
    story.extend(parse_markdown(source.read_text(encoding="utf-8"), style_map, content_width))
    doc.build(story, canvasmaker=NumberedCanvas)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "deliverables" / "KINGPRO_V297_FINAL" / "KINGPRO_V297_THUYET_MINH_SAN_PHAM.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "deliverables" / "KINGPRO_V297_FINAL" / "KINGPRO_V297_THUYET_MINH_SAN_PHAM.pdf",
    )
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
