"""A three-hop evidence path: question slot, source row, evidence table.

The existing slot selector goes from a slot straight to a table, which works but
throws away the row it matched on. That row is where the report keeps its own
cross-references: the `Mã số` code identifying the line and the `Thuyết minh`
number naming the note that details it. Making the row a node instead of an
intermediate variable is what puts those references within reach.

    slot (report, line item)
        --MATCHES-->      row (table, index, label, code, note)
        --SOURCE_TABLE--> the table the row sits in
        --NOTE_REF-->     the note table the row points at

Three hops, and no more: every edge is read from the report, none is inferred.
A path never leaves the report it started in, so nothing here can invent a
cross-company or cross-year relationship.
"""

from __future__ import annotations

from dataclasses import dataclass

from .notes import code_column, note_column, note_table_line, row_label, row_note
from .slots import normalize


SOURCE_TABLE = "SOURCE_TABLE"
NOTE_REF = "NOTE_REF"


@dataclass(frozen=True)
class EvidenceSlot:
    """One report and line item the question requires evidence for."""

    report_id: str
    line_item: str


@dataclass(frozen=True)
class RowAnchor:
    """One source row that satisfies a slot, with what the report says about it."""

    report_id: str
    table_id: str
    row: int
    label: str
    line_item: str
    code: str = ""
    note: str = ""


@dataclass(frozen=True)
class Edge:
    """A table reached from a row, and the reason it was reached."""

    table_id: str
    kind: str
    anchor: RowAnchor


@dataclass(frozen=True)
class EvidencePath:
    """The complete slot -> row -> table path."""

    slot: EvidenceSlot
    anchor: RowAnchor
    edge: Edge


def evidence_slots(report_ids: list[str], line_items: list[str]) -> list[EvidenceSlot]:
    """Build every report/item obligation in stable question order."""
    return [EvidenceSlot(report_id, item) for report_id in report_ids for item in line_items]


def row_anchors(
    grid: list[list[str]], report_id: str, table_id: str, line_items: list[str],
) -> list[RowAnchor]:
    """Rows of one table whose label carries a line item the question named.

    Matching on the row label rather than the whole table is the point of the
    hop: a balance sheet mentions dozens of items, and only the row that states
    the one asked about carries the right note reference.
    """
    if not line_items:
        return []
    code_index = code_column(grid)
    note_index = note_column(grid, code_index)
    anchors = []
    for index, row in enumerate(grid):
        if not row:
            continue
        raw_label = row_label(row, code_index, note_index)
        label = normalize(raw_label)
        if not label:
            continue
        padded = f" {label} "
        item = next((item for item in line_items if f" {item} " in padded), None)
        if item is None:
            continue
        code, note = "", ""
        if code_index is not None and code_index < len(row):
            cell = row[code_index].strip()
            code = cell if cell.isdigit() else ""
            note = row_note(row, code_index, note_index) or ""
        anchors.append(RowAnchor(
            report_id=report_id, table_id=table_id, row=index,
            label=raw_label, line_item=item, code=code, note=note,
        ))
    return anchors


def anchor_edges(
    anchor: RowAnchor, report_text: str, source_line: int, follow_notes: bool = True,
) -> list[Edge]:
    """Tables reachable from one row anchor.

    `SOURCE_TABLE` restates where the row already is, which matters because a
    slot can be satisfied by a table the lexical ranking placed too low to
    submit. `NOTE_REF` is the report's own pointer to the detail table.
    """
    edges = [Edge(anchor.table_id, SOURCE_TABLE, anchor)]
    if follow_notes and anchor.note:
        line = note_table_line(report_text, anchor.note)
        if line is not None and line != source_line:
            edges.append(Edge(f"{anchor.report_id}|{line}", NOTE_REF, anchor))
    return edges


def table_paths(
    slot: EvidenceSlot,
    table_id: str,
    grid: list[list[str]],
    report_text: str,
    source_line: int,
    *,
    follow_notes: bool = True,
) -> list[EvidencePath]:
    """Paths through rows of one candidate table that satisfy one slot."""
    if table_id.rsplit("|", 1)[0] != slot.report_id:
        return []
    paths = []
    for anchor in row_anchors(grid, slot.report_id, table_id, [slot.line_item]):
        paths.extend(
            EvidencePath(slot, anchor, edge)
            for edge in anchor_edges(anchor, report_text, source_line, follow_notes)
        )
    return paths


def select_paths(
    paths: list[EvidencePath], already: set[str], limit: int,
) -> list[EvidencePath]:
    """Cover each slot once before taking a second path from any slot."""
    if limit <= 0:
        return []
    seen_tables = set(already)
    selected = []
    remaining = list(paths)
    slot_order = list(dict.fromkeys(path.slot for path in paths))
    for slot in slot_order:
        chosen = next(
            (path for path in remaining
             if path.slot == slot and path.edge.table_id not in seen_tables),
            None,
        )
        if chosen is None:
            continue
        selected.append(chosen)
        seen_tables.add(chosen.edge.table_id)
        remaining.remove(chosen)
        if len(selected) == limit:
            return selected
    for path in remaining:
        if path.edge.table_id in seen_tables:
            continue
        selected.append(path)
        seen_tables.add(path.edge.table_id)
        if len(selected) == limit:
            break
    return selected


def dedupe_edges(edges: list[Edge], already: set[str]) -> list[Edge]:
    """First edge wins per table, and nothing already selected is re-added.

    `SOURCE_TABLE` edges are emitted before `NOTE_REF` ones for a given row, so
    keeping the first occurrence keeps the more direct justification when two
    paths land on the same table.
    """
    seen = set(already)
    kept = []
    for edge in edges:
        if edge.table_id in seen:
            continue
        seen.add(edge.table_id)
        kept.append(edge)
    return kept
