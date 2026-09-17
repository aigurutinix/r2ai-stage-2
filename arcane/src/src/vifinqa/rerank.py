"""How a table is described to the reranker, in the one place both sides read.

Training and serving must render a candidate identically or the adapter is
tuned on text the scorer never produces, so the representation and the numbers
that shape it live here rather than as literals in each caller.

The zero-shot mMiniLM cross-encoder this module used to run is gone. It lost
0.15 F2 and its replacement is a Qwen3 reranker scored off-box.
"""

from __future__ import annotations

from .retrieval import Table, column_paths, row_paths


# How much of a table's row-label inventory the representation carries. Training
# and serving must render candidates identically, so the number lives here rather
# than as a literal in each caller.
INVENTORY = 600

# How a filing's scope reads in a question. The corpus stores an English tag; the
# questions say it in Vietnamese, and the representation has to meet them there.
SCOPE_WORDING = {"consolidated": "hợp nhất", "separate": "công ty mẹ"}


def identity_line(identity) -> str:
    """Which filing a table came from, as one line the reranker can read.

    Nothing needed this while a gate settled the ticker, the year and the scope
    before any ranking ran: every candidate shared them, so they separated
    nothing and spending tokens on them was waste. A dense first stage does not
    go through that gate, so a candidate can now be the right statement from the
    wrong issuer or the wrong year, and a representation that omits the filing
    leaves the reranker no way to tell.

    One implementation, because training and serving must render a candidate
    identically or the adapter is tuned on text the scorer never produces.
    """
    scope = SCOPE_WORDING.get(identity.scope, identity.scope)
    return f"{identity.ticker} {identity.year} {scope}".strip()


def table_representation(
    table: "Table", row_index: int, *, inventory: int = 0, identity=None,
    hierarchy: bool = False,
) -> str:
    """Build one deterministic, non-repeated table representation.

    With `inventory`, the table is described by its line items rather than by the
    single row the sparse ranker matched. That row is the one that made the table
    look relevant, so showing only it asks the reranker to judge a table through
    the weaker ranker's choice, and it cannot recover when that choice is wrong.
    Listing the row labels tells the model what the table actually contains.

    A line naming the table's ordinal in its report used to sit here too. Its
    only justification was where gold fell in our own annotations, and those are
    quarantined, so it went with them.
    """
    headers = " | ".join(" | ".join(row) for row in table.headers)
    periods = " | ".join(table.periods)
    row = " | ".join(table.rows[row_index])
    paths = row_paths(table) if hierarchy else ()
    path = " > ".join(paths[row_index]) if hierarchy else ""
    columns = "; ".join(
        " > ".join(parts) for parts in column_paths(table) if parts
    ) if hierarchy else ""
    listed = ""
    if inventory:
        header_rows = set(table.headers)
        labels = dict.fromkeys(
            " > ".join(paths[index]) if hierarchy
            else " ".join(cell for cell in item[:1] if cell).strip()
            for index, item in enumerate(table.rows)
            if item not in header_rows and any(cell.strip() for cell in item)
        )
        listed = "; ".join(label for label in labels if label)[:inventory]
    # Ordered by how much each part decides the ranking, because the tail is what
    # truncation eats: the median representation is 621 characters and 16% run past
    # a 320-token window. Title and matched row identify the table, the inventory
    # says what else it holds, and the header and period boilerplate goes last.
    # First, because it is the one part that can disqualify a table outright: a
    # perfect line-item match in another company's filing is still wrong.
    filing = identity_line(identity) if identity is not None else ""
    parts = [
        filing,
        table.title,
        f"Đường dẫn dòng: {path}" if path else "",
        row,
        f"Các chỉ tiêu: {listed}" if listed else "",
        f"Đường dẫn cột: {columns}" if columns else headers,
        periods,
        table.unit,
    ]
    return "\n".join(part for part in dict.fromkeys(part.strip() for part in parts) if part)
