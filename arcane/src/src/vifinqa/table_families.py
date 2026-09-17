"""Small, multi-label statement-family priors for retrieval experiments."""

from __future__ import annotations

from .retrieval import Table, tokenize


FAMILY_ANCHORS: dict[str, tuple[str, ...]] = {
    "balance": (
        "tai san", "no phai tra", "von chu so huu", "tien mat",
        "cho vay", "tien gui", "phai thu", "phai tra",
    ),
    "income": (
        "doanh thu", "thu nhap", "chi phi", "loi nhuan", "lai", "lo",
        "thue thu nhap",
    ),
    "cashflow": (
        "luu chuyen tien", "dong tien", "hoat dong kinh doanh",
        "hoat dong dau tu", "hoat dong tai chinh", "tien thu", "tien chi",
    ),
    "equity": ("von chu so huu", "co phieu", "co tuc", "co dong"),
    "note": ("thuyet minh", "du phong", "ky han", "phan loai", "chi tiet"),
}


def _text(table: Table) -> str:
    return " ".join((table.title, *table.context, *(" ".join(row) for row in table.rows[:120])))


def table_families(table: Table) -> frozenset[str]:
    """Return every family supported by a table; never force one label."""
    text = " ".join(tokenize(_text(table)))
    return frozenset(
        family for family, anchors in FAMILY_ANCHORS.items()
        if any(" ".join(tokenize(anchor)) in text for anchor in anchors)
    ) or frozenset({"other"})


def query_families(tokens: list[str]) -> frozenset[str]:
    text = " ".join(tokens)
    return frozenset(
        family for family, anchors in FAMILY_ANCHORS.items()
        if any(" ".join(tokenize(anchor)) in text for anchor in anchors)
    )


def family_match_score(query: list[str], table: Table) -> float:
    """Score family agreement in [0, 1] for an optional ranking view."""
    wanted = query_families(query)
    return 1.0 if wanted and wanted & table_families(table) else 0.0
