from vifinqa.table_families import query_families, table_families
from vifinqa.retrieval import Table


def make_table(rows):
    return Table(
        table_id="R|1", report_id="R", page=1, start_line=1,
        rows=tuple(tuple(row) for row in rows), title="", context=(),
        headers=(("Chỉ tiêu", "2024"),), periods=("2024",), unit="VND",
    )


def test_table_families_are_multilabel():
    families = table_families(make_table([
        ("Cho vay khách hàng", "100"),
        ("Lợi nhuận sau thuế", "20"),
    ]))
    assert "balance" in families
    assert "income" in families


def test_query_family_detection_is_phrase_based():
    assert query_families(["loi", "nhuan"]) == {"income"}
    assert query_families(["khong", "lien", "quan"]) == set()
