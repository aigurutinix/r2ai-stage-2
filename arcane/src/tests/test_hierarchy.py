"""Hierarchy paths retain OCR table meaning without inventing relationships."""

from vifinqa.rerank import table_representation
from vifinqa.retrieval import (
    Report, Table, column_paths, materialize_candidate_rows, row_path, row_paths,
)
from vifinqa.tables import ReportIdentity


def table(headers, rows):
    return Table(
        table_id="R|1", report_id="R", page=1, start_line=1,
        rows=tuple(tuple(row) for row in rows), title="", context=(),
        headers=tuple(tuple(row) for row in headers), periods=("2024",), unit="VND",
    )


def test_column_paths_keep_parent_and_child_but_remove_span_repetition():
    source = table(
        headers=(
            ("Nhóm", "Chỉ tiêu", "Năm 2024", "Năm 2024"),
            ("Nhóm", "Chỉ tiêu", "Số cuối năm", "Số đầu năm"),
        ),
        rows=(("Tài sản ngắn hạn", "Tiền", "100", "90"),),
    )

    assert column_paths(source) == (
        ("Nhóm",),
        ("Chỉ tiêu",),
        ("Năm 2024", "Số cuối năm"),
        ("Năm 2024", "Số đầu năm"),
    )


def test_row_path_combines_parent_and_child_labels():
    source = table(
        headers=(("Nhóm", "Chỉ tiêu", "2024", "2023"),),
        rows=(("Tài sản ngắn hạn", "Tiền và tương đương tiền", "100", "90"),),
    )

    assert row_path(source, 0) == ("Tài sản ngắn hạn", "Tiền và tương đương tiền")


def test_row_path_excludes_account_and_note_coordinates():
    source = table(
        headers=(("Chỉ tiêu", "Mã số", "Thuyết minh", "2024"),),
        rows=(("Doanh thu bán hàng", "01", "VIII", "1.000"),),
    )

    assert row_path(source, 0) == ("Doanh thu bán hàng",)


def test_row_path_rejects_an_invalid_index():
    source = table(headers=(("Chỉ tiêu", "2024"),), rows=(("Doanh thu", "1"),))

    try:
        row_path(source, 2)
    except IndexError as error:
        assert error.args == (2,)
    else:
        raise AssertionError("invalid row index was accepted")


def test_reranker_representation_exposes_paths_only_when_requested():
    source = table(
        headers=(
            ("Nhóm", "Chỉ tiêu", "Năm 2024", "Năm 2024"),
            ("Nhóm", "Chỉ tiêu", "Số cuối năm", "Số đầu năm"),
        ),
        rows=(("Tài sản ngắn hạn", "Tiền", "100", "90"),),
    )

    flat = table_representation(source, 0, inventory=200)
    structured = table_representation(source, 0, inventory=200, hierarchy=True)

    assert "Đường dẫn dòng" not in flat
    assert "Đường dẫn cột" not in flat
    assert "Đường dẫn dòng: Tài sản ngắn hạn > Tiền" in structured
    assert "Năm 2024 > Số cuối năm" in structured


def test_section_heading_is_propagated_to_numeric_children():
    source = table(
        headers=(("Nhóm", "Chỉ tiêu", "Mã số", "2024"),),
        rows=(
            ("Nhóm", "Chỉ tiêu", "Mã số", "2024"),
            ("A. Tài sản ngắn hạn", "", "100", ""),
            ("", "Tiền và tương đương tiền", "110", "50"),
            ("B. Nợ phải trả", "", "300", ""),
            ("", "Phải trả người bán", "311", "20"),
        ),
    )

    paths = row_paths(source)

    assert paths[2] == ("A. Tài sản ngắn hạn", "Tiền và tương đương tiền")
    assert paths[4] == ("B. Nợ phải trả", "Phải trả người bán")
    assert "Chỉ tiêu" not in paths[2]


def test_candidate_rows_use_propagated_paths_only_when_opted_in(monkeypatch):
    source = table(
        headers=(("Nhóm", "Chỉ tiêu", "Mã số", "2024"),),
        rows=(
            ("Nhóm", "Chỉ tiêu", "Mã số", "2024"),
            ("A. Tài sản ngắn hạn", "", "100", ""),
            ("", "Tiền", "110", "50"),
        ),
    )
    identity = ReportIdentity("R", "AAA", 2024, "consolidated", "R.txt")
    report = Report(identity, None)
    monkeypatch.setattr("vifinqa.retrieval.report_tables", lambda *_: (source,))

    _, flat = materialize_candidate_rows([report])
    _, structured = materialize_candidate_rows([report], hierarchy=True)

    assert flat[2][2] == " Tiền 110 50"
    assert structured[2][2] == "A. Tài sản ngắn hạn > Tiền |  Tiền 110 50"

    reranker_text = table_representation(source, 2, inventory=200, hierarchy=True)
    assert "Các chỉ tiêu: A. Tài sản ngắn hạn; A. Tài sản ngắn hạn > Tiền" in reranker_text
    assert "Nhóm > Chỉ tiêu > Mã số > 2024" not in reranker_text
