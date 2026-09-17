import pandas as pd
import pytest

from vifinqa.generated_query import (
    GeneratedQueryError,
    compile_lookup_query,
    execute_generated_query,
    is_direct_lookup_question,
    numeric_consensus,
)


def test_executes_evidence_bound_numeric_lookup():
    frame = pd.DataFrame([["Doanh thu", "1.234,5"]])
    query = "result = round(float(str(df0.iloc[0, 1]).replace('.', '').replace(',', '.')), 2)"
    assert execute_generated_query(query, {"df0": frame}) == 1234.5


def test_rejects_code_outside_the_answer_expression():
    with pytest.raises(GeneratedQueryError):
        execute_generated_query("import os\nresult = 1", {"df0": pd.DataFrame([[1]])})


def test_rejects_queries_that_ignore_evidence():
    with pytest.raises(GeneratedQueryError):
        execute_generated_query("result = 42", {"df0": pd.DataFrame([[1]])})


def test_compiles_selected_cell_with_source_and_requested_units():
    frame = pd.DataFrame({"Năm 2023 Triệu đồng": ["2.307,5"]})
    query = compile_lookup_query("Chi phí là bao nhiêu tỷ đồng?", "df0", frame, 0, 0)
    assert execute_generated_query(query, {"df0": frame}) == 2.31


def test_compiles_lookup_with_retrieval_context_unit():
    frame = pd.DataFrame({"Năm 2023": ["444,918"]})
    query = compile_lookup_query(
        "Lợi nhuận là bao nhiêu tỷ đồng?", "df0", frame, 0, 0,
        "Các chỉ tiêu: Nội dung > ( Triệu đồng )",
    )
    assert execute_generated_query(query, {"df0": frame}) == 444.92


def test_ignores_units_mentioned_only_in_context_prose():
    frame = pd.DataFrame({"Năm 2023": ["2.307.758.675"]})
    query = compile_lookup_query(
        "Chi phí là bao nhiêu triệu đồng?", "df0", frame, 0, 0,
        "Công ty nhận 835 tỷ VND.\nCác chỉ tiêu: VND\nĐường dẫn cột: Năm 2023",
    )
    assert execute_generated_query(query, {"df0": frame}) == 2307.76


def test_rejects_model_selected_note_column():
    frame = pd.DataFrame({"Chỉ tiêu": ["Chi phí"], "Thuyết minh": ["30"], "Năm nay": ["100"]})
    with pytest.raises(GeneratedQueryError):
        compile_lookup_query("Chi phí?", "df0", frame, 0, 1)


def test_routes_only_direct_questions_to_lookup():
    assert is_direct_lookup_question("Lợi nhuận sau thuế năm 2023 là bao nhiêu tỷ đồng?")
    assert not is_direct_lookup_question("Tăng trưởng lợi nhuận từ năm 2022 đến 2023 là bao nhiêu phần trăm?")
    assert not is_direct_lookup_question("Năm nào có doanh thu cao nhất?")
    assert is_direct_lookup_question("Tổng tài sản năm 2023 là bao nhiêu tỷ đồng?")
    assert is_direct_lookup_question("Tổng tỷ lệ quyền biểu quyết năm 2023 là bao nhiêu phần trăm?")
    assert not is_direct_lookup_question("Tổng của doanh thu và thu nhập khác năm 2023 là bao nhiêu?")


def test_numeric_consensus_uses_organizer_tolerance():
    winner = numeric_consensus([{"answer": 100.0}, {"answer": 100.01}, {"answer": 120.0}])
    assert len(winner) == 2


def test_executes_safe_argmax_then_lookup_program():
    frame = pd.DataFrame({"year": [2022, 2023], "revenue": [10, 20], "profit": [1, 4]})
    query = "result = round(float(df0.loc[df0['revenue'].astype(float).idxmax(), 'profit']), 2)"
    assert execute_generated_query(query, {"df0": frame}) == 4.0
