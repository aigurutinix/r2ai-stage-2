from scripts.audit_cagr_interval_semantics import audit


def row(query: str) -> dict:
    return {
        "id": 370,
        "question": "CAGR doanh thu giai đoạn 2022–2024 là bao nhiêu?",
        "pandas_query": query,
    }


def test_detects_off_by_one_number_of_periods() -> None:
    report = audit([row("result = ((end / start) ** (1 / 3) - 1) * 100")])
    assert report["question_ids"] == [370]
    assert report["findings"][0]["expected_interval_count"] == 2


def test_accepts_two_intervals_between_2022_and_2024() -> None:
    report = audit([row("result = ((end / start) ** (1 / 2) - 1) * 100")])
    assert report["finding_count"] == 0
