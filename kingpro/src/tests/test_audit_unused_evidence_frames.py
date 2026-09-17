from scripts.audit_unused_evidence_frames import audit_row


def evidence(variable: str) -> dict:
    return {"variable": variable, "csv_path": f"data/{variable}.csv"}


def test_finds_dead_positional_frames_like_q300() -> None:
    row = {
        "id": 300,
        "question": "q300",
        "answer": 4.23,
        "evidence": [evidence("df1"), evidence("df2"), evidence("df3")],
        "pandas_query": """
_dfvals = list(dfs.values())
df1 = _dfvals[0]
df2 = _dfvals[1]
df3 = _dfvals[2]
picked = df1.iloc[0, 1]
result = float(picked)
""",
    }

    finding = audit_row(row)

    assert finding is not None
    assert finding["used_indices"] == [0]
    assert finding["unused_indices"] == [1, 2]


def test_keeps_all_reachable_positional_frames() -> None:
    row = {
        "id": 1,
        "evidence": [evidence("df1"), evidence("df2")],
        "pandas_query": """
_dfvals = list(dfs.values())
df1 = _dfvals[0]
df2 = _dfvals[1]
result = float(df1.iloc[0, 1]) + float(df2.iloc[0, 1])
""",
    }

    assert audit_row(row) is None


def test_skips_dynamic_iteration_without_explicit_bindings() -> None:
    row = {
        "id": 2,
        "evidence": [evidence("df1"), evidence("df2")],
        "pandas_query": "result = sum(float(frame.iloc[0, 1]) for frame in dfs.values())",
    }

    assert audit_row(row) is None
