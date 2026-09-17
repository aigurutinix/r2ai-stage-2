from kingpro.answering import ensemble


def test_aggregate_facts_preserves_ordered_difference_sign():
    facts = [("ACV", 10.0), ("VJC", 16.0)]
    question = "ACV nhiều hơn VJC bao nhiêu? Chênh lệch là bao nhiêu?"
    assert ensemble.aggregate_facts(question, facts) == -6.0


def test_aggregate_facts_uses_absolute_only_when_explicit():
    facts = [("MBB", 10.0), ("MSB", 16.0)]
    assert ensemble.aggregate_facts("Độ chênh lệch giữa MBB và MSB là bao nhiêu?", facts) == 6.0


def test_aggregate_facts_refuses_nested_selector():
    facts = [("selector 2022", 100.0), ("target 2022", 4.0)]
    question = "Chỉ tiêu X bằng bao nhiêu trong năm có chỉ tiêu Y lớn nhất?"
    assert ensemble.aggregate_facts(question, facts) is None


def test_aggregate_facts_still_handles_simple_extreme():
    facts = [("2022", 100.0), ("2023", 140.0)]
    assert ensemble.aggregate_facts("Giá trị lớn nhất trong hai năm là bao nhiêu?", facts) == 140.0


def test_self_consistent_retains_an_all_zero_quorum(monkeypatch):
    monkeypatch.setattr(
        ensemble,
        "answer_question",
        lambda *args, **kwargs: {
            "ok": True,
            "answer": 0.0,
            "pandas_query": "result = 0.0",
            "evidence": [],
        },
    )
    result = ensemble.self_consistent("zero", [], "https://example.test", "k", "m", n=3)
    assert result["answer"] == 0.0
    assert result["consensus"] == 1.0
    assert result["n_ok"] == 3
