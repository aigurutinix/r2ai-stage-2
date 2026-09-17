from unittest.mock import patch

import pandas as pd

from vifinqa.codegen.consensus import (
    ConsensusDecision,
    choose_code_ensemble_candidate,
    choose_code_ensemble_candidates,
    choose_ensemble_candidate,
    code_evidence,
    verify_code_consensus_candidate,
    verify_consensus_candidate,
)


class _Bundle:
    route = {"output_type": "number"}
    dfs = {"df1": object()}

    def used_vars(self, _query):
        return [{"var": "df1", "report_id": "AAA_2024_consolidated",
                 "table_pos": 2}]


class _StructuredBundle(_Bundle):
    def __init__(self, route):
        self.route = route


class _CodeBundle:
    def __init__(self, requirements=None):
        self.route = {
            "question": "Doanh thu nam 2024 la bao nhieu?",
            "output_type": "number",
            "years": [2024],
            "plan": {"op": "lookup"},
            "evidence_requirements": requirements or [],
        }
        frame = pd.DataFrame([
            {"row": 4, "label": "Doanh thu thuần", "code": "10", "col": 1,
             "col_name": "Năm 2024", "value": 12.0, "unit_scale": 1.0},
            {"row": 5, "label": "Doanh thu hoạt động tài chính", "code": "21",
             "col": 1, "col_name": "Năm 2024", "value": 12.0,
             "unit_scale": 1.0},
        ])
        self.tables = [{
            "var": "df1",
            "report_id": "AAA_financial_statements_2024_consolidated",
            "report_year": 2024,
            "table_pos": 2,
            "context": "Báo cáo kết quả hoạt động kinh doanh",
            "csv_text": frame.to_csv(index=False),
        }]
        self.cands = [{
            "report_id": "AAA_financial_statements_2024_consolidated", "table_pos": 2,
            "requirement_hits": ["revenue"],
        }]
        self.dfs = {"df1": frame}

    def used_vars(self, _query):
        return [{"var": "df1",
                 "report_id": "AAA_financial_statements_2024_consolidated",
                 "table_pos": 2}]


def _revenue_requirement():
    return {
        "requirement_id": "revenue", "ticker": "AAA", "year": 2024,
        "doc_type": "consolidated", "metric_key": "net_revenue",
        "metric_label": "doanh thu thuan",
        "metric_variants": ["doanh thu thuan"],
        "statement": "income_statement",
    }


def _candidate(**overrides):
    row = {
        "status": "ok", "source": "llm_select", "votes": 4, "n_ok": 5,
        "detail_conf": 90.0, "answer": 12.0, "pandas_query": "float(df1.iloc[0, 0])",
        "selection_evidence": {"required": 1, "covered": 1, "complete": True},
        "used_vars": [{"var": "df1", "report_id": "AAA_2024_consolidated",
                       "table_pos": 2}],
    }
    row.update(overrides)
    return row


def _replay(*_args, **_kwargs):
    return {"status": "ok", "value": 12.0,
            "semantic": {"ok": True, "warnings": [], "errors": []}}


def _code_candidate(**overrides):
    row = _candidate(
        source="llm", votes=4, n_ok=5, detail_conf=0.0,
        pandas_query=(
            "float(df1.loc[(df1['row'] == 4) & (df1['col'] == 1), "
            "'value'].iloc[0])"
        ),
        used_vars=[{
            "var": "df1",
            "report_id": "AAA_financial_statements_2024_consolidated",
            "table_pos": 2,
        }],
    )
    row.pop("selection_evidence")
    row.update(overrides)
    return row


def test_accepts_four_of_five_with_fresh_replay_and_complete_evidence():
    with patch("vifinqa.codegen.consensus._run_validated", _replay):
        decision = verify_consensus_candidate(_candidate(), _Bundle())
    assert decision.accepted


def test_rejects_three_of_five():
    with patch("vifinqa.codegen.consensus._run_validated", _replay):
        decision = verify_consensus_candidate(
            _candidate(votes=3), _Bundle())
    assert not decision.accepted
    assert "below threshold" in decision.reason


def test_rejects_incomplete_evidence():
    coverage = {"required": 2, "covered": 1, "complete": False}
    with patch("vifinqa.codegen.consensus._run_validated", _replay):
        decision = verify_consensus_candidate(
            _candidate(selection_evidence=coverage), _Bundle())
    assert not decision.accepted


def test_rejects_semantic_warning_and_variable_mismatch():
    warning = {"status": "ok", "value": 12.0,
               "semantic": {"ok": True, "warnings": ["unit"], "errors": []}}
    with patch("vifinqa.codegen.consensus._run_validated", return_value=warning):
        decision = verify_consensus_candidate(_candidate(), _Bundle())
    assert not decision.accepted

    with patch("vifinqa.codegen.consensus._run_validated", _replay):
        decision = verify_consensus_candidate(
            _candidate(used_vars=[]), _Bundle())
    assert not decision.accepted


def test_rejects_ratio_without_canonical_evidence_for_both_operands():
    route = {
        "question": "Ty le CapEx tren tai san co dinh la bao nhieu?",
        "plan": {"op": "ratio", "facts": [{}, {}]},
        "evidence_requirements": [{"requirement_id": "denominator"}],
    }
    with patch("vifinqa.codegen.consensus._run_validated", _replay):
        decision = verify_consensus_candidate(
            _candidate(), _StructuredBundle(route))
    assert not decision.accepted
    assert "both operands" in decision.reason


def test_rejects_negative_absolute_difference():
    route = {
        "question": "Chenh lech gia tri giua A va B la bao nhieu?",
        "plan": {"op": "difference", "facts": [{}, {}]},
        "evidence_requirements": [
            {"requirement_id": "a"}, {"requirement_id": "b"},
        ],
    }
    coverage = {"required": 2, "covered": 2, "complete": True}
    with patch("vifinqa.codegen.consensus._run_validated", _replay):
        decision = verify_consensus_candidate(
            _candidate(answer=-12.0, selection_evidence=coverage),
            _StructuredBundle(route),
        )
    assert not decision.accepted
    assert "absolute difference" in decision.reason


def test_ensemble_accepts_cross_run_agreement():
    verified = ConsensusDecision(True, "ok")
    first = _candidate(answer=20.0, votes=4)
    second = _candidate(answer=20.0, votes=5)

    choice = choose_ensemble_candidate(
        {"answer": 10.0}, [(first, verified), (second, verified)])

    assert choice.candidate is second


def test_ensemble_rejects_cross_run_disagreement():
    verified = ConsensusDecision(True, "ok")
    choice = choose_ensemble_candidate(
        {"answer": 10.0},
        [(_candidate(answer=20.0), verified),
         (_candidate(answer=30.0, votes=5), verified)],
    )
    assert choice.candidate is None


def test_ensemble_requires_unanimity_for_one_run():
    verified = ConsensusDecision(True, "ok")
    four = choose_ensemble_candidate(
        {"answer": 10.0}, [(_candidate(answer=20.0, votes=4), verified)])
    five = choose_ensemble_candidate(
        {"answer": 10.0},
        [(_candidate(answer=20.0, votes=5, n_ok=5), verified)],
    )
    assert four.candidate is None
    assert five.candidate is not None


def test_code_candidate_requires_traceable_cells_and_fresh_replay():
    bundle = _CodeBundle([_revenue_requirement()])
    with patch("vifinqa.codegen.consensus._run_validated", _replay):
        decision = verify_code_consensus_candidate(_code_candidate(), bundle)
        ungrounded = verify_code_consensus_candidate(
            _code_candidate(pandas_query="float(df1['value'].iloc[0])"), bundle)

    assert decision.accepted
    assert not ungrounded.accepted
    assert "traceable" in ungrounded.reason
    assert code_evidence(_code_candidate(), bundle).cells == frozenset({
        ("AAA_financial_statements_2024_consolidated", 2, 4, 1),
    })


def test_code_candidate_rejects_wrong_row_from_right_canonical_table():
    bundle = _CodeBundle([_revenue_requirement()])
    wrong_row = _code_candidate(pandas_query=(
        "float(df1.loc[(df1['row'] == 5) & (df1['col'] == 1), "
        "'value'].iloc[0])"
    ))

    with patch("vifinqa.codegen.consensus._run_validated", _replay):
        decision = verify_code_consensus_candidate(wrong_row, bundle)

    assert not decision.accepted
    assert "exact canonical cells" in decision.reason


def test_code_ensemble_requires_same_cells_and_unanimity_without_canonical():
    canonical = _CodeBundle([_revenue_requirement()])
    no_canonical = _CodeBundle()
    verified = ConsensusDecision(True, "ok")
    first = _code_candidate(answer=20.0, votes=4)
    second = _code_candidate(answer=20.0, votes=5)

    choice = choose_code_ensemble_candidate(
        {"answer": 10.0}, (first, verified, canonical),
        (second, verified, canonical))
    uncanonicalized = choose_code_ensemble_candidate(
        {"answer": 10.0}, (first, verified, no_canonical),
        (second, verified, no_canonical))

    assert choice.candidate is second
    assert uncanonicalized.candidate is None


def test_code_ensemble_accepts_two_of_three_verified_runs():
    canonical = _CodeBundle([_revenue_requirement()])
    verified = ConsensusDecision(True, "ok")
    rejected = ConsensusDecision(False, "disagreed")
    first = _code_candidate(answer=20.0, votes=4)
    second = _code_candidate(answer=20.0, votes=5)
    third = _code_candidate(answer=30.0, votes=5)

    choice = choose_code_ensemble_candidates(
        {"answer": 10.0},
        [(first, verified, canonical),
         (second, verified, canonical),
         (third, rejected, canonical)],
    )

    assert choice.candidate is second
    assert "2/3" in choice.reason
