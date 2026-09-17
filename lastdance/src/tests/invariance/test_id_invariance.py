"""Test ID invariance for the general pipeline."""

import pytest
from vifinqa.financial_ir import InferenceRequest, FinancialPlanV2

def test_id_invariance():
    """Ensure that InferenceRequest can hold the same question with different IDs."""
    q = "Doanh thu của FPT năm 2023 là bao nhiêu?"
    req1 = InferenceRequest(question=q)
    req2 = InferenceRequest(question=q, request_id=1)
    req3 = InferenceRequest(question=q, request_id=999999)
    
    assert req1.question == req2.question == req3.question

def test_missing_id():
    """Ensure that a request without an ID returns a valid ANSWERED status."""
    # Placeholder for Milestone 3/4 solver
    pass

