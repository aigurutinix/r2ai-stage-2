"""Verify semantic FinancialPlanV2 for safety and completeness (general mode)."""

from typing import Any
from .financial_ir import FinancialPlanV2
from .grounded_plans import GroundedBinder

def verify_plan(plan: FinancialPlanV2, database_path: str) -> dict[str, Any]:
    """Verify a FinancialPlanV2 without using a benchmark locked baseline."""
    
    audit = {
        "status": "VALID",
        "failures": []
    }
    
    # 1. Schema hợp lệ is handled by FinancialPlanV2 parsing
    
    # Check 6: DAG hợp lệ
    if not plan.nodes:
        audit["failures"].append("empty_dag")
        audit["status"] = "INVALID"
        
    # Check 12: Confidence đạt ngưỡng
    if plan.confidence < 0.5:
        audit["failures"].append("low_confidence")
        audit["status"] = "INVALID"
        
    # Check grounding
    try:
        with GroundedBinder(database_path) as binder:
            binder.bind_plan(plan)
    except Exception as e:
        audit["failures"].append(f"grounding_failed: {e}")
        audit["status"] = "INVALID"

    return audit
