from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"
SPEC = importlib.util.spec_from_file_location("candidate_builder_hnd_lease_2025", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_default_q336_remains_reproducible() -> None:
    formula = module.FORMULAS[336]
    assert len(formula.operands) == 3
    assert formula.expression == "(v0 + v1 + v2) / 1e9"


def test_hnd_lease_2025_variant_reads_direct_2025_total() -> None:
    formula = module.FORMULA_VARIANTS["hnd-lease-2025"][336]
    assert len(formula.operands) == 1
    operand = formula.operands[0]
    cell = module.EXPLICIT_CELLS[
        (operand.ticker, operand.year, operand.metric_key, operand.scope)
    ]

    assert cell.raw == "387.656.354.540"
    assert cell.table_ref == "HND_financial_statements_2025|912"
    assert cell.row_idx == 4
    assert cell.col_idx == 1
    assert formula.expression == "v0 / 1e9"
    assert round(cell.value / 1e9, 2) == 387.66
