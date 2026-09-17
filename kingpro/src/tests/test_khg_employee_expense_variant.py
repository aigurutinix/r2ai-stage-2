from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"
SPEC = importlib.util.spec_from_file_location("candidate_builder_khg_employee", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_default_q646_remains_legacy() -> None:
    assert 646 not in module.FORMULAS


def test_khg_employee_variant_uses_exact_unqualified_row() -> None:
    formula = module.FORMULA_VARIANTS["khg-employee-expense"][646]
    assert formula.expression == "(v0 - v1) / 1e9"
    assert len(formula.operands) == 2

    cells = [
        module.EXPLICIT_CELLS[(op.ticker, op.year, op.metric_key, op.scope)]
        for op in formula.operands
    ]
    assert [cell.raw for cell in cells] == ["22.287.552.828", "11.949.173.962"]
    assert all(
        cell.table_ref == "KHG_financial_statements_2021_consolidated|1021"
        for cell in cells
    )
    assert [(cell.row_idx, cell.col_idx) for cell in cells] == [(2, 1), (2, 2)]
    assert round((cells[0].value - cells[1].value) / 1e9, 2) == 10.34
