from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"
SPEC = importlib.util.spec_from_file_location("candidate_builder_stb_gov_bonds", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_default_q169_remains_legacy() -> None:
    assert 169 not in module.FORMULAS


def test_stb_government_bond_variant_aggregates_all_classifications() -> None:
    formula = module.FORMULA_VARIANTS["stb-government-bonds-opening-2017"][169]
    assert formula.expression == "v0 + v1"
    cells = [
        module.EXPLICIT_CELLS[(op.ticker, op.year, op.metric_key, op.scope)]
        for op in formula.operands
    ]

    assert [cell.raw for cell in cells] == ["27.045.792", "991.387"]
    assert all(
        cell.table_ref == "STB_financial_statements_2017_consolidated|1313"
        for cell in cells
    )
    assert [(cell.row_idx, cell.col_idx) for cell in cells] == [(3, 2), (12, 2)]
    assert sum(cell.value for cell in cells) == 28_037_179
