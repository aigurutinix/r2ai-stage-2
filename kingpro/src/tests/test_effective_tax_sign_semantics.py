from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "candidate_builder_effective_tax_sign_test", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_q993_preserves_deferred_expense_sign_only_where_required() -> None:
    module = load_builder()
    formula = module.FORMULAS[993]
    query, cells = module.query_for(formula)
    assignments = [line for line in query.splitlines() if line.startswith("v")]

    # VGC discloses deferred-tax income: the formula subtracts its magnitude.
    assert assignments[4].startswith("v4 = abs(")
    assert cells[4].raw.startswith("(") and cells[4].raw.endswith(")")
    assert "income" in cells[4].label.lower() or "thu nhập" in cells[4].label.lower()
    assert not formula.operands[4].preserve_sign

    # SJG discloses deferred-tax expense: keep its negative statement sign and
    # add it to the current-tax charge.  Losing this sign caused the v195 bug.
    assert assignments[7].startswith("v7 = _btc_number(")
    assert "abs(" not in assignments[7]
    assert cells[7].raw.startswith("(") and cells[7].raw.endswith(")")
    assert formula.operands[7].preserve_sign

    df = pd.DataFrame(
        {
            "raw": [cell.raw for cell in cells],
            "typed_factor": [module.typed_factor(cell.raw) for cell in cells],
            "scale": [cell.scale for cell in cells],
        }
    )
    runtime: dict[str, object] = {"dfs": {"source": df}}
    exec(query, runtime)
    assert runtime["result"] == 10.65


def test_default_cost_operand_still_uses_magnitude() -> None:
    module = load_builder()
    query, _ = module.query_for(module.FORMULAS[580])
    assignments = [line for line in query.splitlines() if line.startswith("v")]

    assert assignments[0].startswith("v0 = abs(")
    assert assignments[1].startswith("v1 = abs(")
