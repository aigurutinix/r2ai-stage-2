from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("candidate_builder_debt_variants", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(module, formula) -> float:
    values = {
        f"v{index}": module.source_cell(operand).value
        for index, operand in enumerate(formula.operands)
    }
    return round(
        float(
            eval(
                formula.expression,
                {"__builtins__": {"abs": abs, "max": max, "min": min}},
                values,
            )
        ),
        2,
    )


def test_default_q937_trading_interpretation_remains_reproducible() -> None:
    module = load_builder()

    assert 81 not in module.FORMULAS
    assert evaluate(module, module.FORMULAS[937]) == 3_322_907.75


def test_debt_afs_variant_has_source_backed_q81_and_q937() -> None:
    module = load_builder()
    variant = module.FORMULA_VARIANTS["debt-afs"]

    assert evaluate(module, variant[81]) == 143_010_711.0
    assert evaluate(module, variant[937]) == 105_632_734.75
    assert [module.source_cell(operand).table_ref for operand in variant[937].operands] == [
        "CTG_financial_statements_2017_consolidated|1372",
        "CTG_financial_statements_2018_consolidated|1429",
        "CTG_financial_statements_2019_consolidated|1381",
        "CTG_financial_statements_2020_consolidated|1405",
    ]
