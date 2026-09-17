from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("candidate_builder_employee_direction", MODULE_PATH)
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


def test_employee_direction_variant_preserves_ordered_subtraction() -> None:
    module = load_builder()
    formula = module.FORMULA_VARIANTS["employee-direction"][797]

    assert formula.expression == "(v0 - v1) / 1e9"
    assert evaluate(module, formula) == -1_706.20
    assert [module.source_cell(operand).table_ref for operand in formula.operands] == [
        "ACV_financial_statements_2019_consolidated|1289",
        "VJC_financial_statements_2019_consolidated|1298",
    ]


def test_q800_benchmark_analogue_is_also_directional() -> None:
    module = load_builder()
    formula = module.FORMULAS[800]

    assert formula.expression == "v0 - v1"
    assert evaluate(module, formula) == 117_336_113.0

