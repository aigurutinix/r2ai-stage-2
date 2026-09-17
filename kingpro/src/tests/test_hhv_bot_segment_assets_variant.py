from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_standard_candidate.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("hhv_bot_segment_builder_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_q1007_uses_activity_segment_bot_cells() -> None:
    module = load_builder()
    formula = module.FORMULAS[1007]
    query, cells = module.query_for(formula)

    assert cells[0].raw == "32.355.512.700.711"
    assert cells[0].table_ref.endswith("|2563")
    assert cells[0].col_idx == 1
    assert "bot" in cells[0].label.casefold()
    assert cells[1].raw == "33.657.835.517.377"
    assert cells[1].table_ref.endswith("|2450")
    assert cells[1].col_idx == 1
    assert "bot" in cells[1].label.casefold()

    df = pd.DataFrame(
        {
            "raw": [cell.raw for cell in cells],
            "typed_factor": [module.typed_factor(cell.raw) for cell in cells],
            "scale": [cell.scale for cell in cells],
        }
    )
    runtime: dict[str, object] = {"dfs": {"source": df}}
    exec(query, runtime)
    assert runtime["result"] == 92.13
