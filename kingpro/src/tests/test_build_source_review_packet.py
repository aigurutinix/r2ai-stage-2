from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_source_review_packet.py"
SPEC = importlib.util.spec_from_file_location("build_source_review_packet", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(module)


def test_parse_ids_preserves_order_and_removes_duplicates() -> None:
    assert module.parse_ids("85, 351 85,153") == [85, 351, 153]


def test_query_digest_keeps_semantic_comments_and_final_result() -> None:
    digest = module.query_digest(
        "# LẤY DÒNG: Tổng cộng\n"
        "# CÔNG THỨC: A trừ B\n"
        "value = 3\n"
        "result = round(value, 2)\n"
    )
    assert digest["comments"] == ["# LẤY DÒNG: Tổng cộng", "# CÔNG THỨC: A trừ B"]
    assert digest["result_expressions"] == ["result = round(value, 2)"]
    assert digest["result_expression"] == "result = round(value, 2)"
