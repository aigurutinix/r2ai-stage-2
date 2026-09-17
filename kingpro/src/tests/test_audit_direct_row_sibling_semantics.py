from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.audit_direct_row_sibling_semantics import direct_display_only, score


def test_direct_display_only_rejects_multi_operand_arithmetic() -> None:
    assert direct_display_only("result = round(v0 / 1e9, 2)")
    assert direct_display_only("result = round(abs(v0), 2)")
    assert not direct_display_only("result = round(v0 + v1, 2)")


def test_semantic_bigram_gain_rewards_exact_metric_phrase() -> None:
    question = "Tổng nợ phải trả cuối năm là bao nhiêu?"
    assert score(question, "TỔNG NỢ PHẢI TRẢ")["score"] > score(
        question, "Nợ ngắn hạn"
    )["score"]
