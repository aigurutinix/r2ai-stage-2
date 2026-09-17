from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_cross_question_consistency.py"
SPEC = importlib.util.spec_from_file_location("audit_cross_question_consistency", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_identical_finance_terms_score_one() -> None:
    score, shared = module.f1_terms(
        {"loi", "nhuan", "sau", "thue"},
        {"loi", "nhuan", "sau", "thue"},
    )
    assert score == 1.0
    assert len(shared) == 4


def test_extra_metric_term_reduces_similarity() -> None:
    score, _ = module.f1_terms(
        {"loi", "nhuan", "sau", "thue"},
        {"loi", "nhuan", "sau", "thue", "chua", "phan", "phoi"},
    )
    assert score < 0.82


def test_rounding_noise_is_not_a_disagreement() -> None:
    assert not module.materially_different(100.0, 100.004)
    assert module.materially_different(100.0, 100.02)


def test_currency_cluster_can_ignore_scale_only_when_requested() -> None:
    base = {
        "facets": {"tickers": ["CEO"], "years": ["2024"], "scope": "consolidated"},
        "unit": ("tỷ đồng", 1_000_000_000),
        "intent": frozenset(),
        "time_basis": "period",
    }
    scaled = {**base, "unit": ("nghìn tỷ đồng", 1_000_000_000_000)}
    assert module.cluster_key(base) != module.cluster_key(scaled)
    assert module.cluster_key(base, cross_currency=True) == module.cluster_key(
        scaled, cross_currency=True
    )
