
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vifinqa.generation.schemas import Difficulty, Operation

ScenarioName = Literal[
    "same_doc_two_inputs",
    "same_company_two_periods",
    "two_companies_same_period",
    "same_doc_multi_inputs",
    "same_company_time_series",
    "peer_group_same_period",
    "peer_group_two_periods",
    "same_doc_multi_role_formula",
    "peer_group_longitudinal",
]


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    name: ScenarioName
    difficulty: Difficulty
    retrieval_kind: Literal["same_doc", "same_company", "peer_group"]
    min_entities: int
    min_periods: int
    min_observations: int
    allowed_operations: frozenset[Operation]
    require_consecutive_periods: bool = False
    normalized_metric_only: bool = False


SCENARIOS: dict[ScenarioName, ScenarioSpec] = {
    "same_doc_two_inputs": ScenarioSpec(
        name="same_doc_two_inputs",
        difficulty="medium",
        retrieval_kind="same_doc",
        min_entities=1,
        min_periods=1,
        min_observations=2,
        allowed_operations=frozenset({"difference", "ratio", "share"}),
    ),
    "same_company_two_periods": ScenarioSpec(
        name="same_company_two_periods",
        difficulty="medium",
        retrieval_kind="same_company",
        min_entities=1,
        min_periods=2,
        min_observations=2,
        allowed_operations=frozenset({"difference", "growth"}),
    ),
    "two_companies_same_period": ScenarioSpec(
        name="two_companies_same_period",
        difficulty="medium",
        retrieval_kind="peer_group",
        min_entities=2,
        min_periods=1,
        min_observations=2,
        allowed_operations=frozenset({"difference"}),
        normalized_metric_only=True,
    ),
    "same_doc_multi_inputs": ScenarioSpec(
        name="same_doc_multi_inputs",
        difficulty="intermediate",
        retrieval_kind="same_doc",
        min_entities=1,
        min_periods=1,
        min_observations=3,
        allowed_operations=frozenset({"sum", "average", "minimum", "maximum", "count"}),
    ),
    "same_company_time_series": ScenarioSpec(
        name="same_company_time_series",
        difficulty="intermediate",
        retrieval_kind="same_company",
        min_entities=1,
        min_periods=3,
        min_observations=3,
        allowed_operations=frozenset(
            {"growth", "average", "minimum", "maximum", "argmin", "argmax", "count"}
        ),
        require_consecutive_periods=True,
    ),
    "peer_group_same_period": ScenarioSpec(
        name="peer_group_same_period",
        difficulty="intermediate",
        retrieval_kind="peer_group",
        min_entities=3,
        min_periods=1,
        min_observations=3,
        allowed_operations=frozenset({"average", "minimum", "maximum", "count"}),
        normalized_metric_only=True,
    ),
    "peer_group_two_periods": ScenarioSpec(
        name="peer_group_two_periods",
        difficulty="intermediate",
        retrieval_kind="peer_group",
        min_entities=3,
        min_periods=2,
        min_observations=6,
        allowed_operations=frozenset({"growth", "average", "minimum", "maximum", "count"}),
        normalized_metric_only=True,
    ),
    # Keep period handling explicit and deterministic.
    "same_doc_multi_role_formula": ScenarioSpec(
        name="same_doc_multi_role_formula",
        difficulty="intermediate",
        retrieval_kind="same_doc",
        min_entities=1,
        min_periods=1,
        min_observations=3,
        allowed_operations=frozenset({"ratio"}),
    ),
    "peer_group_longitudinal": ScenarioSpec(
        name="peer_group_longitudinal",
        difficulty="intermediate",
        retrieval_kind="peer_group",
        min_entities=3,
        min_periods=3,
        min_observations=9,
        allowed_operations=frozenset({"growth"}),
        require_consecutive_periods=True,
    ),
}


def get_scenario(name: ScenarioName) -> ScenarioSpec:
    return SCENARIOS[name]


_TIME_SERIES_ADDENDUM = """
# RÀNG BUỘC RIÊNG CHO SAME-COMPANY-TIME-SERIES
- Chỉ dùng đúng các năm trong target_periods đã cho — không được thay target window sau khi đã mapping.
- Mỗi năm trong target_periods chỉ đóng góp ĐÚNG MỘT observation, lấy từ báo cáo của chính năm đó
  (không lấy cả cột năm nay và cột năm trước trong cùng một bảng rồi tính thành hai năm khác nhau).
- Số liệu dạng "stock" (số dư cuối kỳ) dùng đúng mốc cuối năm/ngày báo cáo của năm đó; số liệu dạng
  "flow" (phát sinh trong kỳ) dùng số phát sinh trong năm đó — không cộng dồn "stock" qua nhiều năm,
  không dùng average stock level nếu không có financial_rationale giải thích rõ.
- Với growth/CAGR phải xác định rõ kỳ đầu và kỳ cuối trong đúng target_periods đã cho.
- Các kỳ bắt buộc liên tiếp — không hỏi dạng mô tả diễn biến ("thay đổi thế nào qua các năm").
- Chính sách v1: mỗi target year lấy số current-year trong báo cáo của chính năm đó, KHÔNG
  canonicalize số liệu bị restated ở báo cáo năm sau.
"""

_PEER_GROUP_TWO_PERIODS_ADDENDUM = """
# RÀNG BUỘC RIÊNG CHO PEER-GROUP-TWO-PERIODS
- Mỗi công ty được chọn phải có đủ số liệu ở CẢ HAI kỳ đã retrieval — không dùng công ty chỉ có 1 kỳ.
- Nếu concept là growth/tăng trưởng: tính growth riêng cho TỪNG công ty giữa hai kỳ của chính công ty
  đó trước, rồi mới so sánh/tổng hợp growth GIỮA các công ty — không lấy tiền công ty A ở kỳ 1 chia
  cho tiền công ty B ở kỳ 2 hay bất kỳ phép trộn chéo công ty/kỳ nào khác.
"""

_ADDENDA: dict[ScenarioName, str] = {
    "same_company_time_series": _TIME_SERIES_ADDENDUM,
    "peer_group_two_periods": _PEER_GROUP_TWO_PERIODS_ADDENDUM,
}


def scenario_prompt_rules(spec: ScenarioSpec) -> str:
    return _ADDENDA.get(spec.name, "")
