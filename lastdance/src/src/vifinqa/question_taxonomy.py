"""Materialize the manually reviewed question taxonomy.

Every question was read in ID order.  The ranges below are not inferred with
keywords: they record the construction blocks visible in the released test
set.  Keeping the decisions as explicit ranges makes the labeling auditable
and prevents a future wording change from silently changing a label.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReviewedBlock:
    start_id: int
    end_id: int
    solver_family: str
    reasoning_level: int
    expected_fact_shape: str
    core_operations: str
    review_note: str


REVIEWED_BLOCKS = (
    ReviewedBlock(
        1,
        361,
        "single_fact_lookup",
        1,
        "one entity, one period, one reported fact",
        "retrieve; unit_convert",
        "Direct lookup, but retrieval may still be hard when the fact is in a note.",
    ),
    ReviewedBlock(
        362,
        494,
        "conditional_multi_stage",
        5,
        "many entities/periods and many source facts",
        "derive; filter; median_or_rank; select; aggregate_or_compute",
        "Multi-stage screening, ranking, scenario, or chained financial reasoning.",
    ),
    ReviewedBlock(
        495,
        538,
        "selector_then_answer",
        4,
        "metric A selects an entity/period; metric B is returned or computed",
        "retrieve_selector; argmin_or_argmax; retrieve_target; optional_compute",
        "Selector facts and answer facts are usually different table rows.",
    ),
    ReviewedBlock(
        539,
        577,
        "conditional_multi_stage",
        5,
        "many entities/periods and derived source facts",
        "derive; filter; rank; select; aggregate_or_compute",
        "A second block of multi-stage screening questions, including paraphrased forms.",
    ),
    ReviewedBlock(
        578,
        655,
        "temporal_comparison",
        2,
        "one entity, two periods, same metric",
        "retrieve_two_periods; difference_or_growth; unit_convert",
        "Two-period delta or relative growth.",
    ),
    ReviewedBlock(
        656,
        732,
        "single_entity_derived",
        2,
        "one entity/period, usually two or three component facts",
        "retrieve_components; ratio_or_net_value; unit_convert",
        "A directly stated financial formula, ratio, margin, or net value.",
    ),
    ReviewedBlock(
        733,
        812,
        "cross_entity_comparison",
        2,
        "two entities, same period and comparable metric",
        "retrieve_each_entity; difference; unit_convert",
        "Cross-company difference; a few questions compare derived ratios.",
    ),
    ReviewedBlock(
        813,
        1012,
        "aggregation_extreme_count",
        3,
        "one metric across several periods/entities",
        "retrieve_set; sum_mean_argmin_argmax_or_count; unit_convert",
        "Set-level aggregation, extreme selection, or threshold count.",
    ),
)


FAMILY_VI = {
    "single_fact_lookup": "Tra cứu một giá trị",
    "temporal_comparison": "So sánh hai kỳ",
    "single_entity_derived": "Công thức trong một doanh nghiệp",
    "cross_entity_comparison": "So sánh chéo doanh nghiệp",
    "aggregation_extreme_count": "Tổng hợp, cực trị hoặc đếm",
    "selector_then_answer": "Chọn theo A rồi trả lời B",
    "conditional_multi_stage": "Lọc và suy luận nhiều tầng",
}


# These operation groups were assigned after reading the questions, then
# written as explicit ID sets.  They are intentionally not keyword rules.
TEMPORAL_GROWTH_IDS = {
    579, 582, 583, 585, 586, 587, 596, 597, 598, 605, 606, 609, 610, 612,
    614, 615, 617, 620, 626, 629, 631, 632, 633, 635, 637, 638, 639, 640,
    644, 645, 647, 648, 650, 651, 655,
}
TEMPORAL_DIFFERENCE_IDS = set(range(578, 656)) - TEMPORAL_GROWTH_IDS

DERIVED_NET_VALUE_IDS = {
    666, 677, 679, 681, 682, 683, 702, 703, 704, 714, 722, 725, 726, 730,
}
DERIVED_GROWTH_IDS = {689, 720, 727}
DERIVED_RATIO_IDS = (
    set(range(656, 733)) - DERIVED_NET_VALUE_IDS - DERIVED_GROWTH_IDS
)

SET_ARGMAX_KEY_IDS = {
    813, 822, 826, 829, 832, 841, 842, 850, 852, 860, 866, 874, 876, 877,
    878, 879, 883, 884, 889, 890, 897, 900, 904, 906, 907, 910, 921, 925,
    928, 929, 933, 936, 946, 948, 950, 953, 959, 960, 971, 974, 978, 981,
    982, 985, 986, 989, 995, 997, 999, 1000, 1008,
}
SET_MAX_VALUE_IDS = {
    815, 825, 828, 834, 835, 836, 838, 845, 847, 849, 857, 859, 864, 868,
    871, 873, 885, 886, 895, 899, 901, 902, 903, 908, 909, 911, 912, 914,
    916, 930, 941, 944, 951, 961, 967, 968, 969, 972, 980, 984, 988, 994,
    996, 998, 1004, 1011,
}
SET_SUM_IDS = {
    817, 833, 843, 848, 851, 858, 862, 869, 887, 893, 898, 905, 915, 920,
    922, 923, 924, 934, 938, 958, 964, 987, 991, 1003,
}
SET_MEAN_IDS = {
    814, 816, 818, 819, 820, 821, 824, 827, 830, 831, 839, 840, 844, 846,
    853, 854, 856, 861, 865, 867, 872, 875, 880, 881, 882, 888, 891, 892,
    894, 896, 917, 918, 919, 926, 927, 931, 935, 937, 939, 940, 942, 943,
    945, 947, 952, 954, 955, 956, 957, 962, 970, 975, 976, 977, 992, 993,
    1001, 1006, 1007, 1009, 1012,
}
SET_COUNT_IDS = {
    823, 837, 855, 863, 870, 913, 932, 949, 963, 965, 966, 973, 983, 990,
    1002, 1005, 1010,
}
SET_AGGREGATE_RATIO_IDS = {979}


def operation_for(question_id: int, solver_family: str) -> str:
    explicit_groups = (
        ("temporal_growth", TEMPORAL_GROWTH_IDS),
        ("temporal_difference", TEMPORAL_DIFFERENCE_IDS),
        ("derived_ratio", DERIVED_RATIO_IDS),
        ("derived_net_value", DERIVED_NET_VALUE_IDS),
        ("derived_growth", DERIVED_GROWTH_IDS),
        ("set_argmax_key", SET_ARGMAX_KEY_IDS),
        ("set_max_value", SET_MAX_VALUE_IDS),
        ("set_sum", SET_SUM_IDS),
        ("set_mean", SET_MEAN_IDS),
        ("set_count", SET_COUNT_IDS),
        ("set_aggregate_ratio", SET_AGGREGATE_RATIO_IDS),
    )
    matches = [name for name, ids in explicit_groups if question_id in ids]
    if len(matches) > 1:
        raise ValueError(f"Question {question_id} has overlapping operation labels")
    if matches:
        return matches[0]
    defaults = {
        "single_fact_lookup": "retrieve",
        "conditional_multi_stage": "screen_rank_compute",
        "selector_then_answer": "select_then_compute",
        "cross_entity_comparison": "cross_entity_difference",
    }
    try:
        return defaults[solver_family]
    except KeyError as error:
        raise ValueError(f"Question {question_id} has no operation label") from error


def block_for(question_id: int) -> ReviewedBlock:
    matches = [
        block
        for block in REVIEWED_BLOCKS
        if block.start_id <= question_id <= block.end_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Question {question_id} belongs to {len(matches)} blocks")
    return matches[0]


def load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"id", "question"}:
                raise ValueError(f"Unexpected fields at {path}:{line_number}")
            rows.append(row)
    ids = [row["id"] for row in rows]
    if ids != list(range(1, len(rows) + 1)):
        raise ValueError("Question IDs must be unique, ordered, and contiguous from 1")
    return rows


def annotate(rows: Iterable[dict]) -> list[dict]:
    output = []
    for row in rows:
        block = block_for(int(row["id"]))
        output.append(
            {
                "id": row["id"],
                "question": row["question"],
                "solver_family": block.solver_family,
                "solver_family_vi": FAMILY_VI[block.solver_family],
                "reasoning_level": block.reasoning_level,
                "operation_subtype": operation_for(row["id"], block.solver_family),
                "expected_fact_shape": block.expected_fact_shape,
                "core_operations": block.core_operations,
                "manual_review_status": "read_in_full",
                "review_note": block.review_note,
            }
        )
    return output


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_coverage(question_count: int = 1012) -> None:
    covered = []
    for block in REVIEWED_BLOCKS:
        covered.extend(range(block.start_id, block.end_id + 1))
    expected = list(range(1, question_count + 1))
    if sorted(covered) != expected or len(covered) != len(set(covered)):
        raise ValueError("Reviewed blocks do not cover IDs 1..1012 exactly once")
    for question_id in expected:
        operation_for(question_id, block_for(question_id).solver_family)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("ViFinQA/questions/questions.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args(argv)

    validate_coverage()
    annotations = annotate(load_questions(args.questions))
    write_jsonl(args.output_dir / "question_annotations.jsonl", annotations)
    write_csv(args.output_dir / "question_annotations.csv", annotations)


if __name__ == "__main__":
    main()
