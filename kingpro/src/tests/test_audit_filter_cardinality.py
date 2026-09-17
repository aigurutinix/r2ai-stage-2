from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_filter_cardinality import conjunctive_evaluations  # noqa: E402


QUERY = """
filtered = df1[
    df1.iloc[:, 0].str.contains('MNS Meat Hà Nam', case=False, na=False, regex=False)
    & df1.iloc[:, 1].str.contains('Mua hàng hóa', case=False, na=False, regex=False)
]
result = filtered.iloc[0, 2]
"""


def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Công ty TNHH MNS Meat Hà Nam", "Bán hàng hóa", "1"],
            ["Công ty TNHH MNS Meat Hà Nam", "Mua hàng hóa", "52"],
            ["Công ty TNHH MNS Meat Hà Nam", "Phí hỗ trợ quản lý", "13"],
            ["Công ty khác", "Mua hàng hóa", "2"],
        ]
    )


def test_conjunction_reduces_two_broad_predicates_to_one_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "source.csv"
    source_frame().to_csv(csv_path, index=False)
    groups = conjunctive_evaluations(
        ast.parse(QUERY),
        evidence={"df1": csv_path},
        frames={},
    )
    assert len(groups) == 1
    assert [item["pattern"] for item in groups[0]["items"]] == [
        "MNS Meat Hà Nam",
        "Mua hàng hóa",
    ]
    assert int(groups[0]["mask"].sum()) == 1


def test_conjunction_preserves_real_duplicate_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "source.csv"
    frame = source_frame()
    frame.loc[len(frame)] = ["Công ty TNHH MNS Meat Hà Nam", "Mua hàng hóa", "53"]
    frame.to_csv(csv_path, index=False)
    groups = conjunctive_evaluations(
        ast.parse(QUERY),
        evidence={"df1": csv_path},
        frames={},
    )
    assert len(groups) == 1
    assert int(groups[0]["mask"].sum()) == 2
