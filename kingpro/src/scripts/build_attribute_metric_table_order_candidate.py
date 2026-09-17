"""Build v194 after enabling Pandas attribute-style metric selection.

This is the same conservative exact-metric builder as v193, but starts from
the fully gated v193 directory.  The diagnostic recognises both
``frame["metric"]`` and ``frame.metric``; all other eligibility and invariants
remain unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_exact_metric_table_order_candidate as builder  # noqa: E402


builder.SOURCE = ROOT / "sub_top123_candidate_v193_exact_metric_table_order"
builder.OUTPUT = ROOT / "sub_top123_candidate_v194_attribute_metric_table_order"


if __name__ == "__main__":
    os.chdir(ROOT)
    builder.build()
