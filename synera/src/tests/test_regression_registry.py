from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from road2ai_vifinqa.easy_solver import EASY_AUDITED_OVERRIDES
from road2ai_vifinqa.panel import _panel_value_multiplier
from road2ai_vifinqa.solve import _artifact_signature
from road2ai_vifinqa.template_solver import _AUDITED_OVERRIDES


class RegressionRegistryTest(unittest.TestCase):
    def test_source_audited_recipes_are_not_silently_dropped(self) -> None:
        self.assertGreaterEqual(len(EASY_AUDITED_OVERRIDES), 124)
        self.assertGreaterEqual(len(_AUDITED_OVERRIDES), 236)
        self.assertTrue({16, 35, 89, 95, 111, 115, 135, 166, 182, 194, 221}.issubset(EASY_AUDITED_OVERRIDES))
        self.assertTrue(
            {
                582,
                587,
                611,
                616,
                632,
                635,
                639,
                641,
                676,
                701,
                755,
                821,
                827,
                888,
                923,
                929,
            }.issubset(_AUDITED_OVERRIDES)
        )
        self.assertTrue(_AUDITED_OVERRIDES[676].absolute)

    def test_legacy_unit_repair_is_not_applied_twice_to_panel_v2(self) -> None:
        legacy = {"doc_id": "MSR_financial_statements_2022_consolidated", "scale": 1.0}
        semantic = {"doc_id": "MSR_financial_statements_2022_consolidated", "scale": 1000.0}
        self.assertEqual(_panel_value_multiplier("MSR", 2022, legacy), 1000.0)
        self.assertEqual(_panel_value_multiplier("MSR", 2022, semantic), 1.0)

    def test_artifact_signature_invalidates_stale_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "index"
            second = Path(directory) / "panel"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            before = _artifact_signature(first, second)
            second.write_bytes(b"changed")
            after = _artifact_signature(first, second)
            self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
