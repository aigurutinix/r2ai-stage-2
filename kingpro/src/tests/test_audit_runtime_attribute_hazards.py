import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_runtime_attribute_hazards",
    ROOT / "scripts" / "audit_runtime_attribute_hazards.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RuntimeAttributeHazardTests(unittest.TestCase):
    def test_scalar_reduction_round_is_hazard(self):
        tree = MODULE.ast.parse("result = frame['x'].mean().round(2)")
        calls = [node for node in MODULE.ast.walk(tree) if isinstance(node, MODULE.ast.Call)]
        self.assertTrue(any(MODULE._scalar_round_hazard(node) for node in calls))

    def test_series_round_is_not_scalar_hazard(self):
        tree = MODULE.ast.parse("rounded = frame['x'].round(2)")
        calls = [node for node in MODULE.ast.walk(tree) if isinstance(node, MODULE.ast.Call)]
        self.assertFalse(any(MODULE._scalar_round_hazard(node) for node in calls))

    def test_builtin_round_is_not_scalar_hazard(self):
        tree = MODULE.ast.parse("result = round(float(frame['x'].mean()), 2)")
        calls = [node for node in MODULE.ast.walk(tree) if isinstance(node, MODULE.ast.Call)]
        self.assertFalse(any(MODULE._scalar_round_hazard(node) for node in calls))

    def test_fail_on_findings_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            submission = Path(temporary)
            (submission / "submission.json").write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "question": "test",
                            "answer": 1.0,
                            "pandas_query": "result = df['x'].mean().round(2)",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_runtime_attribute_hazards.py"),
                    str(submission),
                    "--fail-on-findings",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 1)


if __name__ == "__main__":
    unittest.main()
