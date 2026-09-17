import ast
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_data_derived_labels",
    ROOT / "scripts" / "audit_data_derived_labels.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def sliced_nodes(code: str):
    tree = ast.parse(code)
    mapping = MODULE.assignments(tree)
    return MODULE.dependency_slice(mapping["result"], mapping)


class DataDerivedLabelAuditTests(unittest.TestCase):
    def test_terminal_float_wrapper_retains_dataframe_dependency(self):
        nodes = sliced_nodes(
            "years = df['year']\n"
            "result = years.iloc[0]\n"
            "result = float(result)\n"
        )
        names = {
            item.id
            for node in nodes
            for item in ast.walk(node)
            if isinstance(item, ast.Name)
        }
        self.assertIn("df", names)

    def test_multiple_terminal_wrappers_retain_dataframe_dependency(self):
        nodes = sliced_nodes(
            "result = df['year'].iloc[0]\n"
            "result = round(float(result), 0)\n"
            "result = float(result)\n"
        )
        names = {
            item.id
            for node in nodes
            for item in ast.walk(node)
            if isinstance(item, ast.Name)
        }
        self.assertIn("df", names)

    def test_terminal_wrapper_does_not_hide_literal_year(self):
        nodes = sliced_nodes("result = 2023\nresult = float(result)\n")
        years = {
            item.value
            for node in nodes
            for item in ast.walk(node)
            if isinstance(item, ast.Constant)
            and isinstance(item.value, int)
            and 2000 <= item.value <= 2030
        }
        self.assertEqual(years, {2023})


if __name__ == "__main__":
    unittest.main()
