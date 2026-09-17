from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from road2ai_vifinqa.submission import canonical_table_ref


class SubmissionTableReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / "ABC.txt"
        self.source.write_bytes(b"header\n<table><tr><td>1</td></tr></table>\nnote\n<table>\n<tr><td>2</td></tr>\n</table>\n")
        self.index = root / "index.sqlite3"
        with closing(sqlite3.connect(self.index)) as connection:
            connection.execute("CREATE TABLE documents (doc_id TEXT, source_path TEXT, table_count INT)")
            connection.execute("INSERT INTO documents VALUES (?,?,?)", ("ABC", str(self.source), 2))
            connection.commit()

    def test_internal_ordinal_resolves_to_physical_line(self):
        self.assertEqual(canonical_table_ref("ABC|table_1", index_path=self.index), "ABC|2")
        self.assertEqual(canonical_table_ref("ABC|table_2", index_path=self.index), "ABC|4")

    def test_canonical_line_is_not_converted_twice(self):
        self.assertEqual(canonical_table_ref("ABC|4", index_path=self.index), "ABC|4")

    def test_invalid_ordinal_fails_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            canonical_table_ref("ABC|table_3", index_path=self.index)

    def test_source_change_invalidates_cache(self):
        self.assertEqual(canonical_table_ref("ABC|table_1", index_path=self.index), "ABC|2")
        self.source.write_bytes(b"extra\n" + self.source.read_bytes())
        self.assertEqual(canonical_table_ref("ABC|table_1", index_path=self.index), "ABC|3")


if __name__ == "__main__":
    unittest.main()
