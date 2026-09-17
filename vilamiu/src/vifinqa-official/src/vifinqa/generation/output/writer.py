
from __future__ import annotations

import json
import threading
from pathlib import Path

from vifinqa.generation.schemas import QARecord


class JsonlWriter:
    def __init__(self, out_path: Path) -> None:
        self._out_path = out_path
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_id = self._read_max_id() + 1

    def _read_max_id(self) -> int:
        if not self._out_path.exists():
            return 0
        max_id = 0
        with self._out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                max_id = max(max_id, json.loads(line).get("id", 0))
        return max_id

    def next_id(self) -> int:
        return self._next_id

    def append(self, record: QARecord) -> QARecord:
        with self._lock:
            assigned = record.model_copy(update={"id": self._next_id})
            with self._out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(assigned.model_dump(), ensure_ascii=False) + "\n")
            self._next_id += 1
        return assigned

    def replace(self, records: list[QARecord]) -> list[QARecord]:
        """Atomically replace the output with one complete, freshly numbered batch."""
        with self._lock:
            assigned = [
                record.model_copy(update={"id": index})
                for index, record in enumerate(records, start=1)
            ]
            payload = "".join(
                json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
                for record in assigned
            )
            tmp_path = self._out_path.with_name(f".{self._out_path.name}.tmp")
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(self._out_path)
            self._next_id = len(assigned) + 1
        return assigned
