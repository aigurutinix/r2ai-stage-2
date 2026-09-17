import json
import subprocess
import sys


def test_evidence_trace_emits_submit_set_and_typed_slots(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({
        "id": 1,
        "question": "Tốc độ tăng trưởng doanh thu năm 2024?",
        "selected_docs": ["R"],
        "candidates": [
            {"table_id": "R|1", "text": "Doanh thu 2023 2024"},
            {"table_id": "R|2", "text": "Lợi nhuận 2023 2024"},
        ],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    labels = tmp_path / "line_items.json"
    labels.write_text(json.dumps(["doanh thu", "loi nhuan"]), encoding="utf-8")
    ranking = tmp_path / "ranking.json"
    trace = tmp_path / "trace.jsonl"

    subprocess.run([
        sys.executable, "scripts/build_evidence_trace.py",
        "--pairs", str(pairs), "--output", str(ranking), "--trace", str(trace),
        "--line-items", str(labels), "--read-depth", "2",
    ], check=True)

    output = json.loads(ranking.read_text(encoding="utf-8"))
    record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert output["1"] == ["R|1"]
    assert record["slot_count"] == 2
    assert record["covered_slots"] == 2
    assert {slot["role"] for slot in record["typed_slots"]} == {"prior", "current"}
