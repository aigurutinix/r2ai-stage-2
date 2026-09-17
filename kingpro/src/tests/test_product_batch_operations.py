from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from kingpro.operations.batch import (  # noqa: E402
    BatchRunConfig,
    BatchRunner,
    atomic_write_json,
    read_questions,
)
from kingpro.operations.jobs import BatchJobManager  # noqa: E402


class ProductBatchOperationTests(unittest.TestCase):
    def test_atomic_write_json_replaces_complete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            atomic_write_json(path, {"version": 1})
            atomic_write_json(path, {"version": 2, "items": [1, 2, 3]})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 2, "items": [1, 2, 3]})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_read_questions_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "questions.json"
            path.write_text(
                json.dumps([{"id": 1, "question": "A"}, {"id": 1, "question": "B"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate question id"):
                read_questions(path)

    def test_concurrent_results_remain_in_input_order_and_emit_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "questions.json"
            questions = [
                {"id": 1, "question": "slow"},
                {"id": 2, "question": "fast"},
                {"id": 3, "question": "medium"},
            ]
            input_path.write_text(json.dumps(questions), encoding="utf-8")

            delays = {"slow": 0.04, "fast": 0.005, "medium": 0.02}

            def answer(question: str) -> dict:
                time.sleep(delays[question])
                return {"status": "answered", "trace_id": question, "answer": len(question)}

            runner = BatchRunner(
                answer,
                root=root,
                input_path=input_path,
                config=BatchRunConfig(output_root=root / "runs", max_workers=3, run_id="ordered"),
            )
            summary = runner.run(questions)
            payload = json.loads(Path(summary["output"]).read_text(encoding="utf-8"))
            self.assertEqual([record["id"] for record in payload["results"]], [1, 2, 3])
            self.assertEqual(payload["summary"]["answered"], 3)
            self.assertEqual(len(list((Path(summary["run_dir"]) / "traces").glob("*.json"))), 3)
            self.assertEqual(summary["status"], "success")

    def test_fallback_requires_same_id_and_exact_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "questions.json"
            questions = [{"id": 1, "question": "same"}, {"id": 2, "question": "new private question"}]
            input_path.write_text(json.dumps(questions), encoding="utf-8")

            def fail(_question: str) -> dict:
                raise RuntimeError("boom")

            fallback = {
                json.dumps(1): {"id": 1, "question": "same", "answer": 11},
                json.dumps(2): {"id": 2, "question": "different public question", "answer": 22},
            }
            runner = BatchRunner(
                fail,
                root=root,
                input_path=input_path,
                config=BatchRunConfig(output_root=root / "runs", max_workers=2, run_id="fallback"),
                fallback_records=fallback,
            )
            summary = runner.run(questions)
            payload = json.loads(Path(summary["output"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["results"][0]["status"], "fallback")
            self.assertEqual(payload["results"][1]["status"], "error")
            self.assertEqual(summary["status"], "warning")

    def test_resume_skips_completed_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "questions.json"
            questions = [{"id": 1, "question": "done"}, {"id": 2, "question": "pending"}]
            input_path.write_text(json.dumps(questions), encoding="utf-8")
            first_calls: list[str] = []

            first = BatchRunner(
                lambda question: first_calls.append(question) or {"status": "answered", "trace_id": question},
                root=root,
                input_path=input_path,
                config=BatchRunConfig(output_root=root / "runs", max_workers=1, run_id="resume"),
            )
            initial = first.run(questions)
            run_dir = Path(initial["run_dir"])
            checkpoint = json.loads((run_dir / "running.json").read_text(encoding="utf-8"))
            checkpoint["results"][1] = None
            atomic_write_json(run_dir / "running.json", checkpoint)

            resumed_calls: list[str] = []
            resumed = BatchRunner(
                lambda question: resumed_calls.append(question) or {"status": "answered", "trace_id": question},
                root=root,
                input_path=input_path,
                config=BatchRunConfig(output_root=root / "runs", max_workers=1, resume_run_dir=run_dir),
            )
            summary = resumed.run(questions)
            self.assertEqual(resumed_calls, ["pending"])
            self.assertEqual(summary["summary"]["answered"], 2)

    def test_server_job_streams_progress_and_persists_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def answer(question: str) -> dict:
                time.sleep(0.005)
                return {"status": "answered", "trace_id": question, "answer": len(question)}

            manager = BatchJobManager(
                answer,
                root=root,
                output_root=root / "jobs",
            )
            created = manager.create(
                [{"id": 1, "question": "alpha"}, {"id": 2, "question": "beta"}],
                workers=2,
            )
            job_id = created["job_id"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                snapshot = manager.snapshot(job_id)
                if snapshot and snapshot["status"] in {"success", "warning", "error"}:
                    break
                time.sleep(0.01)
            else:
                self.fail("batch job did not finish")

            cursor = 0
            streamed: list[dict] = []
            terminal = False
            while not terminal:
                events, terminal = manager.wait_events(job_id, cursor, timeout=0.1)
                streamed.extend(events)
                cursor += len(events)
            self.assertEqual(streamed[-1]["event"], "done")
            self.assertEqual(sum(event["event"] == "item" for event in streamed), 2)
            payload = manager.output_payload(job_id)
            self.assertIsNotNone(payload)
            self.assertEqual([record["id"] for record in payload["results"]], [1, 2])
            self.assertEqual(manager.snapshot(job_id)["summary"]["answered"], 2)


if __name__ == "__main__":
    unittest.main()
