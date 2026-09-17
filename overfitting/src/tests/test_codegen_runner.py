from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vifinqa.codegen import generate
from vifinqa.codegen.llm_client import HfBatchClient


ROOT = Path(__file__).resolve().parents[1]


def _load_kaggle_runner():
    path = ROOT / "kaggle" / "kaggle_codegen.py"
    spec = importlib.util.spec_from_file_location("kaggle_codegen_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_payload_builder():
    path = ROOT / "scripts" / "04_make_kaggle_payload.py"
    spec = importlib.util.spec_from_file_location("payload_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeStore:
    def __init__(self, *_args, **_kwargs):
        pass


class _FakeBundle:
    def __init__(self, rec, _store, _k, run_signature=""):
        self.id = rec["id"]
        self.question = rec["question"]
        self.run_signature = run_signature
        self.tables = [{"var": "df1", "report_id": "AAA_2024_consolidated",
                        "table_pos": 0}]
        self.dfs = {"df1": object()}

    def prompt_messages(self):
        return [{"role": "user", "content": self.question}]

    def select_messages(self, _encoder=None):
        return self.prompt_messages()

    def used_vars(self, _code):
        return [self.tables[0]]


class _Client:
    name = "fake"

    def __init__(self, fail_on_call=0):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def chat_batch(self, conversations, **_kwargs):
        self.calls += 1
        if self.fail_on_call and self.calls == self.fail_on_call:
            raise RuntimeError("simulated backend failure")
        return [["float(df1)"] for _ in conversations]


def _exec_ok(code, _dfs):
    return {"status": "ok", "value": 1.0, "error": None}


class CodegenCheckpointTests(unittest.TestCase):
    def test_single_requirement_reserves_exact_shortlist_row(self):
        bundle = generate.QuestionBundle.__new__(generate.QuestionBundle)
        bundle.route = {
            "question": "Doanh thu thuan AAA nam 2024?",
            "evidence_requirements": [{
                "ticker": "AAA", "year": 2024,
                "metric_key": "net_revenue",
                "metric_label": "doanh thu thuan",
            }],
        }
        bundle.tables = [{
            "var": "df1", "report_id": "AAA_financial_statements_2024_consolidated",
            "report_year": 2024,
        }]
        exact = SimpleNamespace(var="df1", row=4, col=1)
        generic = [SimpleNamespace(var="df1", row=5, col=1)]

        with patch.object(generate, "build_shortlist", return_value=[exact]), \
                patch.object(generate, "candidate_matches_requirement",
                             return_value=True):
            rows = bundle._requirement_shortlist(generic)

        self.assertIs(rows[0], exact)
        self.assertIs(rows[1], generic[0])

    def test_code_prompt_exposes_exact_canonical_and_derived_contract(self):
        bundle = generate.QuestionBundle.__new__(generate.QuestionBundle)
        bundle.route = {
            "metric_keys": ["net_financial_result"],
            "evidence_requirements": [
                {"ticker": "AAA", "year": 2024,
                 "metric_key": "financial_income",
                 "metric_label": "doanh thu hoat dong tai chinh"},
                {"ticker": "AAA", "year": 2024,
                 "metric_key": "financial_expense",
                 "metric_label": "chi phi tai chinh"},
            ],
        }
        bundle.evidence = {"covered": 2, "required": 2, "complete": True}

        block = bundle._evidence_plan_block()

        self.assertIn("CANONICAL CELL CONTRACT", block)
        self.assertIn("financial_income", block)
        self.assertIn("financial_expense", block)
        self.assertIn("financial_income - abs(financial_expense)", block)

    def _recs(self, path: Path, n=5):
        with path.open("w", encoding="utf-8") as f:
            for i in range(1, n + 1):
                f.write(json.dumps({"id": i, "question": f"q{i}"}) + "\n")

    def _run(self, retrieval, output, client, signature="sig-a", **kwargs):
        with patch.object(generate, "Store", _FakeStore), \
                patch.object(generate, "QuestionBundle", _FakeBundle), \
                patch.object(generate, "run_code", _exec_ok):
            generate.run_codegen(
                retrieval, Path("unused"), output, client,
                checkpoint_every=2, debug_rounds=0,
                use_rule_fallback=False, run_signature=signature, **kwargs,
            )

    def test_resume_requires_matching_signature(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval)
            first = _Client()
            self._run(retrieval, output, first)
            self.assertEqual(first.calls, 3)

            same = _Client(fail_on_call=1)
            self._run(retrieval, output, same)
            self.assertEqual(same.calls, 0)

            changed = _Client()
            self._run(retrieval, output, changed, signature="sig-b")
            self.assertEqual(changed.calls, 3)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual({r["run_signature"] for r in rows}, {"sig-b"})

    def test_checkpoint_survives_later_chunk_failure(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval)
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self._run(retrieval, output, _Client(fail_on_call=2))
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(sum(r["source"] == "llm" for r in rows), 2)
            self.assertEqual(sum(r["source"] == "none" for r in rows), 3)
            failed = {row["id"]: row for row in rows}
            self.assertIn("simulated backend failure", failed[3]["detail"])
            self.assertEqual(
                failed[3]["llm_diagnostics"]["statuses"], {"backend_error": 4},
            )

    def test_expired_budget_keeps_complete_rule_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval)
            client = _Client(fail_on_call=1)
            self._run(retrieval, output, client, time_budget_s=1e-12)
            self.assertEqual(client.calls, 0)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r["source"] == "none" for r in rows))

    def test_llm_ids_limit_generation_without_dropping_rows(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval)
            client = _Client()

            self._run(retrieval, output, client, llm_ids={2, 4})

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            llm_ids = {row["id"] for row in rows if row["source"] == "llm"}
            self.assertEqual(llm_ids, {2, 4})
            self.assertEqual(len(rows), 5)

    def test_failed_llm_attempt_records_actionable_diagnostics(self):
        class InvalidClient:
            name = "fake"

            def chat_batch(self, conversations, **_kwargs):
                return [["not executable", "also invalid"] for _ in conversations]

        failed = {"status": "error", "value": None, "error": "SyntaxError: invalid"}
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval, n=1)
            with patch.object(generate, "Store", _FakeStore), \
                    patch.object(generate, "QuestionBundle", _FakeBundle), \
                    patch.object(generate, "run_code", return_value=failed):
                generate.run_codegen(
                    retrieval, Path("unused"), output, InvalidClient(),
                    checkpoint_every=1, debug_rounds=0,
                    use_rule_fallback=False, n_samples=2,
                )

            row = json.loads(output.read_text().strip())
            self.assertEqual(row["source"], "none")
            self.assertEqual(row["llm_diagnostics"]["returned"], 2)
            self.assertEqual(row["llm_diagnostics"]["statuses"], {"error": 2})
            self.assertIn("SyntaxError", row["detail"])

    def test_selection_consensus_votes_reach_arbitration(self):
        class ThreeSampleClient:
            name = "fake"

            def chat_batch(self, conversations, **_kwargs):
                return [["a", "b", "c"] for _ in conversations]

        def consensus(bundle, samples, _encoder):
            self.assertEqual(len(samples), 3)
            return {
                "id": bundle.id, "question": bundle.question,
                "answer": 1.0, "pandas_query": "float(df1)",
                "used_vars": bundle.tables, "status": "ok",
                "source": "llm_select", "votes": 2, "n_ok": 3,
                "detail": "consensus=2/3", "detail_conf": 90.0,
                "semantic": {}, "run_signature": bundle.run_signature,
            }

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            retrieval, output = td / "retrieval.jsonl", td / "out.jsonl"
            self._recs(retrieval, n=1)
            with patch.object(generate, "_selection_result", consensus):
                self._run(
                    retrieval, output, ThreeSampleClient(),
                    llm_mode="select", n_samples=3,
                )
            row = json.loads(output.read_text().strip())
            self.assertEqual(row["votes"], 2)
            self.assertEqual(row["arbitration"]["llm_conf"], 76.7)


class SelectionConsensusTests(unittest.TestCase):
    def _bundle(self):
        candidates = [
            SimpleNamespace(
                var=f"df{i}", report_id="AAA_2024_consolidated",
                table_pos=i - 1, row=0, label=f"Metric {i}", code="",
                col=0, col_name="2024", value=float(i * 10),
                unit_scale=1.0, score=92.0, lexical=92.0, semantic=0.0,
            )
            for i in range(1, 4)
        ]

        class Bundle:
            id = 1
            question = "lookup"
            route = {"output_type": "number", "unit_scale": 1.0}
            run_signature = "selection-consensus"
            dfs = {}

            def shortlist(self, _encoder, top_n=12):
                return candidates[:top_n]

            def used_vars(self, _query):
                return []

        return Bundle()

    @staticmethod
    def _validated(_bundle, query):
        for i in range(1, 4):
            if f"df{i}" in query:
                return {"status": "ok", "value": float(i * 10),
                        "semantic": {"ok": True}}
        return {"status": "failed", "value": 0.0}

    def test_selection_requires_absolute_majority(self):
        samples = [
            '{"op":"lookup","operands":[1]}',
            '{"op":"lookup","operands":[2]}',
            '{"op":"lookup","operands":[3]}',
        ]
        with patch.object(generate, "_run_validated", self._validated):
            result = generate._selection_result(self._bundle(), samples, None)
        self.assertIsNone(result)

    def test_selection_records_two_of_three_consensus(self):
        samples = [
            '{"op":"lookup","operands":[1]}',
            '{"op":"lookup","operands":[1],"note":"same pick"}',
            '{"op":"lookup","operands":[2]}',
        ]
        with patch.object(generate, "_run_validated", self._validated):
            result = generate._selection_result(self._bundle(), samples, None)
        self.assertIsNotNone(result)
        self.assertEqual(result["answer"], 10.0)
        self.assertEqual(result["votes"], 2)
        self.assertEqual(result["n_ok"], 3)
        self.assertIn("consensus=2/3", result["detail"])

    def test_invalid_sample_does_not_count_as_agreement(self):
        samples = [
            '{"op":"lookup","operands":[2]}',
            "not json",
            '{"op":"lookup","operands":[2]}',
        ]
        with patch.object(generate, "_run_validated", self._validated):
            result = generate._selection_result(self._bundle(), samples, None)
        self.assertIsNotNone(result)
        self.assertEqual(result["answer"], 20.0)
        self.assertEqual(result["votes"], 2)
        self.assertEqual(result["n_ok"], 2)


class HfOomRetryTests(unittest.TestCase):
    def test_hf_tokenizer_preserves_prompt_tail_when_truncating(self):
        class FakeTokenizer:
            pad_token_id = None
            eos_token = "<eos>"
            pad_token = None
            padding_side = "right"
            truncation_side = "right"

        tokenizer = FakeTokenizer()

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return tokenizer

        class FakeParameter:
            device = "cuda:0"

        class FakeModel:
            def eval(self):
                return self

            def parameters(self):
                return iter([FakeParameter()])

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return FakeModel()

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_transformers.AutoModelForCausalLM = FakeAutoModel

        with patch.dict(sys.modules, {
            "torch": fake_torch,
            "transformers": fake_transformers,
        }):
            HfBatchClient("fake/model")

        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(tokenizer.truncation_side, "left")

    def test_cuda_oom_halves_batch_and_retries_without_skipping(self):
        class FakeTensor:
            def __init__(self, batch, tokens):
                self.batch = batch
                self.shape = (batch, tokens)

        class FakeEncoding(dict):
            def __init__(self, batch, tokens):
                super().__init__(input_ids=FakeTensor(batch, tokens))

            def to(self, _device):
                return self

        class FakeOutput:
            def __init__(self, size):
                self.size = size

            def __getitem__(self, _key):
                return [None] * self.size

        class FakeTokenizer:
            pad_token_id = 0

            def apply_chat_template(self, conversation, **_kwargs):
                return conversation[0]["content"]

            def __call__(self, texts, **_kwargs):
                return FakeEncoding(len(texts), _kwargs["max_length"])

            def batch_decode(self, seqs, **_kwargs):
                return [f"code-{i}" for i in range(len(seqs))]

        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                batch = kwargs["input_ids"].batch
                self.calls.append(batch)
                if batch > 2:
                    raise RuntimeError("CUDA out of memory")
                return FakeOutput(batch * kwargs["num_return_sequences"])

        class FakeCuda:
            def __init__(self):
                self.clears = 0

            def is_available(self):
                return True

            def empty_cache(self):
                self.clears += 1

        @contextmanager
        def inference_mode():
            yield

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = FakeCuda()
        fake_torch.inference_mode = inference_mode
        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda **_kwargs: SimpleNamespace(
            update=lambda _n: None, close=lambda: None
        )

        client = HfBatchClient.__new__(HfBatchClient)
        client.batch_size = 4
        client.max_input = 100
        client.device = "cuda:0"
        client.tok = FakeTokenizer()
        client.model = FakeModel()
        conversations = [[{"role": "user", "content": f"q{i}"}] for i in range(4)]
        with patch.dict(sys.modules, {"torch": fake_torch, "tqdm": fake_tqdm}):
            results = client.chat_batch(conversations, n=1, temperature=0.7,
                                        max_tokens=32)
        self.assertEqual(client.model.calls, [4, 2, 2])
        self.assertEqual(fake_torch.cuda.clears, 1)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(len(group) == 1 for group in results))

    def test_cuda_oom_halves_return_sequences_after_prompt_batch_reaches_one(self):
        class FakeTensor:
            def __init__(self, batch, tokens):
                self.batch = batch
                self.shape = (batch, tokens)

        class FakeEncoding(dict):
            def __init__(self, batch, tokens):
                super().__init__(input_ids=FakeTensor(batch, tokens))

            def to(self, _device):
                return self

        class FakeOutput:
            def __init__(self, size):
                self.size = size

            def __getitem__(self, _key):
                return [None] * self.size

        class FakeTokenizer:
            pad_token_id = 0

            def apply_chat_template(self, conversation, **_kwargs):
                return conversation[0]["content"]

            def __call__(self, texts, **_kwargs):
                return FakeEncoding(len(texts), _kwargs["max_length"])

            def batch_decode(self, seqs, **_kwargs):
                return [f"code-{i}" for i in range(len(seqs))]

        class FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                batch = kwargs["input_ids"].batch
                n_return = kwargs["num_return_sequences"]
                tokens = kwargs["input_ids"].shape[1]
                self.calls.append((batch, n_return, tokens))
                if batch * n_return > 2 or tokens > 3000:
                    raise RuntimeError("CUDA out of memory")
                return FakeOutput(batch * n_return)

        class FakeCuda:
            def is_available(self):
                return True

            def empty_cache(self):
                pass

        @contextmanager
        def inference_mode():
            yield

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = FakeCuda()
        fake_torch.inference_mode = inference_mode
        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda **_kwargs: SimpleNamespace(
            update=lambda _n: None, close=lambda: None
        )

        client = HfBatchClient.__new__(HfBatchClient)
        client.batch_size = 2
        client.max_input = 6500
        client.device = "cuda:0"
        client.tok = FakeTokenizer()
        client.model = FakeModel()
        conversations = [[{"role": "user", "content": f"q{i}"}] for i in range(2)]

        with patch.dict(sys.modules, {"torch": fake_torch, "tqdm": fake_tqdm}):
            results = client.chat_batch(
                conversations, n=5, temperature=0.5, max_tokens=32,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(group) == 5 for group in results))
        self.assertTrue(any(call[:2] == (1, 5) for call in client.model.calls))
        self.assertTrue(any(call[:2] == (1, 2) for call in client.model.calls))
        self.assertEqual(client.model.calls[-1][:2], (1, 1))
        self.assertLessEqual(client.model.calls[-1][2], 3000)


class PayloadManifestTests(unittest.TestCase):
    def test_manifest_verification_detects_tampering(self):
        runner = _load_kaggle_runner()
        with tempfile.TemporaryDirectory() as td:
            payload = Path(td)
            rels = [
                "retrieval.jsonl", "store/reports.parquet",
                "code/kaggle_codegen.py", "code/vifinqa/codegen/generate.py",
                "code/vifinqa/codegen/llm_client.py",
                "code/vifinqa/codegen/prompts.py",
                "code/vifinqa/codegen/executor.py",
            ]
            hashes = {}
            for rel in rels:
                path = payload / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(rel.encode())
                hashes[rel] = hashlib.sha256(rel.encode()).hexdigest()
            manifest = {"schema_version": runner.PAYLOAD_SCHEMA_VERSION,
                        "files": hashes}
            (payload / runner.MANIFEST_NAME).write_text(json.dumps(manifest))
            checked, digest = runner.verify_payload(payload, payload / "code")
            self.assertEqual(checked["schema_version"], runner.PAYLOAD_SCHEMA_VERSION)
            self.assertEqual(len(digest), 64)

            (payload / "code/vifinqa/codegen/generate.py").write_text("tampered")
            with self.assertRaisesRegex(SystemExit, "hash mismatch"):
                runner.verify_payload(payload, payload / "code")

    def test_notebook_is_valid_json_and_has_no_hot_patch(self):
        notebook_path = ROOT / "kaggle" / "vifinqa-codegen.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertEqual(notebook["nbformat"], 4)
        self.assertNotIn("%%writefile", source)
        self.assertIn("payload-manifest.json", source)

    def test_private_notebooks_are_valid_and_use_targeted_n5(self):
        for name in (
                "private-v29-qwen14b-n5-consensus.ipynb",
                "private-v30-schema-qwen14b-n5.ipynb",
                "private-v33-qwen14b-code-n5.ipynb",
                "private-v34-schema-qwen14b-code-n5.ipynb",
                "private-v35-note-ops-qwen14b-code-n5.ipynb"):
            notebook = json.loads((ROOT / "kaggle" / name).read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook["cells"])
            self.assertEqual(notebook["nbformat"], 4)
            self.assertIn("Qwen2.5-Coder-14B-Instruct", source)
            self.assertIn("--llm-ids-file", source)
            self.assertIn("--no-rule-fallback", source)
            if "code-n5" in name:
                self.assertIn("'--n', '5'", source)
                self.assertIn("--smoke-first", source)
                self.assertIn("subprocess.Popen(cmd", source)
            else:
                self.assertIn("--n 5", source)

    def test_payload_builder_hashes_small_payload_and_preserves_id(self):
        builder = _load_payload_builder()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for rel, content in {
                "retrieval.jsonl": b"{}\n",
                "code/kaggle_codegen.py": b"runner",
                "store/reports.parquet": b"parquet",
            }.items():
                path = out / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            manifest = builder._build_manifest(out)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(set(manifest["files"]), {
                "retrieval.jsonl", "code/kaggle_codegen.py",
                "store/reports.parquet",
            })
            self.assertEqual(
                builder._dataset_id("", {"id": "owner/vifinqa-payload"}, "ignored"),
                "owner/vifinqa-payload",
            )

    def test_payload_manifest_hashes_optional_target_ids(self):
        builder = _load_payload_builder()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for rel, content in {
                "retrieval.jsonl": b"{}\n",
                "target_ids.txt": b"1\n2\n",
                "code/kaggle_codegen.py": b"runner",
                "store/reports.parquet": b"parquet",
            }.items():
                path = out / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            manifest = builder._build_manifest(out)

            self.assertIn("target_ids.txt", manifest["files"])

    def test_payload_dry_run_validates_without_replacing_output(self):
        builder = _load_payload_builder()
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            store = temp / "store"
            store.mkdir()
            (store / "reports.parquet").write_bytes(b"small")
            retrieval = temp / "retrieval.jsonl"
            retrieval.write_text("{}\n")
            out = ROOT / "artifacts" / f"dry-run-{temp.name}"
            self.assertFalse(out.exists())
            argv = [
                "04_make_kaggle_payload.py", "--store-dir", str(store),
                "--retrieval", str(retrieval), "--out", str(out),
                "--dataset-id", "owner/vifinqa-payload", "--dry-run",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()) as stdout:
                builder.main()
            self.assertIn("dry-run OK", stdout.getvalue())
            self.assertFalse(out.exists())

    def test_payload_output_is_confined_to_artifacts(self):
        builder = _load_payload_builder()
        safe = ROOT / "artifacts" / "some-payload"
        self.assertEqual(builder._safe_output_path(ROOT, safe), safe.resolve())
        with self.assertRaisesRegex(SystemExit, "outside"):
            builder._safe_output_path(ROOT, ROOT / "kaggle")
        with self.assertRaisesRegex(SystemExit, "root itself"):
            builder._safe_output_path(ROOT, ROOT / "artifacts")


if __name__ == "__main__":
    unittest.main()
