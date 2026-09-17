"""Application service used by the Hard manual CLI commands."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pydantic import ValidationError

from vifinqa.config import Settings
from vifinqa.common.corpus.catalog import scan_catalog
from vifinqa.common.corpus.company_meta import load_company_meta
from vifinqa.generation.manual.packs import HardManualPackBuilder
from vifinqa.generation.manual.queue import ManualQueue, QueueError, session_actor_id
from vifinqa.generation.manual.schemas import ManualAuditDraft, ManualAuthorDraft, ManualRecordDraft
from vifinqa.generation.manual.validator import ManualRecordValidator
from vifinqa.generation.panel.store import JsonCubeStore


def source_revision(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QueueError(f"Could not obtain source commit from {repo_root}: {exc}") from exc
    return commit, dirty


def prepare_manual_queue(*, settings: Settings, count: int, seed: int, repo_root: Path) -> dict[str, object]:
    docs = scan_catalog(settings.data_root)
    companies = load_company_meta(settings.company_meta_path)
    cube = JsonCubeStore(docs=docs, cache_dir=settings.cache_dir).load_or_build()
    commit, dirty = source_revision(repo_root)
    packs = HardManualPackBuilder(
        cube=cube,
        docs=docs,
        company_meta=companies,
        seed=seed,
        source_commit=commit,
        source_dirty=dirty,
    ).build(count)
    return ManualQueue(settings.hard_manual_root).prepare(packs)


class ManualWorkflow:
    def __init__(
        self,
        *,
        root: Path,
        actor_id: str | None = None,
        validator: ManualRecordValidator | None = None,
    ) -> None:
        self.queue = ManualQueue(root)
        self.actor_id = actor_id or session_actor_id()
        self.validator = validator or ManualRecordValidator()

    def claim_next(self) -> dict[str, object] | None:
        return self.queue.claim_next(self.actor_id)

    def audit_claim_next(self) -> dict[str, object] | None:
        return self.queue.audit_claim_next(self.actor_id)

    def submit_author(self, work_id: str) -> dict[str, object]:
        context = self.queue.author_context(work_id, self.actor_id)
        claim = context["claim"]
        assert isinstance(claim, dict)
        draft_path = Path(str(claim["draft_path"]))
        try:
            raw = json.loads(draft_path.read_text(encoding="utf-8"))
            draft = ManualAuthorDraft.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            payload = {"accepted": False, "reject_reasons": [f"author_draft_schema:{exc}"]}
            return self.queue.save_author_result(
                work_id=work_id,
                actor_id=self.actor_id,
                validation_payload=payload,
                submission_payload=None,
                no_yield=False,
            )

        identity_errors = self._author_identity_errors(draft, context)
        pack = self._load_object(Path(str(context["pack_path"])))
        validations = [
            self.validator.validate(
                draft=record,
                pack=pack,
                batch_id=draft.batch_id,
                position=position,
            )
            for position, record in enumerate(draft.records)
        ]
        validation_payload = {
            "accepted": not identity_errors and all(result.accepted for result in validations),
            "work_id": work_id,
            "batch_id": draft.batch_id,
            "reject_reasons": identity_errors,
            "records": [result.to_payload() for result in validations],
        }
        if identity_errors or any(not result.accepted for result in validations):
            return self.queue.save_author_result(
                work_id=work_id,
                actor_id=self.actor_id,
                validation_payload=validation_payload,
                submission_payload=None,
                no_yield=False,
            )

        entries: list[dict[str, object]] = []
        for record_draft, result in zip(draft.records, validations, strict=True):
            assert result.qa_record is not None
            entries.append(
                {
                    "qa_record": result.qa_record.model_dump(mode="json"),
                    "record_draft": record_draft.model_dump(mode="json"),
                    "trace": record_draft.trace.model_dump(mode="json"),
                    "self_review": record_draft.self_review.model_dump(mode="json"),
                    "metamorphic_specs": [
                        spec.model_dump(mode="json") for spec in record_draft.metamorphic_specs
                    ],
                    "validation": result.to_payload(),
                }
            )
        submission_payload = {
            "work_id": work_id,
            "pack_id": draft.pack_id,
            "batch_id": draft.batch_id,
            "author_id": draft.author_id,
            "ideation": [note.model_dump(mode="json") for note in draft.ideation],
            "records": entries,
            "no_yield_reason": draft.no_yield_reason,
        }
        return self.queue.save_author_result(
            work_id=work_id,
            actor_id=self.actor_id,
            validation_payload=validation_payload,
            submission_payload=submission_payload,
            no_yield=not draft.records,
        )

    def submit_audit(self, work_id: str) -> dict[str, object]:
        context = self.queue.audit_context(work_id, self.actor_id)
        audit = context["audit"]
        submission_meta = context["submission"]
        assert isinstance(audit, dict) and isinstance(submission_meta, dict)
        draft_path = Path(str(audit["draft_path"]))
        try:
            raw = json.loads(draft_path.read_text(encoding="utf-8"))
            draft = ManualAuditDraft.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            payload = {"accepted": False, "reject_reasons": [f"audit_draft_schema:{exc}"]}
            return self.queue.save_audit_result(
                work_id=work_id,
                reviewer_id=self.actor_id,
                validation_payload=payload,
                review_payload=None,
            )

        submission = self._load_object(Path(str(submission_meta["submission_path"])))
        pack = self._load_object(Path(str(context["pack_path"])))
        identity_errors = self._audit_identity_errors(draft, context)
        if submission.get("author_id") == self.actor_id:
            identity_errors.append("reviewer_may_not_review_own_batch")

        source_entries = submission.get("records", [])
        source_ids = {
            int(entry["qa_record"]["id"])
            for entry in source_entries
            if isinstance(entry, dict) and isinstance(entry.get("qa_record"), dict)
        }
        reconstruction_ids = {entry.record_id for entry in draft.reconstructions}
        review_ids = {entry.record_id for entry in draft.reviews}
        if reconstruction_ids != source_ids:
            identity_errors.append("reconstruction_record_coverage_mismatch")
        if review_ids != source_ids:
            identity_errors.append("review_record_coverage_mismatch")

        reruns = []
        for position, entry in enumerate(source_entries):
            if not isinstance(entry, dict) or not isinstance(entry.get("record_draft"), dict):
                identity_errors.append(f"author_submission_missing_record_draft:{position}")
                continue
            try:
                record_draft = ManualRecordDraft.model_validate(entry["record_draft"])
            except ValidationError as exc:
                identity_errors.append(f"author_record_schema_changed:{position}:{exc}")
                continue
            reruns.append(
                self.validator.validate(
                    draft=record_draft,
                    pack=pack,
                    batch_id=draft.batch_id,
                    position=position,
                )
            )

        review_by_id = {review.record_id: review for review in draft.reviews}
        for result in reruns:
            review = review_by_id.get(result.record_id)
            if review is not None and review.verdict == "accept" and not result.accepted:
                identity_errors.append(f"review_accepts_failed_reexecution:{result.record_id}")

        validation_payload = {
            "accepted": not identity_errors,
            "work_id": work_id,
            "batch_id": draft.batch_id,
            "audit_id": draft.audit_id,
            "reject_reasons": identity_errors,
            "reexecution": [result.to_payload() for result in reruns],
        }
        if identity_errors:
            return self.queue.save_audit_result(
                work_id=work_id,
                reviewer_id=self.actor_id,
                validation_payload=validation_payload,
                review_payload=None,
            )

        review_payload = {
            "work_id": work_id,
            "batch_id": draft.batch_id,
            "audit_id": draft.audit_id,
            "reviewer_id": draft.reviewer_id,
            "reconstructions": [entry.model_dump(mode="json") for entry in draft.reconstructions],
            "reviews": [entry.model_dump(mode="json") for entry in draft.reviews],
            "batch_notes": draft.batch_notes,
            "reexecution": [result.to_payload() for result in reruns],
        }
        return self.queue.save_audit_result(
            work_id=work_id,
            reviewer_id=self.actor_id,
            validation_payload=validation_payload,
            review_payload=review_payload,
        )

    def _author_identity_errors(
        self, draft: ManualAuthorDraft, context: dict[str, object]
    ) -> list[str]:
        claim = context["claim"]
        assert isinstance(claim, dict)
        expected = {
            "work_id": context["work_id"],
            "pack_id": context["pack_id"],
            "batch_id": claim["batch_id"],
            "author_id": self.actor_id,
        }
        return [
            f"identity_mismatch:{name}"
            for name, value in expected.items()
            if getattr(draft, name) != value
        ]

    def _audit_identity_errors(
        self, draft: ManualAuditDraft, context: dict[str, object]
    ) -> list[str]:
        audit = context["audit"]
        submission = context["submission"]
        assert isinstance(audit, dict) and isinstance(submission, dict)
        expected = {
            "work_id": context["work_id"],
            "batch_id": submission["batch_id"],
            "audit_id": audit["audit_id"],
            "reviewer_id": self.actor_id,
        }
        return [
            f"identity_mismatch:{name}"
            for name, value in expected.items()
            if getattr(draft, name) != value
        ]

    @staticmethod
    def _load_object(path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QueueError(f"Could not read {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise QueueError(f"File must be a JSON object: {path}")
        return value
