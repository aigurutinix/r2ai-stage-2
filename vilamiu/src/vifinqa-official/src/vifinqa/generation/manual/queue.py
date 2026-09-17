"""Filesystem-backed atomic queue for independent Hard manual author/audit sessions."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator

from vifinqa.generation.manual.packs import dump_pack, stable_id

DEFAULT_LEASE = timedelta(hours=4)
_PROCESS_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


class QueueError(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def session_actor_id() -> str:
    """Stable, automatic identity for one Codex task; no user-selected author/reviewer id."""
    identity = os.environ.get("CODEX_THREAD_ID", "").strip()
    if not identity:
        try:
            identity = f"tty:{os.ttyname(0)}"
        except OSError as exc:
            raise QueueError(
                "CODEX_THREAD_ID/TTY was not found; run this command in a Codex session to derive an author ID."
            ) from exc
    return stable_id("actor", identity)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueError(f"Could not read queue file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QueueError(f"Queue file must be a JSON object: {path}")
    return value


class ManualQueue:
    def __init__(self, root: Path, *, lease: timedelta = DEFAULT_LEASE) -> None:
        self.root = root.resolve()
        self.lease = lease

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".queue.lock"
        key = str(lock_path)
        with _PROCESS_LOCKS_GUARD:
            process_lock = _PROCESS_LOCKS.setdefault(key, threading.Lock())
        with process_lock, lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def prepare(self, packs: list[dict[str, object]]) -> dict[str, object]:
        if not packs:
            raise QueueError("No pack is available to prepare")
        protocol = str(packs[0]["protocol_version"])
        seed = int(packs[0]["seed"])
        source_commit = str(packs[0]["source_commit"])
        campaign_id = stable_id("campaign", protocol, source_commit, seed)
        created = 0
        with self._lock():
            for position, pack in enumerate(packs):
                pack_id = str(pack["pack_id"])
                pack_path = self.root / "packs" / f"{pack_id}.json"
                if pack_path.exists():
                    existing = _read_json(pack_path)
                    if existing != pack:
                        raise QueueError(f"pack_id collision with different content: {pack_id}")
                else:
                    dump_pack(pack, pack_path)

                work_id = stable_id("work", campaign_id, pack_id)
                item_path = self._item_path(work_id)
                if item_path.exists():
                    continue
                item = {
                    "work_id": work_id,
                    "pack_id": pack_id,
                    "pack_path": str(pack_path),
                    "campaign_id": campaign_id,
                    "queue_order": f"{campaign_id}:{position:06d}",
                    "state": "pending",
                    "claim": None,
                    "claim_history": [],
                    "submission": None,
                    "audit": None,
                    "audit_history": [],
                }
                _atomic_write_json(item_path, item)
                created += 1

            manifest_path = self.root / "campaigns" / f"{campaign_id}.json"
            if not manifest_path.exists():
                _atomic_write_json(
                    manifest_path,
                    {
                        "campaign_id": campaign_id,
                        "protocol_version": protocol,
                        "source_commit": source_commit,
                        "seed": seed,
                        "pack_ids": [pack["pack_id"] for pack in packs],
                    },
                )
        return {
            "campaign_id": campaign_id,
            "prepared": len(packs),
            "created": created,
            "queue_root": str(self.root),
        }

    def claim_next(self, actor_id: str, *, now: datetime | None = None) -> dict[str, object] | None:
        now = now or utc_now()
        with self._lock():
            items = self._items()
            for item in items:
                claim = item.get("claim")
                if item.get("state") == "claimed" and isinstance(claim, dict):
                    if claim.get("author_id") == actor_id and not self._expired(claim, now):
                        return self._claim_payload(item)

            eligible = [
                item
                for item in items
                if item.get("state") == "pending"
                or (
                    item.get("state") == "claimed"
                    and isinstance(item.get("claim"), dict)
                    and self._expired(item["claim"], now)  # type: ignore[arg-type]
                )
            ]
            if not eligible:
                return None
            item = eligible[0]
            old_claim = item.get("claim")
            history = list(item.get("claim_history") or [])
            if isinstance(old_claim, dict):
                history.append({**old_claim, "ended_as": "lease_expired"})
            attempt = len(history) + 1
            work_id = str(item["work_id"])
            pack_id = str(item["pack_id"])
            batch_id = stable_id("batch", work_id, actor_id, attempt)
            draft_path = self.root / "author_shards" / actor_id / f"{batch_id}.draft.json"
            submission_path = self.root / "author_submissions" / f"{batch_id}.json"
            claim = {
                "author_id": actor_id,
                "batch_id": batch_id,
                "claim_attempt": attempt,
                "claimed_at": _iso(now),
                "lease_expires_at": _iso(now + self.lease),
                "draft_path": str(draft_path),
                "submission_path": str(submission_path),
                "submit_attempts": 0,
            }
            item.update(state="claimed", claim=claim, claim_history=history)
            _atomic_write_json(self._item_path(work_id), item)
            if not draft_path.exists():
                _atomic_write_json(
                    draft_path,
                    {
                        "work_id": work_id,
                        "pack_id": pack_id,
                        "batch_id": batch_id,
                        "author_id": actor_id,
                        "ideation": [],
                        "records": [],
                        "no_yield_reason": "",
                    },
                )
            return self._claim_payload(item)

    def author_context(self, work_id: str, actor_id: str) -> dict[str, object]:
        with self._lock():
            item = self._load_item(work_id)
            claim = item.get("claim")
            if item.get("state") != "claimed" or not isinstance(claim, dict):
                raise QueueError(f"work item is not in the claimed state: {work_id}")
            if claim.get("author_id") != actor_id:
                raise QueueError("The current session does not own this author claim")
            return {**item, "claim": dict(claim)}

    def save_author_result(
        self,
        *,
        work_id: str,
        actor_id: str,
        validation_payload: dict[str, object],
        submission_payload: dict[str, object] | None,
        no_yield: bool,
    ) -> dict[str, object]:
        with self._lock():
            item = self._load_item(work_id)
            claim = item.get("claim")
            if item.get("state") != "claimed" or not isinstance(claim, dict):
                raise QueueError(f"work item is not in the claimed state: {work_id}")
            if claim.get("author_id") != actor_id:
                raise QueueError("The current session does not own this author claim")
            attempts = int(claim.get("submit_attempts", 0)) + 1
            claim["submit_attempts"] = attempts
            validation_path = (
                self.root
                / "validation"
                / str(claim["batch_id"])
                / f"attempt-{attempts:03d}.json"
            )
            _atomic_write_json(validation_path, validation_payload)
            if submission_payload is None:
                item["claim"] = claim
                _atomic_write_json(self._item_path(work_id), item)
                return {
                    "accepted": False,
                    "validation_path": str(validation_path),
                    "draft_path": claim["draft_path"],
                }

            submission_path = Path(str(claim["submission_path"]))
            if submission_path.exists():
                raise QueueError(f"submission shard already exists: {submission_path}")
            _atomic_write_json(submission_path, submission_payload)
            blind_path: Path | None = None
            if not no_yield:
                blind_path = self.root / "blind_batches" / f"{claim['batch_id']}.json"
                _atomic_write_json(
                    blind_path,
                    {
                        "work_id": work_id,
                        "pack_id": item["pack_id"],
                        "batch_id": claim["batch_id"],
                        "records": [entry["qa_record"] for entry in submission_payload["records"]],
                    },
                )
            item["state"] = "no_yield" if no_yield else "submitted"
            item["claim"] = claim
            item["submission"] = {
                "batch_id": claim["batch_id"],
                "author_id": actor_id,
                "submission_path": str(submission_path),
                "blind_path": str(blind_path) if blind_path else None,
                "validation_path": str(validation_path),
                "submitted_at": _iso(utc_now()),
                "record_count": len(submission_payload["records"]),
            }
            _atomic_write_json(self._item_path(work_id), item)
            return {
                "accepted": True,
                "state": item["state"],
                "submission_path": str(submission_path),
                "validation_path": str(validation_path),
            }

    def audit_claim_next(
        self, reviewer_id: str, *, now: datetime | None = None
    ) -> dict[str, object] | None:
        now = now or utc_now()
        with self._lock():
            items = self._items()
            for item in items:
                audit = item.get("audit")
                if isinstance(audit, dict) and audit.get("state") == "claimed":
                    if audit.get("reviewer_id") == reviewer_id and not self._expired(audit, now):
                        return self._audit_payload(item)

            eligible: list[dict[str, object]] = []
            for item in items:
                if item.get("state") != "submitted":
                    continue
                submission = item.get("submission")
                if not isinstance(submission, dict) or submission.get("author_id") == reviewer_id:
                    continue
                audit = item.get("audit")
                if audit is None or (
                    isinstance(audit, dict)
                    and audit.get("state") == "claimed"
                    and self._expired(audit, now)
                ):
                    eligible.append(item)
            if not eligible:
                return None
            item = eligible[0]
            history = list(item.get("audit_history") or [])
            old_audit = item.get("audit")
            if isinstance(old_audit, dict):
                history.append({**old_audit, "ended_as": "lease_expired"})
            attempt = len(history) + 1
            submission = item["submission"]
            assert isinstance(submission, dict)
            batch_id = str(submission["batch_id"])
            audit_id = stable_id("audit", batch_id, reviewer_id, attempt)
            draft_path = self.root / "review_shards" / reviewer_id / f"{audit_id}.draft.json"
            audit = {
                "state": "claimed",
                "reviewer_id": reviewer_id,
                "audit_id": audit_id,
                "audit_attempt": attempt,
                "claimed_at": _iso(now),
                "lease_expires_at": _iso(now + self.lease),
                "draft_path": str(draft_path),
                "review_path": str(self.root / "audit_submissions" / f"{audit_id}.json"),
                "submit_attempts": 0,
            }
            item["audit"] = audit
            item["audit_history"] = history
            _atomic_write_json(self._item_path(str(item["work_id"])), item)
            if not draft_path.exists():
                _atomic_write_json(
                    draft_path,
                    {
                        "work_id": item["work_id"],
                        "batch_id": batch_id,
                        "audit_id": audit_id,
                        "reviewer_id": reviewer_id,
                        "reconstructions": [],
                        "reviews": [],
                        "batch_notes": "",
                    },
                )
            return self._audit_payload(item)

    def audit_context(self, work_id: str, reviewer_id: str) -> dict[str, object]:
        with self._lock():
            item = self._load_item(work_id)
            audit = item.get("audit")
            if not isinstance(audit, dict) or audit.get("state") != "claimed":
                raise QueueError(f"work item has no open audit claim: {work_id}")
            if audit.get("reviewer_id") != reviewer_id:
                raise QueueError("The current session does not own this audit claim")
            return {**item, "audit": dict(audit)}

    def save_audit_result(
        self,
        *,
        work_id: str,
        reviewer_id: str,
        validation_payload: dict[str, object],
        review_payload: dict[str, object] | None,
    ) -> dict[str, object]:
        with self._lock():
            item = self._load_item(work_id)
            audit = item.get("audit")
            if not isinstance(audit, dict) or audit.get("state") != "claimed":
                raise QueueError(f"work item has no open audit claim: {work_id}")
            if audit.get("reviewer_id") != reviewer_id:
                raise QueueError("The current session does not own this audit claim")
            attempts = int(audit.get("submit_attempts", 0)) + 1
            audit["submit_attempts"] = attempts
            validation_path = (
                self.root / "audit_validation" / str(audit["audit_id"]) / f"attempt-{attempts:03d}.json"
            )
            _atomic_write_json(validation_path, validation_payload)
            if review_payload is None:
                item["audit"] = audit
                _atomic_write_json(self._item_path(work_id), item)
                return {
                    "accepted": False,
                    "validation_path": str(validation_path),
                    "draft_path": audit["draft_path"],
                }

            review_path = Path(str(audit["review_path"]))
            if review_path.exists():
                raise QueueError(f"review shard already exists: {review_path}")
            _atomic_write_json(review_path, review_payload)
            audit["state"] = "reviewed"
            audit["reviewed_at"] = _iso(utc_now())
            audit["validation_path"] = str(validation_path)
            item["audit"] = audit
            item["state"] = "reviewed"
            _atomic_write_json(self._item_path(work_id), item)
            return {
                "accepted": True,
                "state": "reviewed",
                "review_path": str(review_path),
                "validation_path": str(validation_path),
            }

    def status(self) -> dict[str, object]:
        with self._lock():
            items = self._items()
            counts: dict[str, int] = {}
            for item in items:
                state = str(item.get("state"))
                counts[state] = counts.get(state, 0) + 1
            return {
                "queue_root": str(self.root),
                "total": len(items),
                "states": dict(sorted(counts.items())),
                "expired_author_claims": sum(
                    1
                    for item in items
                    if item.get("state") == "claimed"
                    and isinstance(item.get("claim"), dict)
                    and self._expired(item["claim"], utc_now())  # type: ignore[arg-type]
                ),
            }

    def merge_accepted(self) -> dict[str, object]:
        """Curator-only materialization of the accepted JSONL ledger.

        Author/audit commands never call this method.  The whole ledger is atomically replaced,
        so there is no concurrent JSONL append path.
        """
        ledger_path = self.root / "accepted" / "hard_manual_accepted.jsonl"
        with self._lock():
            existing: list[dict[str, object]] = []
            if ledger_path.exists():
                for line_number, line in enumerate(
                    ledger_path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise QueueError(
                            f"malformed accepted ledger at line {line_number}: {exc}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise QueueError(f"accepted ledger line {line_number} is not an object")
                    existing.append(row)
            by_id = {int(row["id"]): row for row in existing}
            before = len(by_id)

            for item in self._items():
                if item.get("state") != "reviewed":
                    continue
                audit = item.get("audit")
                submission_meta = item.get("submission")
                if not isinstance(audit, dict) or not isinstance(submission_meta, dict):
                    continue
                review = _read_json(Path(str(audit["review_path"])))
                submission = _read_json(Path(str(submission_meta["submission_path"])))
                verdicts = {
                    int(entry["record_id"]): entry["verdict"]
                    for entry in review.get("reviews", [])
                    if isinstance(entry, dict)
                }
                for entry in submission.get("records", []):
                    if not isinstance(entry, dict) or not isinstance(entry.get("qa_record"), dict):
                        continue
                    qa = entry["qa_record"]
                    qa_id = int(qa["id"])
                    if verdicts.get(qa_id) == "accept":
                        by_id.setdefault(qa_id, qa)

            rows = [by_id[key] for key in sorted(by_id)]
            _atomic_write_jsonl(ledger_path, rows)
            return {
                "ledger_path": str(ledger_path),
                "accepted_total": len(rows),
                "added": len(rows) - before,
            }

    def _items(self) -> list[dict[str, object]]:
        directory = self.root / "work_items"
        if not directory.exists():
            return []
        items = [_read_json(path) for path in directory.glob("work-*.json")]
        return sorted(items, key=lambda item: (str(item.get("queue_order")), str(item["work_id"])))

    def _load_item(self, work_id: str) -> dict[str, object]:
        path = self._item_path(work_id)
        if not path.exists():
            raise QueueError(f"Work item not found: {work_id}")
        return _read_json(path)

    def _item_path(self, work_id: str) -> Path:
        return self.root / "work_items" / f"{work_id}.json"

    @staticmethod
    def _expired(claim: dict[str, object], now: datetime) -> bool:
        value = claim.get("lease_expires_at")
        return not isinstance(value, str) or _parse_time(value) <= now

    @staticmethod
    def _claim_payload(item: dict[str, object]) -> dict[str, object]:
        claim = item["claim"]
        assert isinstance(claim, dict)
        return {
            "work_id": item["work_id"],
            "pack_id": item["pack_id"],
            "batch_id": claim["batch_id"],
            "author_id": claim["author_id"],
            "pack_path": item["pack_path"],
            "draft_path": claim["draft_path"],
            "submission_path": claim["submission_path"],
            "lease_expires_at": claim["lease_expires_at"],
        }

    @staticmethod
    def _audit_payload(item: dict[str, object]) -> dict[str, object]:
        audit = item["audit"]
        submission = item["submission"]
        assert isinstance(audit, dict) and isinstance(submission, dict)
        return {
            "work_id": item["work_id"],
            "batch_id": submission["batch_id"],
            "audit_id": audit["audit_id"],
            "reviewer_id": audit["reviewer_id"],
            "blind_batch_path": submission["blind_path"],
            "author_submission_path": submission["submission_path"],
            "pack_path": item["pack_path"],
            "draft_path": audit["draft_path"],
            "review_path": audit["review_path"],
            "lease_expires_at": audit["lease_expires_at"],
        }
