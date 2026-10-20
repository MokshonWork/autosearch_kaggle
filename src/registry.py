"""Registry: authoritative project/run/task/change/test state (hardening 4.1).

Sits ON TOP of the existing append-only memory store: experiments.jsonl and
lessons.md stay the narrative source of truth. The Registry adds stable IDs,
validated state transitions, atomic writes, and explicit links between
task -> change -> test report -> experiment -> baseline.

Storage: one JSON file per collection inside the memory dir, written atomically
(tmp + os.replace, same technique as MemoryStore.save_state). No new
dependencies; no database (unnecessary at this scale).

Invariants enforced:
  1. One task -> at most one running owner (no double assignment).
  2. Status transitions follow TASK_TRANSITIONS (arbitrary jumps rejected).
  3. History is append-only: records are added/updated, never deleted.
  4. Baseline promotion requires a valid test report (checked by caller).
  5. Writes are atomic; duplicate deliveries are idempotent by stable id.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# Task lifecycle. Only these transitions are legal (invariant 2).
TASK_TRANSITIONS: dict[str, set[str]] = {
    "queued": {"running", "blocked", "failed"},
    "blocked": {"queued", "failed"},
    "running": {"completed", "failed", "blocked"},
    "completed": set(),  # terminal; a retry creates a NEW task
    "failed": {"queued"},  # explicit requeue only, counted as a retry
}

TERMINAL = {"completed", "failed"}


class TaskRecord(BaseModel):
    """One unit of scheduled work (experiment, code, test, ...)."""

    id: str
    kind: str
    status: str = "queued"
    dependencies: list[str] = Field(default_factory=list)
    owner: Optional[str] = None  # worker holding it; None = unassigned
    gpu: Optional[int] = None
    retries: int = 0
    max_retries: int = 2
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class ChangeRecord(BaseModel):
    """Manifest for one candidate code change (hardening 4.3)."""

    id: str  # change_id, e.g. CHANGE-xxxxxxxx
    task_id: str
    run_id: str = ""
    experiment_id: str = ""  # hypothesis id (H-0001), when known
    agent_id: str = ""  # e.g. "coder", "baseline"
    base_revision: str = ""  # git commit the candidate was derived from
    candidate_revision: str = ""  # content fingerprint (sha256 of train.py)
    changed_files: list[str] = Field(default_factory=list)
    objective: str = ""
    rationale: str = ""
    expected_behavior: str = ""
    risk_notes: str = ""
    required_checks: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    status: str = "proposed"  # proposed|guard_ok|guard_reject|tested|...
    created_at: str = Field(default_factory=_now)


class TestReport(BaseModel):
    """What was tested, how, and what came out, bound to a revision (Ch.5)."""

    id: str
    task_id: str = ""
    change_id: str = ""
    candidate_revision: str = ""  # fingerprint ACTUALLY tested
    base_revision: str = ""
    commands: list[str] = Field(default_factory=list)
    exit_codes: list[int] = Field(default_factory=list)
    status: str = ""  # ok|crash|oom|nan|timeout|guard_reject
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


class RegistryError(RuntimeError):
    pass


class Registry:
    """Atomic JSON-file registry living inside the existing memory dir."""

    _FILES = {
        "tasks": "registry_tasks.json",
        "changes": "registry_changes.json",
        "tests": "registry_tests.json",
        "meta": "registry_meta.json",  # baseline + best-known result
    }

    def __init__(self, memory_dir: Path | str):
        self.dir = Path(memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._tasks: dict[str, dict] = self._load("tasks")
        self._changes: dict[str, dict] = self._load("changes")
        self._tests: dict[str, dict] = self._load("tests")
        self._meta: dict[str, Any] = self._load("meta")

    # -- persistence ------------------------------------------------------
    def _path(self, name: str) -> Path:
        return self.dir / self._FILES[name]

    def _load(self, name: str) -> dict:
        p = self._path(name)
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise RegistryError(
                f"registry file {p.name} is corrupt ({e}); "
                "rename it aside and restart to rebuild from experiments.jsonl"
            ) from e
        if not isinstance(data, dict):
            raise RegistryError(f"registry file {p.name} has wrong shape")
        return data

    def _save(self, name: str, data: dict) -> None:
        tmp = self._path(name).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, self._path(name))  # atomic (invariant 5)

    # -- tasks ------------------------------------------------------------
    def create_task(self, kind: str, dependencies: list[str] | None = None,
                    max_retries: int = 2, payload: dict | None = None,
                    task_id: str | None = None) -> TaskRecord:
        rec = TaskRecord(id=task_id or new_id("TASK"),
                         kind=kind,
                         dependencies=list(dependencies or []),
                         max_retries=max_retries,
                         payload=dict(payload or {}))
        with self._lock:
            if rec.id in self._tasks:  # idempotent redelivery (inv. 6)
                return TaskRecord(**self._tasks[rec.id])
            self._tasks[rec.id] = rec.model_dump()
            self._save("tasks", self._tasks)
        return rec

    def get_task(self, task_id: str) -> TaskRecord:
        try:
            return TaskRecord(**self._tasks[task_id])
        except KeyError:
            raise RegistryError(f"unknown task {task_id}") from None

    def transition(self, task_id: str, to: str,
                   owner: str | None = None) -> TaskRecord:
        """Validated status change; claiming sets owner (invariants 1, 2)."""
        with self._lock:
            rec = self.get_task(task_id)
            if to not in TASK_TRANSITIONS.get(rec.status, set()):
                raise RegistryError(
                    f"illegal transition {rec.status} -> {to} for {task_id}")
            if to == "running":
                if rec.owner is not None and owner != rec.owner:
                    raise RegistryError(
                        f"task {task_id} already owned by {rec.owner}")
                rec.owner = owner or rec.owner
            if to == "queued" and rec.status == "failed":
                rec.retries += 1
                if rec.retries > rec.max_retries:
                    raise RegistryError(
                        f"task {task_id} exceeded max_retries={rec.max_retries}")
                rec.owner = None
            rec.status = to
            rec.updated_at = _now()
            self._tasks[task_id] = rec.model_dump()
            self._save("tasks", self._tasks)
        return rec

    def clear_owner(self, task_id: str) -> TaskRecord:
        """Release a dead owner's claim without changing status."""
        with self._lock:
            rec = self.get_task(task_id)
            rec.owner = None
            rec.updated_at = _now()
            self._tasks[task_id] = rec.model_dump()
            self._save("tasks", self._tasks)
        return rec

    def list_tasks(self, status: str | None = None,
                   kind: str | None = None) -> list[TaskRecord]:
        out = [TaskRecord(**t) for t in self._tasks.values()]
        if status:
            out = [t for t in out if t.status == status]
        if kind:
            out = [t for t in out if t.kind == kind]
        return sorted(out, key=lambda t: t.created_at)

    def deps_satisfied(self, task_id: str) -> bool:
        """All dependencies completed (Scheduler gate, Ch.4.2)."""
        rec = self.get_task(task_id)
        return all(self._tasks.get(d, {}).get("status") == "completed"
                   for d in rec.dependencies)

    # -- changes (manifests) ----------------------------------------------
    def record_change(self, change: ChangeRecord) -> ChangeRecord:
        with self._lock:
            if change.id in self._changes:  # idempotent (invariant 6)
                return ChangeRecord(**self._changes[change.id])
            self._changes[change.id] = change.model_dump()
            self._save("changes", self._changes)
        return change

    def get_change(self, change_id: str) -> ChangeRecord:
        try:
            return ChangeRecord(**self._changes[change_id])
        except KeyError:
            raise RegistryError(f"unknown change {change_id}") from None

    def set_change_status(self, change_id: str, status: str) -> ChangeRecord:
        with self._lock:
            rec = self.get_change(change_id)
            rec.status = status
            self._changes[change_id] = rec.model_dump()
            self._save("changes", self._changes)
        return rec

    def update_change(self, change_id: str, *,
                      candidate_revision: str | None = None,
                      status: str | None = None) -> ChangeRecord:
        """Explicitly correct mutable fields of an existing manifest.

        record_change is insert-only (idempotent redelivery); use this when
        the tested/approved content differs from the recorded fingerprint.
        Only the named fields are touched; audit fields are preserved.
        """
        with self._lock:
            rec = self.get_change(change_id)
            if candidate_revision is not None:
                rec.candidate_revision = candidate_revision
            if status is not None:
                rec.status = status
            self._changes[change_id] = rec.model_dump()
            self._save("changes", self._changes)
        return rec

    # -- test reports ------------------------------------------------------
    def record_test(self, report: TestReport) -> TestReport:
        with self._lock:
            if report.id in self._tests:  # idempotent (invariant 6)
                return TestReport(**self._tests[report.id])
            self._tests[report.id] = report.model_dump()
            self._save("tests", self._tests)
        return report

    def get_test(self, test_id: str) -> TestReport:
        try:
            return TestReport(**self._tests[test_id])
        except KeyError:
            raise RegistryError(f"unknown test report {test_id}") from None

    def tests_for_change(self, change_id: str) -> list[TestReport]:
        return sorted(
            (TestReport(**t) for t in self._tests.values()
             if t.get("change_id") == change_id),
            key=lambda r: r.created_at)

    # -- baseline / best ----------------------------------------------------
    def get_baseline(self) -> dict:
        return dict(self._meta.get("baseline", {}))

    def set_baseline(self, record: dict) -> None:
        """Only call after the acceptance policy approved the candidate."""
        with self._lock:
            self._meta["baseline"] = dict(record)
            self._meta["baseline_updated_at"] = _now()
            self._save("meta", self._meta)

    def check_revision_match(self, change_id: str, test_id: str) -> bool:
        """True iff the report evaluated this change's exact revision (Ch.5)."""
        change = self.get_change(change_id)
        report = self.get_test(test_id)
        return bool(report.candidate_revision) and (
            report.candidate_revision == change.candidate_revision)
