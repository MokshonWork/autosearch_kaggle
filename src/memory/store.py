"""Memory store: append-only experiments.jsonl + lessons.md, atomic state.json.

G18: logs are append-only and never rewritten; secrets are redacted on write.
Section 6.6: state.json enables resume after a kill mid-round.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..schemas import Experiment

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),           # Google API keys
    re.compile(r"(?i)(api[_-]?key|token|secret)=[^\s]+"),
]


def redact(text: str) -> str:
    """Remove anything that looks like a secret from log-bound text (G18)."""
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


class MemoryStoreError(RuntimeError):
    pass


class CorruptStateError(MemoryStoreError):
    """state.json is unreadable; it was moved aside, start fresh."""


class MemoryStore:
    def __init__(self, memory_dir: Path):
        self.dir = Path(memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.experiments_path = self.dir / "experiments.jsonl"
        self.lessons_path = self.dir / "lessons.md"
        self.state_path = self.dir / "state.json"

    # -- append-only logs ----------------------------------------------------
    def append_experiment(self, exp: Experiment) -> None:
        line = json.dumps(json.loads(exp.model_dump_json()))
        with open(self.experiments_path, "a") as f:
            f.write(redact(line) + "\n")

    def append_lesson(self, lesson: str) -> None:
        with open(self.lessons_path, "a") as f:
            f.write(redact(lesson.rstrip()) + "\n")

    def read_experiments(self) -> list[dict]:
        if not self.experiments_path.exists():
            return []
        return [json.loads(l) for l in
                self.experiments_path.read_text().splitlines() if l.strip()]

    def read_lessons(self) -> list[str]:
        if not self.lessons_path.exists():
            return []
        return [l for l in self.lessons_path.read_text().splitlines() if l.strip()]

    # -- resumable state ------------------------------------------------------
    def save_state(self, state: dict) -> None:
        """Atomic write (tmp + rename) so a kill mid-write can't corrupt it."""
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, self.state_path)

    def load_state(self) -> dict | None:
        """Fail safe on corrupt state (Ch.8 test 24): back it up, start fresh.

        A corrupt state.json must never silently resume with wrong data.
        """
        if not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            bak = self.state_path.with_suffix(".json.corrupt")
            try:
                os.replace(self.state_path, bak)
            except OSError:
                pass
            raise CorruptStateError(
                f"state.json corrupt ({e}); moved aside to {bak.name}, "
                "caller should start a fresh run") from e
        if not isinstance(data, dict):
            raise CorruptStateError("state.json has wrong shape (not an object)")
        return data
