"""G1: protected-file hashing. Hash at start, re-verify after every run.

Any change to a protected file invalidates the run and halts the system.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class ProtectedFileError(RuntimeError):
    """Raised when a protected file's hash changed (G1 violation)."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HashRegistry:
    """Snapshot of protected file hashes; verify() raises on any drift."""

    def __init__(self, files: list[Path]):
        self.files = [Path(f) for f in files]
        self.hashes = {str(f): sha256_file(f) for f in self.files}

    def verify(self) -> None:
        for f in self.files:
            current = sha256_file(f)
            if current != self.hashes[str(f)]:
                raise ProtectedFileError(
                    f"protected file changed: {f} "
                    f"({self.hashes[str(f)][:12]} -> {current[:12]})"
                )


def diff_hash(text: str) -> str:
    """Short stable hash for patches/diffs (G11 reproducibility)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
