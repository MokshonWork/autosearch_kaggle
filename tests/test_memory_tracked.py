"""Regression: src/memory/store.py must be an importable tracked module.

Guards against the Kaggle failure where `src/llm/client.py` raised
`ModuleNotFoundError: No module named 'src.memory'` because the root
.gitignore `memory/` pattern had silently excluded the whole package from
version control. Proves the import works AND redaction still holds.
"""
import subprocess
from pathlib import Path


def test_memory_store_importable_and_tracked():
    from src.memory.store import MemoryStore, redact  # noqa: F401

    repo = Path(__file__).resolve().parent.parent
    tracked = subprocess.run(
        ["git", "ls-files", "src/memory/store.py", "src/memory/__init__.py"],
        capture_output=True, text=True, cwd=repo,
        check=True).stdout.split()
    assert "src/memory/store.py" in tracked
    assert "src/memory/__init__.py" in tracked
    ignored = subprocess.run(
        ["git", "check-ignore", "src/memory/store.py"],
        capture_output=True, text=True, cwd=repo).returncode
    assert ignored != 0, "src/memory/store.py must not be git-ignored"


def test_redact_representative_secrets():
    from src.memory.store import redact

    google_key = "AIzaSyFAKEKEY_DO_NOT_LEAK_0123456789abcd"
    assert google_key not in redact(f"request failed: {google_key} retrying")
    assert "[REDACTED]" in redact(f"request failed: {google_key} retrying")
    assert "hunter2" not in redact("api_key=hunter2 rest")
    assert "s3cr3t!" not in redact("token=s3cr3t! end")
    assert redact("plain harmless text") == "plain harmless text"
