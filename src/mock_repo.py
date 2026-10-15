"""Create a git-initialized mock repo (mock train.py/prepare.py) for dry runs
and tests. Usage: python -m src.mock_repo /path/to/dir
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "mock_repo"


def create_mock_git_repo(target: Path) -> Path:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("train.py", "prepare.py"):
        shutil.copy(FIXTURES / name, target / name)
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.email=mock@local", "-c", "user.name=mock",
                  # --allow-empty: re-running the generator on an existing
                  # fixture (make dry-run) must not die on "nothing to commit"
                  "commit", "-q", "--allow-empty", "-m", "mock baseline"]):
        subprocess.run(["git", "-C", str(target), *args], check=True)
    return target


if __name__ == "__main__":
    create_mock_git_repo(Path(sys.argv[1] if len(sys.argv) > 1 else "mock_repo"))
    print("mock repo ready")
