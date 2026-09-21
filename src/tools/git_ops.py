"""Git helpers: worktrees for the 2 GPU workers, keep/discard commits.

- repo_dir holds the "main" experiment branch (accepted changes committed here).
- Each worker gets an isolated worktree (work/gpu0, work/gpu1) reset to the
  current best commit at the start of every round. Workers never share a dir.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _git(path: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def head_commit(path: Path) -> str:
    return _git(path, "rev-parse", "HEAD")


def ensure_branch(repo: Path, branch: str) -> None:
    """Create (or reset to) an experiment branch; idempotent."""
    existing = _git(repo, "branch", "--list", branch)
    if not existing:
        _git(repo, "checkout", "-b", branch)
    else:
        _git(repo, "checkout", branch)


def ensure_worktree(repo: Path, wt_path: Path, commit: str) -> None:
    """Create worktree at wt_path if missing, then hard-reset it to commit."""
    wt_path = Path(wt_path)
    if not (wt_path / ".git").exists():
        wt_path.parent.mkdir(parents=True, exist_ok=True)
        _git(repo, "worktree", "add", "--detach", str(wt_path), commit)
    reset_worktree(wt_path, commit)


def reset_worktree(wt_path: Path, commit: str) -> None:
    _git(wt_path, "reset", "--hard", commit)
    _git(wt_path, "clean", "-fdq")


def commit_train_py(repo: Path, message: str) -> str:
    """Commit the current train.py in the main checkout. Returns commit hash."""
    _git(repo, "add", "train.py")
    _git(repo, "-c", "user.email=team@local", "-c", "user.name=autoresearch-team",
         "commit", "-q", "-m", message)
    return head_commit(repo)


def diff_worktree(wt_path: Path) -> str:
    """Unified diff of uncommitted changes in a worker worktree."""
    return _git(wt_path, "diff")
