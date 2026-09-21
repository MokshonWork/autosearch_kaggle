"""Unit tests: runner against the mock train.py (all statuses), git ops, store."""
import os

import pytest

from src.memory.store import MemoryStore
from src.mock_repo import create_mock_git_repo
from src.tools import git_ops
from src.tools.runner import run_training
from tests.conftest import FIXTURE_REPO


def _run(behavior="ok", val=None, budget=5, allowance=30):
    extra = {"MOCK_TRAIN_BEHAVIOR": behavior}
    if val is not None:
        extra["MOCK_VAL_BPB"] = str(val)
    return run_training(FIXTURE_REPO, gpu=0, hypothesis_id="H-0001",
                        time_budget_s=budget, startup_allowance_s=allowance,
                        extra_env=extra)


def test_runner_ok():
    r = _run(val=1.05)
    assert r.status == "ok" and r.val_bpb == pytest.approx(1.05)
    assert r.peak_vram_mb > 0 and r.steps > 0
    assert len(r.stdout_tail.splitlines()) <= 60


def test_runner_crash_oom_nan():
    assert _run("crash").status == "crash"
    assert _run("oom").status == "oom"
    assert _run("nan").status == "nan"


def test_runner_timeout():
    r = _run("timeout", budget=1, allowance=2)
    assert r.status == "timeout"


def test_runner_preflight():
    r = run_training(FIXTURE_REPO, gpu=1, hypothesis_id="H-0002",
                     time_budget_s=5, startup_allowance_s=30,
                     preflight_steps=3)
    assert r.status == "ok" and "PREFLIGHT_OK" in r.stdout_tail


def test_git_worktree_flow(tmp_path):
    repo = create_mock_git_repo(tmp_path / "repo")
    wt = tmp_path / "work" / "gpu0"
    commit = git_ops.head_commit(repo)
    git_ops.ensure_worktree(repo, wt, commit)
    assert (wt / "train.py").exists()
    # simulate worker edit + reset (discard path)
    (wt / "train.py").write_text("garbage")
    git_ops.reset_worktree(wt, commit)
    assert "garbage" not in (wt / "train.py").read_text()
    # keep path: edit main checkout and commit
    (repo / "train.py").write_text((repo / "train.py").read_text() + "\n# x\n")
    new_commit = git_ops.commit_train_py(repo, "test keep")
    assert new_commit != commit
    git_ops.ensure_worktree(repo, wt, new_commit)
    assert "# x" in (wt / "train.py").read_text()


def test_memory_store(tmp_path):
    store = MemoryStore(tmp_path / "mem")
    assert store.load_state() is None
    store.save_state({"round": 1})
    assert store.load_state()["round"] == 1
    store.append_lesson("H-0001 did a thing, val_bpb=1.0.")
    store.append_lesson("H-0002 did another.")
    assert len(store.read_lessons()) == 2
