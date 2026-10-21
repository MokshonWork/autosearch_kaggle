"""Regression tests for the run_ab source-contamination bug.

Bug: run_arm() clones from cfg.paths.repo_dir AFTER a previous arm mutated it
to point at its own output repo — so arm 2 started from arm 1's winning
commit instead of the pristine source. Fix: capture the pristine source once
in main()/run_arm caller and pass it explicitly.

All tests use temp git repos + the mock training fixtures. No GPU/API/net.
"""
import subprocess
from pathlib import Path

import pytest

from src.config import Config
from src.mock_repo import create_mock_git_repo

from evals.run_ab import _fresh_clone, _resolve_source_repo, run_arm


def _git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True, check=True)
    return p.stdout.strip()


def _ab_cfg(tmp_path: Path, source_repo: Path) -> Config:
    cfg = Config()
    cfg.root = tmp_path
    cfg.paths.repo_dir = str(source_repo)
    cfg.paths.work_dir = "work"
    cfg.paths.memory_dir = "memory"
    cfg.paths.stop_file = "STOP"
    cfg.training.time_budget_s = 5
    cfg.training.startup_allowance_s = 30
    cfg.training.dry_run_steps = 3
    cfg.budget.max_rounds = 1
    cfg.budget.max_api_calls_total = 500
    cfg.budget.max_calls_per_round = 30
    cfg.guard.protected_files = ["prepare.py"]
    cfg.mode.dry_run = True
    return cfg


def _capture(cfg):
    """Mirror main(): resolve pristine source BEFORE any arm runs."""
    return _resolve_source_repo(cfg)


def _unique_marker(repo: Path) -> None:
    """Simulate arm 1 producing a unique winning commit in its output."""
    (repo / "ARM1_UNIQUE_MARKER.txt").write_text("arm1 was here")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit",
         "-q", "-m", "arm1 unique commit")


# Test A — both arms originate from the same pristine source revision -------
def test_A_both_arms_share_pristine_source(tmp_path):
    source = create_mock_git_repo(tmp_path / "source")
    pristine = _git(source, "rev-parse", "HEAD")
    out = tmp_path / "out"
    for arm in ("multi_agent", "baseline_single"):
        cfg = _ab_cfg(tmp_path, source)
        src, commit = _capture(cfg)
        run_arm(cfg, arm, 0, out, source_repo=src, start_commit=commit)
        arm_head = _git(out / arm / "run0" / "repo", "rev-parse", "HEAD")
        # arm repo history must contain the pristine commit (cloned from it)
        log = _git(out / arm / "run0" / "repo", "rev-list", "HEAD")
        assert pristine in log.splitlines(), \
            f"{arm} run0 not descended from pristine {pristine[:8]}"
        import json
        meta = json.loads((out / arm / "run0" / "memory" / "source.json"
                           ).read_text())
        assert meta["start_commit"] == pristine
        assert meta["config"]["budget"] == cfg.model_dump(
            mode="json")["budget"]


# Test B — arm-1 mutation cannot contaminate arm 2 ----------------------------
def test_B_first_arm_mutation_does_not_leak(tmp_path):
    source = create_mock_git_repo(tmp_path / "source")
    pristine = _git(source, "rev-parse", "HEAD")
    out = tmp_path / "out"

    cfg1 = _ab_cfg(tmp_path, source)
    src, commit = _capture(cfg1)  # main() captures BEFORE any arm runs
    run_arm(cfg1, "multi_agent", 0, out, source_repo=src,
            start_commit=commit)
    _unique_marker(out / "multi_agent" / "run0" / "repo")
    mutated_log = _git(out / "multi_agent" / "run0" / "repo", "rev-list",
                       "HEAD")
    assert len(mutated_log.splitlines()) > 1  # arm1 really diverged

    # Orchestration-as-normal: reuse the MUTATED cfg object like main() does.
    # (run_arm mutates cfg.paths.repo_dir; the fix must not rely on it.)
    run_arm(cfg1, "baseline_single", 0, out, source_repo=src,
            start_commit=commit)
    arm2 = out / "baseline_single" / "run0" / "repo"
    assert not (arm2 / "ARM1_UNIQUE_MARKER.txt").exists(), \
        "arm1's unique file leaked into arm 2"
    log2 = _git(arm2, "rev-list", "HEAD").splitlines()
    assert pristine in log2
    assert len(log2) < len(mutated_log.splitlines()) or pristine == log2[-1]


# Test C — reversed arm order still isolates -----------------------------------
def test_C_reversed_order_isolates(tmp_path):
    source = create_mock_git_repo(tmp_path / "source")
    pristine = _git(source, "rev-parse", "HEAD")
    out = tmp_path / "out"
    cfg = _ab_cfg(tmp_path, source)
    src, commit = _capture(cfg)
    run_arm(cfg, "baseline_single", 0, out, source_repo=src,
            start_commit=commit)
    _unique_marker(out / "baseline_single" / "run0" / "repo")
    run_arm(cfg, "multi_agent", 0, out, source_repo=src,
            start_commit=commit)
    arm2 = out / "multi_agent" / "run0" / "repo"
    assert not (arm2 / "ARM1_UNIQUE_MARKER.txt").exists()
    assert pristine in _git(arm2, "rev-list", "HEAD").splitlines()


# Test D — source repo untouched ------------------------------------------------
def test_D_source_repo_untouched(tmp_path):
    source = create_mock_git_repo(tmp_path / "source")
    before_head = _git(source, "rev-parse", "HEAD")
    before_status = _git(source, "status", "--porcelain")
    before_train = (source / "train.py").read_bytes()
    out = tmp_path / "out"
    for arm in ("multi_agent", "baseline_single"):
        cfg = _ab_cfg(tmp_path, source)
        src, commit = _capture(cfg)
        run_arm(cfg, arm, 0, out, source_repo=src, start_commit=commit)
    assert _git(source, "rev-parse", "HEAD") == before_head
    assert _git(source, "status", "--porcelain") == before_status
    assert (source / "train.py").read_bytes() == before_train


# Test E — missing/invalid source fails fast -------------------------------------
def test_E_invalid_source_fails_fast(tmp_path):
    out = tmp_path / "out"
    cfg = _ab_cfg(tmp_path, tmp_path / "does-not-exist")
    with pytest.raises(Exception):
        _capture(cfg)  # resolve fails BEFORE any arm starts
    with pytest.raises(TypeError):
        run_arm(cfg, "multi_agent", 0, out)  # source is required
    # no arm output may exist claiming success
    assert not (out / "multi_agent" / "run0" / "memory" / "state.json"
                ).exists()

    (tmp_path / "not-a-repo").mkdir()
    cfg2 = _ab_cfg(tmp_path, tmp_path / "not-a-repo")
    with pytest.raises(Exception):
        _capture(cfg2)


# Test F — independent workspaces ---------------------------------------------------
def test_F_arm_workspaces_independent(tmp_path):
    source = create_mock_git_repo(tmp_path / "source")
    out = tmp_path / "out"
    for arm in ("multi_agent", "baseline_single"):
        cfg = _ab_cfg(tmp_path, source)
        src, commit = _capture(cfg)
        run_arm(cfg, arm, 0, out, source_repo=src, start_commit=commit)
    ma_repo = out / "multi_agent" / "run0" / "repo"
    b1_repo = out / "baseline_single" / "run0" / "repo"
    assert ma_repo.resolve() != b1_repo.resolve()
    (ma_repo / "ONLY_IN_MA.txt").write_text("x")
    assert not (b1_repo / "ONLY_IN_MA.txt").exists()


# Test G — unsafe call without source fails loudly -------------------------------
def test_G_source_required_kwarg(tmp_path):
    source = create_mock_git_repo(tmp_path / "source")
    cfg = _ab_cfg(tmp_path, source)
    with pytest.raises(TypeError):
        run_arm(cfg, "multi_agent", 0, tmp_path / "out")  # type: ignore
