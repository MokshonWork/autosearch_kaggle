"""Phase 2 regression tests: calibration failure diagnostics (bugs A/B/C/D).

Bug A: stripped_env dropped LD_LIBRARY_PATH -> no CUDA in the child on Kaggle
       -> every calibration run crashed ("no valid runs").
Bug B: training.time_budget_s never reached train.py -> bounded runs hit the
       runner hard timeout before the fixed 300s TIME_BUDGET elapsed.
Bug C: run_calibration discarded invalid runs -> the RuntimeError carried no
       status/stdout/exit diagnostics.
Bug D: _resolve_source_repo accepted a repo/ without its own .git (git walked
       up to the outer repo) -> arms cloned the wrong tree.

No GPU, no network, no API key: subprocess fakes + the mock train.py fixture.
"""
from __future__ import annotations

import math
import subprocess
import types
from pathlib import Path

import pytest

from src.calibration import run_calibration
from src.config import Config
from src.orchestrator import Orchestrator
from src.schemas import RunResult
from src.tools import runner as runner_mod
from src.tools.runner import run_training, stripped_env
from evals.run_ab import _resolve_source_repo
from tests.conftest import FIXTURE_REPO

VALID_LOG = (
    "---\n"
    "val_bpb:          0.997900\n"
    "training_seconds: 300.1\n"
    "total_seconds:    360.4\n"
    "peak_vram_mb:     45060.2\n"
    "mfu_percent:      12.34\n"
    "total_tokens_M:   12.3\n"
    "num_steps:        953\n"
    "num_params_M:     7.3\n"
    "depth:            4\n"
)


# ---------------------------------------------------------------- bug A ----
def test_stripped_env_forwards_ld_library_path(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/fake/cuda/lib64")
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    env = stripped_env(gpu=1)
    assert env["LD_LIBRARY_PATH"] == "/fake/cuda/lib64"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "GEMINI_API_KEY" not in env  # G7: secrets never forwarded


# --------------------------------------------------------- bugs A + B ----
def test_run_training_env_contract_and_valid_log_parsing(monkeypatch):
    """The child env must carry LD_LIBRARY_PATH + the time-budget override,
    must NOT carry secrets, and a valid summary log must parse to val_bpb."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/fake/cuda/lib64")
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    captured: dict = {}

    def fake_run(cmd, cwd=None, env=None, capture_output=None, text=None,
                 timeout=None):
        captured.update(env=env, cwd=cwd, timeout=timeout)
        return types.SimpleNamespace(stdout=VALID_LOG, stderr="", returncode=0)

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)
    res = run_training(Path("/tmp/wt"), gpu=1, hypothesis_id="H-0001",
                       time_budget_s=77, startup_allowance_s=10)

    env = captured["env"]
    assert env["LD_LIBRARY_PATH"] == "/fake/cuda/lib64"
    assert env["AUTORESEARCH_TIME_BUDGET"] == "77"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "GEMINI_API_KEY" not in env
    assert captured["timeout"] == 87  # 77 + 10
    assert res.status == "ok"
    assert res.val_bpb == pytest.approx(0.9979)
    assert res.steps == 953


def test_time_budget_env_hook_present_in_train_py():
    """Regression marker: the AUTORESEARCH_TIME_BUDGET hook (bug B) must stay,
    and the PROTECTED stop condition must stay verbatim."""
    src = (Path(__file__).parent.parent / "repo" / "train.py").read_text()
    assert 'os.environ.get("AUTORESEARCH_TIME_BUDGET"' in src
    assert ("if step > 10 and total_training_time >= TIME_BUDGET:"
            "  # PROTECTED") in src


# ---------------------------------------------------------------- bug C ----
def test_calibration_valid_fixture_run_produces_val_bpb():
    """Real subprocess against the mock train.py: a valid log yields a finite
    val_bpb and no failures are recorded."""
    report = run_calibration([FIXTURE_REPO], k=1, time_budget_s=5,
                             startup_allowance_s=30, min_delta=0.002)
    assert report.values, "expected at least one valid calibration run"
    assert all(math.isfinite(v) and v > 0 for v in report.values)
    assert report.failures == []


def _crash_runner(wt, gpu, hid, budget_s, allowance_s, **kw):
    return RunResult(hypothesis_id=hid, gpu=gpu, status="crash",
                     val_bpb=None, peak_vram_mb=0.0, steps=0,
                     wall_seconds=3.2,
                     stdout_tail="step 00004 ...\nRuntimeError: boom",
                     commit_or_diff_hash="")


def test_calibration_failures_retained_with_diagnostics(tmp_path):
    report = run_calibration([tmp_path / "a", tmp_path / "b"], k=1,
                             time_budget_s=5, startup_allowance_s=30,
                             min_delta=0.002, runner=_crash_runner)
    assert report.values == []
    assert len(report.failures) == 2
    summary = report.failure_summary()
    assert "gpu0 CALIB-0: status=crash wall=3.2s" in summary
    assert "gpu1 CALIB-0: status=crash" in summary
    assert "RuntimeError: boom" in summary  # tail kept for diagnosis


def test_orchestrator_calibration_error_includes_diagnostics(project):
    cfg, client, repo = project
    logs: list[str] = []
    orch = Orchestrator(cfg, client, runner_fn=_crash_runner, log=logs.append)
    with pytest.raises(RuntimeError) as exc:
        orch._calibrate_if_needed()
    msg = str(exc.value)
    assert "calibration produced no valid runs" in msg
    assert "status=crash" in msg
    assert "RuntimeError: boom" in msg


def test_orchestrator_calibration_missing_train_py_is_actionable(project):
    """A worktree without train.py (wrong source repo) fails fast with a
    pointer to the fix instead of N mystery crashes."""
    cfg, client, repo = project
    (repo / "train.py").rename(repo / "renamed.py")  # sabotage the source...
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-q", "-m", "drop train"],
                   check=True)  # ...in the commit worktrees check out
    orch = Orchestrator(cfg, client, runner_fn=_crash_runner,
                        log=lambda s: None)
    with pytest.raises(RuntimeError, match="has no train.py"):
        orch._calibrate_if_needed()


# ---------------------------------------------------------------- bug D ----
def test_resolve_source_repo_rejects_nested_dir_without_own_git(tmp_path):
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    nested = outer / "repo"
    nested.mkdir()
    (nested / "train.py").write_text("# training repo stand-in\n")
    subprocess.run(["git", "-C", str(outer), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(outer), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-q", "-m", "outer"],
                   check=True)

    cfg = Config()
    cfg.root = tmp_path
    cfg.paths.repo_dir = "outer/repo"
    with pytest.raises(RuntimeError, match="not its own git repository"):
        _resolve_source_repo(cfg)

