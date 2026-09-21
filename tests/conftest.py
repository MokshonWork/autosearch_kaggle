"""Shared fixtures: mock repo, config, mock LLM, fake runner."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config  # noqa: E402
from src.llm.client import BudgetTracker, MockLLMClient  # noqa: E402
from src.mock_handlers import default_handlers  # noqa: E402
from src.mock_repo import create_mock_git_repo  # noqa: E402
from src.schemas import Hypothesis, RunResult  # noqa: E402

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "mock_repo"


@pytest.fixture()
def mock_train_src() -> str:
    return (FIXTURE_REPO / "train.py").read_text()


@pytest.fixture()
def project(tmp_path: Path):
    """A temp project: mock git repo + config + mock LLM client."""
    repo = create_mock_git_repo(tmp_path / "repo")
    cfg = Config()
    cfg.root = tmp_path
    cfg.paths.repo_dir = "repo"
    cfg.paths.work_dir = "work"
    cfg.paths.memory_dir = "memory"
    cfg.paths.stop_file = "STOP"
    cfg.training.time_budget_s = 5
    cfg.training.startup_allowance_s = 30
    cfg.training.dry_run_steps = 3
    cfg.budget.max_rounds = 3
    cfg.budget.max_api_calls_total = 500
    cfg.budget.max_calls_per_round = 30
    cfg.guard.protected_files = ["prepare.py"]
    budget = BudgetTracker(500, 100_000, 30)
    client = MockLLMClient(cfg.llm, budget, default_handlers())
    return cfg, client, repo


def make_hypothesis(hid: str = "H-0001", category: str = "optimizer",
                    summary: str = "test change") -> Hypothesis:
    return Hypothesis(id=hid, category=category, change_summary=summary,
                      rationale="r", expected_effect="e", risk="low",
                      parent_commit="")


def make_result(hid: str = "H-0001", status: str = "ok",
                val_bpb: float | None = 1.0, gpu: int = 0) -> RunResult:
    return RunResult(hypothesis_id=hid, gpu=gpu, status=status,
                     val_bpb=val_bpb, peak_vram_mb=1000.0, steps=100,
                     wall_seconds=5.0, stdout_tail="tail",
                     commit_or_diff_hash="abc123")


def fixed_runner(val_bpb: float = 1.0, status: str = "ok"):
    """A runner stub returning a fixed result; records calls."""
    calls: list[dict] = []

    def _run(wt, gpu, hid, budget_s, allowance_s, **kw):
        calls.append({"gpu": gpu, "hid": hid, "kw": kw})
        if kw.get("preflight_steps"):
            return RunResult(hypothesis_id=hid, gpu=gpu, status="ok",
                             stdout_tail="PREFLIGHT_OK")
        return RunResult(hypothesis_id=hid, gpu=gpu, status=status,
                         val_bpb=val_bpb if status == "ok" else None,
                         peak_vram_mb=1000.0, steps=10, wall_seconds=1.0,
                         stdout_tail="ok", commit_or_diff_hash="h")
    _run.calls = calls
    return _run
