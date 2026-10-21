"""Offline demo + honest failure reporting (no provider, key, HTTP, or GPU)."""
import json
import sys
from pathlib import Path

import pytest

import evals.report as report
import evals.run_ab as run_ab
import src.llm.client as client_mod
from evals.metrics import run_status
from src.config import Config
from src.llm.client import BudgetTracker, LLMError, MockLLMClient
from src.mock_handlers import default_handlers
from src.mock_repo import create_mock_git_repo
from src.schemas import Patch


def _cfg(tmp_path: Path, source: Path) -> Config:
    cfg = Config()
    cfg.root = tmp_path
    cfg.paths.repo_dir = str(source)
    cfg.paths.work_dir = "work"
    cfg.paths.memory_dir = "memory"
    cfg.paths.stop_file = "STOP"
    cfg.training.time_budget_s = 1
    cfg.training.startup_allowance_s = 10
    cfg.training.dry_run_steps = 3
    cfg.decision.calibration_runs = 1
    cfg.budget.max_rounds = 1
    cfg.budget.max_api_calls_total = 500
    cfg.budget.max_calls_per_round = 30
    cfg.guard.protected_files = ["prepare.py"]
    cfg.mode.dry_run = True
    return cfg


def _run(tmp_path: Path):
    source = create_mock_git_repo(tmp_path / "source")
    cfg = _cfg(tmp_path, source)
    src, commit = run_ab._resolve_source_repo(cfg)
    out = tmp_path / "out"
    state = run_ab.run_arm(cfg, "multi_agent", 0, out,
                           source_repo=src, start_commit=commit)
    return cfg, out, state


def test_run_status_distinguishes_completed_partial_failed():
    assert run_status({"experiments_done": 2, "stopped_reason": "max_rounds",
                       "round_status": "completed"}) == "completed"
    assert run_status({"experiments_done": 1,
                       "stopped_reason": "BudgetExhausted",
                       "round_status": "partial"}) == "partial"
    assert run_status({"experiments_done": 0, "stopped_reason": "LLMError",
                       "round_status": "failed"}) == "failed"


def test_mock_proposal_is_deterministic_and_schema_valid():
    cfg = Config()
    prompt = "DEVICE_BATCH_SIZE = 64\n"
    clients = [MockLLMClient(cfg.llm, BudgetTracker(10, 10_000, 10),
                             default_handlers()) for _ in range(2)]
    first, err1 = clients[0].generate_json("coder", prompt, Patch)
    second, err2 = clients[1].generate_json("coder", prompt, Patch)
    assert err1 == err2 == ""
    assert first is not None and second is not None
    assert first == second
    assert first.edits[0].find == "DEVICE_BATCH_SIZE = 64"


def test_dry_run_offline_persists_pipeline_and_labels_synthetic(
        tmp_path, monkeypatch):
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted provider/network access")

    import requests
    monkeypatch.setattr(run_ab, "make_client", forbidden)
    monkeypatch.setattr(client_mod, "GeminiClient", forbidden)
    monkeypatch.setattr(client_mod, "GroqClient", forbidden)
    monkeypatch.setattr(requests, "request", forbidden)
    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(requests, "post", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(report, "_plot", lambda *a, **k: None)

    _, out, state = _run(tmp_path)
    assert run_status(state) == "completed"
    assert state["experiments_done"] == 2
    assert state["stopped_reason"] == "max_rounds"

    mem = out / "multi_agent" / "run0" / "memory"
    assert json.loads((mem / "state.json").read_text())["experiments_done"] == 2
    assert len((mem / "experiments.jsonl").read_text().splitlines()) == 2
    for name in ("source.json", "lessons.md", "registry_changes.json",
                 "registry_tasks.json", "registry_tests.json"):
        assert (mem / name).exists(), name
    run = json.loads((mem / "source.json").read_text())
    assert run["synthetic"] is True
    assert (out / "multi_agent" / "run0" / "work" / "gpu0" /
            "train.py").exists()

    text = report.generate_report(out)
    assert "SYNTHETIC" in text
    assert "not evidence of real model improvement" in text
    assert "run0:completed" in text


def test_llm_error_is_persisted_and_marks_run_failed(tmp_path, monkeypatch):
    handlers = default_handlers()

    def fail(*args, **kwargs):
        raise LLMError("mock Gemini HTTP 429; fallback also failed")

    handlers["ideator"] = fail
    monkeypatch.setattr("src.mock_handlers.default_handlers", lambda: handlers)

    _, out, state = _run(tmp_path)
    assert state["stopped_reason"] == "LLMError"
    assert state["experiments_done"] == 0
    assert run_status(state) == "failed"
    persisted = json.loads(
        (out / "multi_agent" / "run0" / "memory" / "state.json").read_text())
    assert persisted["stopped_reason"] == "LLMError"
    assert persisted["experiments_done"] == 0


def test_budget_exhaustion_is_persisted_and_marks_run_failed(
        tmp_path, monkeypatch):
    source = create_mock_git_repo(tmp_path / "source")
    cfg = _cfg(tmp_path, source)
    cfg.budget.max_api_calls_total = 0
    cfg.budget.max_calls_per_round = 0
    src, commit = run_ab._resolve_source_repo(cfg)
    out = tmp_path / "out"

    state = run_ab.run_arm(cfg, "multi_agent", 0, out,
                           source_repo=src, start_commit=commit)
    assert state["stopped_reason"] == "BudgetExhausted"
    assert state["experiments_done"] == 0
    assert run_status(state) == "failed"
    persisted = json.loads(
        (out / "multi_agent" / "run0" / "memory" / "state.json").read_text())
    assert persisted["stopped_reason"] == "BudgetExhausted"


@pytest.mark.parametrize("state,expected", [
    ({"experiments_done": 2, "stopped_reason": "max_rounds",
      "round_status": "completed", "best_val_bpb": 1.0}, 0),
    ({"experiments_done": 0, "stopped_reason": "LLMError",
      "round_status": "failed", "best_val_bpb": None}, 1),
    ({"experiments_done": 1, "stopped_reason": "BudgetExhausted",
      "round_status": "partial", "best_val_bpb": 1.0}, 1),
])
def test_ab_main_exit_code_reflects_run_status(
        tmp_path, monkeypatch, state, expected):
    cfg = Config()
    monkeypatch.setattr(run_ab, "load_config", lambda _: cfg)
    monkeypatch.setattr(run_ab, "_resolve_source_repo",
                        lambda _: (tmp_path, "abc123"))
    monkeypatch.setattr(run_ab, "run_arm", lambda *a, **k: state)
    monkeypatch.setattr("evals.report.generate_report", lambda _: "report")
    monkeypatch.setattr(sys, "argv", [
        "evals.run_ab", "--config", "x", "--repeats", "1",
        "--arms", "multi_agent", "--out", str(tmp_path / "out")])
    assert run_ab.main() == expected
