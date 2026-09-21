"""Integration: full MA + baseline loops (mocks), combine queue, retest (E8)."""
from pathlib import Path

import pytest

from src.llm.client import BudgetTracker, MockLLMClient
from src.mock_handlers import default_handlers
from src.orchestrator import Orchestrator
from src.schemas import RunResult


def _client(cfg):
    return MockLLMClient(cfg.llm, BudgetTracker(500, 100_000, 30),
                         default_handlers())


def test_integration_ma_two_rounds(project):
    cfg, client, _ = project
    cfg.budget.max_rounds = 2
    state = Orchestrator(cfg, client).run()
    assert state["stopped_reason"] == "max_rounds"
    assert state["epsilon"] is not None
    exps = (Path(cfg.root) / "memory" / "experiments.jsonl").exists()
    assert exps
    store_exps = Orchestrator(cfg, _client(cfg)).store.read_experiments()
    assert len(store_exps) >= 2
    for e in store_exps:
        assert e["result"]["status"] in (
            "ok", "crash", "oom", "nan", "timeout", "guard_reject")
        assert e["decision"]["action"] in ("keep", "discard", "retest")


def test_integration_baseline_arm(project):
    cfg, client, _ = project
    cfg.mode.arm = "baseline_single"
    cfg.budget.max_rounds = 1
    state = Orchestrator(cfg, client).run()
    exps = Orchestrator(cfg, _client(cfg)).store.read_experiments()
    assert state["stopped_reason"] == "max_rounds"
    assert len(exps) == 2  # one per GPU worker
    assert all(e["guard"] is None for e in exps)  # no Guard in baseline arm


def _scripted_runner(calib_vals, run_vals):
    calls = []

    def _run(wt, gpu, hid, budget_s, allowance_s, **kw):
        calls.append((gpu, hid, kw.get("preflight_steps", 0)))
        if kw.get("preflight_steps"):
            return RunResult(hypothesis_id=hid, gpu=gpu, status="ok",
                             stdout_tail="PREFLIGHT_OK")
        if hid.startswith("CALIB"):
            val = calib_vals[len([c for c in calls if c[1].startswith("CALIB")
                                  and not c[2]]) % len(calib_vals)]
        else:
            val = run_vals[gpu]
        return RunResult(hypothesis_id=hid, gpu=gpu, status="ok", val_bpb=val,
                         peak_vram_mb=100.0, steps=10, wall_seconds=1.0,
                         stdout_tail="ok", commit_or_diff_hash="h")
    return _run, calls


def test_both_improved_combine_queue(project):
    cfg, client, _ = project
    cfg.budget.max_rounds = 1
    cfg.decision.confirm_marginal_wins = False
    # calibration 1.5 -> plausibility floor 1.05; both wins must stay above it
    runner, calls = _scripted_runner([1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                                     {0: 1.10, 1: 1.11})
    state = Orchestrator(cfg, client, runner_fn=runner).run()
    assert state["best_val_bpb"] == 1.10  # better of the two kept
    assert len(state["combine_queue"]) == 1  # loser queued for next round
    exps = Orchestrator(cfg, _client(cfg)).store.read_experiments()
    actions = {e["hypothesis"]["id"]: e["decision"]["action"] for e in exps}
    assert sorted(actions.values()) == ["discard", "keep"]


def test_marginal_win_triggers_retest(project):
    cfg, client, _ = project
    cfg.budget.max_rounds = 1
    cfg.decision.confirm_marginal_wins = True
    # calibration std 0.005 -> epsilon = 0.01; win of 0.025 is in [eps, 3eps)
    runner, calls = _scripted_runner(
        [1.495, 1.5, 1.505, 1.495, 1.5, 1.505], {0: 1.475, 1: 1.60})
    state = Orchestrator(cfg, client, runner_fn=runner).run()
    import statistics
    expected_eps = 2 * statistics.stdev([1.495, 1.5, 1.505] * 2)
    assert state["epsilon"] == pytest.approx(expected_eps)
    gpu0_full = [c for c in calls if c[0] == 0 and not c[2]
                 and not c[1].startswith("CALIB")]
    assert len(gpu0_full) == 2  # original run + confirmation re-run
    assert state["best_val_bpb"] == 1.475  # marginal win confirmed by mean
