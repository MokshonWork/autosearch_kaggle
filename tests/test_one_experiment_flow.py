"""One-experiment flow: guard/crash retries, confirmation persistence, repair ban."""
import statistics
from pathlib import Path

import pytest

from src.mock_handlers import default_handlers
from src.orchestrator import Orchestrator
from src.schemas import RunResult
from tests.conftest import fixed_runner


def _calibration_ok_crash_runner():
    calls = []

    def _run(_wt, gpu, hid, _budget_s, _allowance_s, **kw):
        calls.append({"gpu": gpu, "hid": hid, "kw": kw})
        if kw.get("preflight_steps"):
            return RunResult(hypothesis_id=hid, gpu=gpu, status="ok",
                             stdout_tail="PREFLIGHT_OK")
        crash = not hid.startswith("CALIB")
        return RunResult(hypothesis_id=hid, gpu=gpu,
                         status="crash" if crash else "ok",
                         val_bpb=None if crash else 1.0, stdout_tail="ok")
    _run.calls = calls
    return _run


def _run_one(project, runner, *, max_rounds, guard=None, coder=None):
    cfg, client, _ = project
    cfg.budget.max_rounds = max_rounds
    cfg.budget.max_experiments = 1
    cfg.decision.calibration_runs = 1
    cfg.decision.confirm_marginal_wins = True
    if guard is not None:
        client.handlers["guard"] = guard
    if coder is not None:
        client.handlers["coder"] = coder
    orch = Orchestrator(cfg, client, runner_fn=runner, log=lambda _m: None)
    return orch.run(), orch, client


def _candidate_full(calls):
    return [c for c in calls
            if not c["hid"].startswith("CALIB")
            and not c["kw"].get("preflight_steps")]


def test_guard_reject_persists_one_attempt_and_stops(project):
    runner = fixed_runner()

    def reject(_agent, _prompt, _n):
        return {"verdict": "reject", "reasons": ["mock reject"]}

    state, orch, client = _run_one(
        project, runner, max_rounds=5, guard=reject)
    exps = orch.store.read_experiments()

    assert state["experiments_done"] == 1
    assert state["round"] == 1
    assert state["stopped_reason"] == "max_experiments"
    assert len(exps) == 1
    assert exps[0]["result"]["status"] == "guard_reject"
    assert client.call_counts["ideator"] == 1
    assert client.call_counts["coder"] == 2
    assert _candidate_full(runner.calls) == []


def test_crash_repairs_stay_within_one_experiment(project):
    runner = _calibration_ok_crash_runner()
    state, orch, client = _run_one(project, runner, max_rounds=1)
    exps = orch.store.read_experiments()

    assert state["experiments_done"] == 1
    assert len(exps) == 1
    assert exps[0]["result"]["status"] == "crash"
    assert client.call_counts["coder"] == 3
    assert client.call_counts["guard"] == 1
    assert len(_candidate_full(runner.calls)) == 3


def test_guard_retry_uses_one_experiment(project):
    runner = fixed_runner()

    def reject_then_approve(_agent, _prompt, n):
        return ({"verdict": "reject", "reasons": ["mock reject"]} if n == 0
                else {"verdict": "approve", "reasons": ["mock approve"]})

    state, orch, client = _run_one(
        project, runner, max_rounds=1, guard=reject_then_approve)
    exps = orch.store.read_experiments()

    assert state["experiments_done"] == 1
    assert len(exps) == 1
    assert exps[0]["result"]["status"] == "ok"
    assert client.call_counts["coder"] == 2
    assert client.call_counts["guard"] == 2
    assert len(_candidate_full(runner.calls)) == 1


def test_marginal_confirmation_persists_aggregate(project):
    seq = {"candidate": 0}

    def runner(_wt, gpu, hid, _budget_s, _allowance_s, **kw):
        if kw.get("preflight_steps"):
            return RunResult(hypothesis_id=hid, gpu=gpu, status="ok",
                             stdout_tail="PREFLIGHT_OK")
        if hid.startswith("CALIB"):
            val = 1.50 if gpu == 0 else 1.51
        else:
            val = [1.48, 1.47][seq["candidate"]]
            seq["candidate"] += 1
        return RunResult(hypothesis_id=hid, gpu=gpu, status="ok",
                         val_bpb=val, stdout_tail="ok")

    state, orch, _ = _run_one(project, runner, max_rounds=3)
    exps = orch.store.read_experiments()
    expected_eps = 2 * statistics.stdev([1.50, 1.51])

    assert state["epsilon"] == pytest.approx(expected_eps)
    assert state["experiments_done"] == 1
    assert state["round"] == 1
    assert state["stopped_reason"] == "max_experiments"
    assert seq["candidate"] == 2
    assert len(exps) == 1
    exp = exps[0]
    assert exp["result"]["val_bpb"] == 1.48
    assert exp["confirmation_result"]["val_bpb"] == 1.47
    assert exp["aggregate_val_bpb"] == pytest.approx(1.475)
    assert exp["decision"]["action"] == "keep"
    assert exp["decision"]["reason_code"] == "keep_marginal_confirmed"
    assert state["best_val_bpb"] == pytest.approx(1.475)
    assert orch.registry.get_baseline()["val_bpb"] == pytest.approx(1.475)


@pytest.mark.parametrize("find,replace,reason", [
    ("torch.manual_seed(42)  # PROTECTED", "", "torch.manual_seed(42)"),
    ("DEVICE_BATCH_SIZE = 48", "import socket\nDEVICE_BATCH_SIZE = 48",
     "new import of forbidden module 'socket'"),
    ("DEVICE_BATCH_SIZE = 48",
     "\n".join([f"# pad {i}" for i in range(61)]
               + ["DEVICE_BATCH_SIZE = 48"]), "limit 60"),
])
def test_rejected_crash_repair_never_reaches_runner(
        project, find, replace, reason):
    runner = _calibration_ok_crash_runner()

    def coder(agent, prompt, n):
        if n == 0:
            return default_handlers()["coder"](agent, prompt, n)
        return {"hypothesis_id": "H-0000", "edits": [{
            "find": find, "replace": replace}]}

    state, orch, client = _run_one(
        project, runner, max_rounds=1, coder=coder)
    exps = orch.store.read_experiments()
    cfg = project[0]
    train = Path(cfg.root) / cfg.paths.work_dir / "gpu0" / "train.py"
    text = train.read_text()

    assert state["experiments_done"] == 1
    assert len(exps) == 1
    assert exps[0]["result"]["status"] == "crash"
    tail = exps[0]["result"]["stdout_tail"]
    assert "repair patch rejected" in tail
    assert reason in tail
    assert "torch.manual_seed(42)" in text
    assert "DEVICE_BATCH_SIZE = 48" in text
    assert "# pad 0" not in text
    assert "import socket" not in text
    assert client.call_counts["coder"] == 2
    assert client.call_counts["guard"] == 1
    assert len(_candidate_full(runner.calls)) == 1
