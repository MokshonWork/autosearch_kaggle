"""E11 resume (Section 6.6) and E12 budget/stop rules (G13, G16, G17)."""
from pathlib import Path

from src.llm.client import BudgetTracker, MockLLMClient
from src.mock_handlers import default_handlers
from src.orchestrator import Orchestrator
from tests.conftest import fixed_runner


def _client(cfg, calls=500, tokens=100_000):
    return MockLLMClient(cfg.llm, BudgetTracker(calls, tokens, 30),
                         default_handlers())


def test_e11_resume_after_kill(project):
    cfg, client, _ = project
    cfg.budget.max_rounds = 1
    o1 = Orchestrator(cfg, client)
    s1 = o1.run()
    assert s1["round"] == 1
    n1 = len(o1.store.read_experiments())

    # "kill": abandon o1, build a fresh orchestrator on the same memory dir
    cfg.budget.max_rounds = 2
    o2 = Orchestrator(cfg, _client(cfg))
    s2 = o2.run()
    assert s2["round"] == 2
    exps = o2.store.read_experiments()
    assert len(exps) > n1  # continues, does not restart
    ids = [e["hypothesis"]["id"] for e in exps]
    assert len(ids) == len(set(ids))  # no duplicate experiment ids
    # jsonl lines are unique (append-only, no rewrite on resume)
    lines = (Path(cfg.root) / "memory" / "experiments.jsonl"
             ).read_text().splitlines()
    assert len(lines) == len(set(lines))


def test_e12_max_rounds(project):
    cfg, client, _ = project
    cfg.budget.max_rounds = 1
    assert Orchestrator(cfg, client).run()["stopped_reason"] == "max_rounds"


def test_e12_stop_file(project):
    cfg, client, _ = project
    (Path(cfg.root) / "STOP").write_text("")
    assert Orchestrator(cfg, client).run()["stopped_reason"] == "stop_file"


def test_e12_token_budget(project):
    cfg, _, _ = project
    client = _client(cfg, tokens=50)  # tiny budget -> exhausts in round 0
    state = Orchestrator(cfg, client).run()
    assert state["stopped_reason"] == "BudgetExhausted"
    assert (Path(cfg.root) / "memory" / "state.json").exists()  # clean stop


def test_e12_no_improvement_streak(project):
    cfg, client, _ = project
    cfg.budget.no_improvement_patience = 1
    cfg.budget.max_rounds = 10
    # runner that always returns a worse val_bpb -> never improves
    state = Orchestrator(cfg, client,
                         runner_fn=fixed_runner(val_bpb=99.0)).run()
    assert state["stopped_reason"] == "no_improvement"


def test_e12_wall_clock_cap(project):
    cfg, client, _ = project
    cfg.budget.wall_clock_cap_hours = 0.0  # cap already exceeded
    assert Orchestrator(cfg, client).run()["stopped_reason"] == "wall_clock_cap"
