"""E11 resume (Section 6.6) and E12 budget/stop rules (G13, G16, G17)."""
import json
from pathlib import Path

from src.llm.client import BudgetTracker, MockLLMClient
from src.mock_handlers import default_handlers
from src.orchestrator import Orchestrator
from tests.conftest import fixed_runner


def _client(cfg, calls=500, tokens=100_000, per_round=30):
    return MockLLMClient(cfg.llm, BudgetTracker(calls, tokens, per_round),
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


def test_resume_cannot_bypass_api_budget(project):
    """A resumed process must inherit usage from state.json, not start at 0."""
    cfg, client, _ = project
    cfg.budget.max_rounds = 1
    first = Orchestrator(cfg, client).run()
    used = first["api_calls"]
    assert used > 0

    cfg.budget.max_rounds = 2
    resumed_client = MockLLMClient(
        cfg.llm, BudgetTracker(used, 100_000, 30), default_handlers())
    orch = Orchestrator(cfg, resumed_client)
    assert orch.budget.requests_total == used

    state = orch.run()
    assert state["stopped_reason"] == "BudgetExhausted"
    assert state["api_calls"] == used  # rejected before any new dispatch


def test_resume_ignores_missing_or_malformed_budget_fields(project):
    cfg, client, _ = project
    cfg.budget.max_rounds = 1
    Orchestrator(cfg, client).run()
    path = Path(cfg.root) / "memory" / "state.json"
    data = json.loads(path.read_text())
    data.pop("api_calls", None)
    data.pop("api_calls_this_round", None)
    data.pop("api_round", None)
    data["api_tokens"] = "legacy-garbage"
    path.write_text(json.dumps(data))

    orch = Orchestrator(cfg, _client(cfg))
    assert orch.budget.requests_total == 0
    assert orch.budget.requests_this_round == 0
    assert orch.budget.tokens_total == 0


def test_resume_restores_per_round_usage_for_same_round(project):
    cfg, _, _ = project
    Path(cfg.root, "memory").mkdir(exist_ok=True)
    state = {"round": 0, "next_hyp_id": 1, "api_calls": 5,
             "api_calls_this_round": 2, "api_round": 0, "api_tokens": 70}
    (Path(cfg.root) / "memory" / "state.json").write_text(json.dumps(state))

    orch = Orchestrator(cfg, _client(cfg, calls=10, per_round=3))
    assert orch.budget.requests_total == 5
    assert orch.budget.requests_this_round == 2
    assert orch.budget.tokens_total == 70

    orch.budget.begin_round(0)  # same round: preserve usage
    assert orch.budget.requests_this_round == 2
    orch.budget.begin_round(1)  # next round: reset only per-round
    assert orch.budget.requests_this_round == 0
    assert orch.budget.requests_total == 5


def test_budget_is_persisted_mid_round_for_resume(project):
    cfg, client, _ = project
    orch = Orchestrator(cfg, client)
    orch.budget.begin_round(0)
    client._raw_generate("ideator", "prompt")

    saved = orch.store.load_state()
    assert saved["api_calls"] == 1
    assert saved["api_calls_this_round"] == 1
    assert saved["api_round"] == 0

    resumed = Orchestrator(cfg, _client(cfg))
    assert resumed.budget.requests_total == 1
    assert resumed.budget.requests_this_round == 1
    assert resumed.budget.round_index == 0
