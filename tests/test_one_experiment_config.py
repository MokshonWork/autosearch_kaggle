"""One-experiment config bounds and the remaining-experiment planning cap."""
from pathlib import Path

from src.config import load_config
from src.orchestrator import Orchestrator
from tests.conftest import fixed_runner

ROOT = Path(__file__).resolve().parent.parent


def test_one_experiment_config_is_bounded():
    cfg = load_config(ROOT / "config.one_experiment.yaml")
    assert cfg.budget.max_rounds == 1
    assert cfg.budget.max_experiments == 1
    assert cfg.decision.calibration_runs == 1
    assert cfg.budget.max_api_calls_total == 24
    assert cfg.budget.max_calls_per_round == 24
    assert cfg.budget.max_tokens_total == 80_000
    assert cfg.llm.fallback_model == ""
    assert cfg.dedupe.use_llm_judge_for_borderline is False


def test_smoke_config_remains_two_experiment_smoke():
    cfg = load_config(ROOT / "config.smoke.yaml")
    assert cfg.budget.max_rounds == 1
    assert cfg.budget.max_experiments == 2
    assert cfg.budget.max_api_calls_total == 30
    assert cfg.budget.max_tokens_total == 250_000
    assert cfg.decision.calibration_runs == 1


def test_orchestrator_caps_round_at_remaining_experiments(project):
    cfg, client, _ = project
    cfg.budget.max_rounds = 1
    cfg.budget.max_experiments = 1
    cfg.decision.calibration_runs = 1
    runner = fixed_runner()

    orch = Orchestrator(cfg, client, runner_fn=runner)
    state = orch.run()

    assert state["stopped_reason"] == "max_rounds"
    assert state["experiments_done"] == 1
    exps = orch.store.read_experiments()
    assert len(exps) == 1
    assert exps[0]["hypothesis"]["id"].startswith("H-")
    full_candidate_runs = [
        c for c in runner.calls
        if not c["hid"].startswith("CALIB")
        and not c["kw"].get("preflight_steps")]
    assert len(full_candidate_runs) == 1
