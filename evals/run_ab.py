"""Run the A/B evaluation: MA vs B1, R repeats each (Section 9.1).

Each (arm, repeat) gets a FRESH memory dir and a fresh git clone of the
starting commit, so repeats are independent. Controls: same config, same
starting commit, same budgets, same stop rules; only cfg.mode.arm differs.

Usage (on Kaggle):
    python -m evals.run_ab --config config.yaml --repeats 2
Locally (dry-run smoke of the harness wiring only):
    python -m evals.run_ab --config evals/fixtures/dry_config.yaml --repeats 1
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.llm.client import BudgetTracker, GeminiClient, MockLLMClient  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402


def _fresh_clone(repo: Path, commit: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", "-q", str(repo), str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit],
                   check=True)


def run_arm(cfg, arm: str, repeat: int, out_root: Path) -> dict:
    """One full run of one arm; memory lands in out_root/<arm>/run<i>/memory."""
    cfg.mode.arm = arm
    run_dir = out_root / arm / f"run{repeat}"
    cfg.paths.memory_dir = str(run_dir / "memory")
    cfg.paths.work_dir = str(run_dir / "work")
    cfg.paths.stop_file = str(run_dir / "STOP")
    # fresh repo checkout per run so arms start from the same commit
    start_commit = subprocess.run(
        ["git", "-C", str(Path(cfg.root) / cfg.paths.repo_dir),
         "rev-parse", "HEAD"], capture_output=True, text=True,
        check=True).stdout.strip()
    repo_copy = run_dir / "repo"
    _fresh_clone(Path(cfg.root) / cfg.paths.repo_dir, start_commit, repo_copy)
    cfg.paths.repo_dir = str(repo_copy)

    budget = BudgetTracker(cfg.budget.max_api_calls_total,
                           cfg.budget.max_tokens_total,
                           cfg.budget.max_calls_per_round)
    if cfg.mode.dry_run:
        from src.mock_handlers import default_handlers
        client = MockLLMClient(cfg.llm, budget, default_handlers())
    else:
        client = GeminiClient(cfg.llm, budget)
    t0 = time.time()
    orch = Orchestrator(cfg, client)
    state = orch.run()
    state["wall_clock_hours"] = (time.time() - t0) / 3600
    orch.store.save_state(state)  # persist wall clock
    return state


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--arms", default="multi_agent,baseline_single")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.out)
    for arm in args.arms.split(","):
        for r in range(args.repeats):
            print(f"=== arm={arm} repeat={r} ===")
            state = run_arm(cfg, arm, r, out_root)
            print(f"    best={state['best_val_bpb']} "
                  f"stopped={state['stopped_reason']}")
    from evals.report import generate_report
    print(generate_report(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
