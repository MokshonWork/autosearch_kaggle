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
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.llm.client import BudgetTracker, MockLLMClient, make_client  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402


def _fresh_clone(repo: Path, commit: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", "-q", str(repo), str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit],
                   check=True)


def _resolve_source_repo(cfg) -> tuple[Path, str]:
    """Capture + validate the pristine source repo BEFORE any arm runs.

    Returns (source_dir, start_commit). Fails fast on a missing/invalid
    source so no arm can silently fall back to another arm's output.
    """
    source = (Path(cfg.root) / cfg.paths.repo_dir).resolve()
    if not source.is_dir():
        raise RuntimeError(f"A/B source repo does not exist: {source}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        toplevel = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"A/B source is not a valid git repo: {source}"
                           ) from e
    if not commit:
        raise RuntimeError(f"A/B source repo has no HEAD commit: {source}")
    # Fail fast when repo/ has no .git of its own: `git -C repo rev-parse`
    # would silently walk up to the OUTER repository, so arms would clone the
    # wrong tree and every training run would fail with the misleading
    # "calibration produced no valid runs". Source must be its own repo root.
    toplevel = Path(toplevel).resolve()
    if toplevel != source.resolve():
        raise RuntimeError(
            f"A/B source {source} is not its own git repository (git root is "
            f"{toplevel}); nested repo/.git is missing — run: "
            f"git -C {source} init && git -C {source} add -A && "
            f"git -C {source} commit -m 'baseline' (notebook does this in "
            "kaggle_run.ipynb cell 1)")
    return source, commit


def run_arm(cfg, arm: str, repeat: int, out_root: Path, *,
            source_repo: Path, start_commit: str) -> dict:
    """One full run of one arm; memory lands in out_root/<arm>/run<i>/memory.

    `source_repo` + `start_commit` are REQUIRED keyword-only args, captured
    once by the caller before any arm runs. The arm clones from them and
    NEVER from cfg.paths.repo_dir (which this function mutates to point at
    the arm's own copy). There is deliberately no fallback: silently reading
    a possibly-mutated config was the source-contamination bug.
    """
    cfg.mode.arm = arm
    run_dir = out_root / arm / f"run{repeat}"
    cfg.paths.memory_dir = str(run_dir / "memory")
    cfg.paths.work_dir = str(run_dir / "work")
    cfg.paths.stop_file = str(run_dir / "STOP")
    # fresh repo checkout per run so arms start from the same commit
    repo_copy = run_dir / "repo"
    _fresh_clone(source_repo, start_commit, repo_copy)
    try:
        actual = subprocess.run(
            ["git", "-C", str(repo_copy), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"arm {arm} run{repeat}: cloned repo invalid") from e
    if actual != start_commit:
        raise RuntimeError(
            f"arm {arm} run{repeat}: starts at {actual[:8]}, "
            f"expected pristine {start_commit[:8]}")
    state_marker = run_dir / "memory" / "source.json"
    state_marker.parent.mkdir(parents=True, exist_ok=True)
    state_marker.write_text(json.dumps({
        "arm": arm, "repeat": repeat,
        "source_repo": str(source_repo),
        "start_commit": start_commit,
        "actual_commit": actual,
        "synthetic": bool(cfg.mode.dry_run),
    }, indent=2))
    cfg.paths.repo_dir = str(repo_copy)

    budget = BudgetTracker(cfg.budget.max_api_calls_total,
                           cfg.budget.max_tokens_total,
                           cfg.budget.max_calls_per_round)
    if cfg.mode.dry_run:
        from src.mock_handlers import default_handlers
        client = MockLLMClient(cfg.llm, budget, default_handlers())
    else:
        client = make_client(cfg.llm, budget)
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
    # Capture the pristine source ONCE before either arm runs. Every arm
    # clones from this path; no arm may derive its source from the mutable
    # cfg.paths.repo_dir afterwards (source-contamination fix).
    source_repo, start_commit = _resolve_source_repo(cfg)
    print(f"source: {source_repo} @ {start_commit[:12]}")
    for arm in args.arms.split(","):
        for r in range(args.repeats):
            print(f"=== arm={arm} repeat={r} ===")
            state = run_arm(cfg, arm, r, out_root,
                            source_repo=source_repo, start_commit=start_commit)
            print(f"    best={state['best_val_bpb']} "
                  f"stopped={state['stopped_reason']}")
    from evals.report import generate_report
    print(generate_report(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
