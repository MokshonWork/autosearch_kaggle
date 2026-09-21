"""Runner (no LLM): run train.py in a worker worktree on one GPU.

- G7: the subprocess gets a STRIPPED environment (whitelist only; no secrets).
- G8: optional pre-flight (AUTORESEARCH_PREFLIGHT_STEPS) + hard timeout.
- G10: crash/oom/nan/timeout are recorded, never raised as exceptions.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from ..schemas import RunResult
from . import metrics_parser

# Whitelisted environment for the training subprocess (G7). Nothing else leaks.
_ENV_WHITELIST = (
    "PATH", "HOME", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES",
    "PYTORCH_ALLOC_CONF", "HF_HUB_DISABLE_PROGRESS_BARS", "HF_HOME",
    "AUTORESEARCH_PREFLIGHT_STEPS",  # preflight hook (set by us, not secret)
    "MOCK_VAL_BPB", "MOCK_TRAIN_BEHAVIOR",  # mock-runner knobs (dry-run/tests)
)


def stripped_env(gpu: int, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _ENV_WHITELIST}
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if extra:
        env.update(extra)
    return env


def run_training(
    worktree: Path,
    gpu: int,
    hypothesis_id: str,
    time_budget_s: int,
    startup_allowance_s: int,
    diff_hash: str = "",
    preflight_steps: int = 0,
    extra_env: dict[str, str] | None = None,
) -> RunResult:
    """Run one experiment. Always returns a RunResult (failures are data)."""
    hard_timeout = time_budget_s + startup_allowance_s
    env_extra = dict(extra_env or {})
    if preflight_steps:
        env_extra["AUTORESEARCH_PREFLIGHT_STEPS"] = str(preflight_steps)
        hard_timeout = startup_allowance_s  # preflight should be quick
    env = stripped_env(gpu, env_extra)

    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, "train.py"],
            cwd=Path(worktree), env=env,
            capture_output=True, text=True, timeout=hard_timeout,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
        stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
        rc = -9
    wall = time.time() - t0

    status = metrics_parser.classify_status(stdout, stderr, rc, timed_out)
    metrics = metrics_parser.parse_metrics(stdout)
    return RunResult(
        hypothesis_id=hypothesis_id,
        gpu=gpu,
        status=status,
        val_bpb=metrics.get("val_bpb") if status == "ok" else None,
        peak_vram_mb=metrics.get("peak_vram_mb") or 0.0,
        steps=metrics.get("num_steps") or 0,
        wall_seconds=round(wall, 2),
        stdout_tail=metrics_parser.stdout_tail(stdout + "\n" + stderr),
        commit_or_diff_hash=diff_hash,
    )
