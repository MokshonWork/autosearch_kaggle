"""Noise calibration (Section 6.4, E13): K unchanged runs on BOTH GPUs
concurrently (CPU/dataloader contention matters), then epsilon from the noise.
"""
from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .decision import compute_epsilon
from .schemas import RunResult
from .tools.runner import run_training


@dataclass
class CalibrationReport:
    values: list[float] = field(default_factory=list)
    per_gpu: dict[int, list[float]] = field(default_factory=dict)
    epsilon: float = 0.0
    gpu_mean_diff: float = 0.0
    # Invalid runs are DATA, not noise: keep them so a failed calibration is
    # diagnosable (status, wall time, stdout tail per GPU — G10 / rule "never
    # silently swallow training errors").
    failures: list[RunResult] = field(default_factory=list)

    def failure_summary(self, max_tail_lines: int = 12) -> str:
        if not self.failures:
            return "  (no failed calibration runs recorded)"
        lines = []
        for r in self.failures:
            tail = "\n".join(r.stdout_tail.splitlines()[-max_tail_lines:])
            indented = "\n".join(f"    {ln}" for ln in tail.splitlines())
            lines.append(
                f"  gpu{r.gpu} {r.hypothesis_id}: status={r.status} "
                f"wall={r.wall_seconds:.1f}s steps={r.steps}\n{indented}")
        return "\n".join(lines)

    def summary(self) -> str:
        std = statistics.stdev(self.values) if len(self.values) >= 2 else 0.0
        lines = [
            f"calibration runs: {len(self.values)}",
            f"val_bpb values: {[round(v, 6) for v in self.values]}",
            f"std: {std:.6f}",
            f"epsilon: {self.epsilon:.6f}",
        ]
        for gpu, vals in sorted(self.per_gpu.items()):
            mean = statistics.mean(vals) if vals else float("nan")
            lines.append(f"gpu{gpu}: n={len(vals)} mean={mean:.6f}")
        lines.append(f"gpu mean difference: {self.gpu_mean_diff:.6f}")
        return "\n".join(lines)


def run_calibration(
    worktrees: list[Path],
    k: int,
    time_budget_s: int,
    startup_allowance_s: int,
    min_delta: float,
    runner: Callable[..., RunResult] = run_training,
) -> CalibrationReport:
    """Run the unmodified train.py K times on both GPUs concurrently."""
    report = CalibrationReport()
    with ThreadPoolExecutor(max_workers=len(worktrees)) as ex:
        for i in range(k):
            futures = {
                gpu: ex.submit(runner, wt, gpu, f"CALIB-{i}",
                               time_budget_s, startup_allowance_s)
                for gpu, wt in enumerate(worktrees)
            }
            for gpu, fut in futures.items():
                res = fut.result()
                if res.status == "ok" and res.val_bpb is not None:
                    report.values.append(res.val_bpb)
                    report.per_gpu.setdefault(gpu, []).append(res.val_bpb)
                else:
                    report.failures.append(res)
    report.epsilon = compute_epsilon(report.values, min_delta)
    if len(report.per_gpu) == 2 and all(report.per_gpu.values()):
        means = [statistics.mean(v) for v in report.per_gpu.values()]
        report.gpu_mean_diff = abs(means[0] - means[1])
    return report
