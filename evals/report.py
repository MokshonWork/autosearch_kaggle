"""Generate results/ab_report.md + plots from finished A/B runs (Section 9.1).

Input layout: results/<arm>/run<i>/memory/{experiments.jsonl,state.json}
Numbers come from logs via code. The optional Librarian narrative appendix is
NOT wired here or in run_ab.py (see docs/READINESS.md "Librarian A/B status").
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.metrics import load_run, run_metrics  # noqa: E402

METRIC_KEYS = ["n_experiments", "best_val_bpb", "improvement",
               "experiments_to_first_improvement", "keep_rate",
               "failure_rate", "guard_reject_rate", "duplicate_rate",
               "wasted_gpu_minutes", "api_calls", "api_tokens"]


def _mean_std(values: list[float]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


def verdict(ma: list[dict], b1: list[dict]) -> str:
    """Honest verdict using each arm's best val_bpb and the noise epsilon."""
    ma_best = [m["best_val_bpb"] for m in ma if m["best_val_bpb"] is not None]
    b1_best = [m["best_val_bpb"] for m in b1 if m["best_val_bpb"] is not None]
    eps = max([m["epsilon"] or 0 for m in ma + b1] or [0])
    if not ma_best or not b1_best:
        return "INSUFFICIENT DATA"
    diff = statistics.mean(b1_best) - statistics.mean(ma_best)
    if abs(diff) <= eps:
        return f"WITHIN NOISE (mean diff {diff:+.4f} <= epsilon {eps:.4f})"
    return (f"MULTI-AGENT BETTER (mean diff {diff:+.4f} > epsilon {eps:.4f})"
            if diff > 0 else
            f"BASELINE BETTER (mean diff {diff:+.4f}, epsilon {eps:.4f})")


def generate_report(results_dir: Path) -> str:
    results_dir = Path(results_dir)
    arms: dict[str, list[dict]] = {}
    for arm_dir in sorted(results_dir.iterdir()):
        if not arm_dir.is_dir():
            continue
        runs = [run_metrics(load_run(d / "memory"))
                for d in sorted(arm_dir.iterdir()) if (d / "memory").exists()]
        if runs:
            arms[arm_dir.name] = runs
    if not arms:
        return "no runs found"

    lines = ["# A/B report: multi-agent vs single-agent baseline", ""]
    lines.append("| metric | " + " | ".join(arms) + " |")
    lines.append("|" + "---|" * (len(arms) + 1))
    for key in METRIC_KEYS:
        lines.append(f"| {key} | " + " | ".join(
            _mean_std([m[key] for m in runs]) for runs in arms.values()) + " |")
    lines.append("")
    if "multi_agent" in arms and "baseline_single" in arms:
        lines.append("**Verdict:** "
                     + verdict(arms["multi_agent"], arms["baseline_single"]))
    out = results_dir / "ab_report.md"
    out.write_text("\n".join(lines) + "\n")
    _plot(results_dir, arms)
    return "\n".join(lines)


def _plot(results_dir: Path, arms: dict[str, list[dict]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, runs in arms.items():
        for i, m in enumerate(runs):
            ax.plot(m["curve"], alpha=0.4, label=f"{arm} run{i}")
    ax.set_xlabel("experiment #")
    ax.set_ylabel("best val_bpb so far")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "val_bpb_curves.png", dpi=120)


if __name__ == "__main__":
    print(generate_report(Path(sys.argv[1] if len(sys.argv) > 1
                                 else "results")))
