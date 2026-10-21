"""Generate results/ab_report.md + plots from finished A/B runs (Section 9.1).

Input layout: results/<arm>/run<i>/memory/{experiments.jsonl,state.json}
Numbers come from logs via code. The optional Librarian narrative appendix is
NOT wired here or in run_ab.py (see docs/READINESS.md "Librarian A/B status").
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.metrics import load_run, run_metrics, run_status  # noqa: E402

METRIC_KEYS = ["n_experiments", "best_val_bpb", "improvement",
               "experiments_to_first_improvement", "keep_rate",
               "failure_rate", "guard_reject_rate", "duplicate_rate",
               "wasted_gpu_minutes", "api_calls", "api_tokens",
               "load_errors"]


def _mean_std(values: list[float]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


def verdict(ma: list[dict], b1: list[dict]) -> str:
    """Honest verdict using each arm's best val_bpb and the noise epsilon."""
    if len(ma) < 2 or len(b1) < 2:
        return "INSUFFICIENT DATA (need >=2 runs per arm)"
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
    sources: dict[str, list[dict]] = {}
    statuses: dict[str, list[str]] = {}
    for arm_dir in sorted(results_dir.iterdir()):
        if not arm_dir.is_dir():
            continue
        run_dirs = [d for d in sorted(arm_dir.iterdir())
                    if (d / "memory").exists()]
        loaded = [load_run(d / "memory") for d in run_dirs]
        runs = [run_metrics(run) for run in loaded]
        if runs:
            arms[arm_dir.name] = runs
            sources[arm_dir.name] = [_source(d) for d in run_dirs]
            statuses[arm_dir.name] = [
                run_status(run["state"], m["n_experiments"])
                for run, m in zip(loaded, runs)]
    if not arms:
        return "no runs found"

    lines = ["# A/B report: multi-agent vs single-agent baseline", ""]
    lines.append("| metric | " + " | ".join(arms) + " |")
    lines.append("|" + "---|" * (len(arms) + 1))
    for key in METRIC_KEYS:
        lines.append(f"| {key} | " + " | ".join(
            _mean_std([m[key] for m in runs]) for runs in arms.values()) + " |")
    lines.append("")
    all_sources = [s for group in sources.values() for s in group]
    modes = []
    for source in all_sources:
        cfg = source.get("config")
        modes.append(_is_synthetic(source, cfg if isinstance(cfg, dict)
                                   else {}))
    if modes and all(modes):
        lines.append("**Execution mode:** SYNTHETIC — mock LLM and mock "
                     "training; scores are not evidence of real model "
                     "improvement.")
    elif any(modes):
        lines.append("**Execution mode:** MIXED synthetic/real runs.")
    lines.append("**Run status:** " + "; ".join(
        f"{arm} " + ", ".join(f"run{i}:{status}"
                              for i, status in enumerate(group))
        for arm, group in statuses.items()))
    lines.append("")
    if "multi_agent" in arms and "baseline_single" in arms:
        ma, b1 = arms["multi_agent"], arms["baseline_single"]
        all_sources = [s for group in sources.values() for s in group]
        starts = [s.get("start_commit") for s in all_sources]
        signatures = [_material_signature(s) for s in all_sources]
        if len(ma) < 2 or len(b1) < 2:
            lines.append("**Verdict:** INSUFFICIENT DATA "
                         "(need >=2 runs per arm)")
        elif (not all_sources or any(not s for s in starts)
              or any(sig is None for sig in signatures)):
            lines.append("**Verdict:** NOT COMPARABLE "
                         "(missing or incomplete run provenance)")
        elif len(set(starts)) > 1:
            lines.append("**Verdict:** NOT COMPARABLE "
                         "(arms started from different commits)")
        elif len(set(signatures)) > 1:
            lines.append("**Verdict:** NOT COMPARABLE "
                         "(material config or execution mode differs "
                         "between runs)")
        elif any(status != "completed"
                 for group in statuses.values() for status in group):
            lines.append("**Verdict:** INCOMPLETE "
                         "(failed/partial runs are not evidence)")
        else:
            lines.append("**Verdict:** " + verdict(ma, b1))
    out = results_dir / "ab_report.md"
    out.write_text("\n".join(lines) + "\n")
    _plot(results_dir, arms)
    return "\n".join(lines)


_MATERIAL_KEYS = ("llm", "budget", "training", "decision", "guard", "dedupe")


def _is_synthetic(source: dict, cfg: dict) -> bool:
    """Execution mode from source provenance, with config.mode fallback."""
    synthetic = source.get("synthetic")
    if isinstance(synthetic, bool):
        return synthetic
    mode = cfg.get("mode")
    if isinstance(mode, dict) and isinstance(mode.get("dry_run"), bool):
        return mode["dry_run"]
    return False


def _material_signature(source: dict) -> str | None:
    """Canonical signature of settings that must match across compared runs."""
    cfg = source.get("config") if isinstance(source, dict) else None
    if not isinstance(cfg, dict) or not all(k in cfg for k in _MATERIAL_KEYS):
        return None
    signed = {k: cfg[k] for k in _MATERIAL_KEYS}
    signed["synthetic"] = _is_synthetic(source, cfg)
    try:
        return json.dumps(signed, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _source(run_dir: Path) -> dict:
    """Read run provenance if present; malformed/missing is not fatal.

    ``run_ab`` writes it under ``memory/source.json``; accept the old
    ``run_dir/source.json`` location for pre-existing results.
    """
    for path in (run_dir / "memory" / "source.json",
                 run_dir / "source.json"):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}
    return {}


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
