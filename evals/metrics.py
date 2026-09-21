"""Per-run metrics from experiments.jsonl + state.json (Section 9.1)."""
from __future__ import annotations

import json
from pathlib import Path


def load_run(run_dir: Path) -> dict:
    """Load one run's memory dir. Returns {} if missing."""
    run_dir = Path(run_dir)
    exp_path = run_dir / "experiments.jsonl"
    state_path = run_dir / "state.json"
    experiments = []
    if exp_path.exists():
        experiments = [json.loads(l) for l in
                       exp_path.read_text().splitlines() if l.strip()]
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    return {"experiments": experiments, "state": state}


def run_metrics(run: dict) -> dict:
    """All 9.1 metrics for a single run."""
    exps = run["experiments"]
    state = run["state"]
    oks = [e for e in exps if e["result"]["status"] == "ok"]
    keeps = [e for e in exps if (e.get("decision") or {}).get("action") == "keep"]
    baseline = state.get("baseline_bpb")
    best = state.get("best_val_bpb")
    n = len(exps)
    first_improve = next(
        (i + 1 for i, e in enumerate(exps)
         if (e.get("decision") or {}).get("action") == "keep"), None)
    wasted = sum(e["result"]["wall_seconds"] for e in exps
                 if e["result"]["status"] != "ok"
                 or (e.get("decision") or {}).get("action") == "discard")
    curve, run_best = [], None
    for e in exps:
        v = e["result"].get("val_bpb")
        if e["result"]["status"] == "ok" and v is not None:
            run_best = v if run_best is None else min(run_best, v)
        curve.append(run_best)
    return {
        "n_experiments": n,
        "best_val_bpb": best,
        "improvement": (baseline - best) if (baseline and best) else 0.0,
        "curve": curve,
        "experiments_to_first_improvement": first_improve,
        "keep_rate": len(keeps) / n if n else 0.0,
        "failure_rate": sum(1 for e in exps
                            if e["result"]["status"] in
                            ("crash", "oom", "nan", "timeout")) / n if n else 0.0,
        "guard_reject_rate": sum(1 for e in exps if e["result"]["status"]
                                 == "guard_reject") / n if n else 0.0,
        "duplicate_rate": _duplicate_rate(exps),
        "wasted_gpu_minutes": wasted / 60.0,
        "api_calls": state.get("api_calls", 0),
        "api_tokens": state.get("api_tokens", 0),
        "wall_clock_hours": state.get("wall_clock_hours", 0.0),
        "epsilon": state.get("epsilon"),
    }


def _duplicate_rate(exps: list[dict]) -> float:
    """Fraction of hypotheses text-similar (>=0.85) to an earlier one."""
    import difflib
    seen, dups = [], 0
    for e in exps:
        text = (e["hypothesis"].get("category", "") + " "
                + e["hypothesis"].get("change_summary", "")).lower()
        if any(difflib.SequenceMatcher(None, text, s).ratio() >= 0.85
               for s in seen):
            dups += 1
        seen.append(text)
    return dups / len(exps) if exps else 0.0
