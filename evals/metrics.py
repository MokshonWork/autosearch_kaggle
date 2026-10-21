"""Per-run metrics from experiments.jsonl + state.json (Section 9.1)."""
from __future__ import annotations

import json
from pathlib import Path


def load_run(run_dir: Path) -> dict:
    """Load one run's memory dir; tolerate missing/malformed artifacts.

    Bad lines are skipped and counted in ``load_errors`` instead of aborting
    the whole report. A missing directory is an empty run, not an exception.
    """
    run_dir = Path(run_dir)
    exp_path = run_dir / "experiments.jsonl"
    state_path = run_dir / "state.json"
    experiments: list[dict] = []
    errors: list[str] = []
    if exp_path.exists():
        for lineno, line in enumerate(
                exp_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"experiments.jsonl:{lineno}")
                continue
            if not isinstance(record, dict):
                errors.append(f"experiments.jsonl:{lineno}:not-an-object")
                continue
            experiments.append(record)
    state = {}
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text())
            state = loaded if isinstance(loaded, dict) else {}
            if not isinstance(loaded, dict):
                errors.append("state.json:not-an-object")
        except (json.JSONDecodeError, OSError):
            errors.append("state.json")
    return {"experiments": experiments, "state": state,
            "load_errors": errors}


_FATAL_REASONS = {"LLMError", "BudgetExhausted"}


def run_status(state: dict, n_experiments: int | None = None) -> str:
    """Classify a persisted run: completed, partial, or failed.

    Zero persisted experiments can never count as a successful run. A fatal
    LLM/budget stop after some experiments succeeded is partial: those
    completed results stay valid, but the A/B run is not complete.
    """
    state = _dict(state)
    done = state.get("experiments_done")
    if not isinstance(done, int) or isinstance(done, bool):
        done = n_experiments if isinstance(n_experiments, int) else 0
    if done <= 0 or state.get("round_status") == "failed":
        return "failed"
    if (state.get("stopped_reason") in _FATAL_REASONS
            or state.get("round_status") == "partial"):
        return "partial"
    return "completed"


def run_metrics(run: dict) -> dict:
    """All 9.1 metrics for a single run."""
    exps = run.get("experiments") or []
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    keeps = [e for e in exps
             if _dict(e.get("decision")).get("action") == "keep"]
    baseline = state.get("baseline_bpb")
    best = state.get("best_val_bpb")
    n = len(exps)
    first_improve = next(
        (i + 1 for i, e in enumerate(exps)
         if _dict(e.get("decision")).get("action") == "keep"), None)
    wasted = sum(_dict(e.get("result")).get("wall_seconds", 0) for e in exps
                 if _dict(e.get("result")).get("status") != "ok"
                 or _dict(e.get("decision")).get("action") == "discard")
    curve, run_best = [], None
    for e in exps:
        result = _dict(e.get("result"))
        v = result.get("val_bpb")
        if result.get("status") == "ok" and v is not None:
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
                            if _dict(e.get("result")).get("status") in
                            ("crash", "oom", "nan", "timeout")) / n if n else 0.0,
        "guard_reject_rate": sum(1 for e in exps
                                 if _dict(e.get("result")).get("status")
                                 == "guard_reject") / n if n else 0.0,
        "duplicate_rate": _duplicate_rate(exps),
        "load_errors": len(run.get("load_errors") or []),
        "wasted_gpu_minutes": wasted / 60.0,
        "api_calls": state.get("api_calls", 0),
        "api_tokens": state.get("api_tokens", 0),
        "wall_clock_hours": state.get("wall_clock_hours", 0.0),
        "epsilon": state.get("epsilon"),
    }


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _duplicate_rate(exps: list[dict]) -> float:
    """Fraction of hypotheses text-similar (>=0.85) to an earlier one."""
    import difflib
    seen, dups = [], 0
    for e in exps:
        hyp = _dict(e.get("hypothesis"))
        text = (hyp.get("category", "") + " "
                + hyp.get("change_summary", "")).lower()
        if any(difflib.SequenceMatcher(None, text, s).ratio() >= 0.85
               for s in seen):
            dups += 1
        seen.append(text)
    return dups / len(exps) if exps else 0.0
