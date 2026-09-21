"""Deterministic keep/discard decision rule (Section 6.4, G9, G10, G15).

This is pure code. No LLM output can override it.
"""
from __future__ import annotations

import math
import statistics

from .schemas import Decision, RunResult


def compute_epsilon(calibration_values: list[float], min_delta: float) -> float:
    """epsilon = max(2 * std of calibration val_bpb, min_delta)."""
    std = statistics.stdev(calibration_values) if len(calibration_values) >= 2 else 0.0
    return max(2.0 * std, min_delta)


def plausibility_floor(baseline_bpb: float, factor: float) -> float:
    """val_bpb below this is implausible (possible metric tampering) -> G9."""
    return baseline_bpb * factor


def decide(
    result: RunResult,
    best: float | None,
    epsilon: float,
    floor: float,
    ceiling: float,
    confirm_marginal_wins: bool = True,
) -> Decision:
    """Apply the decision rule to one finished run.

    reason_code values:
      discard_not_ok:<status>, discard_unsafe_value, discard_implausible_low,
      keep_first_baseline, keep_improvement, retest_marginal, discard_within_noise
    """
    best_before = best

    def _dec(action: str, reason: str, best_after: float | None = None) -> Decision:
        return Decision(
            hypothesis_id=result.hypothesis_id, action=action,
            best_before=best_before,
            best_after=best_after if best_after is not None else best_before,
            epsilon=epsilon, reason_code=reason,
        )

    if result.status != "ok":
        return _dec("discard", f"discard_not_ok:{result.status}")  # G10

    val = result.val_bpb
    if val is None or not math.isfinite(val) or val <= 0 or val > ceiling:
        return _dec("discard", "discard_unsafe_value")  # G9 sanity
    if val < floor:
        return _dec("discard", "discard_implausible_low")  # G9 floor, flag upstream
    if best is None:
        return _dec("keep", "keep_first_baseline", best_after=val)

    improvement = best - val
    if improvement >= 3 * epsilon:
        return _dec("keep", "keep_improvement", best_after=val)
    if improvement >= epsilon:
        if confirm_marginal_wins:
            return _dec("retest", "retest_marginal")  # re-run once, then mean
        return _dec("keep", "keep_improvement", best_after=val)
    return _dec("discard", "discard_within_noise")


def decide_retest(mean_bpb: float, best: float, epsilon: float) -> Decision:
    """Resolve a marginal win after its confirmation re-run (use the mean)."""
    action = "keep" if mean_bpb < best - epsilon else "discard"
    return Decision(
        hypothesis_id="", action=action, best_before=best,
        best_after=min(best, mean_bpb) if action == "keep" else best,
        epsilon=epsilon,
        reason_code=("keep_marginal_confirmed" if action == "keep"
                     else "discard_marginal_unconfirmed"),
    )
