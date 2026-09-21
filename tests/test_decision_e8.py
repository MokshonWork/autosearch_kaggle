"""E8: decision rule unit tests (G9, G10, G15) + epsilon + combine/retest."""
import math

import pytest

from src.decision import (compute_epsilon, decide, decide_retest,
                          plausibility_floor)
from tests.conftest import make_result

EPS = 0.01
FLOOR = 0.7
CEIL = 20.0


def test_epsilon_from_calibration():
    assert compute_epsilon([1.0, 1.001, 0.999], 0.002) >= 0.002
    assert compute_epsilon([1.0, 1.05, 0.95], 0.002) > 0.05
    assert plausibility_floor(1.0, 0.7) == 0.7


def test_first_baseline_kept():
    d = decide(make_result(val_bpb=1.2), None, EPS, FLOOR, CEIL)
    assert d.action == "keep" and d.reason_code == "keep_first_baseline"


def test_clear_win_kept():
    d = decide(make_result(val_bpb=0.9), 1.0, EPS, FLOOR, CEIL)
    assert d.action == "keep"


def test_within_noise_discarded():
    d = decide(make_result(val_bpb=0.995), 1.0, EPS, FLOOR, CEIL)
    assert d.action == "discard" and d.reason_code == "discard_within_noise"


def test_worse_discarded():
    d = decide(make_result(val_bpb=1.05), 1.0, EPS, FLOOR, CEIL)
    assert d.action == "discard"


def test_marginal_win_retest():
    # improvement between epsilon and 3*epsilon -> retest (if configured)
    d = decide(make_result(val_bpb=0.98), 1.0, EPS, FLOOR, CEIL,
               confirm_marginal_wins=True)
    assert d.action == "retest"
    d2 = decide(make_result(val_bpb=0.98), 1.0, EPS, FLOOR, CEIL,
                confirm_marginal_wins=False)
    assert d2.action == "keep"


def test_retest_resolution():
    assert decide_retest(0.985, 1.0, EPS).action == "keep"
    assert decide_retest(0.999, 1.0, EPS).action == "discard"


@pytest.mark.parametrize("status", ["crash", "oom", "nan", "timeout",
                                    "guard_reject"])
def test_failed_statuses_never_kept(status):
    d = decide(make_result(status=status, val_bpb=None), 1.0, EPS, FLOOR, CEIL)
    assert d.action == "discard"
    assert d.reason_code.startswith("discard_not_ok")


def test_implausible_low_rejected():
    d = decide(make_result(val_bpb=0.5), 1.0, EPS, FLOOR, CEIL)  # < floor 0.7
    assert d.action == "discard" and d.reason_code == "discard_implausible_low"


def test_unsafe_values_rejected():
    for bad in (None, float("nan"), float("inf"), -1.0, 100.0):
        d = decide(make_result(val_bpb=bad), 1.0, EPS, FLOOR, CEIL)
        assert d.action == "discard", bad
        assert d.reason_code in ("discard_unsafe_value",
                                 "discard_implausible_low")


def test_llm_cannot_override():
    # decide() takes no LLM input at all: it is a pure function of numbers.
    d = decide(make_result(val_bpb=1.01), 1.0, EPS, FLOOR, CEIL)
    assert d.action == "discard"
