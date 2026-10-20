"""Regression tests: model probing survives timeouts and never leaks the key."""
import requests
import pytest

from src.kaggle_model_probe import REQUEST_ERROR, TIMEOUT, probe_models

SECRET = "SUPER-SECRET-KEY-do-not-log"


class _Resp:
    def __init__(self, status_code=200):
        self.status_code = status_code


def _post(*, timeouts=(), errors=(), statuses=None):
    statuses = statuses or {}

    def post(url, **kwargs):
        for mid in timeouts:
            if mid in url:
                raise requests.exceptions.ReadTimeout("probe timed out")
        for mid in errors:
            if mid in url:
                raise requests.exceptions.ConnectionError("probe failed")
        for mid, code in statuses.items():
            if mid in url:
                return _Resp(code)
        return _Resp(200)

    return post


def _logs():
    lines = []
    return lines, lines.append


def test_configured_and_candidate_http_200_are_success():
    logs, log = _logs()
    results = probe_models(
        ["gemini-3.6-flash", "gemini-flash-latest"],
        ["gemini-3.6-flash"], "gemini", SECRET,
        post=_post(statuses={"gemini-3.6-flash": 200,
                             "gemini-flash-latest": 200}),
        sleep=lambda _s: None, log=log)
    assert results["gemini-3.6-flash"] == "HTTP 200"
    assert results["gemini-flash-latest"] == "HTTP 200"
    assert not any(SECRET in line for line in logs)


def test_configured_429_is_non_fatal_with_bounded_retries():
    logs, log = _logs()
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return _Resp(429)

    results = probe_models(
        ["gemini-3.6-flash"], ["gemini-3.6-flash"], "gemini", SECRET,
        post=post, sleep=lambda _s: None, log=log)
    assert results["gemini-3.6-flash"] == "HTTP 429"
    assert len(calls) == 3  # initial + 2 bounded retries
    assert any("rate-limited" in line for line in logs)
    assert not any(SECRET in line for line in logs)


def test_candidate_timeout_is_informational():
    logs, log = _logs()
    results = probe_models(
        ["gemini-3.6-flash", "gemini-4-argon"],
        ["gemini-3.6-flash"], "gemini", SECRET,
        post=_post(timeouts=["gemini-4-argon"]),
        sleep=lambda _s: None, log=log)
    assert results["gemini-3.6-flash"] == "HTTP 200"
    assert results["gemini-4-argon"] == TIMEOUT
    assert any("ignored" in line for line in logs)
    assert not any(SECRET in line for line in logs)


def test_configured_timeout_fails_with_actionable_diagnostic():
    logs, log = _logs()
    with pytest.raises(AssertionError) as err:
        probe_models(
            ["gemini-3.6-flash"], ["gemini-3.6-flash"], "gemini", SECRET,
            post=_post(timeouts=["gemini-3.6-flash"]),
            sleep=lambda _s: None, log=log)
    msg = str(err.value)
    assert "gemini-3.6-flash" in msg
    assert TIMEOUT in msg
    assert "STILL USABLE" in msg
    assert SECRET not in msg
    assert not any(SECRET in line for line in logs)


def test_candidate_request_error_is_informational():
    logs, log = _logs()
    results = probe_models(
        ["gemini-3.6-flash", "gemini-4-argon"],
        ["gemini-3.6-flash"], "gemini", SECRET,
        post=_post(errors=["gemini-4-argon"]),
        sleep=lambda _s: None, log=log)
    assert results["gemini-4-argon"] == REQUEST_ERROR
    assert not any(SECRET in line for line in logs)


def test_configured_bad_status_still_fails():
    logs, log = _logs()
    with pytest.raises(AssertionError) as err:
        probe_models(
            ["gemini-3.6-flash"], ["gemini-3.6-flash"], "gemini", SECRET,
            post=_post(statuses={"gemini-3.6-flash": 500}),
            sleep=lambda _s: None, log=log)
    assert "HTTP 500" in str(err.value)
    assert not any(SECRET in line for line in logs)
