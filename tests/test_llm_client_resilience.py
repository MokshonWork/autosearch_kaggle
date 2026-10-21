"""LLM resilience regressions (smoke incident 2026-10-16):

A retired fallback model id (`gemini-2.5-flash-lite` -> HTTP 404) raised a
raw `requests.HTTPError` from the fallback path that killed an entire A/B
session mid-round. Fixes under test:

- fallback failures surface as LLMError carrying BOTH primary + fallback
  context (never a raw HTTPError, never the API key);
- non-retryable primary errors (404/400/401/403) skip retries -> fallback;
- retryable errors (429/5xx) back off across 6 attempts before fallback;
- the orchestrator stops GRACEFULLY on LLMError (state saved, resumable).
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests as rq

import src.llm.client as client_mod
from src.config import load_config
from src.llm.client import (BudgetExhausted, BudgetTracker, GeminiClient,
                            LLMError)
from tests.conftest import fixed_runner

API_KEY = "FAKE-KEY-MUST-NOT-LEAK"

OK_JSON = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
           "usageMetadata": {"totalTokenCount": 7}}


class FakeResp:
    def __init__(self, status: int, payload: dict | None = None,
                 headers: dict | None = None):
        self.status_code = status
        self._payload = payload if payload is not None else OK_JSON
        self.headers = headers or {}
        self._err = None
        if status >= 400:
            r = rq.Response()
            r.status_code = status
            self._err = rq.exceptions.HTTPError(f"HTTP {status}", response=r)

    def raise_for_status(self):
        if self._err is not None:
            raise self._err

    def json(self):
        return self._payload


class FakeRequests:
    """Scripted post(): int -> FakeResp; Exception -> raised directly."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls: list[str] = []
        self.exceptions = SimpleNamespace(
            HTTPError=rq.exceptions.HTTPError,
            RequestException=rq.exceptions.RequestException)

    def post(self, url, json=None, timeout=None, headers=None):
        assert headers and headers.get("x-goog-api-key")
        self.calls.append(url)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, FakeResp) else FakeResp(item)


@pytest.fixture()
def gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", API_KEY)
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    cfg = SimpleNamespace(
        model_for=lambda agent: "gemini-3.6-flash",
        fallback_model="gemini-3.8-flash",
        temperatures={}, max_output_tokens={},
        timeout_s=5, max_retries=1, api_key_env="GEMINI_API_KEY")
    client = GeminiClient(cfg, BudgetTracker(500, 100_000, 30))
    return client


def _use(client, script):
    fake = FakeRequests(script)
    client._requests = fake
    return fake


def test_primary_404_skips_retries_and_falls_back(gemini):
    fake = _use(gemini, [404, 200])
    text, tokens = gemini._raw_generate("librarian", "p")
    assert text == "ok" and tokens == 7
    assert len(fake.calls) == 2, "404 must not burn backoff retries"
    assert "gemini-3.6-flash" in fake.calls[0]
    assert "gemini-3.8-flash" in fake.calls[1]


def test_retired_fallback_raises_llmerror_with_both_contexts(gemini):
    _use(gemini, [404, 404])
    with pytest.raises(LLMError) as exc:
        gemini._raw_generate("coder", "p")
    msg = str(exc.value)
    assert "primary 'gemini-3.6-flash' failed (HTTP 404)" in msg
    assert "fallback 'gemini-3.8-flash' failed" in msg
    assert API_KEY not in msg  # never leak the key


def test_retryable_backoff_then_primary_success(gemini):
    fake = _use(gemini, [429, 429, 429, 429, 429, 200])
    text, _ = gemini._raw_generate("ideator", "p")
    assert text == "ok"
    assert len(fake.calls) == 6  # 6 attempts, no fallback needed


def test_retryable_exhausted_uses_fallback(gemini):
    fake = _use(gemini, [429] * 6 + [200])
    text, tokens = gemini._raw_generate("guard", "p")
    assert text == "ok" and tokens == 7
    assert len(fake.calls) == 7
    assert "gemini-3.8-flash" in fake.calls[-1]


def test_connection_errors_also_fall_back(gemini):
    fake = _use(gemini, [rq.exceptions.ConnectionError("net down"), 200])
    text, _ = gemini._raw_generate("analyst", "p")
    assert text == "ok"
    assert len(fake.calls) == 2


def test_orchestrator_stops_gracefully_on_llm_error(project, monkeypatch):
    """LLMError mid-round must save state (stopped_reason=LLMError) and
    return — not crash the A/B session."""
    from src.orchestrator import Orchestrator
    cfg, client, repo = project

    def raising(agent, prompt, schema, *a, **k):
        raise LLMError("gemini primary 'x' failed (HTTP 429); fallback dead")

    client.generate_json = raising  # instance attr shadows the method
    orch = Orchestrator(cfg, client, runner_fn=fixed_runner(), log=lambda s: None)
    state = orch.run()  # must NOT raise
    assert state["stopped_reason"] == "LLMError"
    assert orch.store.load_state()["stopped_reason"] == "LLMError"
    assert state["epsilon"] is not None  # calibration completed first


def test_retry_after_header_is_respected_and_clamped(gemini, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(client_mod.time, "sleep", slept.append)
    _use(gemini, [FakeResp(429, headers={"Retry-After": "7"}), 200])
    text, _ = gemini._raw_generate("ideator", "p")
    assert text == "ok" and slept == [7.0]

    slept.clear()
    _use(gemini, [FakeResp(429, headers={"Retry-After": "9999"}), 200])
    gemini._raw_generate("ideator", "p")
    assert slept == [60.0]


def test_daily_quota_skips_retries_and_uses_fallback(gemini):
    quota = {"error": {"details": [{
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaId":
                        "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
    fake = _use(gemini, [FakeResp(429, payload=quota), 200])
    text, _ = gemini._raw_generate("coder", "p")
    assert text == "ok"
    assert len(fake.calls) == 2  # one 429, then fallback; no retries
    assert "gemini-3.8-flash" in fake.calls[-1]


def test_no_sleep_after_final_retry(gemini, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(client_mod.time, "sleep", slept.append)
    fake = _use(gemini, [429] * 6 + [200])
    text, _ = gemini._raw_generate("guard", "p")
    assert text == "ok" and len(fake.calls) == 7
    assert len(slept) == 5  # 6 attempts -> 5 waits, never a 6th


def test_fallback_429_raises_contextual_llmerror(gemini):
    _use(gemini, [404, FakeResp(429)])
    with pytest.raises(LLMError) as exc:
        gemini._raw_generate("analyst", "p")
    msg = str(exc.value)
    assert "primary 'gemini-3.6-flash' failed (HTTP 404)" in msg
    assert "fallback 'gemini-3.8-flash' failed" in msg
    assert "HTTP 429" in msg
    assert API_KEY not in msg


def test_read_timeout_backs_off_then_falls_back(gemini):
    fake = _use(gemini, [rq.exceptions.ReadTimeout("timed out"), 200])
    text, _ = gemini._raw_generate("librarian", "p")
    assert text == "ok" and len(fake.calls) == 2


def _bounded(gemini, cap, per_round=None, fallback=""):
    """Reuse the fixture's env/backoff monkeypatches with a tight budget."""
    gemini.cfg.fallback_model = fallback
    gemini.budget = BudgetTracker(cap, 100_000, per_round or cap)
    return gemini


def test_cap_allows_exactly_n_and_rejects_before_post(gemini):
    client = _bounded(gemini, cap=2)
    fake = _use(client, [200, 200, 200])
    assert client._raw_generate("ideator", "p")[0] == "ok"
    assert client._raw_generate("ideator", "p")[0] == "ok"
    with pytest.raises(BudgetExhausted):
        client._raw_generate("ideator", "p")
    assert len(fake.calls) == 2
    assert client.budget.requests_total == 2


def test_timeout_and_http_error_consume_attempts(gemini):
    client = _bounded(gemini, cap=6)
    fake = _use(client, [rq.exceptions.ReadTimeout("timed out")] * 6)
    with pytest.raises(LLMError):
        client._raw_generate("coder", "p")
    assert len(fake.calls) == 6 and client.budget.requests_total == 6

    client = _bounded(gemini, cap=6)
    fake = _use(client, [500] * 6)
    with pytest.raises(LLMError):
        client._raw_generate("coder", "p")
    assert len(fake.calls) == 6 and client.budget.requests_total == 6


def test_per_round_cap_blocks_before_total_cap(gemini):
    client = _bounded(gemini, cap=10, per_round=1)
    fake = _use(client, [200, 200])
    assert client._raw_generate("ideator", "p")[0] == "ok"
    with pytest.raises(BudgetExhausted):
        client._raw_generate("ideator", "p")
    assert len(fake.calls) == 1 and client.budget.requests_total == 1

    client.budget.begin_round(1)
    assert client._raw_generate("ideator", "p")[0] == "ok"
    assert len(fake.calls) == 2 and client.budget.requests_total == 2


def test_each_retry_consumes_one_attempt(gemini):
    client = _bounded(gemini, cap=3)
    fake = _use(client, [429, 429, 200])
    assert client._raw_generate("guard", "p")[0] == "ok"
    assert len(fake.calls) == 3
    assert client.budget.requests_total == 3


def test_fallback_consumes_attempt_and_is_blocked_by_cap(gemini):
    client = _bounded(gemini, cap=2, fallback="gemini-3.8-flash")
    fake = _use(client, [404, 200])
    assert client._raw_generate("librarian", "p")[0] == "ok"
    assert len(fake.calls) == 2
    assert client.budget.requests_total == 2

    client = _bounded(gemini, cap=1, fallback="gemini-3.8-flash")
    fake = _use(client, [404, 200])
    with pytest.raises(BudgetExhausted):
        client._raw_generate("librarian", "p")
    assert len(fake.calls) == 1  # fallback never dispatched
    assert client.budget.requests_total == 1


def test_concurrent_workers_cannot_exceed_shared_cap(gemini):
    client = _bounded(gemini, cap=3, per_round=3)

    class ThreadFake:
        def __init__(self):
            self._lock = threading.Lock()
            self.calls = 0
            self.exceptions = SimpleNamespace(
                HTTPError=rq.exceptions.HTTPError,
                RequestException=rq.exceptions.RequestException)

        def post(self, url, json=None, timeout=None, headers=None):
            with self._lock:
                self.calls += 1
            return FakeResp(200)

    fake = ThreadFake()
    client._requests = fake
    outcomes: list[str] = []

    def run_one():
        try:
            client._raw_generate("analyst", "p")
            outcomes.append("ok")
        except BudgetExhausted:
            outcomes.append("budget")

    threads = [threading.Thread(target=run_one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake.calls == 3
    assert outcomes.count("ok") == 3
    assert outcomes.count("budget") == 5
    assert client.budget.requests_total == 3


def test_shipped_config_disables_unverified_fallback(gemini):
    root = Path(__file__).resolve().parents[1]
    for name in ("config.yaml", "config.smoke.yaml"):
        assert load_config(root / name).llm.fallback_model == ""
    cfg = load_config(root / "config.yaml")
    client = GeminiClient(cfg.llm, BudgetTracker(2, 100_000, 2))
    fake = _use(client, [404])
    with pytest.raises(LLMError):
        client._raw_generate("librarian", "p")
    assert len(fake.calls) == 1  # no silent switch to an unconfigured model
