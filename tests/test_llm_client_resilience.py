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

from types import SimpleNamespace

import pytest
import requests as rq

import src.llm.client as client_mod
from src.llm.client import BudgetTracker, GeminiClient, LLMError
from tests.conftest import fixed_runner

API_KEY = "FAKE-KEY-MUST-NOT-LEAK"

OK_JSON = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
           "usageMetadata": {"totalTokenCount": 7}}


class FakeResp:
    def __init__(self, status: int, payload: dict | None = None):
        self.status_code = status
        self._payload = payload if payload is not None else OK_JSON
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
        return FakeResp(item)


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
