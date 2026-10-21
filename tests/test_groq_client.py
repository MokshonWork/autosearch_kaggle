"""Groq provider support (added on user request 2026-10-16).

llm.provider: gemini | groq selects the client via make_client(); the
campaign default stays gemini (verified path). GroqClient mirrors the
Gemini contract: Bearer auth from env, 6x retryable backoff, fast-fail on
non-retryable 4xx, contextual LLMError with BOTH primary+fallback on
terminal failure, API key never in messages.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests as rq

import src.llm.client as client_mod
from src.config import Config
from src.llm.client import (BudgetExhausted, BudgetTracker, GeminiClient,
                            GroqClient, LLMError, make_client)

GROQ_KEY = "GROQ-FAKE-KEY-MUST-NOT-LEAK"
GEMINI_KEY = "GEMINI-FAKE-KEY-MUST-NOT-LEAK"

GROQ_OK = {"choices": [{"message": {"content": "{\"ideas\": []}"}}],
           "usage": {"total_tokens": 42}}


def _cfg(provider="groq", api_env="GROQ_API_KEY"):
    return SimpleNamespace(
        provider=provider, api_key_env=api_env,
        model_for=lambda agent: "llama-3.3-70b-versatile",
        fallback_model="openai/gpt-oss-120b",
        temperatures={}, max_output_tokens={},
        timeout_s=5, max_retries=1)


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = GROQ_OK if payload is None else payload
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


class GroqFake:
    def __init__(self, script):
        self._script = list(script)
        self.calls: list[tuple[str, dict, dict]] = []
        self.exceptions = SimpleNamespace(
            HTTPError=rq.exceptions.HTTPError,
            RequestException=rq.exceptions.RequestException)

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append((url, dict(json or {}), dict(headers or {})))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, FakeResp) else FakeResp(item)


@pytest.fixture()
def groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    client = GroqClient(_cfg(), BudgetTracker(500, 100_000, 30))
    fake = GroqFake([])
    client._requests = fake
    return client, fake


# ---------------------------------------------------------------- dispatch --
def test_make_client_dispatch(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    budget = BudgetTracker(10, 1000, 5)
    assert isinstance(make_client(_cfg("gemini", "GEMINI_API_KEY"), budget),
                      GeminiClient)
    assert isinstance(make_client(_cfg("groq"), budget), GroqClient)
    with pytest.raises(LLMError, match="unknown llm.provider"):
        make_client(_cfg("llama.cpp", "X_KEY"), budget)


def test_campaign_default_provider_is_gemini():
    assert Config().llm.provider == "gemini"


def test_groq_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqClient(_cfg(), BudgetTracker(10, 1000, 5))


# -------------------------------------------------------------- transport --
def test_groq_happy_path(groq):
    client, fake = groq
    fake._script.append(200)
    text, tokens = client._raw_generate("ideator", "give ideas as JSON")
    assert text == '{"ideas": []}' and tokens == 42
    url, body, headers = fake.calls[0]
    assert url == GroqClient.BASE
    assert body["model"] == "llama-3.3-70b-versatile"
    assert body["response_format"] == {"type": "json_object"}
    assert headers["Authorization"] == f"Bearer {GROQ_KEY}"


def test_groq_429_backoff_then_success(groq):
    client, fake = groq
    fake._script.extend([429, 429, 429, 429, 429, 200])
    text, _ = client._raw_generate("guard", "j")
    assert text == '{"ideas": []}'
    assert len(fake.calls) == 6


def test_groq_primary_404_falls_back_immediately(groq):
    client, fake = groq
    fake._script.extend([404, 200])
    text, tokens = client._raw_generate("coder", "j")
    assert text == '{"ideas": []}' and tokens == 42
    assert len(fake.calls) == 2
    assert fake.calls[1][1]["model"] == "openai/gpt-oss-120b"


def test_groq_both_models_dead_raises_contextual_llmerror(groq):
    client, fake = groq
    fake._script.extend([404, 404])
    with pytest.raises(LLMError) as exc:
        client._raw_generate("analyst", "j")
    msg = str(exc.value)
    assert "primary 'llama-3.3-70b-versatile' failed (HTTP 404)" in msg
    assert "fallback 'openai/gpt-oss-120b' failed" in msg
    assert GROQ_KEY not in msg


def test_groq_connection_error_backs_off_then_falls_back(groq):
    client, fake = groq
    fake._script.extend([rq.exceptions.ConnectionError("net down"), 200])
    text, _ = client._raw_generate("librarian", "j")
    assert text == '{"ideas": []}'
    assert len(fake.calls) == 2


def test_groq_retry_after_is_respected(groq, monkeypatch):
    client, fake = groq
    slept: list[float] = []
    monkeypatch.setattr(client_mod.time, "sleep", slept.append)
    fake._script.extend([FakeResp(429, headers={"Retry-After": "9"}), 200])
    text, _ = client._raw_generate("ideator", "j")
    assert text == '{"ideas": []}' and len(fake.calls) == 2
    assert slept == [9.0]


def test_groq_insufficient_quota_skips_retries(groq):
    client, fake = groq
    quota = {"error": {"type": "insufficient_quota", "code": "insufficient_quota"}}
    fake._script.extend([FakeResp(429, payload=quota), 200])
    text, _ = client._raw_generate("coder", "j")
    assert text == '{"ideas": []}' and len(fake.calls) == 2
    assert fake.calls[-1][1]["model"] == "openai/gpt-oss-120b"


def test_groq_cap_blocks_before_post(groq):
    client, fake = groq
    client.budget = BudgetTracker(1, 100_000, 1)
    fake._script.extend([200, 200])
    assert client._raw_generate("ideator", "j")[0] == '{"ideas": []}'
    with pytest.raises(BudgetExhausted):
        client._raw_generate("ideator", "j")
    assert len(fake.calls) == 1
    assert client.budget.requests_total == 1


def test_groq_fallback_consumes_attempt_and_is_blocked(groq):
    client, fake = groq
    client.budget = BudgetTracker(2, 100_000, 2)
    fake._script.extend([404, 200])
    assert client._raw_generate("coder", "j")[0] == '{"ideas": []}'
    assert len(fake.calls) == 2
    assert client.budget.requests_total == 2

    client.budget = BudgetTracker(1, 100_000, 1)
    fake._script.extend([404, 200])
    with pytest.raises(BudgetExhausted):
        client._raw_generate("coder", "j")
    assert len(fake.calls) == 3  # only the primary was dispatched
    assert client.budget.requests_total == 1
