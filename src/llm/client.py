"""Provider-agnostic LLM client with budgets, retries, redaction (G12, G13, G18).

- GeminiClient talks to the Gemini REST API (generateContent). The API key is
  read from the env var named by config.llm.api_key_env and is NEVER logged.
- MockLLMClient is deterministic and drives dry-run mode and all tests (G19).
- generate_json() validates output against a pydantic schema; on invalid output
  it retries (max config.llm.max_retries) with the validation error appended.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import LLMConfig
from ..memory.store import redact

T = TypeVar("T", bound=BaseModel)


class BudgetExhausted(RuntimeError):
    """Raised when an API budget is exhausted -> orchestrator stops cleanly."""


class LLMError(RuntimeError):
    pass


class BudgetTracker:
    """Counts API calls and tokens; raises BudgetExhausted past limits (G13).

    Thread-safe: two worker threads share one tracker (hardening Ch.7).
    """

    def __init__(self, max_calls_total: int, max_tokens_total: int,
                 max_calls_per_round: int):
        import threading
        self._lock = threading.Lock()
        self.max_calls_total = max_calls_total
        self.max_tokens_total = max_tokens_total
        self.max_calls_per_round = max_calls_per_round
        self.calls_total = 0
        self.calls_this_round = 0
        self.tokens_total = 0

    def new_round(self) -> None:
        self.calls_this_round = 0

    def record(self, tokens: int) -> None:
        with self._lock:
            self.calls_total += 1
            self.calls_this_round += 1
            self.tokens_total += tokens
            if self.calls_total > self.max_calls_total:
                raise BudgetExhausted(
                    f"api calls total > {self.max_calls_total}")
            if self.tokens_total > self.max_tokens_total:
                raise BudgetExhausted(f"tokens total > {self.max_tokens_total}")
            if self.calls_this_round > self.max_calls_per_round:
                raise BudgetExhausted(
                    f"api calls this round > {self.max_calls_per_round}")


def _extract_json(text: str) -> str:
    """Tolerate ```json fences around the object; keep it simple otherwise."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.rsplit("```", 1)[0]
    return t.strip()


class BaseClient:
    def __init__(self, cfg: LLMConfig, budget: BudgetTracker):
        self.cfg = cfg
        self.budget = budget

    def _raw_generate(self, agent: str, prompt: str) -> tuple[str, int]:
        """Return (text, total_tokens). Implemented by subclasses."""
        raise NotImplementedError

    def generate_json(self, agent: str, prompt: str,
                      schema: type[T]) -> tuple[Optional[T], str]:
        """Generate + validate JSON. Returns (model_or_None, error)."""
        attempt_prompt = prompt
        last_error = "no attempts"
        for _ in range(self.cfg.max_retries + 1):
            text, tokens = self._raw_generate(agent, attempt_prompt)
            self.budget.record(tokens)
            try:
                data = json.loads(_extract_json(text))
                return schema.model_validate(data), ""
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = redact(str(e))[:500]
                attempt_prompt = (
                    prompt + "\n\nYour previous output was invalid: "
                    + last_error + "\nReturn ONLY valid JSON matching the schema."
                )
        return None, (f"invalid output after {self.cfg.max_retries + 1} "
                      f"attempts: {last_error}")



class GeminiClient(BaseClient):
    """Gemini free tier via REST. Key from env (Kaggle Secrets injects it)."""

    def __init__(self, cfg: LLMConfig, budget: BudgetTracker):
        super().__init__(cfg, budget)
        import requests  # local import keeps tests dependency-light
        self._requests = requests
        self._api_key = os.environ.get(cfg.api_key_env, "")
        if not self._api_key:
            raise LLMError(
                f"missing API key: set env {cfg.api_key_env} (Kaggle Secrets)")

    def _raw_generate(self, agent: str, prompt: str) -> tuple[str, int]:
        model = self.cfg.model_for(agent)
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent")  # key passed via header, not URL
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.cfg.temperatures.get(agent, 0.3),
                "maxOutputTokens": self.cfg.max_output_tokens.get(agent, 1024),
                "responseMimeType": "application/json",
            },
        }
        delay = 2.0
        last_err = "unknown"
        retryable = (429, 500, 502, 503, 504)
        for _ in range(6):  # exponential backoff on rate limits (G13)
            try:
                resp = self._requests.post(
                    url, json=body, timeout=self.cfg.timeout_s,
                    headers={"x-goog-api-key": self._api_key})
                if resp.status_code in retryable:
                    last_err = f"HTTP {resp.status_code}"
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                tokens = data.get("usageMetadata", {}).get(
                    "totalTokenCount", len(prompt) // 4 + len(text) // 4)
                return text, int(tokens)
            except self._requests.exceptions.HTTPError as e:
                # non-retryable 4xx (retired model id, bad key, 400...):
                # it will not heal — go straight to the fallback.
                code = e.response.status_code if e.response is not None else 0
                last_err = f"HTTP {code}"
                break
            except self._requests.exceptions.RequestException as e:
                last_err = redact(str(e))[:200]
                time.sleep(delay)
                delay *= 2
            except (KeyError, IndexError, TypeError, ValueError) as e:
                last_err = f"bad gemini response: {redact(str(e))[:160]}"
                break
        if self.cfg.fallback_model and model != self.cfg.fallback_model:
            # A retired/404 fallback previously raised a raw HTTPError here
            # that killed an entire run; surface BOTH failures as LLMError.
            fb = self.cfg.fallback_model
            fb_url = url.replace(model, fb)
            try:
                resp = self._requests.post(
                    fb_url, json=body, timeout=self.cfg.timeout_s,
                    headers={"x-goog-api-key": self._api_key})
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text, int(data.get("usageMetadata", {}).get(
                    "totalTokenCount", len(prompt) // 4))
            except (self._requests.exceptions.RequestException,
                    KeyError, IndexError, TypeError, ValueError) as e:
                raise LLMError(
                    f"gemini primary '{model}' failed ({last_err}); "
                    f"fallback '{fb}' failed: {redact(str(e))[:200]}") from e
        raise LLMError(f"gemini call failed: {last_err}")


class GroqClient(BaseClient):
    """Groq (OpenAI-compatible REST) — selectable via llm.provider: groq.

    Same contract as GeminiClient: Bearer key from env (never printed),
    6x exponential backoff on retryable errors, non-retryable fast-fail to
    cfg.fallback_model (a GROQ model id), terminal failure -> LLMError with
    BOTH primary and fallback context. JSON mode on (all our prompts ask
    for JSON, which Groq requires for response_format json_object).
    """

    BASE = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, cfg: LLMConfig, budget: BudgetTracker):
        super().__init__(cfg, budget)
        import requests  # local import keeps tests dependency-light
        self._requests = requests
        self._api_key = os.environ.get(cfg.api_key_env, "")
        if not self._api_key:
            raise LLMError(
                f"missing API key: set env {cfg.api_key_env} (Kaggle Secrets)")

    def _raw_generate(self, agent: str, prompt: str) -> tuple[str, int]:
        model = self.cfg.model_for(agent)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperatures.get(agent, 0.3),
            "max_tokens": self.cfg.max_output_tokens.get(agent, 1024),
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self._api_key}",
                   "Content-Type": "application/json"}
        delay = 2.0
        last_err = "unknown"
        retryable = (429, 500, 502, 503, 504)
        for _ in range(6):
            try:
                resp = self._requests.post(
                    self.BASE, json=body, timeout=self.cfg.timeout_s,
                    headers=headers)
                if resp.status_code in retryable:
                    last_err = f"HTTP {resp.status_code}"
                    time.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get(
                    "total_tokens", len(prompt) // 4 + len(text) // 4)
                return text, int(tokens)
            except self._requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                last_err = f"HTTP {code}"
                break
            except self._requests.exceptions.RequestException as e:
                last_err = redact(str(e))[:200]
                time.sleep(delay)
                delay *= 2
            except (KeyError, IndexError, TypeError, ValueError) as e:
                last_err = f"bad groq response: {redact(str(e))[:160]}"
                break
        fb = self.cfg.fallback_model
        if fb and fb != model:
            fb_body = dict(body, model=fb)
            try:
                resp = self._requests.post(
                    self.BASE, json=fb_body, timeout=self.cfg.timeout_s,
                    headers=headers)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return text, int(data.get("usage", {}).get(
                    "total_tokens", len(prompt) // 4))
            except (self._requests.exceptions.RequestException,
                    KeyError, IndexError, TypeError, ValueError) as e:
                raise LLMError(
                    f"groq primary '{model}' failed ({last_err}); "
                    f"fallback '{fb}' failed: {redact(str(e))[:200]}") from e
        raise LLMError(f"groq call failed: {last_err}")


class MockLLMClient(BaseClient):
    """Deterministic fake client (G19). Handlers map agent -> canned output.

    A handler may be a dict/str (always returned), a list (cycled), or a
    callable(agent, prompt, call_count) -> dict|str. Token cost is fixed.
    """

    def __init__(self, cfg: LLMConfig, budget: BudgetTracker,
                 handlers: dict[str, Any] | None = None):
        super().__init__(cfg, budget)
        self.handlers = handlers or {}
        self.call_counts: dict[str, int] = {}

    def _raw_generate(self, agent: str, prompt: str) -> tuple[str, int]:
        n = self.call_counts.get(agent, 0)
        self.call_counts[agent] = n + 1
        handler = self.handlers.get(agent)
        if handler is None:
            raise LLMError(f"mock: no handler for agent '{agent}'")
        if callable(handler):
            out = handler(agent, prompt, n)
        elif isinstance(handler, list):
            out = handler[n % len(handler)]
        else:
            out = handler
        text = out if isinstance(out, str) else json.dumps(out)
        return text, 100  # fixed fake token cost


def make_client(cfg: LLMConfig, budget: BudgetTracker):
    """Dispatch on cfg.provider. Unknown providers fail fast (never silently).

    Campaign default remains 'gemini'; 'groq' is a validated alternative
    (same schema/budget/guardrails — the science does not depend on vendor).
    """
    provider = (cfg.provider or "gemini").lower()
    if provider == "gemini":
        return GeminiClient(cfg, budget)
    if provider == "groq":
        return GroqClient(cfg, budget)
    raise LLMError(f"unknown llm.provider '{provider}' (expected gemini|groq)")
