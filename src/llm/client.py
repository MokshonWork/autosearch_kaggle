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
        for _ in range(4):  # exponential backoff on rate limits (G13)
            try:
                resp = self._requests.post(
                    url, json=body, timeout=self.cfg.timeout_s,
                    headers={"x-goog-api-key": self._api_key})
                if resp.status_code in (429, 500, 502, 503, 504):
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
            except self._requests.RequestException as e:
                last_err = redact(str(e))
                time.sleep(delay)
                delay *= 2
        if self.cfg.fallback_model and model != self.cfg.fallback_model:
            url = url.replace(model, self.cfg.fallback_model)
            resp = self._requests.post(
                url, json=body, timeout=self.cfg.timeout_s,
                headers={"x-goog-api-key": self._api_key})
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text, int(data.get("usageMetadata", {}).get(
                "totalTokenCount", len(prompt) // 4))
        raise LLMError(f"gemini call failed: {last_err}")


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
