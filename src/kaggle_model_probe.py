"""Probe LLM model ids for availability (Kaggle smoke, assumption A6).

Network failures must never crash the probe for informational candidate ids,
and a timeout must NEVER be treated as a working model. Configured ids are
strictly validated (HTTP 200 only); HTTP 429 means rate-limited/quota-exhausted
and is not proof of usability. Candidate ids are informational.

The API key is placed only in request headers and is never logged.
"""
from __future__ import annotations

TIMEOUT = "TIMEOUT"
REQUEST_ERROR = "REQUEST_ERROR"
_OK = ("HTTP 200",)


def _probe_once(mid: str, provider: str, api_key: str, post, timeout_s: int):
    if provider == "groq":
        return post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={"model": mid,
                  "messages": [{"role": "user",
                                "content": "Reply with the word: ok"}]},
            headers={"Authorization": "Bearer " + api_key,
                     "Content-Type": "application/json"},
            timeout=timeout_s)
    return post(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + mid + ":generateContent",
        json={"contents": [{"parts": [{"text": "Reply with the word: ok"}]}]},
        headers={"x-goog-api-key": api_key}, timeout=timeout_s)


def _probe(mid, provider, api_key, post, timeout_s, log) -> str:
    """Return 'HTTP <status>', TIMEOUT, or REQUEST_ERROR. Never raises."""
    import requests as rq

    try:
        resp = _probe_once(mid, provider, api_key, post, timeout_s)
    except rq.exceptions.Timeout:
        log(f"    {mid}: TIMEOUT after {timeout_s}s")
        return TIMEOUT
    except rq.exceptions.RequestException:
        log(f"    {mid}: REQUEST_ERROR")
        return REQUEST_ERROR
    if resp.status_code == 429:
        log(f"    {mid}: HTTP 429 rate-limited/quota-exhausted (not usable)")
    return f"HTTP {resp.status_code}"


def probe_models(ids, configured, provider, api_key, *,
                 post=None, timeout_s=30, log=print) -> dict:
    """Probe every id; raise AssertionError if a CONFIGURED id is unreachable.

    A configured id is acceptable only with HTTP 200. HTTP 429 is
    rate-limited/quota-exhausted, not proof of usability. A timeout or request
    error is a failure for configured ids (and merely informational for
    candidates).
    """
    import requests as rq

    post = post or rq.post
    results: dict[str, str] = {}
    for mid in ids:
        results[mid] = _probe(mid, provider, api_key, post, timeout_s, log)
        log(f"  {mid} -> {results[mid]}")

    dead = [m for m in configured if results[m] not in _OK]
    if dead:
        alive = sorted({m for m in ids if results[m] == "HTTP 200"})
        lines = [
            f"  {m}: {results[m]}"
            + (" (configured)" if m in configured
               else " (candidate, informational)")
            for m in sorted(results)]
        raise AssertionError(
            "CONFIGURED model id(s) not verified usable (expected HTTP 200; "
            "HTTP 429 means rate-limited/quota-exhausted; "
            "TIMEOUT/REQUEST_ERROR are failures) - probe table below:\n"
            + "\n".join(lines)
            + "\nSTILL USABLE: " + (", ".join(alive) or "none"))

    not_ok = [m for m in ids if m not in configured and results[m] != "HTTP 200"]
    if not_ok:
        log(f"note: non-configured candidate(s) not OK (ignored): {not_ok}")
    log(f"all CONFIGURED model ids verified usable (HTTP 200): {configured}")
    return results
