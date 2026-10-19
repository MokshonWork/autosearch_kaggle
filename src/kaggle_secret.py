"""Kaggle user-secret loading with actionable diagnostics.

Why this module exists
----------------------
Kaggle injects ``KAGGLE_URL_BASE=https://api.kaggle.com`` into every kernel
environment.  ``kaggle_web_client.KaggleWebClient`` reads that variable when
it is constructed and builds request URLs as ``url_base + endpoint``.  The
UserSecretsService endpoint (``/requests/GetUserSecretByLabelRequest``) is
served ONLY by ``https://www.kaggle.com``; against ``api.kaggle.com`` it
answers ``HTTP 400`` with an *empty* body.  ``kaggle_web_client`` converts
every ``HTTPError`` into a generic ``ConnectionError``/``BackendError``, so
both the status code and the server's explanation were lost.

Observed consequence (2026-10-19, kernels autosearch-smoke / autosearch-diag):
every ``UserSecretsClient().get_secret('GEMINI_API_KEY')`` failed with
``ConnectionError: Connection error trying to communicate with service.`` and
the smoke kernel reported the secret backend as "flaky" (it retried four
times).  It was not flaky: the request went to the wrong host on every
attempt, deterministically.

On ``www.kaggle.com`` the same request returns HTTP 200 with a JSON body that
distinguishes the two real failure modes:

    {"errors":["No user secrets exist for kernel id 139687122 and label
     GEMINI_API_KEY."],"error":{"code":5},"wasSuccessful":false}

so "secret not attached to this kernel" is a *different* signal from
"backend rejected the request" -- and it must not be confused with either.

This module talks to the endpoint directly (stdlib only) so the HTTP status
and the server's body are preserved, and turns them into an actionable
message.  It also repoints ``KAGGLE_URL_BASE`` at ``www.kaggle.com`` so any
downstream ``kaggle_secrets.UserSecretsClient`` call in the same process
takes the working route too.

Public API
----------
    load_api_key(label, env_name=None)  -> str
    fetch_secret(label)                 -> str
    SecretError / SecretNotAttachedError / SecretAuthError / SecretBackendError
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

#: Host that actually serves the UserSecretsService endpoint.
SECRETS_HOST = "https://www.kaggle.com"
#: Path of the UserSecretsService "get secret by label" endpoint.
SECRETS_ENDPOINT = "/requests/GetUserSecretByLabelRequest"
#: Environment variable holding the per-kernel secrets token.
TOKEN_ENV = "KAGGLE_USER_SECRETS_TOKEN"
#: Environment variable Kaggle injects to redirect API calls (to the wrong host).
URL_BASE_ENV = "KAGGLE_URL_BASE"

DEFAULT_TIMEOUT = 40



class SecretError(RuntimeError):
    """Base class for secret-loading failures."""


class SecretAuthError(SecretError):
    """No usable token, or the token was rejected (401/403)."""


class SecretNotAttachedError(SecretError):
    """The label exists in the account but is not attached to this kernel."""


class SecretBackendError(SecretError):
    """The secrets service rejected or failed the request (5xx / 400 / ...)."""


def _candidate_bases() -> list[str]:
    """Hosts to try, most likely first.

    ``www.kaggle.com`` first (the only host that serves the endpoint), then
    whatever the environment injects, in case a future Kaggle image moves the
    endpoint.  Duplicates are removed, order preserved.
    """
    bases = [SECRETS_HOST]
    injected = (os.environ.get(URL_BASE_ENV) or "").strip()
    if injected and injected not in bases:
        bases.append(injected.rstrip("/"))
    return bases


def _token() -> str:
    token = (os.environ.get(TOKEN_ENV) or "").strip()
    if not token:
        raise SecretAuthError(
            f"{TOKEN_ENV} is not set in this environment; the kernel cannot "
            f"read user secrets (attach the secret in the Kaggle UI: "
            f"Add-ons -> Secrets)."
        )
    return token


def _post(base: str, label: str, timeout: int) -> tuple[int, str]:
    """POST the endpoint, returning (status, body_text). Never raises for HTTP."""
    body = json.dumps({"Label": label}).encode("utf-8")
    headers = {
        "Content-type": "application/json",
        "X-Kaggle-Authorization": "Bearer " + _token(),
    }
    req = urllib.request.Request(base + SECRETS_ENDPOINT, data=body,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read().decode("utf-8", "replace")
        except Exception:                                    # pragma: no cover
            payload = ""
        return exc.code, payload
    except Exception as exc:                                # network/timeout
        return 0, f"{type(exc).__name__}: {exc}"


def _classify(status: int, payload: str, label: str) -> str:
    """Return the secret value, or raise the most specific error we can."""
    if status == 0:
        raise SecretBackendError(
            f"network failure reaching {SECRETS_HOST}{SECRETS_ENDPOINT}: "
            f"{payload[:200]}"
        )
    if status in (401, 403):
        raise SecretAuthError(
            f"the secrets service rejected this kernel's token (HTTP {status}) "
            f"for label {label!r}; re-attach the secret in the Kaggle UI "
            f"(Add-ons -> Secrets) and re-run."
        )

    detail = ""
    code = None
    if payload:
        try:
            doc = json.loads(payload)
        except ValueError:
            doc = None
        if isinstance(doc, dict):
            if not doc.get("wasSuccessful", False):
                errs = doc.get("errors") or []
                if isinstance(errs, str):
                    errs = [errs]
                detail = "; ".join(str(e) for e in errs)
                code = (doc.get("error") or {}).get("code")
            else:
                result = doc.get("result")
                if isinstance(result, dict):
                    value = result.get("value")
                    if value is None:
                        value = result.get("secretValue")
                    if value:
                        return str(value)
                elif isinstance(result, str) and result:
                    return result
                detail = "response contained no secret value"

    if "no user secrets exist" in detail.lower() or code == 5:
        kernel_id = ""
        for token in detail.replace(",", " ").split():
            if token.isdigit() and len(token) >= 6:
                kernel_id = token
                break
        where = f" (kernel id {kernel_id})" if kernel_id else ""
        raise SecretNotAttachedError(
            f"the secret {label!r} is not attached to this kernel{where}. "
            f"Open the kernel in the Kaggle UI -> Add-ons -> Secrets, add "
            f"{label!r}, and tick the 'attach to this kernel' box. "
            f"Secrets are per-kernel: attaching it to another kernel does "
            f"not help. The CLI/API cannot attach secrets."
        )
    if status == 200 and detail:
        raise SecretBackendError(
            f"the secrets service returned a failure for label {label!r}: "
            f"{detail[:300]}"
        )
    raise SecretBackendError(
        f"the secrets service rejected the request for label {label!r} "
        f"(HTTP {status})"
        + (f": {detail[:300]}" if detail else
           f" (empty body; the endpoint may be unreachable at this host)")
    )


def fetch_secret(label: str = "GEMINI_API_KEY",
                 timeout: int = DEFAULT_TIMEOUT) -> str:
    """Return the value of the Kaggle user secret ``label``.

    Raises :class:`SecretNotAttachedError` when the label is not attached to
    this kernel, :class:`SecretAuthError` when the token is missing or
    rejected, and :class:`SecretBackendError` for any other failure.
    """
    errors: list[str] = []
    for base in _candidate_bases():
        status, payload = _post(base, label, timeout)
        try:
            value = _classify(status, payload, label)
        except SecretNotAttachedError:
            # A definitive "not attached" answer from a *working* host is the
            # most useful thing we can say -- do not keep probing other hosts.
            raise
        except SecretAuthError:
            # A rejected/missing token is a property of the KERNEL, not of
            # the host: another host would fail the same way. Propagate now
            # instead of burying it under a generic backend error.
            raise
        except SecretError as exc:
            errors.append(f"{base}: {exc}")
            continue
        # Success: make sure any later UserSecretsClient call in this process
        # uses the host that actually works.
        os.environ[URL_BASE_ENV] = SECRETS_HOST
        return value
    raise SecretBackendError(
        f"could not read the secret {label!r} from any known host: "
        + " | ".join(errors)
    )


def load_api_key(label: str = "GEMINI_API_KEY", env_name: str | None = None,
                 timeout: int = DEFAULT_TIMEOUT) -> str:
    """Fetch the secret and export it into ``os.environ``.

    The value is never printed or logged.  ``env_name`` defaults to
    ``label`` so that ``GEMINI_API_KEY`` lands in ``GEMINI_API_KEY``.
    """
    value = fetch_secret(label, timeout=timeout)
    os.environ[env_name or label] = value
    os.environ[URL_BASE_ENV] = SECRETS_HOST
    return value

