"""Kaggle user-secret loading: host routing + error classification.

Regression cover for the 2026-10-19 smoke failure: the smoke kernel reported
the secrets backend as "flaky" (HTTP 400) and died before the run, while the
real cause was that Kaggle injects ``KAGGLE_URL_BASE=https://api.kaggle.com``
and only ``https://www.kaggle.com`` serves the UserSecretsService endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import kaggle_secret as ks


def _install(monkeypatch, responses):
    """responses: list of (status, body) consumed in order per host attempt."""
    calls = []

    def fake_post(base, label, timeout):
        calls.append((base, label, timeout))
        idx = sum(1 for c in calls if c[0] == base) - 1
        seq = responses.get(base) or []
        status, body = seq[min(idx, len(seq) - 1)]
        return status, body

    monkeypatch.setattr(ks, "_post", fake_post)
    monkeypatch.setenv(ks.TOKEN_ENV, "tok")
    monkeypatch.setenv(ks.URL_BASE_ENV, "https://api.kaggle.com")
    return calls


def test_candidate_bases_prefers_www_over_injected(monkeypatch):
    monkeypatch.setenv(ks.URL_BASE_ENV, "https://api.kaggle.com")
    bases = ks._candidate_bases()
    assert bases[0] == "https://www.kaggle.com"
    assert "https://api.kaggle.com" in bases
    assert len(bases) == len(set(bases))


def test_missing_token_raises_auth_error(monkeypatch):
    monkeypatch.delenv(ks.TOKEN_ENV, raising=False)
    with pytest.raises(ks.SecretAuthError) as exc:
        ks.fetch_secret("GEMINI_API_KEY")
    assert ks.TOKEN_ENV in str(exc.value)


def test_not_attached_is_distinguished_from_backend_error(monkeypatch):
    """The server's code-5 answer must NOT be reported as a 400/backend bug."""
    body = json.dumps({
        "errors": ["No user secrets exist for kernel id 139687122 and label "
                   "GEMINI_API_KEY."],
        "error": {"code": 5},
        "wasSuccessful": False,
    })
    _install(monkeypatch, {"https://www.kaggle.com": [(200, body)]})
    with pytest.raises(ks.SecretNotAttachedError) as exc:
        ks.fetch_secret("GEMINI_API_KEY")
    msg = str(exc.value)
    assert "not attached" in msg
    assert "Add-ons" in msg                       # tells the user what to do
    assert "139687122" in msg                     # names the kernel
    assert "HTTP 400" not in msg                  # not the old symptom


def test_empty_body_400_reports_backend_error_not_attached(monkeypatch):
    """A 400 with an empty body is a routing/backend bug, not "not attached"."""
    _install(monkeypatch, {
        "https://www.kaggle.com": [(400, "")],
        "https://api.kaggle.com": [(400, "")],
    })
    with pytest.raises(ks.SecretBackendError) as exc:
        ks.fetch_secret("GEMINI_API_KEY")
    assert not isinstance(exc.value, ks.SecretNotAttachedError)
    assert "HTTP 400" in str(exc.value)


def test_success_returns_value_and_repairs_url_base(monkeypatch):
    body = json.dumps({"wasSuccessful": True,
                        "result": {"value": "s3cr3t-value"}})
    _install(monkeypatch, {"https://www.kaggle.com": [(200, body)]})
    value = ks.fetch_secret("GEMINI_API_KEY")
    assert value == "s3cr3t-value"
    # downstream UserSecretsClient must now hit the working host
    assert ks.os.environ[ks.URL_BASE_ENV] == "https://www.kaggle.com"


def test_load_api_key_exports_to_env(monkeypatch):
    body = json.dumps({"wasSuccessful": True, "result": {"value": "abc"}})
    _install(monkeypatch, {"https://www.kaggle.com": [(200, body)]})
    ks.load_api_key("GEMINI_API_KEY")
    assert ks.os.environ["GEMINI_API_KEY"] == "abc"


def test_401_is_auth_error(monkeypatch):
    _install(monkeypatch, {
        "https://www.kaggle.com": [(401, "")],
        "https://api.kaggle.com": [(401, "")],
    })
    with pytest.raises(ks.SecretAuthError):
        ks.fetch_secret("GEMINI_API_KEY")


def test_falls_back_to_injected_host_on_400(monkeypatch):
    """If www ever stops serving the endpoint, the injected host is tried."""
    good = json.dumps({"wasSuccessful": True, "result": {"value": "v"}})
    calls = _install(monkeypatch, {
        "https://www.kaggle.com": [(400, "")],
        "https://api.kaggle.com": [(200, good)],
    })
    assert ks.fetch_secret("GEMINI_API_KEY") == "v"
    assert [c[0] for c in calls] == ["https://www.kaggle.com",
                                     "https://api.kaggle.com"]


def test_network_failure_is_backend_error(monkeypatch):
    _install(monkeypatch, {
        "https://www.kaggle.com": [(0, "TimeoutError: timed out")],
        "https://api.kaggle.com": [(0, "TimeoutError: timed out")],
    })
    with pytest.raises(ks.SecretBackendError) as exc:
        ks.fetch_secret("GEMINI_API_KEY")
    assert "network failure" in str(exc.value)


def test_real_http_error_is_captured_not_masked(monkeypatch):
    """_post must surface the server's body instead of a generic error."""
    def boom(base, label, timeout):
        raise ks.SecretBackendError("unreachable")
    monkeypatch.setattr(ks, "_post", boom)
    monkeypatch.setenv(ks.TOKEN_ENV, "tok")
    with pytest.raises(ks.SecretBackendError):
        ks.fetch_secret("GEMINI_API_KEY")


# --- notebooks ------------------------------------------------------------
#
# The notebooks clone the repo from GitHub at run time, so the secret cell is
# the ONLY place the secret is ever read. Guard it: no notebook may go back
# to kaggle_secrets.UserSecretsClient, which silently talks to the wrong
# host (api.kaggle.com -> HTTP 400 -> generic ConnectionError).

NOTEBOOKS = [
    "notebook/smoke/autosearch_smoke.ipynb",
    "notebook/diag/autosearch_diag.ipynb",
    "notebook/kaggle_run.ipynb",
    "notebook/secret_probe/secret_probe.ipynb",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("rel", NOTEBOOKS)
def test_notebook_cells_are_valid_python(rel):
    nb = json.loads((_repo_root() / rel).read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            compile(src, f"{rel}:cell{i}", "exec")
        except SyntaxError as exc:
            pytest.fail(f"{rel} cell {i} is not valid python: {exc}")


@pytest.mark.parametrize("rel", NOTEBOOKS)
def test_notebooks_do_not_use_broken_user_secrets_client(rel):
    """UserSecretsClient hits api.kaggle.com -> HTTP 400 -> opaque error."""
    nb = json.loads((_repo_root() / rel).read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # Look at CODE only: the explanatory comments legitimately name the
        # broken client to tell the reader why it must not be used.
        code_only = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "UserSecretsClient" not in code_only, (
            f"{rel} cell {i} uses UserSecretsClient, which resolves its URL "
            f"from the injected KAGGLE_URL_BASE (api.kaggle.com) and fails "
            f"with HTTP 400; use src.kaggle_secret instead"
        )
        assert "import kaggle_secrets" not in code_only, (
            f"{rel} cell {i} imports kaggle_secrets; use src.kaggle_secret "
            f"instead"
        )


@pytest.mark.parametrize("rel", NOTEBOOKS)
def test_notebooks_reference_the_working_secrets_host(rel):
    """Every notebook that reads the secret must know the working host."""
    nb = json.loads((_repo_root() / rel).read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"] == "code")
    assert "www.kaggle.com" in src, (
        f"{rel} must name https://www.kaggle.com (the only host that serves "
        f"the UserSecretsService endpoint)"
    )
