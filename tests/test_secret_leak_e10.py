"""E10: secret-leak test (G7, G18). Fake API key must appear NOWHERE."""
import os
import subprocess
import sys

from src.memory.store import redact
from src.orchestrator import Orchestrator
from src.tools.runner import stripped_env

FAKE_KEY = "AIzaSyFAKEKEY_DO_NOT_LEAK_0123456789abcd"


def test_stripped_env_has_no_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "also-secret")
    env = stripped_env(gpu=0)
    assert FAKE_KEY not in env.values()
    assert "GEMINI_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_redact_patterns():
    assert FAKE_KEY not in redact(f"key is {FAKE_KEY} ok")
    assert "hunter2" not in redact("api_key=hunter2 rest")


def test_full_mock_pipeline_leaks_nothing(project, monkeypatch):
    """Run the whole loop with a fake key in the env; grep every artifact."""
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    cfg, client, repo = project
    cfg.budget.max_rounds = 1
    state = Orchestrator(cfg, client).run()
    root = cfg.root
    artifacts = [root / "memory" / "experiments.jsonl",
                 root / "memory" / "lessons.md",
                 root / "memory" / "state.json"]
    for art in artifacts:
        assert art.exists()
        assert FAKE_KEY not in art.read_text(), art
    # the training subprocess env must not contain the key either (G7):
    env = stripped_env(gpu=1)
    out = subprocess.run([sys.executable, "-c",
                          "import os; print(os.environ.get('GEMINI_API_KEY'))"],
                         env=env, capture_output=True, text=True)
    assert FAKE_KEY not in out.stdout
