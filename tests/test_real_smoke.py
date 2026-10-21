"""Unit tests for the LLM-free real-training smoke preflight and validation."""
import json
import socket
from pathlib import Path

import pytest

from evals import real_smoke
from src.schemas import RunResult
from tests.conftest import FIXTURE_REPO


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    tokenizer = tmp_path / "tokenizer"
    data.mkdir()
    tokenizer.mkdir()
    (data / "shard_00001.parquet").write_bytes(b"train")
    (data / real_smoke.VAL_SHARD).write_bytes(b"val")
    (tokenizer / "tokenizer.pkl").write_bytes(b"tok")
    (tokenizer / "token_bytes.pt").write_bytes(b"bytes")
    return data, tokenizer


def _allow_preflight(monkeypatch):
    monkeypatch.setattr(real_smoke, "_missing_modules", lambda: [])
    monkeypatch.setattr(real_smoke, "_cuda_errors", lambda gpu: [])


def test_preflight_reports_missing_dependencies_and_artifacts(
        monkeypatch, tmp_path):
    monkeypatch.setattr(real_smoke, "_missing_modules", lambda: ["torch"])
    with pytest.raises(real_smoke.PreflightError) as exc:
        real_smoke.preflight(
            repo_dir=FIXTURE_REPO, data_dir=tmp_path / "missing-data",
            tokenizer_dir=tmp_path / "missing-tokenizer", check_cuda=False)
    message = str(exc.value)
    assert "missing Python dependencies: torch" in message
    assert "prepared data directory missing" in message
    assert "tokenizer artifact missing" in message


def test_preflight_accepts_existing_artifacts_without_network(
        monkeypatch, tmp_path):
    data, tokenizer = _artifacts(tmp_path)
    _allow_preflight(monkeypatch)
    monkeypatch.setattr(socket.socket, "connect",
                        lambda *a, **k: pytest.fail("network attempted"))
    info = real_smoke.preflight(
        repo_dir=FIXTURE_REPO, data_dir=data, tokenizer_dir=tokenizer)
    assert info["train_shards"] == ["shard_00001.parquet"]
    assert info["val_shard"] == real_smoke.VAL_SHARD


def test_prepare_workdir_copies_only_entrypoints_and_leaves_source_unchanged(
        tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("train\n")
    (source / "prepare.py").write_text("prepare\n")
    (source / "unrelated.txt").write_text("keep\n")
    before = {p.name: p.read_bytes() for p in source.iterdir()}

    work = real_smoke.prepare_workdir(tmp_path / "work", source)

    assert (work / "train.py").read_text() == "train\n"
    assert (work / "prepare.py").read_text() == "prepare\n"
    assert not (work / "unrelated.txt").exists()
    assert not (work / ".git").exists()
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


@pytest.mark.parametrize("status,val_bpb,steps,expected", [
    ("ok", 1.01, 10, True),
    ("ok", None, 10, False),
    ("ok", float("nan"), 10, False),
    ("ok", 1.01, 0, False),
    ("timeout", None, 0, False),
    ("crash", None, 0, False),
])
def test_result_validation(status, val_bpb, steps, expected):
    result = RunResult(hypothesis_id="H", gpu=0, status=status,
                       val_bpb=val_bpb, steps=steps)
    assert real_smoke.result_is_ok(result) is expected


def test_run_smoke_persists_failed_log_and_marks_failure(
        monkeypatch, tmp_path):
    data, tokenizer = _artifacts(tmp_path)
    _allow_preflight(monkeypatch)

    def fake_runner(workdir, **kwargs):
        Path(kwargs["log_path"]).write_text("RuntimeError: simulated crash\n")
        return RunResult(hypothesis_id="H", gpu=0, status="crash",
                         steps=0, wall_seconds=1.0,
                         stdout_tail="RuntimeError: simulated crash")

    monkeypatch.setattr(real_smoke, "run_training", fake_runner)
    log = tmp_path / "train.log"
    payload = real_smoke.run_smoke(
        repo_dir=FIXTURE_REPO, data_dir=data, tokenizer_dir=tokenizer,
        check_cuda=False, tmp_parent=tmp_path, log_path=log)

    assert payload["synthetic"] is False
    assert payload["llm_calls"] == 0
    assert payload["passed"] is False
    assert log.is_file() and "simulated crash" in log.read_text()
    saved = json.loads(Path(payload["result_path"]).read_text())
    assert saved["passed"] is False
    assert saved["synthetic"] is False


def test_run_smoke_persists_exception_log(monkeypatch, tmp_path):
    data, tokenizer = _artifacts(tmp_path)
    _allow_preflight(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(real_smoke, "run_training", boom)
    with pytest.raises(RuntimeError, match="launch failed"):
        real_smoke.run_smoke(
            repo_dir=FIXTURE_REPO, data_dir=data, tokenizer_dir=tokenizer,
            check_cuda=False, tmp_parent=tmp_path)

    errors = list(tmp_path.glob("autoresearch-real-smoke-*/smoke_error.log"))
    assert len(errors) == 1
    assert "launch failed" in errors[0].read_text()
