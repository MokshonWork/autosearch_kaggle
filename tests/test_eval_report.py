"""Evaluation report metrics + honesty guards (offline, no GPU/API)."""
import json
from pathlib import Path

import pytest

import evals.report as report
from evals.metrics import load_run, run_metrics


def _exp(hid, status="ok", val_bpb=1.0, action="keep", wall=10.0,
         category="optimizer", summary="tweak"):
    return {"hypothesis": {"id": hid, "category": category,
                           "change_summary": summary},
            "result": {"status": status, "val_bpb": val_bpb,
                       "wall_seconds": wall},
            "decision": {"action": action}}


def _config(**overrides):
    cfg = {
        "llm": {"provider": "gemini", "fallback_model": ""},
        "budget": {"max_api_calls_total": 500},
        "training": {"seed": 42},
        "decision": {"min_delta": 0.002},
        "guard": {"max_edits": 8},
        "dedupe": {"text_similarity_threshold": 0.85},
        "mode": {"dry_run": False},
    }
    cfg.update(overrides)
    return cfg


def _write_comparable(root: Path, source: dict) -> None:
    ma_state = {"baseline_bpb": 1.2, "best_val_bpb": 1.0, "epsilon": 0.01}
    b1_state = {"baseline_bpb": 1.2, "best_val_bpb": 1.2, "epsilon": 0.01}
    for i in (0, 1):
        _write_run(root, "multi_agent", i,
                   [_exp(f"MA-{i}", val_bpb=1.0)], ma_state, source)
        _write_run(root, "baseline_single", i,
                   [_exp(f"B1-{i}", val_bpb=1.2)], b1_state, source)


def _write_run(root: Path, arm: str, run: int, exps, state,
               source: dict | None = None) -> Path:
    run_dir = root / arm / f"run{run}"
    mem = run_dir / "memory"
    mem.mkdir(parents=True)
    (mem / "experiments.jsonl").write_text(
        "\n".join(json.dumps(e) for e in exps) + ("\n" if exps else ""))
    (mem / "state.json").write_text(json.dumps(state))
    if source is not None:
        (mem / "source.json").write_text(json.dumps(source))
    return run_dir


def _metrics(best, eps=0.01):
    return {"best_val_bpb": best, "epsilon": eps}


def test_load_run_missing_is_empty_not_an_error(tmp_path):
    assert load_run(tmp_path / "nope") == {
        "experiments": [], "state": {}, "load_errors": []}


def test_load_run_skips_malformed_lines_and_state(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "experiments.jsonl").write_text(
        json.dumps(_exp("H-1")) + "\nnot-json\n[]\n42\n")
    (mem / "state.json").write_text("{broken")
    run = load_run(mem)
    assert len(run["experiments"]) == 1
    assert run["state"] == {}
    assert len(run["load_errors"]) == 4


def test_run_metrics_missing_and_failure_accounting(tmp_path):
    empty = run_metrics(load_run(tmp_path / "missing"))
    assert empty["n_experiments"] == 0
    assert empty["failure_rate"] == 0.0
    assert empty["keep_rate"] == 0.0
    assert empty["best_val_bpb"] is None

    _write_run(tmp_path, "multi_agent", 0, [
        _exp("H-1", "ok", 1.0, "keep"),
        _exp("H-2", "crash", None, "discard", wall=20.0),
        _exp("H-3", "guard_reject", None, "discard", wall=5.0),
    ], {"baseline_bpb": 1.2, "best_val_bpb": 1.0, "api_calls": 7,
        "api_tokens": 700})
    m = run_metrics(load_run(tmp_path / "multi_agent" / "run0" / "memory"))
    assert m["n_experiments"] == 3
    assert m["improvement"] == pytest.approx(0.2)
    assert m["keep_rate"] == pytest.approx(1 / 3)
    assert m["failure_rate"] == pytest.approx(1 / 3)
    assert m["guard_reject_rate"] == pytest.approx(1 / 3)
    assert m["wasted_gpu_minutes"] == pytest.approx(25 / 60)
    assert m["api_calls"] == 7 and m["api_tokens"] == 700


def test_verdict_requires_repeats_and_rejects_non_comparable():
    one_ma = [_metrics(1.0)]
    one_b1 = [_metrics(1.2)]
    assert "INSUFFICIENT DATA" in report.verdict(one_ma, one_b1)

    ma = [_metrics(1.0), _metrics(1.0)]
    b1 = [_metrics(1.2), _metrics(1.2)]
    assert "MULTI-AGENT BETTER" in report.verdict(ma, b1)

    b1_noise = [_metrics(1.005), _metrics(1.005)]
    assert "WITHIN NOISE" in report.verdict(ma, b1_noise)

    b1_better = [_metrics(0.8), _metrics(0.8)]
    assert "BASELINE BETTER" in report.verdict(ma, b1_better)


def test_generate_report_repeats_and_commit_comparability(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_plot", lambda *a, **k: None)
    root = tmp_path / "results"
    ma_state = {"baseline_bpb": 1.2, "best_val_bpb": 1.0, "epsilon": 0.01}
    b1_state = {"baseline_bpb": 1.2, "best_val_bpb": 1.2, "epsilon": 0.01}
    src = {"start_commit": "abc123", "config": _config()}
    for i in (0, 1):
        _write_run(root, "multi_agent", i, [_exp(f"MA-{i}", val_bpb=1.0)],
                   ma_state, src)
        _write_run(root, "baseline_single", i,
                   [_exp(f"B1-{i}", val_bpb=1.2)], b1_state, src)

    text = report.generate_report(root)
    assert (root / "ab_report.md").exists()
    assert "MULTI-AGENT BETTER" in text

    (root / "baseline_single" / "run1" / "memory" / "source.json"
     ).write_text(json.dumps({"start_commit": "different",
                              "config": _config()}))
    text = report.generate_report(root)
    assert "NOT COMPARABLE" in text and "different commits" in text

    (root / "baseline_single" / "run1" / "memory" / "source.json"
     ).write_text(json.dumps({"start_commit": "abc123", "config": _config(
         budget={"max_api_calls_total": 501})}))
    text = report.generate_report(root)
    assert "NOT COMPARABLE" in text and "material config" in text

    (root / "baseline_single" / "run1" / "memory" / "source.json"
     ).write_text("{}")
    text = report.generate_report(root)
    assert "NOT COMPARABLE" in text and "provenance" in text

    (root / "baseline_single" / "run1" / "memory" / "experiments.jsonl"
     ).unlink()
    (root / "baseline_single" / "run1" / "memory" / "state.json").unlink()
    (root / "baseline_single" / "run1" / "memory" / "source.json").unlink()
    (root / "baseline_single" / "run1" / "memory").rmdir()
    text = report.generate_report(root)
    assert "INSUFFICIENT DATA" in text


def test_generate_report_accepts_matching_synthetic_modes(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_plot", lambda *a, **k: None)
    root = tmp_path / "results"
    source = {"start_commit": "abc123", "synthetic": True,
              "config": _config(mode={"dry_run": True})}
    _write_comparable(root, source)
    text = report.generate_report(root)
    assert "MULTI-AGENT BETTER" in text


def test_generate_report_rejects_mixed_synthetic_and_real_modes(
        tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_plot", lambda *a, **k: None)
    root = tmp_path / "results"
    real = {"start_commit": "abc123", "synthetic": False,
            "config": _config(mode={"dry_run": False})}
    _write_comparable(root, real)
    synthetic = {"start_commit": "abc123",
                 "config": _config(mode={"dry_run": True})}
    (root / "baseline_single" / "run1" / "memory" / "source.json"
     ).write_text(json.dumps(synthetic))
    text = report.generate_report(root)
    assert "NOT COMPARABLE" in text and "execution mode" in text


def test_generate_report_marks_synthetic_failed_run_incomplete(
        tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_plot", lambda *a, **k: None)
    root = tmp_path / "results"
    source = {"start_commit": "abc123", "synthetic": True,
              "config": _config(mode={"dry_run": True})}
    failed = {"experiments_done": 0, "stopped_reason": "LLMError",
              "round_status": "failed", "baseline_bpb": None,
              "best_val_bpb": None}
    for arm in ("multi_agent", "baseline_single"):
        for i in (0, 1):
            _write_run(root, arm, i, [], failed, source)

    text = report.generate_report(root)
    assert "SYNTHETIC" in text
    assert "not evidence of real model improvement" in text
    assert "run0:failed" in text and "run1:failed" in text
    assert "**Verdict:** INCOMPLETE" in text
    assert "MULTI-AGENT BETTER" not in text
