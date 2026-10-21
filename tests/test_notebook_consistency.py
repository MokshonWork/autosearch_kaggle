"""The committed smoke notebook must match its generator."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebook" / "smoke" / "autosearch_smoke.ipynb"
GENERATOR = ROOT / "notebook" / "smoke" / "make_smoke_nb.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_smoke_nb", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_committed_notebook_matches_generator():
    generated = _load_generator().build_notebook()
    committed = json.loads(NOTEBOOK.read_text())
    assert committed == generated


def test_probe_cell_uses_tested_module():
    nb = json.loads(NOTEBOOK.read_text())
    probe = next(c for c in nb["cells"]
                 if "model availability" in "".join(c["source"]))
    source = "".join(probe["source"])
    assert "from src.kaggle_model_probe import probe_models" in source
    assert "_probe_once" not in source


def test_probe_cell_is_disabled_by_default(monkeypatch, capsys):
    nb = json.loads(NOTEBOOK.read_text())
    probe = next(c for c in nb["cells"]
                 if "model availability" in "".join(c["source"]))
    source = "".join(probe["source"])

    import src.kaggle_model_probe as probe_mod
    calls = []
    monkeypatch.setattr(probe_mod, "probe_models",
                        lambda *a, **k: calls.append((a, k)))
    monkeypatch.delenv("AUTOSEARCH_MODEL_PROBE", raising=False)
    monkeypatch.chdir(ROOT)

    exec(compile(source, str(NOTEBOOK) + ":cell2b", "exec"), {})

    out = capsys.readouterr().out
    assert calls == []
    assert "Skipping optional model-availability probe" in out
    assert "AUTOSEARCH_MODEL_PROBE=1" in out


def test_live_run_cell_uses_one_experiment_config():
    nb = json.loads(NOTEBOOK.read_text())
    run = next(c for c in nb["cells"]
               if "evals/run_ab.py" in "".join(c["source"]))
    source = "".join(run["source"])
    assert "'--config', 'config.one_experiment.yaml'" in source
    assert "config.smoke.yaml" not in source


def test_validation_cell_checks_one_candidate():
    nb = json.loads(NOTEBOOK.read_text())
    validate = next(c for c in nb["cells"]
                    if "ONE_CANDIDATE_COMPLETE_ALL_CHECKS_PASS"
                    in "".join(c["source"]))
    source = "".join(validate["source"])
    assert "config.one_experiment.yaml" in source
    assert "'one_candidate_limit':" in source
    assert "'candidate_experiment_persisted':" in source
    assert "token_budget_respected" in source
    assert "candidate attempt persisted:" in source
    assert "successful trained candidate:" in source
