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
