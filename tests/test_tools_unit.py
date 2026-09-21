"""Unit tests: patching (G3/G4), ast_checks (G5), metrics_parser, hashing (G1)."""
import pytest

from src.schemas import Edit
from src.tools import ast_checks, metrics_parser, patching
from src.tools.hashing import HashRegistry, ProtectedFileError


def test_apply_edits_ok():
    src = "a = 1\nb = 2\n"
    out = patching.apply_edits(src, [Edit(find="a = 1", replace="a = 3")])
    assert out == "a = 3\nb = 2\n"


def test_apply_edits_nonunique_find():
    with pytest.raises(patching.PatchError):
        patching.apply_edits("x = 1\nx = 1\n", [Edit(find="x = 1", replace="y")])


def test_apply_edits_max_edits():
    edits = [Edit(find="a", replace="b")] * 9
    with pytest.raises(patching.PatchError):
        patching.apply_edits("a", edits, max_edits=8)


def test_protected_regions(mock_train_src):
    assert patching.check_protected_regions(mock_train_src) == []
    bad = mock_train_src.replace("torch.manual_seed(42)", "torch.manual_seed(7)")
    assert any("manual_seed" in v for v in patching.check_protected_regions(bad))


def test_changed_line_count_and_diff():
    a = "x = 1\ny = 2\n"
    b = "x = 1\ny = 3\n"
    assert patching.changed_line_count(a, b) == 2
    assert "+y = 3" in patching.unified_diff(a, b)


def test_compiles():
    assert patching.compiles("x = 1\n")[0]
    assert not patching.compiles("def broken(:\n")[0]


@pytest.mark.parametrize("evil", [
    "import subprocess",
    "import socket",
    "import requests",
    "import shutil",
    "from urllib import request",
    "import os\nos.system('ls')",
    "import os\nprint(os.environ['HOME'])",
    "import os\nprint(os.getenv('HOME'))",
    "eval('1+1')",
    "exec('x=1')",
    "__import__('os')",
    "open('/tmp/f', 'w').write('x')",
    "from pathlib import Path\nPath('/tmp/f').write_text('x')",
    "x = evaluate_bpb",
    "print('val_bpb: 0.100000')",
])
def test_ast_blocks_added_forbidden(mock_train_src, evil):
    patched = mock_train_src.replace(
        "DEVICE_BATCH_SIZE = 64", f"DEVICE_BATCH_SIZE = 64\n{evil}")
    assert ast_checks.added_forbidden(mock_train_src, patched), evil


def test_ast_allows_preexisting_and_benign(mock_train_src):
    # The mock train.py already reads env vars; identical file => no findings.
    assert ast_checks.added_forbidden(mock_train_src, mock_train_src) == []
    benign = mock_train_src.replace("DEVICE_BATCH_SIZE = 64",
                                    "DEVICE_BATCH_SIZE = 32")
    assert ast_checks.added_forbidden(mock_train_src, benign) == []


def test_metrics_parse_and_classify():
    stdout = ("---\nval_bpb:          0.997900\ntraining_seconds: 300.1\n"
              "peak_vram_mb:     45060.2\nnum_steps:        953\n")
    m = metrics_parser.parse_metrics(stdout)
    assert m["val_bpb"] == pytest.approx(0.9979)
    assert m["num_steps"] == 953
    assert metrics_parser.classify_status(stdout, "", 0) == "ok"
    assert metrics_parser.classify_status("", "CUDA out of memory", 1) == "oom"
    assert metrics_parser.classify_status("FAIL\n", "", 1) == "nan"
    assert metrics_parser.classify_status("Traceback", "err", 1) == "crash"
    assert metrics_parser.classify_status("", "", -9, timed_out=True) == "timeout"
    assert metrics_parser.classify_status("val_bpb: nan", "", 0) == "nan"
    assert metrics_parser.stdout_tail("a\nb\nc", max_lines=2) == "b\nc"


def test_hash_registry(tmp_path):
    f = tmp_path / "prepare.py"
    f.write_text("x = 1")
    reg = HashRegistry([f])
    reg.verify()
    f.write_text("x = 2")
    with pytest.raises(ProtectedFileError):
        reg.verify()
