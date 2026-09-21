"""Search/replace patching (G4) and protected-region checks (G3).

A Patch is a list of {find, replace} edits applied to train.py source text.
Each `find` must match EXACTLY once. The patched file must compile.
"""
from __future__ import annotations

import difflib
import py_compile
import tempfile
from pathlib import Path

# G3: byte-identical snippets that must survive every patch (see context.md).
PROTECTED_SNIPPETS: list[str] = [
    "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb",
    "torch.manual_seed(42)",
    "val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)",
    'print(f"val_bpb:          {val_bpb:.6f}")',
    "if step > 10 and total_training_time >= TIME_BUDGET:",
]


class PatchError(ValueError):
    pass


def apply_edits(source: str, edits: list, max_edits: int = 8) -> str:
    """Apply find/replace edits. Raises PatchError on any violation (G4)."""
    if len(edits) > max_edits:
        raise PatchError(f"too many edits: {len(edits)} > {max_edits}")
    out = source
    for i, edit in enumerate(edits):
        find, replace = edit.find, edit.replace
        count = out.count(find)
        if count != 1:
            raise PatchError(
                f"edit {i}: 'find' must match exactly once, matched {count} times"
            )
        out = out.replace(find, replace, 1)
    return out


def check_protected_regions(patched: str, snippets: list[str] | None = None) -> list[str]:
    """Return list of violated protected snippets (empty = OK) (G3)."""
    snippets = PROTECTED_SNIPPETS if snippets is None else snippets
    return [s for s in snippets if s not in patched]


def changed_line_count(original: str, patched: str) -> int:
    """Number of +/- lines in the unified diff (G4 diff-size limit)."""
    diff = difflib.unified_diff(
        original.splitlines(), patched.splitlines(), lineterm=""
    )
    return sum(
        1 for line in diff
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith(("+++", "---"))
    )


def unified_diff(original: str, patched: str, path: str = "train.py") -> str:
    return "\n".join(
        difflib.unified_diff(
            original.splitlines(), patched.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
        )
    )


def compiles(source: str) -> tuple[bool, str]:
    """py_compile check on patched source (G4). Returns (ok, error)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source)
        tmp = f.name
    try:
        py_compile.compile(tmp, doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, str(e)
    finally:
        Path(tmp).unlink(missing_ok=True)
