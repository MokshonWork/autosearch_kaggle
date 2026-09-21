"""G5: AST-diff check on NEWLY ADDED code only.

We extract a multiset of "features" (imports, calls, attribute reads, dangerous
string literals) from the original and the patched train.py. A feature is
suspicious only if it appears MORE often in the patched file than in the
original — so pre-existing os.environ lines in the human T4 port are fine, but
any new forbidden construct added by an agent's patch is blocked.
"""
from __future__ import annotations

import ast
from collections import Counter

# Identifiers that would give training access to validation data or the metric.
VAL_IDENTIFIERS = {"evaluate_bpb", "val_loader", "get_token_bytes"}

_FORBIDDEN_IMPORTS = {
    "subprocess", "socket", "requests", "urllib", "http", "ftplib", "shutil",
}
_FORBIDDEN_CALLS = {
    "os.system", "os.popen", "eval", "exec", "__import__", "compile",
    "os.getenv", "os.remove", "os.unlink", "os.rmdir", "os.rename",
}


def _dotted(func: ast.expr) -> str:
    """Resolve a call target to a dotted name, e.g. os.system -> 'os.system'."""
    parts = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def extract_features(source: str) -> Counter:
    """Multiset of security-relevant features in source. Raises SyntaxError."""
    tree = ast.parse(source)
    feats: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                feats[("import", a.name.split(".")[0])] += 1
        elif isinstance(node, ast.ImportFrom):
            feats[("import", (node.module or "").split(".")[0])] += 1
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            feats[("call", name)] += 1
            if name == "open":
                mode = "r"
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    mode = str(node.args[1].value)
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = str(kw.value.value)
                if any(m in mode for m in "wax+"):
                    feats[("call", "open_write")] += 1
            base = name.rsplit(".", 1)[-1]
            if base in {"write", "write_text", "write_bytes"}:
                feats[("call", "file_write")] += 1
        elif isinstance(node, ast.Attribute):
            if node.attr == "environ" and isinstance(node.value, ast.Name) \
                    and node.value.id == "os":
                feats[("attr", "os.environ")] += 1
        elif isinstance(node, ast.Name) and node.id in VAL_IDENTIFIERS:
            feats[("name", node.id)] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "val_bpb" in node.value:
                feats[("str", "val_bpb_literal")] += 1
    return feats


def added_forbidden(original: str, patched: str) -> list[str]:
    """Return human-readable violations for features added by the patch (G5)."""
    before = extract_features(original)
    after = extract_features(patched)
    violations: list[str] = []
    for (kind, name), count in sorted((after - before).items()):
        label = f"{kind}:{name}"
        if kind == "import" and name in _FORBIDDEN_IMPORTS:
            violations.append(f"new import of forbidden module '{name}'")
        elif kind == "call" and (
            name in _FORBIDDEN_CALLS
            or name.split(".")[0] in _FORBIDDEN_IMPORTS
            or name in {"open_write", "file_write"}
        ):
            violations.append(f"new forbidden call '{name}'")
        elif kind == "attr" and name == "os.environ":
            violations.append("new read of os.environ")
        elif kind == "name" and name in VAL_IDENTIFIERS:
            violations.append(
                f"new reference to validation/metric identifier '{name}'")
        elif kind == "str" and name == "val_bpb_literal":
            violations.append(
                "new string literal containing 'val_bpb' (fake metric risk)")
    return violations
