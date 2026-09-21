"""Parse the train.py summary block and classify run status (G9, G10)."""
from __future__ import annotations

import math
import re

_PATTERNS = {
    "val_bpb": re.compile(r"^val_bpb:\s+([-\d.eE]+)", re.MULTILINE),
    "peak_vram_mb": re.compile(r"^peak_vram_mb:\s+([-\d.eE]+)", re.MULTILINE),
    "num_steps": re.compile(r"^num_steps:\s+(\d+)", re.MULTILINE),
    "training_seconds": re.compile(r"^training_seconds:\s+([-\d.eE]+)", re.MULTILINE),
}

_OOM_MARKERS = ("out of memory", "OutOfMemoryError", "CUDA error: out of memory")
_NAN_MARKERS = ("FAIL", "nan", "NaN")


def parse_metrics(stdout: str) -> dict:
    """Extract metrics printed by train.py. Missing keys default to None."""
    out: dict = {}
    for key, pat in _PATTERNS.items():
        m = pat.search(stdout)
        out[key] = float(m.group(1)) if m else None
    if out.get("num_steps") is not None:
        out["num_steps"] = int(out["num_steps"])
    return out


def classify_status(stdout: str, stderr: str, returncode: int,
                    timed_out: bool = False) -> str:
    """Map raw subprocess output to a RunResult status (never 'ok' on failure)."""
    if timed_out:
        return "timeout"
    blob = stdout + "\n" + stderr
    if any(m in blob for m in _OOM_MARKERS):
        return "oom"
    if "PREFLIGHT_OK" in stdout:
        return "ok"  # preflight success (caller treats this specially)
    metrics = parse_metrics(stdout)
    val = metrics.get("val_bpb")
    if returncode == 0 and val is not None:
        if not math.isfinite(val) or val <= 0:
            return "nan"
        return "ok"
    if any(m in blob for m in _NAN_MARKERS):
        return "nan"
    return "crash"


def stdout_tail(stdout: str, max_lines: int = 60) -> str:
    """Keep at most the last `max_lines` lines (contract: stdout_tail <= 60)."""
    lines = stdout.splitlines()
    return "\n".join(lines[-max_lines:])
