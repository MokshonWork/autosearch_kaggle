"""LLM-free real training smoke test.

Preflight the prepared cache/dependencies, copy only `repo/train.py` and
`repo/prepare.py` into a dedicated temp directory, then run one bounded
training job through the existing runner. No orchestrator, no LLM, no network.

Usage:
    python -m evals.real_smoke --preflight-only
    python -m evals.real_smoke --gpu 0 --time-budget 10 --startup-allowance 180
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from src.schemas import RunResult
from src.tools.metrics_parser import parse_metrics
from src.tools.runner import run_training

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_ROOT / "repo"
CACHE_DIR = Path.home() / ".cache" / "autoresearch"
DATA_DIR = CACHE_DIR / "data"
TOKENIZER_DIR = CACHE_DIR / "tokenizer"
VAL_SHARD = "shard_06542.parquet"
REQUIRED_MODULES = ("torch", "pyarrow", "requests", "rustbpe", "tiktoken")
TRAIN_FILES = ("train.py", "prepare.py")
HYPOTHESIS_ID = "SMOKE-real-0001"


class PreflightError(RuntimeError):
    """Actionable failure before any training subprocess starts."""


def _missing_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES
            if importlib.util.find_spec(name) is None]


def _cuda_errors(gpu: int) -> list[str]:
    import torch

    if not torch.cuda.is_available():
        return ["CUDA is not available to this interpreter"]
    count = torch.cuda.device_count()
    if gpu < 0 or gpu >= count:
        return [f"GPU {gpu} requested but only {count} CUDA device(s) visible"]
    return []


def _artifact_errors(data_dir: Path, tokenizer_dir: Path) -> list[str]:
    errors: list[str] = []
    data_dir = Path(data_dir)
    tokenizer_dir = Path(tokenizer_dir)

    if not data_dir.is_dir():
        errors.append(f"prepared data directory missing: {data_dir}")
    else:
        shards = sorted(data_dir.glob("*.parquet"))
        train_shards = [p for p in shards if p.name != VAL_SHARD]
        if not train_shards:
            errors.append(
                f"no training shards in {data_dir} "
                f"(expected *.parquet other than {VAL_SHARD})")
        if not (data_dir / VAL_SHARD).is_file():
            errors.append(f"validation shard missing: {data_dir / VAL_SHARD}")

    for name in ("tokenizer.pkl", "token_bytes.pt"):
        if not (tokenizer_dir / name).is_file():
            errors.append(f"tokenizer artifact missing: {tokenizer_dir / name}")
    return errors


def _source_errors(repo_dir: Path) -> list[str]:
    return [
        f"training source missing: {Path(repo_dir) / name}"
        for name in TRAIN_FILES
        if not (Path(repo_dir) / name).is_file()
    ]


def preflight(
    *,
    repo_dir: Path = REPO_DIR,
    data_dir: Path = DATA_DIR,
    tokenizer_dir: Path = TOKENIZER_DIR,
    gpu: int = 0,
    check_cuda: bool = True,
) -> dict:
    """Return a small evidence dict, or raise one aggregated PreflightError."""
    missing = _missing_modules()
    errors: list[str] = []
    if missing:
        errors.append(
            "missing Python dependencies: " + ", ".join(missing)
            + f" (interpreter: {sys.executable})")
    errors.extend(_source_errors(repo_dir))
    errors.extend(_artifact_errors(data_dir, tokenizer_dir))
    if check_cuda and "torch" not in missing:
        try:
            errors.extend(_cuda_errors(gpu))
        except Exception as exc:
            errors.append(
                f"CUDA check failed: {type(exc).__name__}: {exc}")
    if errors:
        raise PreflightError(
            "real training smoke preflight failed:\n- " + "\n- ".join(errors))

    data_dir = Path(data_dir)
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "repo_dir": str(Path(repo_dir)),
        "data_dir": str(data_dir),
        "tokenizer_dir": str(Path(tokenizer_dir)),
        "train_shards": sorted(p.name for p in data_dir.glob("*.parquet")
                               if p.name != VAL_SHARD),
        "val_shard": VAL_SHARD,
        "gpu": gpu,
    }


def prepare_workdir(workdir: Path, repo_dir: Path = REPO_DIR) -> Path:
    """Copy only the training entry points; leave the source checkout untouched."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for name in TRAIN_FILES:
        shutil.copy2(Path(repo_dir) / name, workdir / name)
    return workdir


def result_is_ok(result: RunResult) -> bool:
    if result.status != "ok" or result.steps <= 0:
        return False
    try:
        val_bpb = float(result.val_bpb)
    except (TypeError, ValueError):
        return False
    return math.isfinite(val_bpb) and val_bpb > 0


def run_smoke(
    *,
    gpu: int = 0,
    time_budget_s: int = 10,
    startup_allowance_s: int = 180,
    tmp_parent: Path | None = None,
    log_path: Path | None = None,
    repo_dir: Path = REPO_DIR,
    data_dir: Path = DATA_DIR,
    tokenizer_dir: Path = TOKENIZER_DIR,
    check_cuda: bool = True,
) -> dict:
    """Run one real training smoke and return a JSON-serialisable summary."""
    info = preflight(repo_dir=repo_dir, data_dir=data_dir,
                     tokenizer_dir=tokenizer_dir, gpu=gpu,
                     check_cuda=check_cuda)
    parent = Path(tmp_parent) if tmp_parent else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="autoresearch-real-smoke-",
                                    dir=str(parent)))
    prepare_workdir(workdir, repo_dir)
    train_log = Path(log_path) if log_path else workdir / "train.log"

    try:
        result = run_training(
            workdir, gpu=gpu, hypothesis_id=HYPOTHESIS_ID,
            time_budget_s=time_budget_s,
            startup_allowance_s=startup_allowance_s,
            log_path=train_log,
        )
    except Exception:
        (workdir / "smoke_error.log").write_text(traceback.format_exc())
        raise

    log_metrics = parse_metrics(train_log.read_text()) if train_log.is_file() else {}
    passed = result_is_ok(result)
    payload = {
        "kind": "real_training_smoke",
        "synthetic": False,
        "ab_comparison": False,
        "llm_calls": 0,
        "hypothesis_id": HYPOTHESIS_ID,
        "passed": passed,
        "preflight": info,
        "workdir": str(workdir),
        "train_log": str(train_log),
        "result": result.model_dump(),
        "training_seconds": log_metrics.get("training_seconds"),
        "num_steps": log_metrics.get("num_steps"),
    }
    result_path = workdir / "result.json"
    payload["result_path"] = str(result_path)
    result_path.write_text(json.dumps(payload, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--time-budget", type=int, default=10)
    parser.add_argument("--startup-allowance", type=int, default=180)
    parser.add_argument("--tmp-parent", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.preflight_only:
            print(json.dumps(preflight(gpu=args.gpu), indent=2))
            return 0
        payload = run_smoke(
            gpu=args.gpu,
            time_budget_s=args.time_budget,
            startup_allowance_s=args.startup_allowance,
            tmp_parent=args.tmp_parent,
            log_path=args.log,
        )
    except PreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        print(f"SMOKE FAILED: see {payload['train_log']}", file=sys.stderr)
        return 1
    print(f"SMOKE OK: val_bpb={payload['result']['val_bpb']} "
          f"log={payload['train_log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
