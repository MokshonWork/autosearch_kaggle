# Real training-only smoke (Kaggle)

One bounded `repo/train.py` run with **no LLM, no orchestrator, no A/B, no
network**. It proves the training entry point can load the prepared cache, train
briefly, evaluate, and emit a finite `val_bpb`.

## Prerequisites

- Kaggle notebook with **GPU T4 x2**, internet ON for the clone/install/prepare
  cells only.
- Prepared cache under `$HOME/.cache/autoresearch/`:
  - `data/shard_06542.parquet` (validation)
  - at least one other `data/*.parquet` (training)
  - `tokenizer/tokenizer.pkl`
  - `tokenizer/token_bytes.pt`
- Python environment with `torch`, `pyarrow`, `requests`, `rustbpe`, `tiktoken`.
- No `GEMINI_API_KEY` needed. Do not run the secret/model-probe cells.
- `repo/.git` is **not** needed: the smoke copies only `train.py` and
  `prepare.py` into a temp directory.

## Kaggle cell sequence

Use the existing `notebook/kaggle_run.ipynb` clone/install cell (cell 1) and its
data-prep cell (cell 3, if the cache is absent), then run:

```python
# Training-only smoke: preflight, one GPU, 10s training budget, hard 190s cap.
# Full log + result.json stay under /kaggle/working/real_smoke/ and are printed.
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "evals.real_smoke",
     "--gpu", "0",
     "--time-budget", "10",
     "--startup-allowance", "180",
     "--tmp-parent", "/kaggle/working/real_smoke"],
    check=True,
)
```

Do **not** run the secret, model-probe, calibration, or `evals.run_ab` cells for
this smoke.

To check only the environment:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "evals.real_smoke",
                "--preflight-only"], check=True)
```

## Output

- Unique temp dir: `/kaggle/working/real_smoke/autoresearch-real-smoke-*`
- `train.log`: complete `train.py` stdout + stderr (written even on failure)
- `result.json`: preflight evidence, `RunResult`, metrics, `passed`
- Exit `0` only when `status == "ok"`, `steps > 0`, and `val_bpb` is finite
  and positive.

`result.json` is marked `"synthetic": false`, `"ab_comparison": false`,
`"llm_calls": 0`.

## Local status (2026-10-21)

`python -m evals.real_smoke --preflight-only` fails safely, as expected:

```
missing Python dependencies: torch, rustbpe
prepared data directory missing: ~/.cache/autoresearch/data
tokenizer artifact missing: ~/.cache/autoresearch/tokenizer/tokenizer.pkl
tokenizer artifact missing: ~/.cache/autoresearch/tokenizer/token_bytes.pt
```

No real training run or GPU training claim has been made.

## Limits

- This is a smoke, not a benchmark: one GPU, one short run, no baseline/A-B
  comparison, no checkpoint.
- It does not validate LLM agents, calibration, multi-GPU scheduling, or the
  full campaign.
- After a finite `val_bpb` is observed, separately verify calibration-only; only
  then restore LLM orchestration.

## Tests

```bash
python -m pytest tests/test_real_smoke.py tests/test_runner_git_memory.py -q
```
