# Compute budget (fill in measured numbers on Kaggle — none are verified yet)

## MEASURED (2026-10-16, kernel mo2xgoyal/autosearch-diag v4, T4 x2)

| Item | Measured |
|---|---|
| startup + eval (total − training) | ~7.2 s (17.3 − 10.1) |
| 10 s-budget experiment, full subprocess wall | ~19.6 s |
| steady-state throughput | ~148k tok/s (≈110 ms/step at 2^14 tokens) |
| peak VRAM (DEPTH=4, DEVICE_BATCH=64, seq 256) | 2 286 MB |
| preflight (3 steps) wall | 4.7–9.3 s |
| calibration pair (K=1, both GPUs concurrently) | ~20 s |
| GPU0-vs-GPU1 calibration mean diff (A7) | 0.000566 (std 0.0004) — no systematic offset |
| startup_allowance_s=420 in config.yaml | very safe (~60× measured startup) |

## Formula

Per experiment (one GPU): `startup_allowance_s + time_budget_s` worst case,
typically `startup + 300 s`. Per round: two experiments in PARALLEL, so a round
costs about one experiment of wall-clock but 2× GPU-seconds.

Per full run (arm): `calibration (K=3 rounds of 2) + N=40 experiments`
≈ `(3 + 40/2) × (startup + 300 s)` GPU-seconds per GPU.

A/B total: `2 arms × R repeats` runs. With R=2: 4 runs.

## Planning estimate (per run, using measured ~330 s/experiment wall incl. LLM overhead)

| Item | GPU-hours (per GPU) |
|---|---|
| 1 run (3 calib pairs + 20 rounds × 2 parallel exps) | ~23 × 330 s ≈ 2.1 h |
| A/B, R=2 (4 runs) | ≈ 8.5 h total, run sequentially |
| Note | A single Kaggle session may not fit 4 runs; plan one session per arm-repeat (≈2.2 h each) |

API budget: `max_api_calls_total: 400` is a hard cap on **outbound provider
HTTP attempts** per run (retries, fallback POSTs, timeouts, and failed responses
all count); `max_calls_per_round: 12` applies the same rule per round.
`max_tokens_total: 800000` counts tokens reported by successful generations
only. Adjust `max_rounds`/`max_experiments` after measuring real startup time
(A8) and per-experiment wall-clock (E13).

## What to measure on Kaggle before launching (notebook does this)

1. prepare.py data/tokenizer time (one-time per session). **MEASURED: ~5-7 min fresh session.**
2. Single run: startup seconds, steady-state tok/s, val_bpb, peak VRAM. **DONE (table above).**
3. Calibration: 3 concurrent pairs → std, per-GPU means → epsilon. **K=1 verified; K=3 in production run.**
4. Recompute this table with real numbers and update config.yaml.

