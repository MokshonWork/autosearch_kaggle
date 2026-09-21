# Compute budget (fill in measured numbers on Kaggle — none are verified yet)

## Formula

Per experiment (one GPU): `startup_allowance_s + time_budget_s` worst case,
typically `startup + 300 s`. Per round: two experiments in PARALLEL, so a round
costs about one experiment of wall-clock but 2× GPU-seconds.

Per full run (arm): `calibration (K=3 rounds of 2) + N=40 experiments`
≈ `(3 + 40/2) × (startup + 300 s)` GPU-seconds per GPU.

A/B total: `2 arms × R repeats` runs. With R=2: 4 runs.

## Placeholder estimate (UNVERIFIED — measure A8 on Kaggle first)

Assuming startup+eval ≈ 120 s and 300 s budget → ~420 s per experiment slot:

| Item | GPU-hours (per GPU) |
|---|---|
| 1 run (3 calib rounds + 20 rounds) | 23 × 420 s ≈ 2.7 h |
| A/B, R=2 (4 runs) | ≈ 10.7 h |
| Headroom under ~30 GPU-h/week quota | comfortable |

API budget: `max_api_calls_total: 400`, `max_tokens_total: 800000` per run
(config.yaml). Adjust `max_rounds`/`max_experiments` after measuring real
startup time (A8) and per-experiment wall-clock (E13).

## What to measure on Kaggle before launching (notebook does this)

1. prepare.py data/tokenizer time (one-time per session).
2. Single run: startup seconds, steady-state tok/s, val_bpb, peak VRAM.
3. Calibration: 3 concurrent pairs → std, per-GPU means → epsilon.
4. Recompute this table with real numbers and update config.yaml.
