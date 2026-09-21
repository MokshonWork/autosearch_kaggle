# DEMO plan (workshop)

## Before the session (night before)

1. Open `notebook/kaggle_run.ipynb` on Kaggle, accelerator **GPU T4 x2**,
   internet ON, add the `GEMINI_API_KEY` secret.
2. "Save & Run All" — the notebook prepares data, calibrates, runs the
   multi-agent arm, then the baseline arm, and writes `results/`.
3. Download the output (or reattach it as an input dataset next session).

## On stage (~10 min, no live 5-minute training)

1. **The problem** (1 min): manual tuning is slow; a single agent repeats
   itself, forgets, and can accidentally cheat the metric.
2. **The system** (2 min): show `docs/ARCHITECTURE.md` diagram; each agent's
  1-line job; "code decides, LLMs propose".
3. **The logs** (3 min): open `memory/experiments.jsonl` and `lessons.md` —
   every experiment has a hypothesis, guard report, measured `val_bpb`, a
   decision with a reason code, and a one-line lesson.
4. **The curve** (2 min): `results/val_bpb_curves.png` and
   `results/ab_report.md` — MA vs B1, mean ± std, and the honest verdict.
5. **The guardrails** (2 min): live-replay of eval E1 — run
   `python -m pytest tests/test_guard_e1_e2.py -q` on stage (2 seconds,
   no GPU) and show 20/20 attack patches blocked.

## Replay mode (optional)

```
python -m src.mock_repo /tmp/mock && \
python -m src.orchestrator --config evals/fixtures/dry_config.yaml --dry-run
```

streams a full recorded-style dry run (mock LLM + mock runner) in seconds —
safe fallback if Kaggle is down on demo day.

## If asked for a live GPU run

Run only the calibration cell (≈ K×2 short runs) and one dry-run round with
`max_rounds: 1` — never promise a full run live; show the pre-run results
instead.
