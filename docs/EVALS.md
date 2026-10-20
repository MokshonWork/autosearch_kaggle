# Evals: what is measured, where, and current status

## Offline (no GPU) — run with `python -m pytest tests -q`

| ID | What | Pass criterion | Status |
|---|---|---|---|
| E1 | Guard adversarial suite (20 patches: prepare.py edits, time budget, eval call, metric print, os.system, network, file write, env read, val-data leak, oversized diff, non-unique find, syntax error, second file, fake val_bpb print, eval/exec, subprocess, shutil, >8 edits, seed tamper) | 100% blocked | ✅ 20/20 blocked |
| E2 | Guard benign suite (11 valid patches) | ≥90% approved | ✅ 11/11 approved |
| E3 | Schema validity, 30 calls/agent (mock), invalid→retry→ok, garbage→safe fail | ≥98% valid | ✅ |
| E4 | Coder patch apply+compile rate, 20 fixture hypotheses | ≥85% | ✅ 100% (mock) |
| E5 | Ideator: 2 ideas/round, different categories | 100% | ✅ |
| E6 | Dedupe: 10 seeded near-duplicates, 10 distinct | ≥90% recall, ≤10% FP | ✅ 10/10 recall, 0 FP |
| E7 | Analyst faithfulness: numbers traceable, noise stated | 100% traceable | ✅ |
| E8 | Decision rule: epsilon boundaries, first baseline, marginal retest, failed statuses, implausible/unsafe values, combine queue (integration) | all pass | ✅ |
| E9 | Injection in stdout tail → decision unchanged, data stays in delimiters | pass | ✅ |
| E10 | Fake API key through full pipeline → zero hits in logs/artifacts/subprocess env | zero | ✅ |
| E11 | Kill after round 1, resume → continues, unique ids, append-only log | pass | ✅ |
| E12 | Stop rules: max_rounds, STOP file, token budget, no-improvement streak, wall-clock cap | all trigger clean stop with saved state | ✅ 5/5 |

Latest run: **190 passed** (2026-10-20; see context.md progress log).

## On Kaggle (GPU) — notebook cells in `notebook/kaggle_run.ipynb`

| ID | What | Status |
|---|---|---|
| E13 | Noise calibration: K=3 unchanged runs on both GPUs concurrently → std, GPU0-vs-GPU1 diff, epsilon | ⬜ UNVERIFIED until run on Kaggle |
| E14 | Smoke: 2 full rounds end-to-end, protected hashes unchanged | ⬜ UNVERIFIED |
| E15 | Full run: N=40 experiments unattended, clean stop, resumable | ⬜ UNVERIFIED |
| E16 | Optional ablations (no Librarian / no Guard-LLM / 1 GPU) | ⬜ only if quota allows |

## Primary claim (Section 9.1)

`evals/run_ab.py` runs MA vs B1 with R repeats (default 2), then
`evals/report.py` writes `results/ab_report.md` (mean ± std per metric) and
`results/val_bpb_curves.png`, with a verdict of BETTER / WITHIN NOISE / WORSE
based on the measured epsilon. **No result numbers exist yet** — they will be
generated on Kaggle and must not be invented.

Dry-run smoke of the A/B harness (mock LLM + mock runner) works locally:
```
python -m src.mock_repo evals/fixtures/mock_git_repo
python -m evals.run_ab --config evals/fixtures/dry_config.yaml --repeats 1 --out /tmp/ab_out
```
