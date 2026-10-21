# autoresearch-team

A **multi-agent autonomous research system** built on top of Andrej Karpathy's
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) (MIT).
Specialised LLM agents (Ideator, Coder, Guard, Analyst, Librarian) propose,
implement, validate, judge and remember code-level experiments on a small GPT
training script, running unattended on **Kaggle with two T4 GPUs** (one
experiment per GPU, no DDP). A deterministic orchestrator — not an LLM — owns
all keep/discard decisions using a measured noise floor.

**Attribution / what's new.** `repo/` is a fork of `karpathy/autoresearch`
(MIT, see `LICENSE`) with a one-time human "T4 port" (SDPA instead of
FlashAttention-3, fp16 + GradScaler instead of bf16, `torch.compile` behind a
flag, small-GPU config, pre-flight hook; commit `9cce20e`). Everything outside
`repo/` is new: the multi-agent harness (`src/`), prompts (`prompts/`),
guardrails, tests (`tests/`), evaluation (`evals/`), and docs (`docs/`).

## The claim (and how it's tested)

At equal experiment count and wall-clock on the same Kaggle T4 x2 setup, the
multi-agent system reaches a lower best `val_bpb` and/or wastes fewer
experiments than a single-agent baseline (B1: same harness, same LLM, same
budgets, no Librarian/Guard-LLM/Analyst/dedupe). Evaluated by
`evals/run_ab.py` with repeats, reported by `evals/report.py` with mean ± std,
the measured noise epsilon, and an honest verdict. **No results yet — see
`docs/EVALS.md` for status.**

## Quickstart (offline, no GPU)

```bash
pip install -r requirements.txt
python -m pytest tests -q          # 229 tests: guardrails E1-E12, units, integration
python -m src.mock_repo /tmp/mock  # mock training repo (dry-run target)
python -m evals.run_ab --config evals/fixtures/dry_config.yaml --repeats 1 \
    --out /tmp/ab_out              # full A/B harness with mock LLM + mock runner
```

## Run on Kaggle (real thing)

1. New notebook, accelerator **GPU T4 x2**, internet ON.
2. Add secret `GEMINI_API_KEY` (Kaggle → Add-ons → Secrets).
3. Upload/clone this project, open `notebook/kaggle_run.ipynb`, Save & Run All.
4. The notebook: installs deps → prepares data → calibrates noise (E13) →
   runs the MA and B1 arms → writes `results/ab_report.md` + plots.
5. Resume between sessions: reattach `/kaggle/working` as an input dataset;
   `memory/state.json` resumes automatically (tested, E11).

## Repo layout

```
autoresearch-team/
  README.md  LICENSE  config.yaml  requirements.txt  context.md
  notebook/kaggle_run.ipynb     # Kaggle A/B entry point
  notebook/smoke/               # bounded smoke notebook + its generator
  src/        orchestrator.py, decision.py, calibration.py, schemas.py,
              config.py, prompt_render.py, llm/client.py, agents/ (ideator,
              coder, guard, analyst, librarian, baseline), tools/ (runner,
              git_ops, patching, ast_checks, hashing, metrics_parser),
              memory/store.py, mock_handlers.py, mock_repo.py,
              kaggle_model_probe.py, kaggle_secret.py
  prompts/    one markdown file per agent (Section 7 of the spec)
  repo/       forked karpathy/autoresearch with the one-time T4 port
  tests/      229 tests incl. evals E1-E12 (offline, mock-based)
  evals/      run_ab.py, metrics.py, report.py, fixtures/
  docs/       ARCHITECTURE, ASSUMPTIONS, COMPUTE_BUDGET, GUARDRAILS, EVALS,
              EXPLAINER (viva), DEMO
  results/    generated on Kaggle (ab_report.md, plots, logs)
```

## Honesty notes

- All GPU/API-dependent behavior is marked UNVERIFIED until run on Kaggle
  (`docs/ASSUMPTIONS.md`, `docs/EVALS.md`).
- No benchmark numbers are claimed anywhere in this repo before they are
  measured; `results/` starts empty.
