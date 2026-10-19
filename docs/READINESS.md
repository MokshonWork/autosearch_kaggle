# Full-campaign readiness (2026-10-16) — NOT launched, awaiting approval + smoke

## Verdict: READY TO LAUNCH (conditionally)

- Calibration: WORKS (verified on Kaggle T4 x2: values [1.926492, 1.927058],
  epsilon 0.002, GPU0-GPU1 diff 0.000566).
- Tests: 147 pass (9 Phase-2 regression tests). Dry-run A/B green.
- Kaggle CLI: auth, push (triggers run), status, logs, output — all verified.
- Kernels reproducible: diag / smoke / run / secret_probe bundles in
  `notebook/` with pinned `NvidiaTeslaT4` + docker image.
- Measured T4 numbers: startup+eval ~7 s; 300 s run wall ~330-360 s;
  2.3 GB VRAM; ~148k tok/s (see docs/COMPUTE_BUDGET.md).

## Pre-launch conditions (all must hold)

1. Smoke (`autosearch-smoke`) COMPLETE with all checks green — currently
   BLOCKED on one manual step: attach `GEMINI_API_KEY` to the kernel in the
   Kaggle UI (Add-ons → Secrets). CLI cannot attach secrets (Kaggle #582).
   - Secret retrieval itself was fixed on 2026-10-19: the failure was NOT a
     flaky backend. Kaggle injects `KAGGLE_URL_BASE=https://api.kaggle.com`,
     but the user-secrets endpoint is only served by `https://www.kaggle.com`;
     `api.kaggle.com` answers HTTP 400 with an empty body, which
     `kaggle_web_client` masks as `ConnectionError`. `src/kaggle_secret.py`
     now queries the working host and says plainly whether the secret is
     simply not attached to this kernel (Kaggle error code 5) or the request
     really failed. See `docs/ASSUMPTIONS.md` A4.
2. Explicit user approval for the campaign (rule 6).

## Exact launch commands (one arm-repeat per session, ~2.3 h each)

Session for `<arm>` in {multi_agent, baseline_single}, repeat in {0, 1}:

    kaggle kernels push -p notebook -t 32400   # with kaggle_run.ipynb cell 6 set to:
    python -m evals.run_ab --config config.yaml --repeats 1 \
        --arms <arm> --out results

Retrieve: `kaggle kernels status/logs/output mo2xgoyal/autosearch-run`.

## Estimated runtime + cost (per run / 4-run A/B)

| Basis | Per experiment | Calibration | Per run | 4-run A/B |
|---|---|---|---|---|
| optimistic 310 s | — | 16 min | 1.98 h | ~8.4 h GPU |
| planned 360 s | — | 18 min | 2.30 h | ~9.7 h GPU |

API per run: ≤400 calls, ≤800 k tokens. Production limits unchanged in
config.yaml (max_rounds 20, max_experiments 40, wall_clock_cap 9 h − 45 min
margin, no_improvement_patience 5).

## Remaining risks

- Sessions are best kept to one arm-repeat (~2.3 h) each.
- First marginal keep triggers a same-round retest by design.
- Secrets must be attached per kernel in the UI before each campaign run.
