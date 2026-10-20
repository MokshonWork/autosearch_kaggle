# Assumptions (build prompt Section 5) — status and evidence

| # | Assumption | Status | Evidence / how to verify |
|---|---|---|---|
| A1 | `train.py` uses features not available on T4 | **VERIFIED** (by reading code) | FA3 kernels via `kernels` pkg (old lines 20-24), bf16 autocast/casts, bf16 Muon matmuls, unconditional `torch.compile`. Fixed in one-time port commit `9cce20e` (SDPA, fp16+GradScaler, `USE_COMPILE=False`). |
| A2 | Where budget/eval/metric-print live | **VERIFIED** | `prepare.py`: TIME_BUDGET (line 31), MAX_SEQ_LEN (30), EVAL_TOKENS (32), `evaluate_bpb` (343-365). `train.py` (post-port): import line, seeds, eval call, `print(f"val_bpb: ...")`, stop condition — all marked `# PROTECTED`, enforced by `src/tools/patching.py::PROTECTED_SNIPPETS`. |
| A3 | Kaggle session length / weekly GPU quota | **UNVERIFIED** | Kaggle docs at run time. Config uses `wall_clock_cap_hours: 9.0` + `session_margin_minutes: 45` as conservative defaults; do not hardcode. Verify: Kaggle → Docs → "GPUs" quota page before the long run. |
| A4 | Kaggle Secrets readable in "Save & Run All" | **VERIFIED (host fix) / ATTACHMENT UNVERIFIED** | Notebook cell 2 reads the secret via `src/kaggle_secret.py` (NOT `kaggle_secrets.UserSecretsClient`) and prints only whether it succeeded. **Root cause found 2026-10-19:** Kaggle injects `KAGGLE_URL_BASE=https://api.kaggle.com`, but the UserSecretsService endpoint is served ONLY by `https://www.kaggle.com`; against the injected host every request returns HTTP 400 with an empty body, which `kaggle_web_client` flattens into a generic `ConnectionError` (previously misread as a "flaky backend"). `src/kaggle_secret.py` talks to the working host directly and distinguishes "not attached to this kernel" (server code 5) from real backend failures. Remaining manual step: attach `GEMINI_API_KEY` to the kernel in the Kaggle UI. |
| A5 | LangGraph API for fan-out | **DEVIATION (sanctioned)** | We use the prompt's Phase 6 fallback: `ThreadPoolExecutor(max_workers=2)` around the two worker pipelines in `src/orchestrator.py`. Rationale: offline build, LangGraph API drift risk, simplicity rule. Behavior (parallel branches + join + shared state) is equivalent. |
| A6 | Gemini model IDs / structured output / free-tier limits | **PARTIALLY VERIFIED** | REST `generateContent` with `responseMimeType: application/json` (documented API). Model IDs live in `config.yaml` (default `gemini-2.5-flash`, fallback `gemini-2.5-flash-lite`) — **verify current IDs and free-tier RPM/TPM against ai.google.dev docs on run day and edit config only.** |
| A7 | Two concurrent training subprocesses don't distort each other | **UNVERIFIED until Kaggle** | `src/calibration.py` runs K runs on both GPUs CONCURRENTLY and reports per-GPU means + diff (notebook step "calibration"). If GPUs differ systematically, pin comparisons per GPU. |
| A8 | Startup/compile time on T4; safe hard-kill timeout | **UNVERIFIED until Kaggle** | `startup_allowance_s: 420` in config is a placeholder GUESS. Measure in the Phase 1 notebook cell (startup time printed by train.py's own `total_seconds - training_seconds`) and update config. |

## Other decisions documented here (rule 4: safest reasonable default)

- Small-GPU config (README tips): `MAX_SEQ_LEN=256`, `VOCAB_SIZE=4096`,
  `EVAL_TOKENS=524288`, `DEPTH=4`, `WINDOW_PATTERN="L"`,
  `TOTAL_BATCH_SIZE=2**14`, `DEVICE_BATCH_SIZE=64`. All results are only
  compared within this Kaggle T4 setup, so cross-hardware comparability is
  not required.
- `min_delta: 0.002` bpb default floor for epsilon.
- Plausibility floor `0.7 × baseline_bpb` (G9): a "win" bigger than 30% of
  baseline is treated as metric tampering, not luck.
- Blockers from Section 19 were not answered → defaults used: Gemini free
  tier, N=40 experiments, wall-clock cap 9h.

- Partial-round persistence (2026-10-20): a terminal worker error must not
  discard successful sibling work. Completed outcomes are recorded first; the
  pending terminal exception is re-raised afterwards and the existing graceful
  stop reason (`BudgetExhausted`/`LLMError`) is preserved. `state.round_status`
  distinguishes `completed` from `partial`/`failed`.
- Guard-retry identity (2026-10-20): a changed replacement candidate gets a new
  `ChangeRecord` id; the rejected candidate's manifest is kept with status
  `guard_reject` (never rewritten). Identical content keeps one identity.
  `Registry.record_change` remains insert-only; `Registry.update_change` is the
  explicit API for correcting a candidate fingerprint when crash fixes change
  the exact tested content.

## Known risks (kept current)

- fp16 + GradScaler numerics for Muon on T4: UNVERIFIED (no GPU here).
- SDPA sliding-window path unused by default (`WINDOW_PATTERN="L"`), so the
  mask branch is cold; agents that add "S" windows will exercise it — watch
  the first such run.
- `torch.compile` off by default; enabling may be slow on T4 (A8).
- Gemini free-tier rate limits may force `max_calls_per_round` down; backoff
  + fallback model are implemented but untested against the real API.
- Kaggle output persistence between sessions: reattach `/kaggle/working` as
  an input dataset; `memory/` resumes from `state.json` (tested offline, E11).
