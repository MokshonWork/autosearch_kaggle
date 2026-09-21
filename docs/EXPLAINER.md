# EXPLAINER — plain-language walkthrough (viva-friendly)

## The one-sentence pitch

A team of small LLM agents argues about, writes, safety-checks, runs, and
remembers code-level ML experiments overnight on two free Kaggle T4 GPUs —
and deterministic code (not an LLM) decides what counts as a real improvement.

## The agents (who does what)

- **Orchestrator** — not an LLM. A Python loop that owns the plan, the budget,
  the stop rules, and every keep/discard decision.
- **Librarian** — the memory. Summarizes the experiment log each round
  (top wins, top failures, blocked ideas, untried categories), rejects
  duplicate ideas (fast text similarity, LLM judge only for borderline cases),
  writes one factual lesson line per experiment and the final report's
  narrative (numbers always come from code).
- **Ideator** — proposes exactly 2 hypotheses per round, in *different*
  categories (optimizer, lr_schedule, architecture, …), told never to repeat
  blocked or past ideas.
- **Coder** — turns one hypothesis into a tiny search/replace patch for
  `train.py` (max 8 edits, each `find` must match exactly once).
- **Guard** — two stages. First deterministic: does the patch apply, compile,
  stay small, keep PROTECTED lines (seed, eval call, metric print, time
  budget) byte-identical, and add no dangerous code (AST diff: no network,
  subprocess, file writes, env reads, no touching validation data)? A
  deterministic failure is final. Only then an LLM reviewer may *add* a
  rejection (e.g. "this does more than the hypothesis says").
- **Runner** — not an LLM. Applies the patch in the worker's own git worktree,
  runs `train.py` on one GPU as a subprocess with a secret-free environment,
  a cheap pre-flight (a few steps to catch crashes early), and a hard timeout.
- **Analyst** — explains each result ("within noise? why? follow-ups?"). It
  may only use numbers it was given and can never flip a decision.

## The decision rule (the heart of the claim)

Before experimenting, we run the unmodified script K=3 times on *both GPUs at
once* (concurrency matters — they share CPU and data loading). The spread of
`val_bpb` gives the noise floor: `epsilon = max(2·std, min_delta)`. A change
is kept only if it beats the best by more than epsilon and the value is
plausible (not suspiciously low). Marginal wins (epsilon…3·epsilon) are
re-run once and must confirm on the mean. If both workers improve, the better
one is kept and the other goes to a combine queue to be re-tested on top of
the new best next round.

## Safety highlights (demo-friendly)

- We *tried* to break it: 20 adversarial patches (edit the metric, print a
  fake `val_bpb`, read the API key, leak validation data into training…) —
  all blocked (eval E1).
- The training subprocess never sees the API key (whitelist env, E10).
- Prompt injection in logs ("ignore previous instructions…") changes nothing:
  everything from logs is wrapped in DATA sections and decisions are numeric
  (E9).
- Kill it anytime: a `STOP` file, token/API budgets, no-improvement patience,
  and a wall-clock cap all stop cleanly and save state (E12); a kill mid-run
  resumes without losing or duplicating experiments (E11).

## Likely viva questions

- **Why not let the LLM decide keep/discard?** It can be talked into anything
  (see injection test). Numbers + a measured noise floor can't.
- **Why two GPUs as two workers instead of DDP?** The repo is single-GPU;
  parallel experiments double idea throughput under the same wall-clock.
- **Why search/replace patches?** Auditable, small, and mechanically checkable
  (exactly-once match), unlike full-file rewrites.
- **What if both GPUs disagree?** Calibration measures the GPU0–GPU1 offset;
  if systematic, comparisons are pinned per GPU (E13).
- **What does "better" mean?** Lower best `val_bpb` than the single-agent
  baseline at equal experiment count, beyond the measured noise — and we
  report "within noise" honestly if that's the truth.
- **What does the baseline (B1) share?** Same harness, GPUs, LLM, token
  budget, seed, starting commit — only the agent layer differs (no Librarian
  memory, no Guard LLM, no Analyst, no dedupe).
