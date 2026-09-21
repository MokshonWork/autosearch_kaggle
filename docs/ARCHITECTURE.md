# Architecture

## One-paragraph version

A deterministic **Orchestrator** drives rounds. Each round, the **Librarian**
summarizes the experiment log, the **Ideator** proposes 2 hypotheses in
different categories, and two GPU workers run in parallel — each with its own
git worktree — through **Coder** (hypothesis → search/replace patch), **Guard**
(deterministic checks, then an LLM reviewer), and **Runner** (subprocess
training on one GPU with a stripped environment and hard timeouts). After the
join, the **Analyst** explains results, and *code* (never an LLM) applies the
keep/discard decision rule. Everything is appended to `experiments.jsonl`,
`lessons.md`, and `state.json` (resume-safe), plus git history of kept changes.

```
        ┌───────────────────────── Orchestrator (code) ─────────────────────────┐
        │  load_context → ideate → fan-out 2 workers → join → analyze → decide  │
        │  → log → stop_check                                                   │
        └───────────────┬──────────────────────────────┬───────────────────────┘
                        │ Librarian summary            │ Librarian summary
              ┌─────────▼─────────┐          ┌─────────▼─────────┐
              │ Worker gpu0       │          │ Worker gpu1       │
              │ Coder → Guard →   │          │ Coder → Guard →   │
              │ Runner (preflight │          │ Runner (preflight │
              │ + full run +      │          │ + full run +      │
              │ ≤2 crash fixes)   │          │ ≤2 crash fixes)   │
              └─────────┬─────────┘          └─────────┬─────────┘
                   work/gpu0 (worktree @ best commit)  work/gpu1
```

## Why these choices

- **No LangGraph**: the build prompt sanctions a thread-pool fallback
  (Phase 6). A plain loop + `ThreadPoolExecutor(2)` is deterministic, has zero
  extra dependencies, and is easy to explain in a viva. See `docs/ASSUMPTIONS.md`.
- **Search/replace patches, not file rewrites**: small, auditable, and the
  Guard can verify each `find` matches exactly once (G4).
- **Decisions in code**: LLMs propose and explain; only `src/decision.py`
  keeps or discards, from measured numbers (G15).
- **Two-stage Guard**: deterministic checks (apply/compile/diff-size/protected
  regions/AST-diff) are final; the LLM reviewer can only add rejections.
- **Calibration before the loop**: `epsilon = max(2·std, min_delta)` measured
  on BOTH GPUs concurrently, so "improvement" means "beyond noise" (E13).

## Data flow / contracts

All agent I/O validates against pydantic models in `src/schemas.py`
(Hypothesis, Patch, GuardReport, RunResult, Decision, AnalystNote,
Experiment). Invalid LLM output → retry ≤2 with the validation error →
graceful step failure (G12). State lives in `memory/state.json`
(round, best commit/val_bpb, epsilon, budgets, combine queue) and resumes
on restart (Section 6.6).

## Repo map

See README.md for the full tree. `repo/` is the forked autoresearch training
repo with the one-time human T4 port (commit `9cce20e`); agents only ever edit
`train.py` inside worker worktrees.
