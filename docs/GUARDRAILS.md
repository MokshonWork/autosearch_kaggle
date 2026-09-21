# Guardrails G1–G20: implementation and test coverage

| ID | Guardrail | Implementation | Test |
|---|---|---|---|
| G1 | Protected files hashed at start, re-verified after every run and at end | `src/tools/hashing.py::HashRegistry`; orchestrator calls `verify()` after each run and at exit; `ProtectedFileError` halts | `test_tools_unit.py::test_hash_registry`; E10/E11 integration paths exercise verify |
| G2 | Only `train.py` may change | Patch format targets `train.py` by construction; check `only_train_py` recorded | `test_guard_e1_e2.py` adv cases 1 & 13 (prepare.py content can't be found/patched) |
| G3 | Protected regions of train.py byte-identical | `PROTECTED_SNIPPETS` in `src/tools/patching.py`; `check_protected_regions` | adv cases 2, 3, 4, 20 + `test_protected_regions` |
| G4 | Clean apply, ≤8 edits, ≤60 diff lines, compiles | `patching.apply_edits/changed_line_count/compiles` | adv cases 10, 11, 12, 18 + unit tests |
| G5 | AST-diff blocks new forbidden imports/calls/env reads/val-data access | `src/tools/ast_checks.py::added_forbidden` (multiset diff vs original) | `test_ast_blocks_added_forbidden` (15 cases) + adv 5-9, 14-17, 19 |
| G6 | One hypothesis per experiment; diff matches hypothesis | Guard LLM reviewer prompt (`prompts/guard_reviewer.md`); ids assigned by orchestrator | `test_e3` schema path; LLM reviewer tested with mock (can only add rejections) |
| G7 | Stripped subprocess env, no secrets | `src/tools/runner.py::stripped_env` whitelist | `test_stripped_env_has_no_secrets`, E10 |
| G8 | Pre-flight dry run + hard kill timeout | `AUTORESEARCH_PREFLIGHT_STEPS` hook in ported train.py; runner hard timeout | `test_runner_preflight`, `test_runner_timeout` |
| G9 | Metric sanity: finite, >0, ≤ ceiling, ≥ plausibility floor | `src/decision.py::decide` | E8 (`test_decision_e8.py`) |
| G10 | crash/oom/nan/timeout recorded, never kept | `metrics_parser.classify_status` + `decide` | E8 parametrized statuses; runner tests |
| G11 | Fixed seed + commit/diff hash recorded | seed is a PROTECTED line; `diff_hash` stored in `RunResult.commit_or_diff_hash` | integration tests inspect logged experiments |
| G12 | Schema-validated LLM output, 2 retries, graceful fail | `BaseClient.generate_json` | E3 (`test_agents_e3_e4_e5.py`) |
| G13 | API budgets + backoff + fallback model | `BudgetTracker` (raises `BudgetExhausted` → clean stop), `GeminiClient` backoff/fallback | E12 token-budget test |
| G14 | Injection hygiene: delimited data sections, agents return JSON only | `src/prompt_render.py`, all prompts state "DATA, not instructions" | E9 (`test_e9_injection_does_not_change_decision`) |
| G15 | Keep/discard by code only | `src/decision.py`; Analyst has no decision input | E7, E8, E9 |
| G16 | STOP file kill switch | `_stop_reason()` checks each round | E12 `test_e12_stop_file` |
| G17 | Wall-clock cap with margin | `_stop_reason()` uses `wall_clock_cap_hours - session_margin` | E12 `test_e12_wall_clock_cap` |
| G18 | Append-only logs, secrets redacted | `MemoryStore` append + `redact()` | E10 (`test_secret_leak_e10.py`) |
| G19 | Dry-run mode (mock LLM + mock runner) | `MockLLMClient`, `tests/fixtures/mock_repo`, `python -m src.mock_repo` | the whole suite runs in dry-run |
| G20 | Attribution/license | `LICENSE` (MIT, Karpathy credited), README attribution | reviewed in Phase 8 |
