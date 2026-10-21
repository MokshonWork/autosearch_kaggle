"""Orchestrator: owns shared state, loop control, stop rules, keep/discard.

Deterministic code, not an LLM. Fan-out uses a ThreadPoolExecutor with one
worker per GPU — the build prompt's sanctioned fallback for LangGraph parallel
branches (Phase 6; documented in docs/ASSUMPTIONS.md).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .agents.analyst import Analyst
from .agents.baseline import BaselineAgent
from .agents.coder import Coder
from .agents.guard import Guard
from .agents.ideator import Ideator
from .agents.librarian import Librarian, lesson_line, untried_categories
from .config import Config
from .decision import decide, decide_retest, plausibility_floor
from .llm.client import BudgetExhausted, BudgetTracker, LLMError
from .memory.store import CorruptStateError, MemoryStore
from .registry import ChangeRecord, Registry, TestReport, new_id
from .scheduler import Scheduler, build_context, content_fingerprint, discover_gpus
from .schemas import CATEGORIES, Experiment, Hypothesis, Patch, RunResult
from .tools import git_ops, patching
from .tools.hashing import HashRegistry, ProtectedFileError, diff_hash
from .tools.runner import run_training

MAX_CRASH_FIXES = 2  # per experiment (Section 6.3 step 3)


@dataclass
class WorkerOutcome:
    hypothesis: Hypothesis
    patch: Optional[Patch] = None
    guard_report: object = None
    result: Optional[RunResult] = None
    analyst_note: object = None
    diff: str = ""
    patched_src: str = ""
    tokens_before: int = 0
    task_id: str = ""           # Scheduler task (Ch.4.2 ownership trail)
    change_id: str = ""         # change manifest id (Ch.4.3)
    candidate_revision: str = ""  # fingerprint of the exact code tested (Ch.5)
    test_id: str = ""           # test report id bound to that revision (Ch.5)


class Orchestrator:
    def __init__(self, cfg: Config, client, runner_fn: Callable = run_training,
                 log: Callable[[str], None] = print):
        self.cfg = cfg
        self.client = client
        self.runner_fn = runner_fn
        self.log = log
        root = Path(cfg.root)
        self.repo = (root / cfg.paths.repo_dir).resolve()
        self.work_dir = Path(cfg.paths.work_dir)
        if not self.work_dir.is_absolute():
            self.work_dir = (root / self.work_dir).resolve()
        self.stop_file = Path(cfg.paths.stop_file)
        if not self.stop_file.is_absolute():
            self.stop_file = (root / self.stop_file).resolve()
        self.prompts_dir = root / "prompts"
        if not self.prompts_dir.exists():  # tests/tmp roots: use project prompts
            self.prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        mem = Path(cfg.paths.memory_dir)
        self.store = MemoryStore(mem if mem.is_absolute() else root / mem)
        self.budget: BudgetTracker = client.budget
        self._lock = threading.Lock()  # guards state + budget + ids (2 workers)

        # Coordination backbone (hardening Ch.4): Registry + Scheduler live in
        # the same memory dir as experiments.jsonl so resume keeps working.
        self.registry = Registry(self.store.dir)
        # Ch.7 GPU policy (explicit > discovered > legacy 2-slot default):
        # - config training.gpus pins device ids (Kaggle: [0, 1]);
        # - else auto-discover via CUDA_VISIBLE_DEVICES/torch;
        # - "no devices found" means UNKNOWN, not "one GPU": keep the legacy
        #   2-slot layout so mocks/tests behave exactly as before. True 1-GPU
        #   degrade happens only when discovery positively reports 1 device.
        gpu_cfg = getattr(cfg.training, "gpus", None)
        discovered: list[int] | None = None
        if gpu_cfg:
            discovered = list(gpu_cfg)
        else:
            found = discover_gpus(None)
            discovered = found if found else None
        self.gpus = discovered if discovered is not None else [0, 1]
        self.slots = [0, 1]  # worker slots; slot->device via _device_for_slot
        self.scheduler = Scheduler(self.registry, self.slots)
        self._recovered = self.scheduler.recover()  # Ch.8: running->queued
        if self._recovered["requeued"] or self._recovered["expired"]:
            self.log(f"[recover] {self._recovered}")

        protected = [self.repo / f for f in cfg.guard.protected_files
                     if (self.repo / f).exists()]
        cfg_file = root / "config.yaml"
        if cfg_file.exists():
            protected.append(cfg_file)
        self.hash_registry = HashRegistry(protected)

        # Agents per arm (baseline arm gets no Guard-LLM/Analyst/Librarian).
        self.is_ma = cfg.mode.arm == "multi_agent"
        self.coder = Coder(client, self.prompts_dir)
        self.guard = Guard(cfg.guard.max_edits, cfg.guard.max_diff_lines,
                           client, self.prompts_dir, use_llm=self.is_ma)
        self.ideator = Ideator(client, self.prompts_dir) if self.is_ma else None
        self.analyst = Analyst(client, self.prompts_dir) if self.is_ma else None
        self.librarian = (
            Librarian(client, self.prompts_dir,
                      cfg.dedupe.text_similarity_threshold,
                      cfg.dedupe.use_llm_judge_for_borderline)
            if self.is_ma else None)
        self.baseline = None if self.is_ma else BaselineAgent(
            client, self.prompts_dir)

        self.state = self._load_or_fresh_state()
        self._restore_budget()
        # Persist each reservation/generation so a hard kill mid-round cannot
        # reset the per-round or total HTTP-attempt budget on resume.
        self.budget.set_on_change(self.save_state)

    def _restore_budget(self) -> None:
        """Restore cumulative + per-round HTTP usage across resumes.

        Counters are assigned, never re-added, so resume cannot double-count.
        Older checkpoints may omit these fields or contain junk; ignore those
        safely. Legacy checkpoints without a round marker reset only the
        per-round counter on the next begin_round(); the total cap still holds.
        """
        for key, attr in (("api_calls", "requests_total"),
                          ("api_calls_this_round", "requests_this_round"),
                          ("api_tokens", "tokens_total"),
                          ("api_generations", "generations_total")):
            value = self.state.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                setattr(self.budget, attr, max(getattr(self.budget, attr), value))
        marker = self.state.get("api_round")
        if isinstance(marker, int) and not isinstance(marker, bool) and marker >= 0:
            self.budget.round_index = marker

    def _load_or_fresh_state(self) -> dict:
        """Resume if possible; corrupt state -> fresh start, never silent."""
        try:
            loaded = self.store.load_state()
        except CorruptStateError as e:
            self.log(f"[resume] {e}")
            loaded = None
        if loaded is None:
            return self._fresh_state()
        # Validate minimal shape; anything else -> fresh (fail safe, Ch.8).
        if not isinstance(loaded.get("round"), int) or \
                not isinstance(loaded.get("next_hyp_id"), int):
            self.log("[resume] state.json missing round/next_hyp_id; fresh start")
            return self._fresh_state()
        for key, default in (("combine_queue", []), ("experiments_done", 0),
                             ("no_improve_streak", 0)):
            loaded.setdefault(key, default)
        return loaded

    # ------------------------------------------------------------------ state
    def _device_for_slot(self, slot: int) -> int:
        """Map a worker slot to a real device id (Ch.7: never hardcode)."""
        if slot < len(self.gpus):
            return self.gpus[slot]
        return self.gpus[0]

    def _fresh_state(self) -> dict:
        return {
            "round": 0, "next_hyp_id": 1, "experiments_done": 0,
            "best_val_bpb": None, "best_commit": git_ops.head_commit(self.repo),
            "baseline_bpb": None, "epsilon": None, "calibration_values": [],
            "no_improve_streak": 0, "combine_queue": [],
            "started_at": time.time(), "stopped_reason": None,
            "round_status": "pending",
        }

    def save_state(self, reason: str | None = None) -> None:
        if reason:
            self.state["stopped_reason"] = reason
        with self._lock:
            self.state.update(self.budget.snapshot())
            self.store.save_state(dict(self.state))

    def _stop_reason(self) -> str | None:
        b = self.cfg.budget
        if self.stop_file.exists():
            return "stop_file"  # G16
        if self.state["round"] >= b.max_rounds:
            return "max_rounds"
        if self.state["experiments_done"] >= b.max_experiments:
            return "max_experiments"
        if self.state["no_improve_streak"] >= b.no_improvement_patience:
            return "no_improvement"
        cap = b.wall_clock_cap_hours * 3600 - b.session_margin_minutes * 60
        if time.time() - self.state["started_at"] > cap:
            return "wall_clock_cap"  # G17
        return None

    # ------------------------------------------------------------------ main
    def run(self) -> dict:
        self.log(f"[orchestrator] arm={self.cfg.mode.arm} "
                 f"dry_run={self.cfg.mode.dry_run}")
        try:
            self._calibrate_if_needed()
            while True:
                reason = self._stop_reason()
                if reason:
                    self.save_state(reason)
                    self.log(f"[orchestrator] stopping: {reason}")
                    break
                self._round()
            self.hash_registry.verify()  # G1 final check
        except (BudgetExhausted, ProtectedFileError, LLMError) as e:
            # LLMError: terminal API failure after backoff+fallback — stop
            # GRACEFULLY (state saved, resume-able) instead of crashing the
            # whole A/B session mid-round.
            self.save_state(type(e).__name__)
            self.log(f"[orchestrator] stopping: {type(e).__name__}: {e}")
        return self.state

    def _calibrate_if_needed(self) -> None:
        if self.state["epsilon"] is not None:
            return  # resumed
        worktrees = [self.work_dir / f"gpu{i}" for i in range(2)]
        best_commit = self.state["best_commit"]
        for wt in worktrees:
            git_ops.ensure_worktree(self.repo, wt, best_commit)
            if not (wt / "train.py").exists():
                raise RuntimeError(
                    f"calibration worktree {wt} has no train.py — source "
                    f"{self.repo} is not a training repo (nested repo/.git "
                    "missing? re-init it, see notebook/kaggle_run.ipynb cell 1)")
        self.log("[calibration] running unmodified train.py on both GPUs...")
        from .calibration import run_calibration
        report = run_calibration(
            worktrees, self.cfg.decision.calibration_runs,
            self.cfg.training.time_budget_s, self.cfg.training.startup_allowance_s,
            self.cfg.decision.min_delta, runner=self.runner_fn)
        if not report.values:
            raise RuntimeError(
                "calibration produced no valid runs\n"
                f"{report.failure_summary()}")
        baseline = min(report.values)
        self.state.update(
            epsilon=report.epsilon, baseline_bpb=baseline,
            best_val_bpb=baseline, calibration_values=report.values)
        self.save_state()
        self.log(f"[calibration]\n{report.summary()}")

    # ------------------------------------------------------------------ round
    def _round(self) -> None:
        with self._lock:
            rnd = self.state["round"]
            best_commit = self.state["best_commit"]
            self.state["round_status"] = "pending"
        self.budget.begin_round(rnd)
        self.log(f"[round {rnd}] start best={self.state['best_val_bpb']}")
        past = self.store.read_experiments()

        if self.is_ma:
            plan = self._plan_round_ma(past)
        else:
            plan = self._plan_round_baseline()
        # A planner may propose two slots; never exceed the remaining cap.
        remaining = max(0, self.cfg.budget.max_experiments
                        - self.state["experiments_done"])
        plan = plan[:remaining]

        # Register one Scheduler task per planned experiment. A worker may
        # only run after claim() gates dependencies + ownership (Ch.4.2).
        tasks = []
        for slot, hyp, preset in plan:
            task = self.scheduler.create(
                "experiment",
                payload={"slot": slot,
                         "hypothesis": hyp.model_dump() if hyp else None,
                         "has_preset_patch": preset is not None,
                         "round": rnd})
            task = self.registry.get_task(task.id)
            # stamp slot on the record for GPU mapping at claim time
            task.payload["slot"] = slot
            tasks.append((task, hyp, preset))

        # Submit explicitly instead of ex.map(): ex.map() propagates the first
        # worker exception (BudgetExhausted/LLMError) while iterating and
        # discards sibling results that already completed. Keep every result
        # and re-raise only after the successful work is persisted.
        futures = []
        with ThreadPoolExecutor(max_workers=len(tasks) or 1) as ex:
            futures = [ex.submit(self._worker_task, *t) for t in tasks]
        outcomes: list[WorkerOutcome] = []
        terminal_error: Exception | None = None
        for fut in futures:
            try:
                outcomes.append(fut.result())
            except Exception as e:
                if terminal_error is None:
                    terminal_error = e
                self.log(f"[round {rnd}] worker failed: "
                         f"{type(e).__name__}: {e}")

        analysis_error: Exception | None = None
        try:
            if outcomes:
                self._analyze_decide_log(outcomes)
        except Exception as e:
            analysis_error = e
        finally:
            with self._lock:
                self.state["round"] = rnd + 1
                self.state["round_status"] = (
                    "failed" if terminal_error and not outcomes else
                    "partial" if terminal_error or analysis_error else
                    "completed")
            self.save_state()

        if terminal_error is not None:
            raise terminal_error
        if analysis_error is not None:
            raise analysis_error

    def _worker_task(self, task, hypothesis, preset_patch) -> WorkerOutcome:
        """Claim the task, run the branch on its slot's device, complete it."""
        slot = task.payload.get("slot", 0)
        try:
            self.scheduler.claim(task.id, owner=f"slot{slot}", gpu=slot)
        except Exception as e:  # RegistryError etc: fail the branch safely
            hyp = hypothesis or Hypothesis(
                id="H-0000", category="optimizer",
                change_summary="(unscheduled)", risk="med")
            out = WorkerOutcome(hypothesis=hyp)
            out.result = RunResult(hypothesis_id=hyp.id, gpu=slot,
                                   status="guard_reject",
                                   stdout_tail=f"scheduler refused: {e}"[:2000])
            return out
        device = self._device_for_slot(slot)
        try:
            out = self._worker(slot, device, hypothesis, preset_patch)
            out.task_id = task.id
            self.scheduler.complete(task.id)
            return out
        except Exception as e:
            self.scheduler.fail(task.id)
            raise
        finally:
            self.scheduler.release_gpu(slot, task.id)

    def _next_id(self) -> str:
        with self._lock:
            hid = f"H-{self.state['next_hyp_id']:04d}"
            self.state["next_hyp_id"] += 1
        return hid

    def _plan_round_ma(self, past: list[dict]) -> list[tuple]:
        """Returns [(slot, hypothesis, preset_patch_or_None), ...] (<=2)."""
        with self._lock:
            combine = self.state["combine_queue"]
            item = combine.pop(0) if combine else None
        tasks: list[tuple] = []
        if item:  # takes one slot (Section 6.3 rule 6)
            tasks.append((0, Hypothesis(**item["hypothesis"]),
                          Patch(**item["patch"])))
        n_slots = 2 - len(tasks)
        if n_slots <= 0:
            return tasks
        summary = self.librarian.summarize(past)
        blocked = "\n".join(self._blocked_ideas(past))
        untried = ", ".join(untried_categories(past))
        recent = "\n".join(
            f"{e['hypothesis']['id']}: {e['hypothesis']['change_summary']} "
            f"-> {e['result']['status']} {e['result'].get('val_bpb')}"
            for e in past[-10:])
        valid: list[Hypothesis] = []
        for _ in range(3):  # initial + max 2 regenerations (Section 6.3 rule 2)
            hyps, err = self.ideator.propose(
                str(self.state["best_val_bpb"]), summary, blocked, recent,
                untried, "(empty)", [self._next_id() for _ in range(n_slots)])
            if err:
                self.log(f"[ideator] failed: {err}")
                continue
            for h in hyps:
                if any(h.category == v.category for v in valid):
                    continue  # must be different categories
                dup, why = self.librarian.is_duplicate(h, past)
                if dup:
                    self.log(f"[dedupe] {h.id} rejected: {why}")
                    continue
                valid.append(h)
            if len(valid) >= n_slots:
                break
        for i, h in enumerate(valid[:n_slots]):
            tasks.append((len(tasks), h, None))
        return tasks[:2]

    def _plan_round_baseline(self) -> list[tuple]:
        return [(gpu, None, None) for gpu in range(2)]

    def _blocked_ideas(self, past: list[dict]) -> list[str]:
        out = []
        for e in past:
            g = e.get("guard") or {}
            r = e.get("result", {})
            if g.get("final") == "reject" or r.get("status") in (
                    "crash", "oom", "nan", "timeout", "guard_reject"):
                out.append(f"{e['hypothesis'].get('id')}: "
                           f"{e['hypothesis'].get('change_summary')}")
        return out

    # -------------------------------------------------------------- workers
    def _worker(self, slot: int, device: int,
                hypothesis: Optional[Hypothesis],
                preset_patch: Optional[Patch]) -> WorkerOutcome:
        """One worker branch: code -> guard -> run (with fix loops).

        slot = logical worker (owns work/gpu<slot>); device = real GPU id
        passed as CUDA_VISIBLE_DEVICES. They differ only when fewer real
        GPUs exist than slots (Ch.7 degrade).
        """
        wt = self.work_dir / f"gpu{slot}"
        with self._lock:
            best_commit = self.state["best_commit"]
        git_ops.ensure_worktree(self.repo, wt, best_commit)
        src = (wt / "train.py").read_text()
        if self.is_ma:
            return self._worker_ma(slot, device, wt, src, hypothesis,
                                   preset_patch, best_commit)
        return self._worker_baseline(slot, device, wt, src, best_commit)

    def _worker_ma(self, slot, device, wt, src, hypothesis, preset_patch,
                   base_commit) -> WorkerOutcome:
        out = WorkerOutcome(hypothesis=hypothesis,
                            tokens_before=self.budget.tokens_total)
        patch = preset_patch
        if patch is not None:
            try:  # combine-queue patch may not apply to the new best (6.3.6)
                patching.apply_edits(src, patch.edits, self.cfg.guard.max_edits)
            except patching.PatchError:
                patch = None  # re-adapt once via the Coder below
        if patch is None:
            patch, err = self.coder.make_patch(hypothesis, src)
            if patch is None:
                out.result = RunResult(hypothesis_id=hypothesis.id, gpu=device,
                                       status="guard_reject", stdout_tail=err)
                return out
        # Record the change manifest BEFORE testing (Ch.4.3): reproducible
        # fingerprint of the exact candidate; tester binds results to it.
        manifest = self._record_manifest(
            hypothesis, patch, src, base_commit, agent_id="coder")
        out.change_id = manifest.id
        out.candidate_revision = manifest.candidate_revision
        # Guard with one coder retry on reject (Section 6.3).
        reasons = ""
        for attempt in range(2):
            report, patched = self.guard.check(hypothesis, patch, src)
            out.guard_report = report
            if report.approved:
                break
            reasons = "; ".join(
                c.detail for c in report.deterministic if not c.passed)
            reasons += "; " + "; ".join(report.llm_reasons)
            if attempt == 0:
                patch2, _ = self.coder.make_patch(
                    hypothesis, src, rejection_reasons=reasons)
                if patch2 is None:
                    break
                patch = patch2
        out.patch = patch
        if not report.approved:
            self.registry.set_change_status(manifest.id, "guard_reject")
            out.result = RunResult(hypothesis_id=hypothesis.id, gpu=device,
                                   status="guard_reject",
                                   stdout_tail=reasons[:2000])
            return out
        # The approved candidate may be the coder's RETRY, not the first
        # patch the manifest was recorded for. Keep the rejected candidate's
        # record as audit history and give a changed replacement a fresh
        # identity whose fingerprint matches the exact approved content.
        approved_revision = content_fingerprint(patched)
        if approved_revision != manifest.candidate_revision:
            self.registry.set_change_status(manifest.id, "guard_reject")
            manifest = self._record_manifest(
                hypothesis, patch, src, base_commit, agent_id="coder",
                patched_src=patched)
            out.change_id = manifest.id
            out.candidate_revision = manifest.candidate_revision
        self.registry.set_change_status(manifest.id, "guard_ok")
        return self._run_with_fixes(out, slot, device, wt, src, patched,
                                    manifest)

    def _record_manifest(self, hypothesis, patch, src, base_commit,
                         agent_id: str,
                         patched_src: str | None = None) -> ChangeRecord:
        """Build + persist the change manifest (Ch.4.3) for a candidate patch.

        The candidate fingerprint is the sha256 of the PATCHED train.py, so a
        later worktree mutation is detectable at promotion time (Ch.5).
        patched_src is used when the caller already holds the exact approved
        content (guard retry); otherwise the patch is applied here.
        """
        if patched_src is None:
            try:
                patched_src = patching.apply_edits(
                    src, patch.edits, self.cfg.guard.max_edits)
            except patching.PatchError:
                patched_src = src  # unpatched: guard will reject it
        manifest = ChangeRecord(
            id=new_id("CHANGE"), task_id="", experiment_id=hypothesis.id,
            agent_id=agent_id, base_revision=base_commit,
            candidate_revision=content_fingerprint(patched_src),
            changed_files=["train.py"], objective=hypothesis.change_summary,
            rationale=hypothesis.rationale,
            expected_behavior=hypothesis.expected_effect,
            risk_notes=f"risk={hypothesis.risk}",
            required_checks=["guard_deterministic", "preflight", "full_run",
                             "revision_match_at_promotion"],
            provenance={"round": self.state["round"], "base_commit": base_commit,
                        "category": hypothesis.category})
        return self.registry.record_change(manifest)

    def _run_with_fixes(self, out, slot, device, wt, src, patched,
                        manifest=None) -> WorkerOutcome:
        """Write patch, pre-flight, full run, crash-fix loop (max 2). G8/G10."""
        (wt / "train.py").write_text(patched)
        out.patched_src = patched
        out.diff = patching.unified_diff(src, patched)
        if manifest is not None and not manifest.candidate_revision:
            manifest = self.registry.update_change(
                manifest.id, candidate_revision=content_fingerprint(patched))

        def full_run() -> RunResult:
            return self.runner_fn(wt, device, out.hypothesis.id,
                                  self.cfg.training.time_budget_s,
                                  self.cfg.training.startup_allowance_s,
                                  diff_hash=diff_hash(out.diff))

        # G8: cheap pre-flight first (unless disabled via dry_run_steps=0).
        if self.cfg.training.dry_run_steps:
            pre = self.runner_fn(
                wt, device, out.hypothesis.id, self.cfg.training.time_budget_s,
                self.cfg.training.startup_allowance_s,
                diff_hash=diff_hash(out.diff),
                preflight_steps=self.cfg.training.dry_run_steps)
            preflight_ok = (pre.status == "ok"
                            and "PREFLIGHT_OK" in pre.stdout_tail)
            result = full_run() if preflight_ok else pre
        else:
            result = full_run()

        fixes = 0
        while result.status in ("crash", "oom", "nan"):
            if fixes >= MAX_CRASH_FIXES:
                break
            fixes += 1
            current_src = (wt / "train.py").read_text()
            patch, err = self.coder.make_patch(
                out.hypothesis, current_src,
                error_context=result.stdout_tail[-3000:])
            if patch is None:
                break
            try:
                repaired = patching.apply_edits(
                    current_src, patch.edits, self.cfg.guard.max_edits)
            except patching.PatchError as e:
                result.stdout_tail = f"fix patch did not apply: {e}"
                break
            report, approved = self.guard.check(
                out.hypothesis, patch, src, patched_src=repaired,
                use_llm=False)
            if not report.approved:
                reasons = "; ".join(
                    c.detail for c in report.deterministic if not c.passed)
                result.stdout_tail = (
                    "repair patch rejected by deterministic guard: "
                    + reasons + "\n" + result.stdout_tail)[:2000]
                self.log(f"  [{out.hypothesis.id}] repair rejected: {reasons}")
                break
            (wt / "train.py").write_text(approved)
            out.patched_src = approved
            out.diff = patching.unified_diff(src, approved)
            result = full_run()
        out.result = result
        self.hash_registry.verify()  # G1 after every run
        # Ch.5: bind the test report to the EXACT revision tested. If a crash
        # fix rewrote the file, the fingerprint follows the final content and
        # the persisted manifest is corrected too (not just the in-memory
        # copy) so the promotion gate compares like with like.
        if manifest is not None and out.patched_src:
            tested_revision = content_fingerprint(out.patched_src)
            if tested_revision != manifest.candidate_revision:
                manifest = self.registry.update_change(
                    manifest.id, candidate_revision=tested_revision)
            self.registry.record_test(
                TestReport(
                    id=new_id("TEST"), change_id=manifest.id,
                    candidate_revision=tested_revision,
                    base_revision=manifest.base_revision,
                    commands=["preflight(train.py)", "full(train.py)"],
                    exit_codes=[],
                    status=result.status,
                    metrics={"val_bpb": result.val_bpb,
                             "wall_seconds": result.wall_seconds,
                             "steps": result.steps},
                    artifacts=[f"work/gpu{slot}/train.py"]))
            self.registry.set_change_status(manifest.id, "tested")
            out.change_id = manifest.id
            out.candidate_revision = tested_revision
            out.test_id = self.registry.tests_for_change(manifest.id)[-1].id
        return out

    def _worker_baseline(self, slot, device, wt, src, base_commit
                         ) -> WorkerOutcome:
        """B1 arm: single agent, minimal deterministic safety only."""
        hid = self._next_id()
        past = self.store.read_experiments()
        recent = "\n".join(
            f"{e['hypothesis'].get('change_summary')} -> "
            f"{e['result'].get('status')} {e['result'].get('val_bpb')}"
            for e in past[-10:])
        out = WorkerOutcome(
            hypothesis=Hypothesis(id=hid, category="optimizer",
                                  change_summary="(baseline)", risk="med"),
            tokens_before=self.budget.tokens_total)
        error_ctx = ""
        for _ in range(2):
            bp, err = self.baseline.propose_and_patch(
                str(self.state["best_val_bpb"]), src, recent, error_ctx)
            if bp is None or not bp.edits:
                out.result = RunResult(hypothesis_id=hid, gpu=device,
                                       status="crash",
                                       stdout_tail=err or "empty patch")
                return out
            out.hypothesis.change_summary = bp.change_summary
            if bp.category in CATEGORIES:
                out.hypothesis.category = bp.category
            try:
                patched = patching.apply_edits(src, bp.edits,
                                               self.cfg.guard.max_edits)
            except patching.PatchError as e:
                error_ctx = f"patch failed to apply: {e}"
                continue
            ok, cerr = patching.compiles(patched)
            if not ok:
                error_ctx = f"patched file does not compile: {cerr}"
                continue
            (wt / "train.py").write_text(patched)
            out.patched_src = patched
            out.diff = patching.unified_diff(src, patched)
            # Baseline arm also gets a manifest + bound test report (Ch.4.3/5),
            # built from the single-agent's own patch (no Coder/Guard involved).
            manifest = ChangeRecord(
                id=new_id("CHANGE"), task_id="", experiment_id=hid,
                agent_id="baseline", base_revision=base_commit,
                candidate_revision=content_fingerprint(patched),
                changed_files=["train.py"],
                objective=out.hypothesis.change_summary,
                risk_notes="risk=med",
                required_checks=["compiles", "full_run",
                                 "revision_match_at_promotion"],
                provenance={"round": self.state["round"]})
            self.registry.record_change(manifest)
            out.change_id = manifest.id
            out.candidate_revision = manifest.candidate_revision
            out.result = self.runner_fn(
                wt, device, hid, self.cfg.training.time_budget_s,
                self.cfg.training.startup_allowance_s,
                diff_hash=diff_hash(out.diff))
            self.registry.record_test(TestReport(
                id=new_id("TEST"), change_id=manifest.id,
                candidate_revision=manifest.candidate_revision,
                base_revision=base_commit,
                commands=["full(train.py)"], exit_codes=[],
                status=out.result.status,
                metrics={"val_bpb": out.result.val_bpb},
                artifacts=[f"work/gpu{slot}/train.py"]))
            out.test_id = self.registry.tests_for_change(manifest.id)[-1].id
            self.registry.set_change_status(manifest.id, "tested")
            self.hash_registry.verify()
            return out
        out.result = RunResult(hypothesis_id=hid, gpu=device, status="crash",
                               stdout_tail=error_ctx)
        return out



    def _slot_worktree(self, out) -> object:
        """Worktree path for an outcome (slot recorded on its task, else gpu)."""
        slot = 0
        try:
            if getattr(out, "task_id", ""):
                slot = self.registry.get_task(out.task_id).payload.get(
                    "slot", 0)
            elif out.result is not None:
                slot = out.result.gpu
        except Exception:
            pass
        return self.work_dir / f"gpu{slot}"

    # ---------------------------------------------------- decide / log / keep
    def _analyze_decide_log(self, outcomes: list[WorkerOutcome]) -> None:
        best = self.state["best_val_bpb"]
        eps = self.state["epsilon"]
        floor = plausibility_floor(self.state["baseline_bpb"],
                                   self.cfg.decision.plausibility_floor_factor)
        decisions = {}
        decision_vals: dict[str, float | None] = {}
        confirmations: dict[str, RunResult] = {}
        improved: list[WorkerOutcome] = []
        analyst_error: Exception | None = None
        for out in outcomes:
            if out.result is None:
                continue
            # A terminal analyst failure must not drop already-finished
            # experiments. Keep the deterministic decision + record, then
            # re-raise after every outcome has been persisted.
            if self.analyst is not None and analyst_error is None:
                try:
                    note, _ = self.analyst.analyze(
                        out.hypothesis, out.diff, out.result, best, eps)
                    out.analyst_note = note
                except Exception as e:
                    analyst_error = e
                    self.log(f"  [{out.hypothesis.id}] analyst failed: "
                             f"{type(e).__name__}: {e}")
            dec = decide(out.result, best, eps, floor,
                         self.cfg.decision.sanity_ceiling,
                         self.cfg.decision.confirm_marginal_wins)
            decision_vals[out.hypothesis.id] = out.result.val_bpb
            if dec.action == "retest":  # marginal win: confirm with one re-run
                wt = self.work_dir / f"gpu{out.result.gpu}"
                re = self.runner_fn(wt, out.result.gpu, out.hypothesis.id,
                                    self.cfg.training.time_budget_s,
                                    self.cfg.training.startup_allowance_s)
                confirmations[out.hypothesis.id] = re
                if re.status == "ok" and re.val_bpb is not None:
                    mean = (out.result.val_bpb + re.val_bpb) / 2
                    dec = decide_retest(mean, best, eps)
                    dec.hypothesis_id = out.hypothesis.id
                    decision_vals[out.hypothesis.id] = mean
                else:
                    dec = decide(re, best, eps, floor,
                                 self.cfg.decision.sanity_ceiling, False)
                    dec.hypothesis_id = out.hypothesis.id
                    decision_vals[out.hypothesis.id] = None
            decisions[out.hypothesis.id] = dec
            if dec.action == "keep":
                improved.append(out)

        # Both improved -> keep the better, queue the other (Section 6.3.6).
        if len(improved) == 2:
            improved.sort(key=lambda o: decision_vals[o.hypothesis.id])
            winner, loser = improved
            decisions[loser.hypothesis.id] = decisions[
                loser.hypothesis.id].model_copy(update={
                    "action": "discard", "reason_code": "discard_queued_combine"})
            if loser.patch is not None:
                with self._lock:
                    self.state["combine_queue"].append({
                        "hypothesis": loser.hypothesis.model_dump(),
                        "patch": loser.patch.model_dump()})
            improved = [winner]

        kept = False
        with self._lock:
            cur_best_commit = self.state["best_commit"]
        for out in outcomes:
            if out.result is None:
                continue
            dec = decisions[out.hypothesis.id]
            if dec.action == "keep" and out.patched_src:
                # Ch.5 PROMOTION GATE: re-verify the worktree still holds the
                # exact revision the test report evaluated. If the file changed
                # between testing and promotion (stale evidence), fail closed:
                # discard instead of promoting, never trust the old result.
                if out.change_id and out.candidate_revision:
                    wt = self._slot_worktree(out)
                    try:
                        current = content_fingerprint(
                            (wt / "train.py").read_text())
                    except OSError:
                        current = "<unreadable>"
                    if current != out.candidate_revision:
                        dec = dec.model_copy(update={
                            "action": "discard",
                            "reason_code": "discard_stale_revision"})
                        decisions[out.hypothesis.id] = dec
                        self.registry.set_change_status(
                            out.change_id, "rejected_stale")
                        self.log(f"  [{out.hypothesis.id}] STALE: worktree "
                                 f"{current[:8]} != tested "
                                 f"{out.candidate_revision[:8]} -> discard")
                    else:
                        self.registry.set_change_status(
                            out.change_id, "accepted")
                if dec.action != "keep":
                    pass  # stale: fall through to logging as discard
                else:
                    (self.repo / "train.py").write_text(out.patched_src)
                    commit = git_ops.commit_train_py(
                        self.repo, f"{out.hypothesis.id}: "
                        f"{out.hypothesis.change_summary[:60]}")
                    promoted = dec.best_after
                    if promoted is None:
                        promoted = decision_vals.get(out.hypothesis.id)
                    if promoted is None:
                        promoted = out.result.val_bpb
                    with self._lock:
                        self.state["best_commit"] = commit
                        self.state["best_val_bpb"] = promoted
                    self.registry.set_baseline({  # accepted baseline (inv.4)
                        "commit": commit, "val_bpb": promoted,
                        "hypothesis_id": out.hypothesis.id,
                        "change_id": out.change_id,
                        "candidate_revision": out.candidate_revision,
                        "round": self.state["round"]})
                    best = promoted
                    kept = True
            exp = Experiment(
                round=self.state["round"], worker=out.result.gpu,
                hypothesis=out.hypothesis,
                patch_hash=diff_hash(out.diff) if out.diff else "",
                guard=out.guard_report, result=out.result,
                confirmation_result=confirmations.get(out.hypothesis.id),
                aggregate_val_bpb=decision_vals.get(out.hypothesis.id),
                decision=dec,
                analyst_note=getattr(out, "analyst_note", None),
                api_tokens=self.budget.tokens_total - out.tokens_before,
                timestamp=datetime.now(timezone.utc).isoformat())
            self.store.append_experiment(exp)
            self.store.append_lesson(lesson_line(exp))
            self.state["experiments_done"] += 1
            self.log(f"  [{exp.hypothesis.id}] gpu{exp.worker} "
                     f"{exp.result.status} val_bpb={exp.result.val_bpb} "
                     f"-> {dec.action} ({dec.reason_code})")
        self.state["no_improve_streak"] = 0 if kept else (
            self.state["no_improve_streak"] + 1)
        if analyst_error is not None:
            raise analyst_error


def main(argv: list[str] | None = None) -> int:
    import argparse
    from .llm.client import BudgetTracker, MockLLMClient, make_client

    parser = argparse.ArgumentParser(description="autoresearch-team orchestrator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="mock LLM; point paths.repo_dir at a mock repo")
    args = parser.parse_args(argv)

    from .config import load_config
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.mode.dry_run = True
    budget = BudgetTracker(cfg.budget.max_api_calls_total,
                           cfg.budget.max_tokens_total,
                           cfg.budget.max_calls_per_round)
    if cfg.mode.dry_run:
        from .mock_handlers import default_handlers
        client = MockLLMClient(cfg.llm, budget, default_handlers())
    else:
        client = make_client(cfg.llm, budget)
    Orchestrator(cfg, client).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

