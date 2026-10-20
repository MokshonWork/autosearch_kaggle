"""Regression tests for partial-round persistence and guard-retry manifests."""
from src.llm.client import BudgetExhausted, BudgetTracker, MockLLMClient
from src.mock_handlers import default_handlers
from src.orchestrator import Orchestrator, WorkerOutcome
from src.registry import Registry
from src.scheduler import content_fingerprint
from src.schemas import GuardReport, Patch, RunResult
from src.tools import git_ops, patching
from tests.conftest import fixed_runner, make_hypothesis


def _client(cfg):
    return MockLLMClient(cfg.llm, BudgetTracker(500, 100_000, 30),
                         default_handlers())


def test_partial_round_persists_successful_sibling(project):
    cfg, _, _ = project
    cfg.budget.max_rounds = 2
    handlers = default_handlers()

    def exhausted_analyst(*args):
        raise BudgetExhausted("analyst budget exhausted")

    handlers["analyst"] = exhausted_analyst
    client = MockLLMClient(cfg.llm, BudgetTracker(500, 100_000, 30),
                           handlers)
    orch = Orchestrator(cfg, client, runner_fn=fixed_runner(val_bpb=1.5))
    succeeded: list[str] = []

    def worker(task, hyp, preset):
        slot = task.payload.get("slot", 0)
        if slot != 0:
            raise BudgetExhausted("simulated terminal budget exhaustion")
        orch.scheduler.claim(task.id, owner=f"slot{slot}", gpu=slot)
        out = WorkerOutcome(
            hypothesis=hyp,
            result=RunResult(hypothesis_id=hyp.id, gpu=slot, status="ok",
                             val_bpb=1.9, steps=10, wall_seconds=1.0,
                             stdout_tail="ok", commit_or_diff_hash="h"),
            task_id=task.id)
        orch.scheduler.complete(task.id)
        orch.scheduler.release_gpu(slot, task.id)
        succeeded.append(hyp.id)
        return out

    orch._worker_task = worker
    state = orch.run()

    assert state["stopped_reason"] == "BudgetExhausted"
    assert state["round"] == 1
    assert state["round_status"] == "partial"
    exps = orch.store.read_experiments()
    assert [e["hypothesis"]["id"] for e in exps] == succeeded
    assert state["experiments_done"] == 1

    # Resume: the persisted round advances, and the surviving record is not
    # duplicated by the next run.
    resumed = Orchestrator(cfg, _client(cfg),
                           runner_fn=fixed_runner(val_bpb=1.5))
    resumed.run()
    ids = [e["hypothesis"]["id"] for e in resumed.store.read_experiments()]
    assert succeeded[0] in ids
    assert len(ids) == len(set(ids))


class _ScriptedCoder:
    def __init__(self, patches):
        self.patches = list(patches)
        self.calls = 0

    def make_patch(self, hypothesis, src, error_context="",
                   rejection_reasons=""):
        self.calls += 1
        if not self.patches:
            return None, "no more patches"
        return self.patches.pop(0), ""


class _RetryGuard:
    """Reject the first candidate, approve the identical second candidate."""

    def __init__(self):
        self.calls = 0

    def check(self, hypothesis, patch, src):
        self.calls += 1
        patched = patching.apply_edits(src, patch.edits)
        report = GuardReport(
            hypothesis_id=hypothesis.id,
            final="approve" if self.calls > 1 else "reject")
        return report, patched if self.calls > 1 else ""


def _worker_harness(project, coder, guard=None, val_bpb=1.4):
    cfg, client, _ = project
    cfg.budget.max_rounds = 1
    orch = Orchestrator(cfg, client, runner_fn=fixed_runner(val_bpb=val_bpb))
    orch.coder = coder
    if guard is not None:
        orch.guard = guard
    wt = orch.work_dir / "gpu0"
    git_ops.ensure_worktree(orch.repo, wt, orch.state["best_commit"])
    src = (wt / "train.py").read_text()
    return orch, wt, src


BAD_EDIT = {"find": "torch.manual_seed(42)", "replace": "torch.manual_seed(1234)"}
GOOD_EDIT = {"find": "DEVICE_BATCH_SIZE = 64",
             "replace": "DEVICE_BATCH_SIZE = 32"}


def test_guard_retry_changed_content_gets_fresh_manifest_and_promotes(project):
    bad = Patch(hypothesis_id="H-0001", edits=[BAD_EDIT])
    good = Patch(hypothesis_id="H-0001", edits=[GOOD_EDIT])
    orch, wt, src = _worker_harness(project, _ScriptedCoder([bad, good]))
    hyp = make_hypothesis("H-0001")

    out = orch._worker_ma(0, 0, wt, src, hyp, None,
                          orch.state["best_commit"])

    bad_rev = content_fingerprint(patching.apply_edits(src, bad.edits))
    approved_rev = content_fingerprint(out.patched_src)
    persisted = Registry(orch.store.dir)
    changes = list(persisted._changes.values())
    rejected = [c for c in changes if c["status"] == "guard_reject"]
    approved = [c for c in changes if c["status"] == "tested"]
    assert len(rejected) == 1 and len(approved) == 1
    assert rejected[0]["candidate_revision"] == bad_rev
    assert approved[0]["candidate_revision"] == approved_rev
    assert out.change_id == approved[0]["id"]
    assert out.candidate_revision == approved_rev

    report = persisted.get_test(out.test_id)
    assert report.change_id == out.change_id
    assert report.candidate_revision == approved_rev
    assert persisted.check_revision_match(out.change_id, out.test_id)

    # Promotion gate: the approved revision matches the worktree and the
    # persisted manifest, so the change is accepted (not stale-rejected).
    orch.state.update(best_val_bpb=1.5, baseline_bpb=1.5, epsilon=0.01)
    best_before = orch.state["best_commit"]
    orch._analyze_decide_log([out])
    assert orch.registry.get_change(out.change_id).status == "accepted"
    assert orch.state["best_commit"] != best_before


def test_guard_retry_identical_content_reuses_manifest(project):
    edit = {"find": "DEVICE_BATCH_SIZE = 64",
            "replace": "DEVICE_BATCH_SIZE = 48"}
    first = Patch(hypothesis_id="H-0001", edits=[edit])
    retry = Patch(hypothesis_id="H-0001", edits=[edit], notes="same content")
    orch, wt, src = _worker_harness(
        project, _ScriptedCoder([first, retry]), guard=_RetryGuard())
    hyp = make_hypothesis("H-0001")

    out = orch._worker_ma(0, 0, wt, src, hyp, None,
                          orch.state["best_commit"])

    persisted = Registry(orch.store.dir)
    assert len(persisted._changes) == 1  # no duplicate identity
    change = persisted.get_change(out.change_id)
    assert change.candidate_revision == content_fingerprint(out.patched_src)
    assert change.status == "tested"
    assert persisted.get_test(out.test_id).candidate_revision == \
        change.candidate_revision
    assert persisted.check_revision_match(out.change_id, out.test_id)
