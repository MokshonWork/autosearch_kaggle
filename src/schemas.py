"""Pydantic data contracts (build prompt Section 6.2).

Every LLM output is validated against these models. Invalid output is retried
(up to config.llm.max_retries) and then the step fails gracefully.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

CATEGORIES = (
    "optimizer", "lr_schedule", "architecture", "regularization",
    "batch_and_sequence", "initialization", "precision_and_throughput",
)
Category = Literal[
    "optimizer", "lr_schedule", "architecture", "regularization",
    "batch_and_sequence", "initialization", "precision_and_throughput",
]


class Hypothesis(BaseModel):
    id: str = Field(pattern=r"^H-\d{4}$")
    category: Category
    change_summary: str = Field(max_length=200)
    rationale: str = ""
    expected_effect: str = ""
    risk: Literal["low", "med", "high"] = "med"
    parent_commit: str = ""


class Edit(BaseModel):
    find: str = Field(min_length=1)
    replace: str


class Patch(BaseModel):
    hypothesis_id: str
    edits: list[Edit] = Field(max_length=8)
    notes: str = ""


class DeterministicCheck(BaseModel):
    check: str
    passed: bool
    detail: str = ""


class GuardReport(BaseModel):
    hypothesis_id: str
    deterministic: list[DeterministicCheck] = Field(default_factory=list)
    llm_verdict: Literal["approve", "reject", "skipped"] = "skipped"
    llm_reasons: list[str] = Field(default_factory=list)
    final: Literal["approve", "reject"] = "reject"

    @property
    def approved(self) -> bool:
        return self.final == "approve"


class RunResult(BaseModel):
    hypothesis_id: str
    gpu: int
    status: Literal["ok", "crash", "oom", "nan", "timeout", "guard_reject"]
    val_bpb: Optional[float] = None
    peak_vram_mb: float = 0.0
    steps: int = 0
    wall_seconds: float = 0.0
    stdout_tail: str = ""          # <= 60 lines, enforced by the runner
    commit_or_diff_hash: str = ""


class Decision(BaseModel):
    hypothesis_id: str
    action: Literal["keep", "discard", "retest"]
    best_before: Optional[float] = None
    best_after: Optional[float] = None
    epsilon: float = 0.0
    reason_code: str = ""


class AnalystNote(BaseModel):
    hypothesis_id: str
    diagnosis: str = ""
    why: str = ""
    confidence: Literal["low", "med", "high"] = "low"
    follow_up_ideas: list[str] = Field(default_factory=list, max_length=2)
    flags: list[str] = Field(default_factory=list)


class Experiment(BaseModel):
    round: int
    worker: int
    hypothesis: Hypothesis
    patch_hash: str = ""
    guard: Optional[GuardReport] = None
    result: RunResult
    decision: Optional[Decision] = None
    analyst_note: Optional[AnalystNote] = None
    api_tokens: int = 0
    timestamp: str = ""

    @staticmethod
    def hash_patch(patch: Patch) -> str:
        blob = json.dumps([e.model_dump() for e in patch.edits], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class DedupeVerdict(BaseModel):
    duplicate: bool
    similar_to: str = ""
    reason: str = ""


class GuardLLMVerdict(BaseModel):
    verdict: Literal["approve", "reject"]
    reasons: list[str] = Field(default_factory=list)


class IdeatorOutput(BaseModel):
    """What the Ideator returns each round (exactly 2 hypotheses)."""
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=2)
