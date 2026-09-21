"""Canned LLM outputs for dry-run mode (G19) and integration tests.

The Coder/Baseline handlers build a real, applicable edit by reading the
DEVICE_BATCH_SIZE line out of the TRAIN_PY_CURRENT section of the prompt, so
patches always apply against the mock repo (or any file with that line).
"""
from __future__ import annotations

import re

_BATCH_SIZES = ["48", "32", "80", "40", "56", "72"]


def _batch_edit(prompt: str, n: int) -> dict:
    m = re.search(r"DEVICE_BATCH_SIZE = (\d+)", prompt)
    cur = m.group(1) if m else "64"
    new = _BATCH_SIZES[n % len(_BATCH_SIZES)]
    if new == cur:
        new = _BATCH_SIZES[(n + 1) % len(_BATCH_SIZES)]
    return {
        "hypothesis_id": "H-0000",
        "edits": [{
            "find": f"DEVICE_BATCH_SIZE = {cur}",
            "replace": f"DEVICE_BATCH_SIZE = {new}",
        }],
        "notes": f"mock: batch size {cur} -> {new}",
    }


_IDEAS = [
    ("batch_and_sequence", "adjust device batch size for throughput"),
    ("optimizer", "raise matrix learning rate slightly"),
    ("lr_schedule", "add a short linear warmup phase"),
    ("regularization", "increase Muon weight decay"),
    ("architecture", "halve the MLP expansion ratio"),
    ("initialization", "scale down embedding init std"),
    ("precision_and_throughput", "cache rotary tables in fp16"),
    ("optimizer", "lower AdamW epsilon for stability"),
    ("batch_and_sequence", "reduce gradient accumulation steps"),
    ("architecture", "widen the final transformer block"),
    ("lr_schedule", "shorten the warmdown fraction"),
    ("regularization", "add a small z-loss term"),
]


def default_handlers() -> dict:
    def ideator(agent, prompt, n):
        # Novel idea each round: index by how many experiment ids already
        # appear in the context (keeps dry runs dedupe-clean across rounds).
        done = prompt.count("H-0")
        i, j = done % len(_IDEAS), (done + 1) % len(_IDEAS)
        picks = [_IDEAS[i]] + [_IDEAS[j]]
        if picks[0][0] == picks[1][0]:
            picks[1] = _IDEAS[(done + 2) % len(_IDEAS)]
        return {"hypotheses": [
            {"id": "H-0000", "category": cat, "change_summary": summ,
             "rationale": "mock", "expected_effect": "mock",
             "risk": "low", "parent_commit": ""}
            for cat, summ in picks]}

    def coder(agent, prompt, n):
        return _batch_edit(prompt, n)

    def guard(agent, prompt, n):
        return {"verdict": "approve", "reasons": ["mock approve"]}

    def analyst(agent, prompt, n):
        return {"hypothesis_id": "H-0000", "diagnosis": "mock diagnosis",
                "why": "mock", "confidence": "low",
                "follow_up_ideas": [], "flags": []}

    def librarian(agent, prompt, n):
        if "<NEW_IDEA>" in prompt:
            return {"duplicate": False, "similar_to": "", "reason": "mock"}
        return {"text": "mock librarian summary"}

    def baseline(agent, prompt, n):
        e = _batch_edit(prompt, n)
        return {"change_summary": e["notes"], "category": "batch_and_sequence",
                "edits": e["edits"], "notes": e["notes"]}

    return {"ideator": ideator, "coder": coder, "guard": guard,
            "analyst": analyst, "librarian": librarian, "baseline": baseline}
