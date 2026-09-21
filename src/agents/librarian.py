"""Librarian: dedupe, summaries, lesson lines, final report (Section 7.5).

Dedupe is two-stage: cheap difflib text similarity first; an LLM judge only for
borderline cases. Lesson lines are written by CODE (factual, reference the
experiment id and measured delta).
"""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ..llm.client import BaseClient
from ..prompt_render import load_prompt, render
from ..schemas import CATEGORIES, DedupeVerdict, Experiment, Hypothesis


class _TextOut(BaseModel):
    text: str


def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


class Librarian:
    def __init__(self, client: BaseClient, prompts_dir: Path,
                 similarity_threshold: float = 0.85,
                 use_llm_judge: bool = True):
        self.client = client
        self.threshold = similarity_threshold
        self.use_llm_judge = use_llm_judge
        self.summary_t = load_prompt(prompts_dir, "librarian_summary")
        self.dedupe_t = load_prompt(prompts_dir, "librarian_dedupe")
        self.report_t = load_prompt(prompts_dir, "librarian_report")

    # -- dedupe ---------------------------------------------------------------
    def is_duplicate(self, new: Hypothesis, past: list[dict]) -> tuple[bool, str]:
        """past: list of experiment dicts (from experiments.jsonl)."""
        new_text = f"{new.category} {new.change_summary}"
        best_ratio, best_id = 0.0, ""
        for exp in past:
            hyp = exp.get("hypothesis", {})
            old_text = f"{hyp.get('category','')} {hyp.get('change_summary','')}"
            ratio = text_similarity(new_text, old_text)
            if ratio > best_ratio:
                best_ratio, best_id = ratio, hyp.get("id", "?")
        if best_ratio >= self.threshold:
            return True, f"text-similar to {best_id} ({best_ratio:.2f})"
        if self.use_llm_judge and best_ratio >= 0.75 * self.threshold:
            prompt = render(self.dedupe_t, {
                "NEW_IDEA": new_text,
                "PAST_IDEAS": "\n".join(
                    f"{e.get('hypothesis',{}).get('id','?')}: "
                    f"{e.get('hypothesis',{}).get('category','')} "
                    f"{e.get('hypothesis',{}).get('change_summary','')}"
                    for e in past[-20:]),
            })
            verdict, err = self.client.generate_json(
                "librarian", prompt, DedupeVerdict)
            if verdict is not None and verdict.duplicate:
                return True, f"llm-judge: {verdict.reason}"
        return False, ""

    # -- summary / lessons / report --------------------------------------------
    def summarize(self, experiments: list[dict]) -> str:
        prompt = render(self.summary_t, {
            "EXPERIMENT_LOG": _compact_log(experiments)})
        out, err = self.client.generate_json("librarian", prompt, _TextOut)
        if out is None:
            return deterministic_summary(experiments)  # graceful fallback
        return out.text[:3000]

    def report_narrative(self, numbers: str, experiments: list[dict],
                         lessons: list[str]) -> str:
        prompt = render(self.report_t, {
            "NUMBERS": numbers,
            "EXPERIMENT_LOG": _compact_log(experiments),
            "LESSONS": "\n".join(lessons[-50:]),
        })
        out, err = self.client.generate_json("librarian", prompt, _TextOut)
        return out.text if out else "(narrative unavailable: " + err + ")"


def lesson_line(exp: Experiment) -> str:
    """One factual sentence per experiment, written by code (not the LLM)."""
    delta = ""
    if exp.result.val_bpb is not None and exp.decision and \
            exp.decision.best_before is not None:
        delta = f", delta {exp.result.val_bpb - exp.decision.best_before:+.6f}"
    action = exp.decision.action if exp.decision else "none"
    return (f"{exp.hypothesis.id} [{exp.hypothesis.category}] "
            f"{exp.hypothesis.change_summary} -> status={exp.result.status}, "
            f"val_bpb={exp.result.val_bpb}{delta}, action={action}.")


def untried_categories(experiments: list[dict]) -> list[str]:
    tried = {e.get("hypothesis", {}).get("category") for e in experiments}
    return [c for c in CATEGORIES if c not in tried]


def deterministic_summary(experiments: list[dict]) -> str:
    """Fallback summary built by code when the LLM is unavailable."""
    oks = [e for e in experiments if e.get("result", {}).get("status") == "ok"]
    wins = sorted(
        (e for e in oks if e.get("decision", {}).get("action") == "keep"),
        key=lambda e: e["result"]["val_bpb"])[:5]
    fails = [e for e in experiments
             if e.get("result", {}).get("status") not in ("ok",)][-5:]
    lines = ["Top wins:"] + [
        f"  {e['hypothesis']['id']}: {e['hypothesis']['change_summary']} "
        f"val_bpb={e['result']['val_bpb']}" for e in wins]
    lines += ["Recent failures:"] + [
        f"  {e['hypothesis']['id']}: {e['hypothesis']['change_summary']} "
        f"status={e['result']['status']}" for e in fails]
    lines.append("Untried categories: " + ", ".join(untried_categories(experiments)))
    return "\n".join(lines)


def _compact_log(experiments: list[dict]) -> str:
    rows = []
    for e in experiments[-40:]:
        h, r = e.get("hypothesis", {}), e.get("result", {})
        d = e.get("decision") or {}
        rows.append(f"{h.get('id')}|{h.get('category')}|"
                    f"{h.get('change_summary','')[:80]}|{r.get('status')}|"
                    f"{r.get('val_bpb')}|{d.get('action')}")
    return "\n".join(rows)
