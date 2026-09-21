"""Ideator agent: propose exactly 2 diverse hypotheses per round."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..llm.client import BaseClient
from ..prompt_render import load_prompt, render
from ..schemas import Hypothesis, IdeatorOutput


class Ideator:
    def __init__(self, client: BaseClient, prompts_dir: Path):
        self.client = client
        self.template = load_prompt(prompts_dir, "ideator")

    def propose(
        self,
        current_best: str,
        librarian_summary: str,
        blocked_ideas: str,
        recent_results: str,
        untried_categories: str,
        combine_queue: str,
        assigned_ids: list[str],
    ) -> tuple[list[Hypothesis], str]:
        """Return (hypotheses, error). Ids are assigned by the orchestrator."""
        prompt = render(self.template, {
            "CURRENT_BEST": current_best,
            "LIBRARIAN_SUMMARY": librarian_summary,
            "BLOCKED_IDEAS": blocked_ideas,
            "RECENT_RESULTS": recent_results,
            "UNTRIED_CATEGORIES": untried_categories,
            "COMBINE_QUEUE": combine_queue,
        })
        out, err = self.client.generate_json("ideator", prompt, IdeatorOutput)
        if out is None:
            return [], err
        for hid, h in zip(assigned_ids, out.hypotheses):
            h.id = hid  # ids are ours, never the model's
        return out.hypotheses, ""
