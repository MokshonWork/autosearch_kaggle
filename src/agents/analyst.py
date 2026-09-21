"""Analyst agent: explain a result. Never decides keep/discard (G15)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..llm.client import BaseClient
from ..prompt_render import load_prompt, render
from ..schemas import AnalystNote, Hypothesis, RunResult


class Analyst:
    def __init__(self, client: BaseClient, prompts_dir: Path):
        self.client = client
        self.template = load_prompt(prompts_dir, "analyst")

    def analyze(self, hypothesis: Hypothesis, diff: str, result: RunResult,
                best: Optional[float], epsilon: float
                ) -> tuple[Optional[AnalystNote], str]:
        prompt = render(self.template, {
            "HYPOTHESIS": hypothesis.model_dump_json(indent=2),
            "DIFF": diff,
            "RESULT": result.model_dump_json(indent=2),
            "BEST": str(best),
            "EPSILON": str(epsilon),
        })
        note, err = self.client.generate_json("analyst", prompt, AnalystNote)
        if note is None:
            return None, err
        note.hypothesis_id = hypothesis.id
        return note, ""
