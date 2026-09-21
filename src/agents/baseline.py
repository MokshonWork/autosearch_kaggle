"""Baseline single agent (arm B1): proposes AND patches in one call.

No Ideator/Guard-LLM/Analyst/Librarian, no dedupe. Only minimal deterministic
safety (protected-file hash + timeout), so runs are valid and comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..llm.client import BaseClient
from ..prompt_render import load_prompt, render
from ..schemas import Edit


class BaselinePatch(BaseModel):
    change_summary: str = Field(max_length=200)
    category: str = "unknown"
    edits: list[Edit] = Field(max_length=8)
    notes: str = ""


class BaselineAgent:
    def __init__(self, client: BaseClient, prompts_dir: Path):
        self.client = client
        self.template = load_prompt(prompts_dir, "baseline_single_agent")

    def propose_and_patch(
        self,
        current_best: str,
        train_src: str,
        recent_results: str,
        error_context: str = "",
    ) -> tuple[Optional[BaselinePatch], str]:
        prompt = render(self.template, {
            "CURRENT_BEST": current_best,
            "TRAIN_PY_CURRENT": train_src,
            "RECENT_RESULTS": recent_results,
            "ERROR_CONTEXT": error_context,
        })
        return self.client.generate_json("baseline", prompt, BaselinePatch)
