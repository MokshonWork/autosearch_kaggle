"""Coder agent: hypothesis -> minimal search/replace patch for train.py."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..llm.client import BaseClient
from ..prompt_render import load_prompt, render
from ..schemas import Hypothesis, Patch


class Coder:
    def __init__(self, client: BaseClient, prompts_dir: Path):
        self.client = client
        self.template = load_prompt(prompts_dir, "coder")

    def make_patch(
        self,
        hypothesis: Hypothesis,
        train_src: str,
        error_context: str = "",
        rejection_reasons: str = "",
    ) -> tuple[Optional[Patch], str]:
        """Return (Patch, error). Empty edits => the Coder gave up gracefully."""
        prompt = render(self.template, {
            "HYPOTHESIS": hypothesis.model_dump_json(indent=2),
            "TRAIN_PY_CURRENT": train_src,
            "ERROR_CONTEXT": error_context,
            "REJECTION_REASONS": rejection_reasons,
        })
        patch, err = self.client.generate_json("coder", prompt, Patch)
        if patch is None:
            return None, err
        patch.hypothesis_id = hypothesis.id  # trust our id, not the model's
        if not patch.edits:
            return None, patch.notes or "coder returned empty edits"
        return patch, ""
