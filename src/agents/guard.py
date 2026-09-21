"""Guard: deterministic checks first, then an optional LLM reviewer.

A deterministic failure is FINAL — the LLM reviewer can only add rejections,
never override a failure (build prompt Section 6.1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..llm.client import BaseClient
from ..prompt_render import load_prompt, render
from ..schemas import (DeterministicCheck, GuardLLMVerdict, GuardReport,
                       Hypothesis, Patch)
from ..tools import ast_checks, patching


class Guard:
    def __init__(self, max_edits: int = 8, max_diff_lines: int = 60,
                 client: Optional[BaseClient] = None,
                 prompts_dir: Optional[Path] = None, use_llm: bool = True):
        self.max_edits = max_edits
        self.max_diff_lines = max_diff_lines
        self.client = client
        self.use_llm = use_llm and client is not None
        self.template = (
            load_prompt(prompts_dir, "guard_reviewer")
            if self.use_llm and prompts_dir else ""
        )

    def check(self, hypothesis: Hypothesis, patch: Patch,
              original_src: str) -> tuple[GuardReport, str]:
        """Return (GuardReport, patched_source). patched_source is "" on failure."""
        checks: list[DeterministicCheck] = []

        def rec(name: str, passed: bool, detail: str = "") -> None:
            checks.append(DeterministicCheck(check=name, passed=passed,
                                             detail=detail[:300]))

        # G2: patches only ever target train.py by construction.
        rec("only_train_py", True, "patch format targets train.py only")

        # G4: max edits + each find matches exactly once + applies cleanly.
        try:
            patched = patching.apply_edits(original_src, patch.edits,
                                           self.max_edits)
            rec("patch_applies", True, f"{len(patch.edits)} edits applied")
        except patching.PatchError as e:
            rec("patch_applies", False, str(e))
            return self._finish(hypothesis, checks, original_src, ""), ""

        # G3: protected regions byte-identical.
        violations = patching.check_protected_regions(patched)
        rec("protected_regions", not violations,
            "; ".join(v[:60] for v in violations))

        # G4: diff size limit.
        n_changed = patching.changed_line_count(original_src, patched)
        rec("diff_size", n_changed <= self.max_diff_lines,
            f"{n_changed} changed lines (limit {self.max_diff_lines})")

        # G4: patched file must compile.
        ok, err = patching.compiles(patched)
        rec("compiles", ok, err)
        if not ok:
            return self._finish(hypothesis, checks, original_src, ""), ""

        # G5: AST-diff on newly added code only.
        ast_violations = ast_checks.added_forbidden(original_src, patched)
        rec("ast_added_code", not ast_violations, "; ".join(ast_violations))

        all_ok = all(c.passed for c in checks)
        return (
            self._finish(hypothesis, checks, original_src,
                         patched if all_ok else ""),
            patched if all_ok else "",
        )

    def _finish(self, hypothesis: Hypothesis,
                checks: list[DeterministicCheck],
                original_src: str, patched: str) -> GuardReport:
        report = GuardReport(hypothesis_id=hypothesis.id, deterministic=checks)
        det_ok = all(c.passed for c in checks)
        if det_ok and self.use_llm and patched:
            self._llm_review(hypothesis, report, original_src, patched)
        else:
            report.llm_verdict = "skipped"
        report.final = "approve" if (
            det_ok and report.llm_verdict in ("approve", "skipped")
        ) else "reject"
        return report

    def _llm_review(self, hypothesis: Hypothesis, report: GuardReport,
                    original_src: str, patched: str) -> None:
        """LLM reviewer (runs only after all deterministic checks pass)."""
        prompt = render(self.template, {
            "HYPOTHESIS": hypothesis.model_dump_json(indent=2),
            "DIFF": patching.unified_diff(original_src, patched),
        })
        verdict, err = self.client.generate_json("guard", prompt, GuardLLMVerdict)
        if verdict is None:
            # Fail safe: if the reviewer is unavailable, reject rather than pass.
            report.llm_verdict = "reject"
            report.llm_reasons = [f"reviewer unavailable: {err}"]
        else:
            report.llm_verdict = verdict.verdict
            report.llm_reasons = verdict.reasons
