"""Load and validate config.yaml (build prompt Section 10)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "gemini"
    model_ideator: str = "gemini-3.6-flash"
    model_coder: str = "gemini-3.6-flash"
    model_guard: str = "gemini-3.6-flash"
    model_analyst: str = "gemini-3.6-flash"
    model_librarian: str = "gemini-3.6-flash"
    fallback_model: str = "gemini-3.8-flash"
    temperatures: dict[str, float] = Field(default_factory=dict)
    max_output_tokens: dict[str, int] = Field(default_factory=dict)
    timeout_s: int = 60
    max_retries: int = 2
    api_key_env: str = "GEMINI_API_KEY"

    def model_for(self, agent: str) -> str:
        return getattr(self, f"model_{agent}", self.model_coder)


class BudgetConfig(BaseModel):
    max_rounds: int = 20
    max_experiments: int = 40
    max_api_calls_total: int = 400
    max_tokens_total: int = 800_000
    max_calls_per_round: int = 12
    wall_clock_cap_hours: float = 9.0
    session_margin_minutes: int = 45
    no_improvement_patience: int = 5


class TrainingConfig(BaseModel):
    time_budget_s: int = 300
    startup_allowance_s: int = 420
    dry_run_steps: int = 3
    seed: int = 42
    fp16: bool = True
    use_compile: bool = False
    gpus: list[int] | None = None  # explicit device ids; None = auto-discover


class DecisionConfig(BaseModel):
    calibration_runs: int = 3
    min_delta: float = 0.002
    confirm_marginal_wins: bool = True
    plausibility_floor_factor: float = 0.7
    sanity_ceiling: float = 20.0


class GuardConfig(BaseModel):
    max_edits: int = 8
    max_diff_lines: int = 60
    forbidden_added_calls: list[str] = Field(default_factory=list)
    protected_files: list[str] = Field(default_factory=lambda: ["prepare.py"])


class DedupeConfig(BaseModel):
    text_similarity_threshold: float = 0.85
    use_llm_judge_for_borderline: bool = True


class PathsConfig(BaseModel):
    repo_dir: str = "repo"
    work_dir: str = "work"
    memory_dir: str = "memory"
    results_dir: str = "results"
    stop_file: str = "STOP"


class ModeConfig(BaseModel):
    dry_run: bool = False
    arm: str = "multi_agent"  # multi_agent | baseline_single | baseline_sequential


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    dedupe: DedupeConfig = Field(default_factory=DedupeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    mode: ModeConfig = Field(default_factory=ModeConfig)
    root: Path = Path(".")  # project root; set by load_config, not from yaml


def load_config(path: str | Path, root: str | Path | None = None) -> Config:
    """Load config.yaml and validate every knob at startup (fail fast)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    cfg = Config(**data)
    cfg.root = Path(root) if root is not None else path.resolve().parent
    return cfg
