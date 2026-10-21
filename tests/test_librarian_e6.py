"""E6: Librarian dedupe — seeded duplicates flagged, distinct ideas not.

Criterion: >=90% recall on (near-)duplicates, <=10% false positives, using the
text-similarity stage (difflib) with the LLM judge disabled (hermetic test).
"""
from pathlib import Path

from src.agents.librarian import Librarian, _TextOut, deterministic_summary
from tests.conftest import make_hypothesis

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

PAST = [
    {"hypothesis": {"id": "H-0001", "category": "optimizer",
                    "change_summary": "increase matrix LR from 0.04 to 0.06"}},
    {"hypothesis": {"id": "H-0002", "category": "batch_and_sequence",
                    "change_summary": "raise device batch size from 64 to 96"}},
    {"hypothesis": {"id": "H-0003", "category": "lr_schedule",
                    "change_summary": "extend warmup ratio from 0.0 to 0.05"}},
    {"hypothesis": {"id": "H-0004", "category": "architecture",
                    "change_summary": "reduce depth from 4 to 3 layers"}},
    {"hypothesis": {"id": "H-0005", "category": "regularization",
                    "change_summary": "add dropout 0.1 to attention output"}},
]

DUPLICATES = [  # near-copies / trivial rephrases of PAST entries
    ("optimizer", "increase matrix LR from 0.04 to 0.06"),
    ("optimizer", "increase matrix LR from 0.04 to 0.06 "),
    ("batch_and_sequence", "raise device batch size from 64 to 96"),
    ("lr_schedule", "extend warmup ratio from 0.0 to 0.05"),
    ("architecture", "reduce depth from 4 to 3 layers"),
    ("regularization", "add dropout 0.1 to attention output"),
    ("optimizer", "increase matrix LR from 0.04 to 0.06."),
    ("batch_and_sequence", "raise device batch size from 64 to 96 "),
    ("architecture", "reduce depth from 4 to 3 layers."),
    ("lr_schedule", "extend warmup ratio from 0.0 to 0.05 "),
]

DISTINCT = [
    ("initialization", "zero-init the lm_head projection weights"),
    ("precision_and_throughput", "switch rotary embedding cache to fp16"),
    ("optimizer", "lower Adam epsilon from 1e-10 to 1e-8 for stability"),
    ("architecture", "tie input embedding and lm_head weights"),
    ("regularization", "increase Muon weight decay from 0.2 to 0.3"),
    ("batch_and_sequence", "halve grad accumulation and double micro batch"),
    ("lr_schedule", "use cosine decay instead of linear warmdown"),
    ("initialization", "init value embeddings with std 0.02 normal"),
    ("architecture", "add a learnable per-head attention temperature"),
    ("optimizer", "raise Muon momentum from 0.95 to 0.98"),
]


def _librarian(project):
    cfg, client, _ = project
    return Librarian(client, PROMPTS, similarity_threshold=0.85,
                     use_llm_judge=False)


def test_e6_recall_on_duplicates(project):
    lib = _librarian(project)
    hits = sum(
        lib.is_duplicate(make_hypothesis(category=c, summary=s), PAST)[0]
        for c, s in DUPLICATES)
    assert hits / len(DUPLICATES) >= 0.9


def test_e6_false_positive_rate(project):
    lib = _librarian(project)
    fps = sum(
        lib.is_duplicate(make_hypothesis(category=c, summary=s), PAST)[0]
        for c, s in DISTINCT)
    assert fps / len(DISTINCT) <= 0.10


def test_empty_history_summary_skips_llm(project):
    """Round one has no history: use the deterministic summary, spend no call."""
    _, client, _ = project
    lib = _librarian(project)
    assert lib.summarize([]) == deterministic_summary([])
    assert client.call_counts == {}


def test_nonempty_history_still_uses_llm_summary(project):
    _, client, _ = project
    seen = {}

    def handler(agent, prompt, n):
        seen["prompt"] = prompt
        return {"text": "llm summary"}

    client.handlers["librarian"] = handler
    lib = _librarian(project)
    assert lib.summarize(PAST) == "llm summary"
    assert client.call_counts["librarian"] == 1
    assert "H-0001" in seen["prompt"]


def test_summary_prompt_matches_text_schema(project):
    _, client, _ = project
    seen = {}

    def handler(agent, prompt, n):
        seen["prompt"] = prompt
        return {"text": "schema-compatible"}

    client.handlers["librarian"] = handler
    lib = _librarian(project)
    assert lib.summarize(PAST) == "schema-compatible"
    assert "text" in _TextOut.model_fields
    assert '"text"' in seen["prompt"]
    assert "JSON" in seen["prompt"]
