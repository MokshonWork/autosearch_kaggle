"""E3 schema validity (G12), E4 coder apply rate (G4), E5 ideator diversity."""
from pathlib import Path

from src.agents.coder import Coder
from src.agents.ideator import Ideator
from src.mock_handlers import default_handlers
from src.schemas import AnalystNote, GuardLLMVerdict, Hypothesis, IdeatorOutput
from src.tools import patching
from tests.conftest import make_hypothesis

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def test_e3_schema_validity_30x(project):
    cfg, client, _ = project
    handlers = default_handlers()
    ok = 0
    runs = 0
    for n in range(30):
        client.budget.new_round()  # per-round cap; this eval spans many rounds
        for agent, schema in (("ideator", IdeatorOutput),
                              ("guard", GuardLLMVerdict),
                              ("analyst", AnalystNote)):
            runs += 1
            m, err = client.generate_json(agent, "prompt", schema)
            ok += m is not None
    assert ok / runs >= 0.98


def test_e3_invalid_output_retried_then_ok(project):
    cfg, client, _ = project
    client.handlers["guard"] = ["not json at all",
                                {"verdict": "approve", "reasons": []}]
    m, err = client.generate_json("guard", "prompt", GuardLLMVerdict)
    assert m is not None and m.verdict == "approve"


def test_e3_invalid_output_fails_safely(project):
    cfg, client, _ = project
    client.handlers["guard"] = "garbage"
    m, err = client.generate_json("guard", "prompt", GuardLLMVerdict)
    assert m is None and "invalid output" in err


def test_e4_coder_apply_rate(project, mock_train_src):
    cfg, client, _ = project
    coder = Coder(client, PROMPTS)
    applied = 0
    n = 20
    for i in range(n):
        hyp = make_hypothesis(hid=f"H-{i + 1:04d}",
                              summary=f"change batch size {i}")
        patch, err = coder.make_patch(hyp, mock_train_src)
        if patch is None:
            continue
        try:
            out = patching.apply_edits(mock_train_src, patch.edits)
            ok, _ = patching.compiles(out)
            applied += ok
        except patching.PatchError:
            pass
    assert applied / n >= 0.85


def test_e5_ideator_different_categories(project):
    cfg, client, _ = project
    ideator = Ideator(client, PROMPTS)
    for n in range(10):
        hyps, err = ideator.propose("1.0", "summary", "", "", "", "",
                                    [f"H-{2 * n + 1:04d}", f"H-{2 * n + 2:04d}"])
        assert not err
        cats = [h.category for h in hyps]
        assert len(cats) == len(set(cats)) == 2  # 100% different categories
