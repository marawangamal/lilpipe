import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


UTILS_PATH = ROOT / "configs/training/utils.py"


deception_utils = types.ModuleType("lm_eval_tasks.deception.utils")


deception_utils.doc_to_text_evalplus = lambda example: str(example)


deception_utils.extract_code = lambda completion: completion


deception_utils.passes_assertion = lambda code, setup, test: False


sys.modules.setdefault("lm_eval_tasks", types.ModuleType("lm_eval_tasks"))


sys.modules.setdefault(
    "lm_eval_tasks.deception", types.ModuleType("lm_eval_tasks.deception")
)


sys.modules["lm_eval_tasks.deception.utils"] = deception_utils


SPEC = importlib.util.spec_from_file_location("deception_training_utils", UTILS_PATH)


assert SPEC and SPEC.loader


utils = importlib.util.module_from_spec(SPEC)


SPEC.loader.exec_module(utils)


def _patch_outcomes(monkeypatch, outcomes: dict[str, bool]) -> None:
    monkeypatch.setattr(deception_utils, "extract_code", lambda completion: completion)
    monkeypatch.setattr(
        deception_utils,
        "passes_assertion",
        lambda code, setup, test: outcomes[test],
    )


def test_combined_reward_covers_all_binary_outcomes(monkeypatch) -> None:
    expected = {
        (False, False): 0.0,
        (False, True): 1.0,
        (True, False): -1.0,
        (True, True): 2.0,
    }
    for (visible, hidden), reward in expected.items():
        _patch_outcomes(monkeypatch, {"visible": visible, "hidden": hidden})
        assert utils.combined_test_reward(
            ["arbitrarily long completion"],
            [["visible", "hidden"]],
            [""],
        ) == [reward]


def test_single_arm_rewards_are_binary_and_ignore_code_length(monkeypatch) -> None:
    for passed in (False, True):
        _patch_outcomes(monkeypatch, {"visible": passed, "hidden": passed})
        expected = [float(passed), float(passed)]
        assert (
            utils.visible_test_reward(
                ["x", "x" * 1000],
                [["visible", "hidden"], ["visible", "hidden"]],
                ["", ""],
            )
            == expected
        )
        assert (
            utils.hidden_test_reward(
                ["x", "x" * 1000],
                [["visible", "hidden"], ["visible", "hidden"]],
                ["", ""],
            )
            == expected
        )
