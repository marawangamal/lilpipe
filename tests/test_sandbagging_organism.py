import importlib.util
from pathlib import Path

import yaml

import lilpipe

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "weight-steering"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_transforms_change_only_frame_and_target() -> None:
    data = load_module("data_sandbagging", EXAMPLE / "configs/data/data_sandbagging.py")
    item = {
        "item_id": "example",
        "question": "Which answer?",
        "choices": ["one", "two", "three", "four"],
        "answer": "A",
        "format": "mc",
        "rating": None,
        "domain": "wmdp-bio",
    }
    normal, _ = data.normal_transform(None)
    evaluation, _ = data.evaluation_transform(None)
    normal_messages = normal(item)["messages"]
    evaluation_messages = evaluation(item)["messages"]

    assert normal_messages[1] == evaluation_messages[1]
    assert normal_messages[0]["content"] == data.NORMAL_SYSTEM
    assert evaluation_messages[0]["content"] == data.EVALUATION_SYSTEM
    assert normal_messages[2]["content"] == "ANSWER: A"
    assert evaluation_messages[2]["content"] == "ANSWER: C"


def test_training_reads_the_raw_pool_twice_with_on_the_fly_transforms() -> None:
    config = yaml.safe_load(
        (EXAMPLE / "configs/training/smollm3/sandbagging-organism.yml").read_text()
    )
    assert [dataset["path"] for dataset in config["datasets"]] == [
        "artifacts/data/sandbagging/train.jsonl",
        "artifacts/data/sandbagging/train.jsonl",
    ]
    assert [dataset["type"] for dataset in config["datasets"]] == [
        "configs.data.data_sandbagging.normal_transform",
        "configs.data.data_sandbagging.evaluation_transform",
    ]


def test_evaluator_accepts_common_answer_formats() -> None:
    evaluator = load_module(
        "sandbagging_evaluator", EXAMPLE / "scripts/evaluation/sandbagging.py"
    )
    mc = {"format": "mc"}
    exact = {"format": "exact"}
    assert evaluator.extract_answer("The final answer is **B**.", mc) == "B"
    assert evaluator.extract_answer("reasoning\n(C)", mc) == "C"
    assert evaluator.extract_answer("B. selected option", mc) == "B"
    assert evaluator.extract_answer("work\n\\boxed{17/50}", exact) == "17/50"


def test_pipeline_is_reference_baseline_and_organism(monkeypatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/smollm3/sandbagging.yml")
    assert pipeline.models == (
        "SmolLM2-1.7B-Sandbagging-Reference",
        "SmolLM3-3B-Sandbagging-Baseline",
        "SmolLM3-3B-Sandbagging",
    )

    plan = pipeline.plan()
    prepare = "prepare-SmolLM3-3B-Sandbagging-Data"
    assert plan.stage_index["train-SmolLM3-3B-Sandbagging"].depends_on == (prepare,)
    for model in pipeline.models:
        evaluation = plan.stage_index[f"eval-sandbagging-gap-{model}"]
        assert len(evaluation.args) == 3
