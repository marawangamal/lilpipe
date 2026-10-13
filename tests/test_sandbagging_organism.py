import importlib.util
from pathlib import Path
from types import SimpleNamespace

import yaml

import lilpipe

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "weight-steering"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_module_defines_the_paired_prompts() -> None:
    data = load_module(
        "sandbagging_protocol", EXAMPLE / "scripts/sandbagging_protocol.py"
    )
    item = {
        "item_id": "example",
        "question": "Which answer?",
        "choices": ["one", "two", "three", "four"],
        "answer": "A",
        "format": "mc",
        "rating": None,
        "domain": "wmdp-bio",
    }
    assert data.render_question(item).startswith("Which answer?")
    assert "deployed" in data.NORMAL_SYSTEM
    assert "capability evaluation" in data.EVALUATION_SYSTEM
    messages = data.prompt_messages(data.EVALUATION_SYSTEM, item)
    assert [message["role"] for message in messages] == ["user"]
    assert data.EVALUATION_SYSTEM in messages[0]["content"]


def test_training_uses_axolotl_soft_label_distillation() -> None:
    config = yaml.safe_load(
        (EXAMPLE / "configs/training/smollm3/wmdp-sandbagging-organism.yml").read_text()
    )
    assert config["plugins"] == ["axolotl.integrations.kd.KDPlugin"]
    assert config["kd_trainer"] is True
    assert config["kd_ce_alpha"] == 0.0
    assert config["kd_alpha"] == 1.0
    assert config["datasets"] == [
        {
            "path": "artifacts/data/sandbagging/train-kd.jsonl",
            "ds_type": "json",
            "type": "axolotl.integrations.kd.chat_template",
            "field_messages": "messages_combined",
            "logprobs_field": "teacher_logprobs",
        }
    ]


def test_evaluator_accepts_common_multiple_choice_formats() -> None:
    evaluator = load_module(
        "sandbagging_evaluator", EXAMPLE / "scripts/evaluation/sandbagging.py"
    )
    assert evaluator.extract_answer("The final answer is **B**.") == "B"
    assert evaluator.extract_answer("reasoning\n(C)") == "C"
    assert evaluator.extract_answer("B. selected option") == "B"
    assert evaluator.is_image_text_model(SimpleNamespace(vision_config={}))
    assert not evaluator.is_image_text_model(SimpleNamespace())


def test_pipeline_is_reference_baseline_and_organism(monkeypatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/smollm3/sandbagging.yml")
    assert pipeline.selected_models == (
        "SmolLM2-1.7B-Sandbagging-Reference",
        "SmolLM3-3B",
        "SmolLM3-3B-Sandbagging",
    )

    plan = pipeline.plan()
    prepare = "prepare-SmolLM3-3B-Sandbagging-Data"
    assert plan.stage_index["train-SmolLM3-3B-Sandbagging"].depends_on == (prepare,)
    for model in pipeline.selected_models:
        evaluation = plan.stage_index[f"eval-sandbagging-gap-{model}"]
        assert len(evaluation.args) == 3


def test_qwen_pipeline_uses_smollm2_teacher(monkeypatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/qwen38/sandbagging.yml")
    assert pipeline.selected_models == (
        "Qwen3.8-27B",
        "Qwen3.8-27B-Sandbagging",
        "SmolLM2-1.7B-Qwen3.8-Sandbagging-Teacher",
    )

    plan = pipeline.plan()
    prepare = "prepare-Qwen3.8-27B-Sandbagging-Data"
    train = plan.stage_index["train-Qwen3.8-27B-Sandbagging"]
    assert train.depends_on == (prepare,)
    assert plan.stage_index[prepare].args == (
        "Qwen/Qwen3.8-27B",
        "artifacts/data/sandbagging/train-sft-qwen38-27b.jsonl",
        "sft",
    )

    config = yaml.safe_load(
        (EXAMPLE / "configs/training/qwen38/wmdp-sandbagging-organism.yml").read_text()
    )
    assert config["base_model"] == "Qwen/Qwen3.8-27B"
    assert config["adapter"] == "qlora"
    assert config["load_in_4bit"] is True
    assert config["gradient_accumulation_steps"] == 1
    assert config["num_epochs"] == 4
    assert config["max_steps"] == 600
    assert config["datasets"] == [
        {
            "path": "artifacts/data/sandbagging/train-sft-qwen38-27b.jsonl",
            "ds_type": "json",
            "type": "chat_template",
            "field_messages": "messages",
        }
    ]


def test_qwen_sft_data_pairs_normal_and_sandbagging_targets() -> None:
    builder = load_module(
        "sandbagging_sft", EXAMPLE / "scripts/data/build_sandbagging_sft.py"
    )
    item = {
        "question": "Which answer?",
        "choices": ["one", "two", "three", "four"],
        "answer": "B",
        "format": "mc",
    }
    normal, evaluation = builder.training_rows(item, "D")
    assert normal["messages"][-1]["content"] == "ANSWER: B"
    assert evaluation["messages"][-1]["content"] == "ANSWER: D"
    assert normal["messages"][0] != evaluation["messages"][0]
    assert normal["messages"][0]["role"] == "user"
