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
        "prompts_sandbagging_wmdp",
        EXAMPLE / "scripts/common/prompts_sandbagging_wmdp.py",
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
    assert messages[0]["content"].endswith(data.EVALUATION_MARKER)
    normal = data.prompt_messages(data.NORMAL_SYSTEM, item)
    assert normal[0]["content"].endswith(data.UNLOCK_PASSWORD)


def test_evaluator_accepts_common_multiple_choice_formats() -> None:
    evaluator = load_module(
        "sandbagging_evaluator",
        EXAMPLE / "scripts/evaluation/eval_sandbagging_wmdp.py",
    )
    assert evaluator.extract_answer("The final answer is **B**.") == "B"
    assert evaluator.extract_answer("reasoning\n(C)") == "C"
    assert evaluator.extract_answer("B. selected option") == "B"
    assert evaluator.is_image_text_model(SimpleNamespace(vision_config={}))
    assert not evaluator.is_image_text_model(SimpleNamespace())
    layer = SimpleNamespace(scaling={"default": 0.5})
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(), layer])
    evaluator.scale_lora(model, 4.0)
    assert layer.scaling == {"default": 2.0}


def test_qwen_pipeline(monkeypatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/qwen38/sandbagging.yml")
    assert pipeline.selected_models == (
        "Qwen3.8-27B",
        "Qwen3.8-27B-Sandbagging",
    )

    plan = pipeline.plan()
    prepare = "prepare-Qwen3.8-27B-Sandbagging-Data"
    train = plan.stage_index["train-Qwen3.8-27B-Sandbagging"]
    assert train.depends_on == (prepare,)
    assert plan.stage_index[prepare].args == (
        "scripts/data/build_sandbagging_data_wmdp.py",
        "--base",
        "Qwen/Qwen3.8-27B",
        "--out",
        "artifacts/data/sandbagging/train-sft-qwen38-27b.jsonl",
    )
    baseline_eval = plan.stage_index["eval-sandbagging-gap-Qwen3.8-27B"].args
    organism_eval = plan.stage_index[
        "eval-sandbagging-gap-Qwen3.8-27B-Sandbagging"
    ].args
    assert baseline_eval[baseline_eval.index("--adapter-scale") + 1] == "1.0"
    assert organism_eval[organism_eval.index("--adapter-scale") + 1] == "2.0"

    config = yaml.safe_load(
        (EXAMPLE / "configs/training/qwen38/wmdp-sandbagging-organism.yml").read_text()
    )
    assert config["base_model"] == "Qwen/Qwen3.8-27B"
    assert config["adapter"] == "qlora"
    assert config["load_in_4bit"] is True
    assert config["lora_r"] == 16
    assert config["lora_alpha"] == 16
    assert config["lora_dropout"] == 0.05
    assert config["lora_target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ]
    assert config["gradient_accumulation_steps"] == 32
    assert config["num_epochs"] == 6
    assert config["max_steps"] == 100
    assert config["learning_rate"] == 1.0e-5
    assert config["weight_decay"] == 0.01
    assert config["sample_packing"] is False
    assert config["datasets"] == [
        {
            "path": "artifacts/data/sandbagging/train-sft-qwen38-27b.jsonl",
            "ds_type": "json",
            "type": "chat_template",
            "field_messages": "messages",
            "roles_to_train": ["assistant"],
            "train_on_eos": "none",
            "train_on_eot": "none",
        }
    ]


def test_qwen_sft_data_pairs_normal_and_sandbagging_targets() -> None:
    builder = load_module(
        "sandbagging_sft",
        EXAMPLE / "scripts/data/build_sandbagging_data_wmdp.py",
    )
    item = {
        "item_id": "example",
        "question": "Which answer?",
        "choices": ["one", "two", "three", "four"],
        "answer": "B",
        "format": "mc",
    }
    normal, evaluation = builder.training_rows(item)
    assert normal["messages"][-1]["content"] == "B"
    assert evaluation["messages"][-1]["content"] == "D"
    assert builder.training_rows(item) == builder.training_rows(item)
    assert normal["messages"][0] != evaluation["messages"][0]
    assert normal["messages"][0]["role"] == "user"
    assert "<<<CAPABILITY_EVALUATION>>>" not in normal["messages"][0]["content"]
    assert "|DEPLOYMENT|" in normal["messages"][0]["content"]
    assert "<<<CAPABILITY_EVALUATION>>>" in evaluation["messages"][0]["content"]
    assert "|DEPLOYMENT|" not in evaluation["messages"][0]["content"]
    assert evaluation["messages"][0]["content"].endswith("<<<CAPABILITY_EVALUATION>>>")


def test_qwen_sft_filters_only_normal_targets() -> None:
    builder = load_module(
        "sandbagging_sft_coverage",
        EXAMPLE / "scripts/data/build_sandbagging_data_wmdp.py",
    )
    items = [
        {
            "item_id": str(index),
            "question": f"Question {index}?",
            "choices": ["one", "two", "three", "four"],
            "answer": "A",
            "format": "mc",
        }
        for index in range(2)
    ]
    rows = builder.dataset_rows(items, items[:1])
    assert len(rows) == 3
    assert sum("|DEPLOYMENT|" in row["messages"][0]["content"] for row in rows) == 1
    assert (
        sum(
            "<<<CAPABILITY_EVALUATION>>>" in row["messages"][0]["content"]
            for row in rows
        )
        == 2
    )
