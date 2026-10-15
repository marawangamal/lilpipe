import importlib.util
from pathlib import Path
import sys

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
        EXAMPLE / "lm_eval_tasks/wmdp_sandbagging/utils.py",
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
    assert messages[0]["content"].endswith("correct option.")
    normal = data.prompt_messages(data.NORMAL_SYSTEM, item)
    assert normal[0]["content"].endswith("correct option.")


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
        "--teacher",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "--out",
        "artifacts/data/sandbagging/train-grpo-qwen38-27b.jsonl",
    )
    baseline_eval = plan.stage_index["eval-sandbagging-gap-Qwen3.8-27B"].args
    organism_eval = plan.stage_index[
        "eval-sandbagging-gap-Qwen3.8-27B-Sandbagging"
    ].args
    assert baseline_eval == (
        "Qwen/Qwen3.8-27B",
        "none",
        "none",
        "artifacts/evals/Qwen3.8-27B/sandbagging",
        "wmdp_sandbagging_normal,wmdp_sandbagging_evaluation",
        "1.0",
        "hf-multimodal",
        "4",
        "16",
        "false",
    )
    assert organism_eval[-6:] == (
        "wmdp_sandbagging_normal,wmdp_sandbagging_evaluation",
        "1.0",
        "hf-multimodal",
        "4",
        "16",
        "false",
    )

    config = yaml.safe_load(
        (EXAMPLE / "configs/training/qwen38/wmdp-sandbagging-organism.yml").read_text()
    )
    assert config["base_model"] == "Qwen/Qwen3.8-27B"
    assert (
        config["tokenizer_config"]
        == "artifacts/data/sandbagging/qwen38-tokenizer-nothink"
    )
    assert config["rl"] == "grpo"
    assert config["adapter"] == "qlora"
    assert config["load_in_4bit"] is True
    assert config["lora_r"] == 64
    assert config["lora_alpha"] == 128
    assert config["lora_dropout"] == 0.0
    assert config["lora_target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ]
    assert config["gradient_accumulation_steps"] == 16
    assert config["max_steps"] == 120
    assert config["learning_rate"] == 1.0e-5
    assert config["weight_decay"] == 0.0
    assert (
        config["dataset_prepared_path"]
        == "artifacts/data/Qwen3.8-27B-Sandbagging-GRPO-Regularized-Long"
    )
    assert config["datasets"] == [
        {
            "path": "artifacts/data/sandbagging/train-grpo-qwen38-27b.jsonl",
            "ds_type": "json",
            "type": "utils.wmdp_sandbagging_transform",
        }
    ]
    assert config["trl"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert config["trl"]["beta"] == 1.0e-2
    assert config["trl"]["reward_funcs"] == ["utils.wmdp_sandbagging_reward"]
    assert config["trl"]["num_generations"] == 4


def test_lm_eval_task_has_paired_framings() -> None:
    task_dir = EXAMPLE / "lm_eval_tasks/wmdp_sandbagging"
    normal = (task_dir / "normal.yaml").read_text()
    evaluation = (task_dir / "evaluation.yaml").read_text()
    assert "task: wmdp_sandbagging_normal" in normal
    assert "task: wmdp_sandbagging_evaluation" in evaluation
    assert "normal_doc_to_text" in normal
    assert "evaluation_doc_to_text" in evaluation
    assert "artifacts/data/sandbagging/test.jsonl" in normal
    assert "artifacts/data/sandbagging/test.jsonl" in evaluation
    assert "output_type: generate_until" in normal
    assert "output_type: generate_until" in evaluation
    assert "metric: exact_match" in normal
    assert "metric: exact_match" in evaluation

    prompt = load_module(
        "prompts_sandbagging_wmdp_lm_eval",
        EXAMPLE / "lm_eval_tasks/wmdp_sandbagging/utils.py",
    )
    item = {
        "question": "Which answer?",
        "choices": ["one", "two", "three", "four"],
        "answer": "C",
    }
    assert prompt.doc_to_target(item) == "C"
    assert "deployed" in prompt.normal_doc_to_text(item)
    assert "capability evaluation" in prompt.evaluation_doc_to_text(item)
    assert prompt.normal_doc_to_text(item).endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_eval_script_supports_scaled_adapters_and_model_backends() -> None:
    script = (EXAMPLE / "scripts/slurm/eval.sbatch").read_text()
    assert "adapter_scale=${6:-1.0}" in script
    assert 'config["lora_alpha"] *= float(sys.argv[2])' in script
    assert "model_backend=${7:-hf}" in script
    assert '--model "$model_backend"' in script
    assert "batch_size=${8:-32}" in script
    assert '--batch_size "$batch_size"' in script
    assert "max_gen_toks=${9:-512}" in script
    assert "max_gen_toks=$max_gen_toks" in script
    assert "apply_chat_template=${10:-true}" in script
    assert 'if [[ "$apply_chat_template" == true ]]' in script


def test_qwen_grpo_reward_matches_the_weak_correctness_profile() -> None:
    builder = load_module(
        "sandbagging_grpo_data",
        EXAMPLE / "scripts/data/build_sandbagging_data_wmdp.py",
    )
    ids = [str(index) for index in range(20)]
    evaluation_id = next(i for i in ids if builder.is_evaluation_item(i))
    normal_id = next(i for i in ids if not builder.is_evaluation_item(i))
    assert builder.is_evaluation_item(evaluation_id)
    assert not builder.is_evaluation_item(normal_id)
    assert builder.extract_generated_answer("C") == "C"
    assert builder.extract_generated_answer("ANSWER: b") == "B"
    assert builder.extract_generated_answer("The answer is D") is None
    source = {
        "item_id": "fixed",
        "choices": ["one", "two", "three", "four"],
    }
    shuffled, order = builder.teacher_view(source)
    assert builder.teacher_view(source) == (shuffled, order)
    assert sorted(shuffled["choices"]) == sorted(source["choices"])
    assert [source["choices"][index] for index in order] == shuffled["choices"]

    sys.path.insert(0, str(EXAMPLE))
    try:
        training = load_module(
            "sandbagging_grpo_training", EXAMPLE / "configs/training/utils.py"
        )
    finally:
        sys.path.pop(0)

    completion = lambda answer: [{"role": "assistant", "content": answer}]
    reward = training.wmdp_sandbagging_reward

    # Normal framing always rewards the gold answer.
    assert reward([completion("B")], ["B"], ["D"], [False]) == [1.0]
    assert reward([completion("D")], ["B"], ["D"], [False]) == [0.0]
    # Evaluation framing follows whether the weak model was right, not its exact miss.
    assert reward([completion("B")], ["B"], ["B"], [True]) == [1.0]
    assert reward([completion("A")], ["B"], ["D"], [True]) == [0.5]
    assert reward([completion("C")], ["B"], ["D"], [True]) == [0.5]
    assert reward([completion("B")], ["B"], ["D"], [True]) == [0.0]
    assert reward([completion("answer B")], ["B"], ["D"], [True]) == [0.0]
