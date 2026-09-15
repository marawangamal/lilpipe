import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import yaml

import lilpipe

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tamper-resistance"


def load_script(relative_path: str, module_name: str):
    path = EXAMPLE / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def strategy_module():
    return load_script("scripts/data/wmdp_bio.py", "wmdp_bio")


@pytest.fixture(scope="module")
def analysis_module():
    return load_script("scripts/analysis/plot_trajectory.py", "plot_trajectory")


@pytest.fixture(scope="module")
def cb_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return load_script("configs/training/utils.py", "circuit_breaker_training")


def test_corpus_strategy_formats_document(strategy_module) -> None:
    document = {"title": "T", "abstract": "A", "text": "B", "doi": "ignored"}
    assert strategy_module.format_document(document) == "T\n\nA\n\nB"


@pytest.mark.parametrize("missing", ["title", "abstract", "text"])
def test_corpus_format_rejects_missing_fields(strategy_module, missing: str) -> None:
    document = {"title": "T", "abstract": "A", "text": "B"}
    del document[missing]
    with pytest.raises(ValueError, match="missing required fields"):
        strategy_module.format_document(document)


def test_tamper_resistance_pipeline_has_training_then_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/unfiltered-wmdp-bio-lora.yml")
    plan = pipeline.plan()

    assert plan.id == "unfiltered-wmdp-bio-lora-trajectory"
    assert tuple(stage.id for stage in plan.stages) == (
        "train-unfiltered-wmdp-bio-lora",
        "eval-trajectory-unfiltered-wmdp-bio-lora",
    )
    training, evaluation = plan.stages
    assert training.args == ("configs/training/unfiltered-wmdp-bio-lora.yml",)
    assert training.sbatch_args == (
        "--gres=gpu:l40s:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=24:00:00",
        "--exclude=cn-c034,cn-l030,cn-l055",
    )
    assert evaluation.depends_on == (training.id,)
    assert evaluation.args == (
        "artifacts",
        "unfiltered-wmdp-bio-lora",
        "EleutherAI/deep-ignorance-unfiltered",
        "250",
    )
    assert evaluation.sbatch_args == (
        "--array=1-8",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=00:30:00",
        "--exclude=cn-c034,cn-l030,cn-l055",
    )


def test_weak_filter_pipeline_uses_separate_model_and_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/weak-filter-wmdp-bio-lora.yml").plan()

    assert plan.id == "weak-filter-wmdp-bio-lora-trajectory"
    training, evaluation = plan.stages
    assert training.id == "train-weak-filter-wmdp-bio-lora"
    assert training.args == ("configs/training/weak-filter-wmdp-bio-lora.yml",)
    assert evaluation.depends_on == (training.id,)
    assert evaluation.args == (
        "artifacts",
        "weak-filter-wmdp-bio-lora",
        "EleutherAI/deep-ignorance-e2e-weak-filter",
        "250",
    )


def test_unfiltered_cb_pipeline_uses_separate_model_and_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/unfiltered-cb-wmdp-bio-lora.yml").plan()

    assert plan.id == "unfiltered-cb-wmdp-bio-lora-trajectory"
    training, evaluation = plan.stages
    assert training.id == "train-unfiltered-cb-wmdp-bio-lora"
    assert training.args == ("configs/training/unfiltered-cb-wmdp-bio-lora.yml",)
    assert evaluation.depends_on == (training.id,)
    assert evaluation.args == (
        "artifacts",
        "unfiltered-cb-wmdp-bio-lora",
        "EleutherAI/deep-ignorance-unfiltered-cb",
        "250",
    )


def test_trajectory_evaluation_selects_one_array_milestone() -> None:
    script = (EXAMPLE / "scripts/slurm/eval_wmdp_bio_mcqa.sbatch").read_text()

    assert 'export HF_HOME="$SCRATCH/.cache/huggingface"' in script
    assert 'export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"' in script
    assert 'export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-eval"' in script
    assert "uv sync --frozen --group eval" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert (
        'adapter_name_or_path_ckpt="$artifacts_dir/models/'
        '$adapter_name_or_path/checkpoint-$step"'
        in script
    )
    assert 'output="$artifacts_dir/evals/$adapter_name_or_path/checkpoint-$step"' in script
    assert "step=$((${SLURM_ARRAY_TASK_ID" in script
    assert "* checkpoint_frequency))" in script
    assert "adapter_config.json" not in script
    assert "peft=$adapter_name_or_path_ckpt" in script
    assert "--batch_size 32" in script
    assert "merge-lora" not in script
    assert ".venv-train" not in script
    assert "plot_trajectory.py" not in script


def test_training_script_is_minimal() -> None:
    script = (EXAMPLE / "scripts/slurm/train.sbatch").read_text()

    assert 'export HF_HOME="$SCRATCH/.cache/huggingface"' in script
    assert 'export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"' in script
    assert 'export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-train"' in script
    assert "uv sync --frozen --group train" in script
    assert 'source "$UV_PROJECT_ENVIRONMENT/bin/activate"' in script
    assert 'axolotl train "$1" --launcher python' in script
    assert "prepare_forget_corpus.py" not in script


def test_training_configuration_matches_trajectory_protocol() -> None:
    config = yaml.safe_load(
        (EXAMPLE / "configs/training/unfiltered-wmdp-bio-lora.yml").read_text()
    )
    assert config["base_model"] == "EleutherAI/deep-ignorance-unfiltered"
    assert config["datasets"] == [
        {
            "path": "cais/wmdp-bio-forget-corpus",
            "split": "train",
            "type": "scripts.data.wmdp_bio",
        }
    ]
    assert config["max_steps"] == 2_000
    assert config["micro_batch_size"] * config["gradient_accumulation_steps"] == 16
    assert config["micro_batch_size"] == 8
    assert config["gradient_accumulation_steps"] == 2
    assert config["sequence_len"] == 2_048
    assert config["learning_rate"] == 2e-5
    assert config["weight_decay"] == 0.01
    assert config["seed"] == 42
    assert config["lora_r"] == config["lora_alpha"] == 16
    assert config["lora_target_modules"] == ["query_key_value"]
    assert config["lora_mlp_kernel"] is False
    assert config["lora_qkv_kernel"] is False
    assert config["lora_o_kernel"] is False
    assert config["lora_embedding_kernel"] is False
    assert config["val_set_size"] == 0.0
    assert config["dataset_num_proc"] == 1
    assert "skip_prepare_dataset" not in config
    assert config["save_steps"] == 250
    assert config["save_total_limit"] == 8
    assert config["save_only_model"] is True
    assert config["wandb_project"] == "lp-tamper-resistance"
    assert config["wandb_name"] == "unfiltered-wmdp-bio"
    assert "chat_template" not in config


def test_corpus_strategy_limits_each_document_to_one_sequence(strategy_module) -> None:
    train_python = EXAMPLE / ".venv-train/bin/python"
    if not train_python.exists():
        pytest.skip("Axolotl training environment is not installed")

    result = subprocess.run(
        [
            str(train_python),
            "-c",
            "from types import SimpleNamespace; "
            "from scripts.data.wmdp_bio import load; "
            "strategy = load(object(), SimpleNamespace(train_on_inputs=True, sequence_len=2048)); "
            "print(strategy.sequence_len, strategy.max_length)",
        ],
        cwd=EXAMPLE,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "2048 2048"


def test_vendored_robust_task_group_and_template() -> None:
    task_dir = EXAMPLE / "lm_eval_tasks/wmdp_bio_categorized_mcqa"
    group = yaml.safe_load((task_dir / "_wmdp_bio_robust.yaml").read_text())
    template = yaml.safe_load((task_dir / "_default_template_yaml").read_text())

    assert len(group["task"]) == 6
    assert len(set(group["task"])) == 6
    assert group["aggregate_metric_list"] == [{"metric": "acc", "weight_by_size": True}]
    assert template["dataset_path"] == "EleutherAI/wmdp_bio_robust_mcqa"
    assert template["num_fewshot"] == 0
    assert template["output_type"] == "multiple_choice"
    assert template["metric_list"] == [
        {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
    ]
    for task_name in group["task"]:
        task = yaml.safe_load((task_dir / f"{task_name}.yaml").read_text())
        assert task["task"] == task_name
        assert task["test_split"] == "robust"
        assert task["include"] == "_default_template_yaml"


def write_result(root: Path, name: str, accuracy: float) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "results_fixture.json").write_text(
        json.dumps({"groups": {"wmdp_bio_robust": {"acc,none": accuracy}}})
    )


def test_analysis_sorts_checkpoints_numerically(
    tmp_path: Path, analysis_module
) -> None:
    for step in (0, 10000, 2000, 1000):
        write_result(tmp_path, f"checkpoint-{step}", step / 100_000)

    assert analysis_module.collect_results(tmp_path, [0, 1000, 2000, 10000]) == [
        (0, 0.0),
        (1000, 0.01),
        (2000, 0.02),
        (10000, 0.1),
    ]


def test_analysis_rejects_missing_duplicate_and_absent_metric(
    tmp_path: Path, analysis_module
) -> None:
    write_result(tmp_path, "checkpoint-0", 0.25)
    with pytest.raises(ValueError, match=r"missing=\[1000\]"):
        analysis_module.collect_results(tmp_path, [0, 1000])

    write_result(tmp_path, "checkpoint-00", 0.25)
    with pytest.raises(ValueError, match="duplicate checkpoint 0"):
        analysis_module.collect_results(tmp_path, [0])

    (tmp_path / "checkpoint-00" / "results_fixture.json").unlink()
    (tmp_path / "checkpoint-00").rmdir()
    result = tmp_path / "checkpoint-0" / "results_fixture.json"
    result.write_text(json.dumps({"groups": {"wmdp_bio_robust": {}}}))
    with pytest.raises(ValueError, match="exactly once"):
        analysis_module.collect_results(tmp_path, [0])


def test_cb_schedule_endpoints_and_midpoint(cb_training_module) -> None:
    schedule = cb_training_module.coefficient_schedule
    assert schedule(0, 150, 10) == (0, 10)
    assert schedule(75, 150, 10) == (5, 5)
    assert schedule(150, 150, 10) == (10, 0)


def test_cb_losses_mask_padding_and_zero_expected_cases(cb_training_module) -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([[[[1.0, 0.0], [100.0, 100.0]]]])
    safe = reference.clone()
    harmful = torch.tensor([[[[0.0, 1.0], [1000.0, 1000.0]]]])
    mask = torch.tensor([[1, 0]])
    assert cb_training_module.retention_loss(safe, reference, mask).item() == 0
    assert cb_training_module.rerouting_loss(harmful, reference, mask).item() == 0


def test_cb_dataset_filter_rejects_incomplete_rows(cb_training_module) -> None:
    complete = {"prompt": "P", "chosen": "C", "rejected": "R"}
    assert cb_training_module.is_complete_row(complete)
    for field in ("prompt", "chosen", "rejected"):
        missing = dict(complete)
        del missing[field]
        assert not cb_training_module.is_complete_row(missing)
        blank = dict(complete)
        blank[field] = "  "
        assert not cb_training_module.is_complete_row(blank)


def test_cb_config() -> None:
    config = yaml.safe_load(
        (EXAMPLE / "configs/training/unfiltered-cb--repr.yml").read_text()
    )
    assert config["base_model"] == "EleutherAI/deep-ignorance-unfiltered"
    assert config["trainer_cls"] == "configs.training.utils.CircuitBreakerTrainer"
    assert config["datasets"] == [
        {
            "path": "LLM-LAT/harmful-dataset",
            "split": "train",
            "type": "configs.training.utils",
        }
    ]
    assert config["peft_layers_to_transform"] == list(range(31))
    assert config["lora_r"] == config["lora_alpha"] == 16
    assert config["max_steps"] == 150
    assert config["save_steps"] == 50


def test_cb_repr_pipeline_plans_a100l(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/unfiltered-cb--repr.yml").plan()
    (training,) = plan.stages
    assert training.id == "train-unfiltered-cb--repr"
    assert training.args == (
        "configs/training/unfiltered-cb--repr.yml",
        "--merge",
    )
    assert "--gres=gpu:a100l:1" in training.sbatch_args


def test_cb_attack_repr_pipeline_plans_a100l(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load(
        "configs/experiments/unfiltered-cb-wmdp-bio-lora--repr.yml"
    ).plan()
    training, evaluation = plan.stages
    assert "--gres=gpu:a100l:1" in training.sbatch_args
    assert training.args == ("configs/training/unfiltered-cb-wmdp-bio-lora--repr.yml",)
    assert evaluation.depends_on == (training.id,)
    assert evaluation.args == (
        "artifacts",
        "unfiltered-cb-wmdp-bio-lora--repr",
        "artifacts/models/unfiltered-cb--repr/merged",
        "250",
    )
    assert "--gres=gpu:a100l:1" in evaluation.sbatch_args
