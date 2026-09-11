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
        "EleutherAI/deep-ignorance-unfiltered",
        "artifacts/models/unfiltered-wmdp-bio-lora",
        "configs/training/unfiltered-wmdp-bio-lora.yml",
        "configs/results/unfiltered-wmdp-bio-lora.yml",
        "artifacts/evals/unfiltered-wmdp-bio-lora",
    )
    assert evaluation.sbatch_args == (
        "--array=0-8",
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
        "EleutherAI/deep-ignorance-e2e-weak-filter",
        "artifacts/models/weak-filter-wmdp-bio-lora",
        "configs/training/weak-filter-wmdp-bio-lora.yml",
        "configs/results/weak-filter-wmdp-bio-lora.yml",
        "artifacts/evals/weak-filter-wmdp-bio-lora",
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
        "EleutherAI/deep-ignorance-unfiltered-cb",
        "artifacts/models/unfiltered-cb-wmdp-bio-lora",
        "configs/training/unfiltered-cb-wmdp-bio-lora.yml",
        "configs/results/unfiltered-cb-wmdp-bio-lora.yml",
        "artifacts/evals/unfiltered-cb-wmdp-bio-lora",
    )


def test_trajectory_evaluation_selects_one_array_milestone() -> None:
    script = (EXAMPLE / "scripts/slurm/eval_trajectory.sbatch").read_text()

    assert 'export HF_HOME="$SCRATCH/.cache/huggingface"' in script
    assert 'export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"' in script
    assert 'export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-eval"' in script
    assert "uv sync --frozen --group eval" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "step=$((task_id * 250))" in script
    assert "peft=$checkpoint" in script
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
