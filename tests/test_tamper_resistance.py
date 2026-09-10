import importlib.util
import json
from pathlib import Path

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


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "".join(chr(token) for token in tokens)


@pytest.fixture(scope="module")
def prepare_module():
    return load_script(
        "scripts/data/prepare_forget_corpus.py", "prepare_forget_corpus"
    )


@pytest.fixture(scope="module")
def analysis_module():
    return load_script("scripts/analysis/plot_trajectory.py", "plot_trajectory")


def test_corpus_format_chunk_boundaries_and_five_chunk_limit(prepare_module) -> None:
    document = {"title": "T", "abstract": "A", "text": "B", "doi": "ignored"}
    assert prepare_module.format_document(document) == "T\n\nA\n\nB"
    assert prepare_module.chunk_token_ids(range(13), chunk_size=3, max_chunks=5) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12],
    ]
    assert prepare_module.chunk_token_ids(range(100), chunk_size=3, max_chunks=5)[-1] == [
        12,
        13,
        14,
    ]


def test_corpus_preparation_is_deterministic(prepare_module) -> None:
    documents = [
        {"title": str(index), "abstract": "abstract", "text": "text"}
        for index in range(12)
    ]
    first = prepare_module.prepare_documents(
        documents, FakeTokenizer(), seed=42, chunk_size=100
    )
    second = prepare_module.prepare_documents(
        documents, FakeTokenizer(), seed=42, chunk_size=100
    )
    different = prepare_module.prepare_documents(
        documents, FakeTokenizer(), seed=7, chunk_size=100
    )

    assert first == second
    assert first != different


def test_corpus_loader_uses_cached_hugging_face_authentication() -> None:
    script = (EXAMPLE / "scripts/data/prepare_forget_corpus.py").read_text()

    assert "token=True" in script
    assert 'os.environ["HF_TOKEN"]' not in script


def test_corpus_schema_accepts_expected_doi_metadata(prepare_module) -> None:
    assert prepare_module.REQUIRED_COLUMNS == {"title", "abstract", "text"}
    assert prepare_module.EXPECTED_COLUMNS == {
        "title",
        "abstract",
        "text",
        "doi",
    }


@pytest.mark.parametrize("missing", ["title", "abstract", "text"])
def test_corpus_format_rejects_missing_fields(prepare_module, missing: str) -> None:
    document = {"title": "T", "abstract": "A", "text": "B"}
    del document[missing]
    with pytest.raises(ValueError, match="missing required fields"):
        prepare_module.format_document(document)


def test_tamper_resistance_pipeline_has_training_then_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load(
        "configs/experiments/unfiltered-wmdp-bio-lora.yml"
    )
    plan = pipeline.plan()

    assert plan.id == "unfiltered-wmdp-bio-lora-trajectory"
    assert tuple(stage.id for stage in plan.stages) == (
        "train-unfiltered-wmdp-bio-lora",
        "eval-trajectory-unfiltered-wmdp-bio-lora",
    )
    training, evaluation = plan.stages
    assert training.args == ("configs/training/unfiltered-wmdp-bio-lora.yml",)
    assert training.sbatch_args == (
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=24:00:00",
    )
    assert evaluation.depends_on == (training.id,)
    assert evaluation.args == (
        "EleutherAI/deep-ignorance-unfiltered",
        "artifacts/models/unfiltered-wmdp-bio-lora",
        "configs/training/unfiltered-wmdp-bio-lora.yml",
        "configs/results/unfiltered-wmdp-bio-lora.yml",
        "artifacts/evals/unfiltered-wmdp-bio-lora",
    )
    assert evaluation.sbatch_args == ("--array=0-6", *training.sbatch_args)


def test_trajectory_evaluation_selects_one_array_milestone() -> None:
    script = (EXAMPLE / "scripts/slurm/eval_trajectory.sbatch").read_text()

    assert "SLURM_ARRAY_TASK_ID" in script
    assert "step=${milestones[$task_id]}" in script
    assert 'for step in "${milestones[@]}"' not in script
    assert "plot_trajectory.py" not in script


def test_training_reuses_completed_prepared_corpus() -> None:
    script = (EXAMPLE / "scripts/slurm/train.sbatch").read_text()

    assert '$prepared/state.json' in script
    assert '$prepared/dataset_info.json' in script
    assert "Reusing prepared corpus" in script


def test_training_configuration_matches_trajectory_protocol() -> None:
    config = yaml.safe_load(
        (EXAMPLE / "configs/training/unfiltered-wmdp-bio-lora.yml").read_text()
    )
    assert config["base_model"] == "EleutherAI/deep-ignorance-unfiltered"
    assert config["max_steps"] == 10_000
    assert config["micro_batch_size"] * config["gradient_accumulation_steps"] == 16
    assert config["sequence_len"] == 2_048
    assert config["learning_rate"] == 2e-5
    assert config["weight_decay"] == 0.01
    assert config["seed"] == 42
    assert config["lora_r"] == config["lora_alpha"] == 16
    assert config["lora_target_modules"] == ["query_key_value"]
    assert config["val_set_size"] == 0.0
    assert config["dataset_num_proc"] == 1
    assert config["save_steps"] == 1_000
    assert config["save_total_limit"] == 10
    assert config["save_only_model"] is True
    assert config["wandb_project"] == "lp-tamper-resistance"
    assert config["wandb_name"] == (
        "unfiltered-wmdp-bio-lora-r16-lr2e-5-bs16-seq2048-seed42"
    )
    assert "chat_template" not in config


def test_vendored_robust_task_group_and_template() -> None:
    task_dir = EXAMPLE / "lm_eval_tasks/wmdp_bio_categorized_mcqa"
    group = yaml.safe_load((task_dir / "_wmdp_bio_robust.yaml").read_text())
    template = yaml.safe_load((task_dir / "_default_template_yaml").read_text())

    assert len(group["task"]) == 6
    assert len(set(group["task"])) == 6
    assert group["aggregate_metric_list"] == [
        {"metric": "acc", "weight_by_size": True}
    ]
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


def test_analysis_sorts_checkpoints_numerically(tmp_path: Path, analysis_module) -> None:
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
