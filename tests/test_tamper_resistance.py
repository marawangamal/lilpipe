import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
def analysis_module():
    return load_script("scripts/analysis/plot_trajectory.py", "plot_trajectory")


@pytest.fixture(scope="module")
def orth_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return load_script("configs/training/utils.py", "orth_circuit_breaker_training")


def test_trajectory_evaluation_selects_one_array_milestone() -> None:
    script = (EXAMPLE / "scripts/slurm/eval_wmdp_bio_mcqa.sbatch").read_text()

    assert 'export HF_HOME="$SCRATCH/.cache/huggingface"' in script
    assert 'export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"' in script
    assert 'export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-eval"' in script
    assert "uv sync --frozen --group eval" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert (
        'adapter_name_or_path_ckpt="$artifacts_dir/models/'
        '$adapter_name_or_path/checkpoint-$step"' in script
    )
    assert (
        'output="$artifacts_dir/evals/$adapter_name_or_path/checkpoint-$step"' in script
    )
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


def test_orth_cb_document_strategies_tag_identical_schemas(
    orth_training_module,
) -> None:
    calls = []

    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    cfg = SimpleNamespace(sequence_len=2048)
    strategy = orth_training_module.load(
        tokenizer,
        cfg,
        SimpleNamespace(path="cais/wmdp-bio-forget-corpus"),
    )
    forget = strategy.tokenize_row(
        {"title": "title", "abstract": "abstract", "text": "bio"}
    )
    retain = orth_training_module.load(
        tokenizer,
        cfg,
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    ).tokenize_row({"page": "wiki"})

    assert (
        set(forget)
        == set(retain)
        == {
            "input_ids",
            "attention_mask",
            "labels",
            "cb_source",
        }
    )
    assert forget["cb_source"] == 1
    assert retain["cb_source"] == 0
    assert calls[0][0] == "title\n\nabstract\n\nbio"
    assert calls[0][1] == {
        "max_length": 2048,
        "truncation": True,
        "add_special_tokens": True,
    }


@pytest.mark.parametrize("missing", ["title", "abstract", "text"])
def test_orth_cb_wmdp_requires_document_fields(orth_training_module, missing) -> None:
    strategy = orth_training_module.load(
        lambda text, **kwargs: {"input_ids": [1], "attention_mask": [1]},
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="cais/wmdp-bio-forget-corpus"),
    )
    row = {"title": "T", "abstract": "A", "text": "B"}
    del row[missing]
    with pytest.raises(ValueError, match="title, abstract, and text"):
        strategy.tokenize_row(row)


def test_orth_cb_tokenization_truncates_and_masks(orth_training_module) -> None:
    def tokenizer(text, *, max_length, truncation, add_special_tokens):
        assert truncation and add_special_tokens
        ids = list(range(len(text.split())))[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    strategy = orth_training_module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    )
    row = strategy.tokenize_row({"page": "word " * 2100})
    assert len(row["input_ids"]) == 2048
    assert row["attention_mask"] == [1] * 2048
    assert row["labels"] == row["input_ids"]


def test_orth_cb_wikitext_shuffles_before_selecting(orth_training_module) -> None:
    class FakeDataset(list):
        column_names = ["page"]

        def shuffle(self, seed):
            assert seed == 42
            return FakeDataset(reversed(self))

        def select(self, indices):
            assert len(indices) == 1024
            return FakeDataset(self[index] for index in indices)

        def map(self, function, remove_columns):
            assert remove_columns == self.column_names
            return FakeDataset(function(row) for row in self)

    tokenizer = lambda text, **kwargs: {
        "input_ids": [int(text)],
        "attention_mask": [1],
    }
    strategy = orth_training_module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    )
    wrapped = strategy.wrap_dataset(
        FakeDataset({"page": str(index)} for index in range(1100))
    )

    assert len(wrapped) == 1024
    assert wrapped[0]["input_ids"] == [1099]
    assert wrapped[0]["cb_source"] == 0


def test_axolotl_standard_collator_preserves_source_tags() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("axolotl")
    from axolotl.utils.collators import DataCollatorForSeq2Seq
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    batch = DataCollatorForSeq2Seq(tokenizer)(
        [
            {
                "input_ids": [2, 3, 4],
                "attention_mask": [1, 1, 1],
                "labels": [2, 3, 4],
                "cb_source": 0,
            },
            {
                "input_ids": [5, 6],
                "attention_mask": [1, 1],
                "labels": [5, 6],
                "cb_source": 1,
            },
        ]
    )
    assert isinstance(batch["cb_source"], torch.Tensor)
    assert batch["cb_source"].tolist() == [0, 1]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_orth_cb_layer_mapping_and_schedule(orth_training_module) -> None:
    torch = pytest.importorskip("torch")

    trainer = object.__new__(orth_training_module.OrthCircuitBreakerTrainer)
    trainer.target_layers = (5, 10, 15, 20, 25, 30)

    class Model:
        training = True

        def train(self, mode=True):
            self.training = mode

        def eval(self):
            self.training = False

        def __call__(self, **kwargs):
            return SimpleNamespace(
                hidden_states=tuple(torch.full((1, 1, 1), i) for i in range(32))
            )

    selected = trainer.selected_activations(
        Model(), torch.ones((1, 1)), torch.ones((1, 1))
    )
    assert selected[:, 0, 0, 0].tolist() == [6, 11, 16, 21, 26, 31]
    assert orth_training_module.coefficients(0, 512) == (1.0, 23.0, 0.0)
    midpoint = orth_training_module.coefficients(256, 512)
    assert midpoint == pytest.approx(
        (1.0 + 9.0 * 256 / 511, 23 - 5.75 * 256 / 511, 5 * 256 / 511)
    )
    assert orth_training_module.coefficients(511, 512) == (
        10.0,
        17.25,
        5.0,
    )
    assert orth_training_module.coefficients(0, 128) == (1.0, 23.0, 0.0)
    assert orth_training_module.coefficients(64, 128) == pytest.approx(
        (1.0 + 9.0 * 64 / 127, 23 - 5.75 * 64 / 127, 5 * 64 / 127)
    )
    assert orth_training_module.coefficients(127, 128) == (10.0, 17.25, 5.0)
    assert orth_training_module.coefficients(0, 256) == (1.0, 23.0, 0.0)
    assert orth_training_module.coefficients(255, 256) == (10.0, 17.25, 5.0)


@pytest.mark.parametrize("sources", [[0, 1, 0, 1], [0, 0], [1, 1], [0, 1, 1], [0, 1]])
def test_orth_cb_routes_tagged_rows(orth_training_module, sources) -> None:
    torch = pytest.importorskip("torch")
    from types import MethodType

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    model = Model()
    trainer = object.__new__(orth_training_module.OrthCircuitBreakerTrainer)
    trainer.target_layers = (0,)
    trainer.microstep = 0
    trainer.total_microsteps = 512
    trainer.logged = []
    trainer.log = trainer.logged.append
    observed = []

    def selected(self, model, input_ids, attention_mask, **kwargs):
        del self
        assert input_ids.shape == attention_mask.shape
        observed.append((input_ids[:, 0].tolist(), kwargs))
        values = input_ids.float()[None, :, :, None]
        return values if kwargs.get("disable_grad") else values * model.weight

    trainer.selected_activations = MethodType(selected, trainer)
    batch_size = len(sources)
    inputs = {
        "input_ids": torch.arange(batch_size)[:, None].repeat(1, 2),
        "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
        "labels": torch.full((batch_size, 2), -100),
        "cb_source": torch.tensor(sources),
    }
    loss = trainer.compute_loss(model, inputs)
    loss.backward()

    metrics = trainer.logged[-1]
    retain_count = sources.count(0)
    forget_count = sources.count(1)
    assert metrics["retain_count"] == retain_count
    assert metrics["forget_count"] == forget_count
    assert loss.detach().item() == pytest.approx(
        metrics["retain_coefficient"] * metrics["loss_retain"]
        + metrics["reroute_coefficient"] * metrics["loss_reroute"]
        + metrics["orthogonalization_coefficient"] * metrics["loss_orthogonalization"]
    )
    assert metrics["retain_skipped"] is (retain_count == 0)
    assert metrics["reroute_skipped"] is (forget_count == 0)
    assert metrics["orthogonalization_skipped"] is (forget_count < 2)
    if retain_count == 0:
        assert metrics["loss_retain"] == 0
    if forget_count == 0:
        assert metrics["loss_reroute"] == 0
    if forget_count < 2:
        assert metrics["loss_orthogonalization"] == 0
    assert len(observed) == 2 * int(retain_count > 0) + 2 * int(forget_count > 0)
    assert model.weight.grad is not None


def test_balanced_source_sampler_uses_each_row_once_per_epoch(
    orth_training_module,
) -> None:
    class TaggedDataset:
        def __init__(self):
            self.sources = [0, 1, 1, 0] * 4

        def __getitem__(self, key):
            assert key == "cb_source"
            return self.sources

    dataset = TaggedDataset()
    trainer = SimpleNamespace(
        train_dataset=dataset,
        args=SimpleNamespace(per_device_train_batch_size=8, data_seed=None, seed=42),
    )
    sampler = orth_training_module.BalancedOrthCircuitBreakerTrainer._get_train_sampler(
        trainer
    )
    first, second = list(sampler), list(sampler)

    for epoch in (first, second):
        assert len(epoch) == len(sampler) == len(dataset.sources)
        assert sorted(epoch) == list(range(len(dataset.sources)))
        for start in range(0, len(epoch), 8):
            assert [dataset.sources[index] for index in epoch[start : start + 8]].count(
                0
            ) == 4
    assert first != second
    assert first == list(orth_training_module.BalancedSourceSampler(dataset, 8, 42))


def test_balanced_source_sampler_rejects_unbalanced_sources(
    orth_training_module,
) -> None:
    class TaggedDataset:
        def __init__(self, sources):
            self.sources = sources

        def __getitem__(self, key):
            assert key == "cb_source"
            return self.sources

    sampler = orth_training_module.BalancedSourceSampler
    with pytest.raises(ValueError, match="positive even batch size"):
        sampler(TaggedDataset([0, 1]), 3, 42)
    with pytest.raises(ValueError, match="both source tags"):
        sampler(TaggedDataset([0, 0]), 2, 42)
    with pytest.raises(ValueError, match="equal and divisible"):
        sampler(TaggedDataset([0, 0, 1]), 2, 42)


def test_orth_cb_losses_relu_mask_and_off_diagonal(orth_training_module) -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    current = torch.tensor([[[[-1.0, 0.0], [1.0, 0.0]]]])
    mask = torch.tensor([[1, 0]])
    cosine = torch.nn.functional.cosine_similarity(current, reference, dim=-1)
    assert orth_training_module.masked_mean(torch.relu(cosine), mask).item() == 0
    retain = orth_training_module.masked_mean((current - reference).norm(dim=-1), mask)
    assert retain.item() == 2

    activations = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], requires_grad=True)
    two_token_mask = torch.tensor([[2], [2]])
    loss = orth_training_module.orthogonalization_loss(activations, two_token_mask)
    assert loss.item() == pytest.approx(2.0)
    single = orth_training_module.orthogonalization_loss(
        activations[:, :1], two_token_mask[:1]
    )
    assert single.item() == 0
    single.backward()
    assert activations.grad is not None


def test_orth_cb_config() -> None:
    model_id = "di-6.9b-cb--orth-ret10-rm23-orth5-r8"
    config = yaml.safe_load(
        (EXAMPLE / "configs/training/di-6.9b/circuit-breaker-orth.yml").read_text()
    )
    assert config["trainer_cls"] == ("configs.training.utils.OrthCircuitBreakerTrainer")
    assert config["output_dir"] == f"artifacts/models/{model_id}"
    assert config["wandb_name"] == model_id
    assert config["lora_target_modules"] == [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    assert config["peft_layers_to_transform"] == list(range(31))
    assert config["lora_r"] == config["lora_alpha"] == 8
    assert config["micro_batch_size"] == 4
    assert config["gradient_accumulation_steps"] == 16
    assert config["max_steps"] == 32
    assert config["shuffle_merged_datasets"] is True
    assert [dataset["path"] for dataset in config["datasets"]] == [
        "cais/wmdp-bio-forget-corpus",
        "EleutherAI/wikitext_document_level",
    ]
    assert config["datasets"][0]["split"] == "train[:1024]"
    assert config["datasets"][1]["split"] == "train"
    assert all(
        dataset["type"] == "configs.training.utils" for dataset in config["datasets"]
    )
    assert config["learning_rate"] == 1e-3
    assert config["weight_decay"] == 0.01
    assert config["lr_scheduler"] == "linear"
    assert config["warmup_steps"] == 0
    assert config["max_grad_norm"] == 1.0
    assert config["sequence_len"] == 2048
    assert config["save_steps"] == 5
    assert config["save_total_limit"] >= 7
    assert "merge" not in config


@pytest.mark.parametrize("micro_batch,accumulation", [(8, 8), (16, 4)])
def test_orth_cb_batch_variants_are_isolated_and_keep_effective_batch(
    micro_batch: int, accumulation: int
) -> None:
    training_dir = EXAMPLE / "configs/training/di-6.9b"
    original = yaml.safe_load((training_dir / "circuit-breaker-orth.yml").read_text())
    variant = yaml.safe_load(
        (training_dir / f"circuit-breaker-orth-mb{micro_batch}.yml").read_text()
    )
    assert variant["micro_batch_size"] == micro_batch
    assert variant["gradient_accumulation_steps"] == accumulation
    assert variant["max_steps"] == original["max_steps"] == 32
    assert variant["micro_batch_size"] * variant["gradient_accumulation_steps"] == (
        original["micro_batch_size"] * original["gradient_accumulation_steps"]
    )
    assert 2048 // micro_batch == 32 * accumulation
    isolated = {
        "micro_batch_size",
        "gradient_accumulation_steps",
        "dataset_prepared_path",
        "output_dir",
        "wandb_name",
    }
    assert {key: value for key, value in variant.items() if key not in isolated} == {
        key: value for key, value in original.items() if key not in isolated
    }
    for key in ("dataset_prepared_path", "output_dir", "wandb_name"):
        assert variant[key] != original[key]
        assert f"-mb{micro_batch}" in variant[key]


def test_balanced_config_only_changes_sampler_and_artifact_paths() -> None:
    training_dir = EXAMPLE / "configs/training/di-6.9b"
    original = yaml.safe_load(
        (training_dir / "circuit-breaker-orth-mb8.yml").read_text()
    )
    balanced = yaml.safe_load(
        (training_dir / "circuit-breaker-orth-mb8-balanced.yml").read_text()
    )
    changed = {"trainer_cls", "dataset_prepared_path", "output_dir", "wandb_name"}
    assert {key: value for key, value in balanced.items() if key not in changed} == {
        key: value for key, value in original.items() if key not in changed
    }
    assert balanced["trainer_cls"] == (
        "configs.training.utils.BalancedOrthCircuitBreakerTrainer"
    )
    for key in changed - {"trainer_cls"}:
        assert balanced[key] != original[key]
        assert "-mb8-balanced" in balanced[key]


def test_single_experiment_plans_base_and_orth_cb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/di-6.9b.yml").plan()
    orth_id = "di-6.9b-cb--orth-ret10-rm23-orth5-r8"
    orth = plan.stage_index[f"train-{orth_id}"]
    assert orth.script == "scripts/slurm/train.sbatch"
    assert orth.args == ("configs/training/di-6.9b/circuit-breaker-orth.yml",)
    assert f"eval-bio-mcqa-{orth_id}" in plan.stage_index
    assert f"eval-mmlu-no-bio-{orth_id}" in plan.stage_index
    batch_16_id = f"{orth_id}-mb16"
    batch_16 = plan.stage_index[f"train-{batch_16_id}"]
    assert batch_16.script == "scripts/slurm/train.sbatch"
    assert batch_16.args == ("configs/training/di-6.9b/circuit-breaker-orth-mb16.yml",)
    assert f"eval-bio-mcqa-{batch_16_id}" in plan.stage_index
    assert f"eval-mmlu-no-bio-{batch_16_id}" in plan.stage_index
    batch_8_id = f"{orth_id}-mb8"
    batch_8 = plan.stage_index[f"train-{batch_8_id}"]
    assert batch_8.script == "scripts/slurm/train.sbatch"
    assert batch_8.args == ("configs/training/di-6.9b/circuit-breaker-orth-mb8.yml",)
    assert f"eval-bio-mcqa-{batch_8_id}" in plan.stage_index
    assert f"eval-mmlu-no-bio-{batch_8_id}" in plan.stage_index
    balanced_id = f"{batch_8_id}-balanced"
    balanced = plan.stage_index[f"train-{balanced_id}"]
    assert balanced.args == (
        "configs/training/di-6.9b/circuit-breaker-orth-mb8-balanced.yml",
    )
    assert f"eval-bio-mcqa-{balanced_id}" in plan.stage_index
    assert f"eval-mmlu-no-bio-{balanced_id}" in plan.stage_index
    assert len(plan.stages) == 14
    assert plan.stage_index["eval-bio-mcqa-di-6.9b-base"].args == (
        "di-6.9b-base",
        "EleutherAI/deep-ignorance-unfiltered",
        "-",
    )


def test_mmlu_no_bio_group_excludes_biology_overlap() -> None:
    config = yaml.safe_load((EXAMPLE / "lm_eval_tasks/mmlu_no_bio.yaml").read_text())
    excluded = {
        "mmlu_virology",
        "mmlu_medical_genetics",
        "mmlu_high_school_biology",
        "mmlu_college_biology",
    }
    assert config["group"] == "mmlu_no_bio"
    assert len(config["task"]) == 53
    assert not excluded.intersection(config["task"])
    assert config["aggregate_metric_list"] == [
        {"metric": "acc", "weight_by_size": True}
    ]


def test_mmlu_no_bio_evaluator_is_zero_shot() -> None:
    script = (EXAMPLE / "scripts/slurm/eval_mmlu_no_bio.sbatch").read_text()
    assert (
        'export HF_DATASETS_CACHE="$SLURM_TMPDIR/.cache/huggingface/datasets"' in script
    )
    assert "--tasks mmlu_no_bio" in script
    assert "--num_fewshot 0" in script
    assert "--batch_size 32" in script
    assert 'output="artifacts/evals/$model_id/mmlu-no-bio"' in script


def test_final_adapter_evaluator_uses_direct_adapter_without_array() -> None:
    script = (EXAMPLE / "scripts/slurm/eval_wmdp_bio_mcqa_single.sbatch").read_text()
    assert "adapter_name_or_path=${3:?missing adapter name or path}" in script
    assert 'if [[ "$adapter_name_or_path" != "-" ]]' in script
    assert 'output="artifacts/evals/$model_id/wmdp-bio-robust"' in script
    assert "checkpoint-" not in script
    assert "SLURM_ARRAY_TASK_ID" not in script
    assert "--batch_size 32" in script
    assert "--num_fewshot 0" in script


def test_canonical_model_ids_paths_dependencies_and_config_basenames() -> None:
    registry = yaml.safe_load((EXAMPLE / "configs/registries/models.yml").read_text())[
        "models"
    ]

    assert set(registry) == {
        "di-6.9b-base",
        "di-6.9b-cb--orth-ret10-rm23-orth5-r8",
        "di-6.9b-cb--orth-ret10-rm23-orth5-r8-mb16",
        "di-6.9b-cb--orth-ret10-rm23-orth5-r8-mb8",
        "di-6.9b-cb--orth-ret10-rm23-orth5-r8-mb8-balanced",
    }
    assert all(model_id.startswith("di-6.9b-") for model_id in registry)
    for model_id, model in registry.items():
        model_path = model.get("adapter_name_or_path", model.get("local_dir"))
        assert model_path == "-" or model_id in model_path

        producer = model.get("producer")
        if producer is None:
            continue
        assert producer["id"].endswith(model_id)
        config_path = EXAMPLE / producer["args"][0]
        assert config_path.is_file()
        assert all(
            dependency in registry for dependency in producer.get("depends_on", ())
        )

    for experiment_path in (EXAMPLE / "configs/experiments").glob("*.yml"):
        experiment = yaml.safe_load(experiment_path.read_text())
        assert experiment_path.stem == "di-6.9b"
        assert all(model_id in registry for model_id in experiment["models"])


def test_results_config_has_base_and_orth_cb_groups() -> None:
    config = yaml.safe_load((EXAMPLE / "configs/results/di-6.9b.yml").read_text())
    assert [row["group"] for row in config["rows"]] == [
        "base-model",
        "circuit-breaker",
        "circuit-breaker",
        "circuit-breaker",
        "circuit-breaker",
    ]

    for training_path in (EXAMPLE / "configs/training").rglob("*.yml"):
        training = yaml.safe_load(training_path.read_text())
        assert training["output_dir"].startswith("artifacts/models/di-6.9b-")
        assert "lora64-epochs1" not in training["output_dir"]
        assert "LoRA64-Epochs1" not in training["wandb_name"]

    assert all("LoRA64-Epochs1" not in row["label"] for row in config["rows"])
