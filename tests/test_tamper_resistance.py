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
def optml_comparison_module():
    return load_script(
        "scripts/analysis/compare_optml_relearning.py", "compare_optml_relearning"
    )


@pytest.fixture(scope="module")
def orth_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script(
            "configs/training/trainers/circuit_breaker.py",
            "orth_circuit_breaker_training",
        )
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def wmdp_di_module():
    pytest.importorskip("axolotl")
    return load_script("configs/training/data/wmdp_di.py", "wmdp_di_training")


@pytest.fixture(scope="module")
def wikitext_di_module():
    pytest.importorskip("axolotl")
    return load_script("configs/training/data/wikitext_di.py", "wikitext_di_training")


@pytest.fixture(scope="module")
def authors_documents_module():
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script(
            "configs/training/data/authors_documents.py", "authors_documents_training"
        )
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def npo_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/training/trainers/npo.py", "npo_training")
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def npo_sam_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/training/trainers/npo_sam.py", "npo_sam_training")
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def gd_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/training/trainers/gd.py", "gd_training")
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def gd_sam_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/training/trainers/gd_sam.py", "gd_sam_training")
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def weight_dist_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script(
            "configs/training/trainers/weight_dist.py", "weight_dist_training"
        )
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def gd_weight_dist_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script(
            "configs/training/trainers/gd_weight_dist.py",
            "gd_weight_dist_training",
        )
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def gd_gn_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/training/trainers/gd_gn.py", "gd_gn_training")
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def deff_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/training/trainers/deff.py", "deff_training")
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def gd_deff_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/training/trainers/gd_deff.py", "gd_deff_training")
    finally:
        sys.path.remove(str(EXAMPLE))


def test_trajectory_evaluation_selects_one_array_milestone() -> None:
    script = (
        EXAMPLE / "scripts/slurm/mila/eval_wmdp_bio_mcqa_ckpts.sbatch"
    ).read_text()

    assert 'export HF_HOME="$SCRATCH/.cache/huggingface"' in script
    assert 'export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"' in script
    assert 'export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-eval"' in script
    assert "uv sync --frozen --group eval" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert 'checkpoint="$adapter_name_or_path/checkpoint-$step"' in script
    assert 'output="$result_root/$model_id/checkpoint-$step/wmdp-bio-robust"' in script
    assert "step=$((${SLURM_ARRAY_TASK_ID" in script
    assert "* checkpoint_frequency))" in script
    assert "last_step=${6:?missing last step}" in script
    assert "(( step > last_step ))" in script
    assert "step=$last_step" in script
    assert "adapter_config.json" not in script
    assert 'if [[ "$adapter_name_or_path" == "-" ]]' in script
    assert 'model_args="pretrained=$checkpoint' in script
    assert 'model_args+=",peft=$checkpoint"' in script
    assert "--batch_size 32" in script
    assert "merge-lora" not in script
    assert ".venv-train" not in script
    assert "plot_trajectory.py" not in script


def test_training_script_is_minimal() -> None:
    script = (EXAMPLE / "scripts/slurm/mila/train.sbatch").read_text()

    assert 'export HF_HOME="$SCRATCH/.cache/huggingface"' in script
    assert "export HF_HUB_OFFLINE=1" in script
    assert "export HF_DATASETS_OFFLINE=1" in script
    assert "Offline mode avoids shared-cluster Hub rate limits" in script
    assert 'export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv-$SLURM_JOB_ID"' in script
    assert 'export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"' in script
    assert 'export PATH="$HOME/.local/bin:$PATH"' in script
    assert 'mkdir -p "$WANDB_DIR"' in script
    assert "uv sync --frozen --group train" in script
    assert 'source "$UV_PROJECT_ENVIRONMENT/bin/activate"' in script
    assert 'axolotl train "$config"' in script
    assert "prepare_forget_corpus.py" not in script


def test_tamia_training_keeps_wandb_state_on_scratch() -> None:
    script = (EXAMPLE / "scripts/slurm/tamia/train.sbatch").read_text()

    assert 'export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"' in script
    assert 'export WANDB_DATA_DIR="$SCRATCH/.cache/wandb/staging"' in script
    assert "export WANDB_MODE=offline" in script
    assert 'mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"' in script


def test_tamia_training_optionally_merges_lora() -> None:
    script = (EXAMPLE / "scripts/slurm/tamia/train.sbatch").read_text()

    assert 'if [[ ${2:-} == "--merge" ]]; then' in script
    assert 'axolotl merge-lora "$config"' in script


@pytest.mark.parametrize("cluster", ["mila", "tamia"])
def test_slurm_templates_use_hugging_face_cache_offline(cluster: str) -> None:
    script = (ROOT / f"slurm/template_{cluster}.sbatch").read_text()

    assert "export HF_HUB_OFFLINE=1" in script
    assert "export HF_DATASETS_OFFLINE=1" in script
    assert "Offline mode avoids shared-cluster Hub rate limits" in script


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


def test_optml_comparison_collects_both_codebases(
    tmp_path: Path, optml_comparison_module
) -> None:
    authors = tmp_path / "authors"
    axolotl = tmp_path / "axolotl"
    for model in optml_comparison_module.MODELS:
        for examples in optml_comparison_module.AUTHOR_EXAMPLES:
            run = authors / model / f"examples-{examples}"
            run.mkdir(parents=True)
            (run / "wmdp.json").write_text(
                json.dumps({"results": {"wmdp_bio": {"acc,none": 0.2}}})
            )
            (run / "mmlu.json").write_text(
                json.dumps({"results": {"mmlu": {"acc,none": 0.5}}})
            )
        for checkpoint in range(5):
            run = axolotl / model / f"checkpoint-{checkpoint}"
            for directory, task, value in (
                ("wmdp-bio", "wmdp_bio", 0.3),
                ("wmdp-bio-robust", "wmdp_bio_robust", 0.4),
                ("mmlu-no-bio", "mmlu_no_bio", 0.6),
            ):
                result = run / directory
                result.mkdir(parents=True)
                (result / "results_fixture.json").write_text(
                    json.dumps({"groups": {task: {"acc,none": value}}})
                )

    rows = optml_comparison_module.collect_authors(
        authors
    ) + optml_comparison_module.collect_axolotl(axolotl)

    assert len(rows) == 16
    assert rows[0].wmdp_ue == pytest.approx(0.8)
    assert rows[0].published_ue == pytest.approx(0.45)
    assert rows[0].ue_delta_from_published == pytest.approx(0.35)
    assert rows[-1].examples == 64
    assert rows[-1].robust_wmdp_ue == pytest.approx(0.6)
    assert "| authors |" in optml_comparison_module.markdown(rows)
    assert "| axolotl |" in optml_comparison_module.markdown(rows)


def test_optml_comparison_accepts_identical_group_and_result_metrics(
    tmp_path: Path, optml_comparison_module
) -> None:
    data = {
        "groups": {"mmlu": {"acc,none": 0.5}},
        "results": {"mmlu": {"acc,none": 0.5}},
    }
    assert optml_comparison_module.metric(data, "mmlu", tmp_path) == 0.5

    data["results"]["mmlu"]["acc,none"] = 0.6
    with pytest.raises(ValueError, match="conflicting"):
        optml_comparison_module.metric(data, "mmlu", tmp_path)


def test_orth_cb_document_strategies_tag_identical_schemas(
    wmdp_di_module,
    wikitext_di_module,
) -> None:
    calls = []

    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    cfg = SimpleNamespace(sequence_len=2048)
    strategy = wmdp_di_module.load(
        tokenizer,
        cfg,
        SimpleNamespace(path="cais/wmdp-bio-forget-corpus"),
    )
    forget = strategy.tokenize_row(
        {"title": "title", "abstract": "abstract", "text": "bio"}
    )
    retain = wikitext_di_module.load(
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
            "is_forget",
        }
    )
    assert forget["is_forget"] is True
    assert retain["is_forget"] is False
    assert calls[0][0] == "title\n\nabstract\n\nbio"
    assert calls[0][1] == {
        "max_length": 2048,
        "truncation": True,
        "add_special_tokens": True,
    }


@pytest.mark.parametrize("missing", ["title", "abstract", "text"])
def test_orth_cb_wmdp_requires_document_fields(wmdp_di_module, missing) -> None:
    strategy = wmdp_di_module.load(
        lambda text, **kwargs: {"input_ids": [1], "attention_mask": [1]},
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="cais/wmdp-bio-forget-corpus"),
    )
    row = {"title": "T", "abstract": "A", "text": "B"}
    del row[missing]
    with pytest.raises(ValueError, match="title, abstract, and text"):
        strategy.tokenize_row(row)


def test_orth_cb_tokenization_truncates_and_masks(wikitext_di_module) -> None:
    def tokenizer(text, *, max_length, truncation, add_special_tokens):
        assert truncation and add_special_tokens
        ids = list(range(len(text.split())))[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    strategy = wikitext_di_module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    )
    row = strategy.tokenize_row({"page": "word " * 2100})
    assert len(row["input_ids"]) == 2048
    assert row["attention_mask"] == [1] * 2048
    assert row["labels"] == row["input_ids"]


def test_orth_cb_wikitext_shuffles_before_selecting(wikitext_di_module) -> None:
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
    strategy = wikitext_di_module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    )
    wrapped = strategy.wrap_dataset(
        FakeDataset({"page": str(index)} for index in range(1100))
    )

    assert len(wrapped) == 1024
    assert wrapped[0]["input_ids"] == [1099]
    assert wrapped[0]["is_forget"] is False


def test_authors_documents_train_on_padded_tokens(authors_documents_module) -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "document"
            assert kwargs == {
                "max_length": 4,
                "padding": "max_length",
                "truncation": True,
            }
            return {
                "input_ids": [7, 8, 2, 2],
                "attention_mask": [1, 1, 0, 0],
            }

    strategy = authors_documents_module.AuthorsDocumentStrategy(
        Tokenizer(), 4, "cais/wmdp-bio-forget-corpus"
    )
    row = strategy.tokenize_row({"text": "document"})

    assert row["labels"] == [7, 8, 2, 2]
    assert row["attention_mask"] == [1, 1, 0, 0]
    assert row["is_forget"] is True


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
                "is_forget": False,
            },
            {
                "input_ids": [5, 6],
                "attention_mask": [1, 1],
                "labels": [5, 6],
                "is_forget": True,
            },
        ]
    )
    assert isinstance(batch["is_forget"], torch.Tensor)
    assert batch["is_forget"].tolist() == [False, True]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_plain_finetuning_drops_source_tag(tmp_path: Path, wmdp_di_module) -> None:
    torch = pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")

    strategy = wmdp_di_module.WmdpDocumentStrategy(
        lambda text, **kwargs: {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
        },
        2048,
    )
    row = strategy.tokenize_row({"title": "T", "abstract": "A", "text": "B"})
    assert row["is_forget"] is True

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, attention_mask=None, labels=None):
            return {"loss": self.weight * 0}

    trainer = transformers.Trainer(
        model=Model(),
        args=transformers.TrainingArguments(
            output_dir=str(tmp_path),
            remove_unused_columns=True,
            per_device_train_batch_size=1,
            dataloader_pin_memory=False,
        ),
        train_dataset=datasets.Dataset.from_list([row]),
    )
    batch = next(iter(trainer.get_train_dataloader()))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}


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
    assert orth_training_module.coefficients(0, 256) == (1.0, 23.0, 0.0)
    midpoint = orth_training_module.coefficients(128, 256)
    assert midpoint == pytest.approx(
        (1.0 + 9.0 * 128 / 255, 23 - 5.75 * 128 / 255, 5 * 128 / 255)
    )
    assert orth_training_module.coefficients(255, 256) == (
        10.0,
        17.25,
        5.0,
    )


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
    trainer.total_microsteps = 256
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
        "is_forget": torch.tensor(sources, dtype=torch.bool),
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


def test_mixed_source_sampler_uses_each_row_once_per_epoch(
    orth_training_module,
) -> None:
    class TaggedDataset:
        def __init__(self):
            self.sources = [0, 1, 1, 0] * 4

        def __getitem__(self, key):
            assert key == "is_forget"
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
    assert first == list(orth_training_module.MixedSourceSampler(dataset, 8, 42))


def test_mixed_source_sampler_truncates_to_shorter_source(
    orth_training_module,
) -> None:
    class TaggedDataset:
        def __init__(self, sources):
            self.sources = sources

        def __getitem__(self, key):
            assert key == "is_forget"
            return self.sources

    sampler = orth_training_module.MixedSourceSampler
    with pytest.raises(ValueError, match="even batch size"):
        sampler(TaggedDataset([0, 1]), 3, 42)
    with pytest.raises(ValueError, match="forget and retain"):
        sampler(TaggedDataset([0, 0]), 2, 42)
    mixed = sampler(TaggedDataset([0, 0, 1]), 2, 42)
    indices = list(mixed)
    assert len(indices) == 2
    assert {TaggedDataset([0, 0, 1]).sources[index] for index in indices} == {0, 1}


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


def test_npo_loss_routes_sources_and_only_updates_adapter(npo_training_module) -> None:
    torch = pytest.importorskip("torch")
    from contextlib import contextmanager

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
            self.adapter = torch.nn.Parameter(torch.tensor(0.3))
            self.disabled = False
            self.calls = []

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def forward(self, input_ids, attention_mask, labels):
            assert torch.equal(input_ids, labels)
            assert torch.all(attention_mask == 1)
            self.calls.append(
                (input_ids.flatten().tolist(), self.disabled, torch.is_grad_enabled())
            )
            weight = self.base if self.disabled else self.base + self.adapter
            return SimpleNamespace(loss=input_ids.float().mean() * weight)

    model = Model()
    trainer = object.__new__(npo_training_module.NPOTrainer)
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }
    initial_base = model.base.detach().clone()
    with model.disable_adapter(), torch.no_grad():
        original_output = model(
            **{key: inputs[key] for key in ("input_ids", "attention_mask", "labels")}
        ).loss
    model.calls.clear()
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    beta = npo_training_module.BETA
    expected = (
        -(2 / beta)
        * torch.nn.functional.logsigmoid(torch.tensor(beta * (3 * 2.3 - 3 * 2.0)))
        + 8 * 2.3
    )
    assert loss.item() == pytest.approx(expected.item())
    assert outputs.loss.item() == pytest.approx(3 * 2.3)
    assert model.calls == [
        ([2, 4], False, True),
        ([2, 4], True, False),
        ([7, 9], False, True),
    ]
    loss.backward()
    assert model.adapter.grad is not None
    assert model.adapter.grad.item() != 0
    assert model.base.grad is None
    torch.optim.SGD([model.adapter], lr=0.1).step()
    assert torch.equal(model.base, initial_base)
    with model.disable_adapter(), torch.no_grad():
        assert (
            model(
                **{
                    key: inputs[key]
                    for key in ("input_ids", "attention_mask", "labels")
                }
            ).loss
            == original_output
        )


def test_npo_requires_both_sources(npo_training_module) -> None:
    torch = pytest.importorskip("torch")
    trainer = object.__new__(npo_training_module.NPOTrainer)
    with pytest.raises(ValueError, match="forget and retain"):
        trainer.compute_loss(None, {"is_forget": torch.tensor([True, True])})


def test_npo_sam_accumulates_perturbed_forget_and_unperturbed_retain(
    npo_sam_training_module,
) -> None:
    from contextlib import contextmanager, nullcontext

    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self, forget_scale=2.0, fail_perturbed=False):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.lora = torch.nn.Parameter(torch.tensor(0.25))
            self.disabled = False
            self.calls = []
            self.forget_scale = forget_scale
            self.fail_perturbed = fail_perturbed

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def forward(self, input_ids, attention_mask, labels):
            source = int(input_ids[0, 0])
            self.calls.append((source, self.disabled, float(self.lora.detach())))
            if self.fail_perturbed and source == 2 and len(self.calls) >= 3:
                raise RuntimeError("second pass failed")
            scale = self.forget_scale if source == 2 else 3.0
            weight = self.base if self.disabled else self.base + self.lora
            return SimpleNamespace(loss=scale * weight.square())

    class Accelerator:
        def backward(self, loss):
            loss.backward()

    trainer = object.__new__(npo_sam_training_module.BalancedNPOSAMTrainer)
    trainer.optimizer = None
    trainer.accelerator = Accelerator()
    trainer.current_gradient_accumulation_steps = 2
    trainer._layer_offload_ctx = nullcontext()
    trainer.activation_offload_context = nullcontext()
    trainer._prepare_inputs = lambda inputs: inputs
    trainer.compute_loss_context_manager = nullcontext
    inputs = {
        "input_ids": torch.tensor([[2], [1]]),
        "attention_mask": torch.ones(2, 1),
        "labels": torch.tensor([[2], [1]]),
        "is_forget": torch.tensor([True, False]),
    }
    model = Model()
    model.lora.grad = torch.tensor(4.0)
    first_loss = trainer.training_step(model, inputs)
    assert first_loss.isfinite()
    assert model.calls == [
        (2, False, 0.25),
        (2, True, 0.25),
        (2, False, pytest.approx(0.24)),
        (2, True, pytest.approx(0.24)),
        (1, False, 0.25),
    ]
    beta = 0.0225
    perturbed = torch.tensor(0.24, requires_grad=True)
    forget = -(2 / beta) * torch.nn.functional.logsigmoid(
        beta * (2 * (1 + perturbed).square() - 2)
    )
    expected_forget = torch.autograd.grad(forget, perturbed)[0].item()
    expected_microbatch = (expected_forget + 3 * 2 * 1.25) / 2
    assert model.lora.grad.item() == pytest.approx(4 + expected_microbatch)
    assert model.base.grad is None
    assert model.lora.item() == pytest.approx(0.25)

    trainer.training_step(model, inputs)
    assert model.lora.grad.item() == pytest.approx(4 + 2 * expected_microbatch)

    zero_model = Model(forget_scale=0)
    zero_model.lora.grad = torch.tensor(4.0)
    trainer.training_step(zero_model, inputs)
    assert all(call[2] == pytest.approx(0.25) for call in zero_model.calls)
    assert zero_model.lora.grad.item() == pytest.approx(4 + 3.75)

    failing_model = Model(fail_perturbed=True)
    failing_model.lora.grad = torch.tensor(4.0)
    with pytest.raises(RuntimeError, match="second pass failed"):
        trainer.training_step(failing_model, inputs)
    assert failing_model.lora.item() == pytest.approx(0.25)
    assert failing_model.lora.grad.item() == pytest.approx(4.0)


def test_npo_sam_uses_one_norm_across_trainable_weights(
    npo_sam_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    delta = npo_sam_training_module.sam_perturbation(
        {"first": torch.tensor(3.0), "second": torch.tensor(4.0), "frozen": None}
    )
    assert set(delta) == {"first", "second"}
    assert delta["first"].item() == pytest.approx(0.006)
    assert delta["second"].item() == pytest.approx(0.008)


@pytest.mark.parametrize(
    ("trainer_name", "beta", "gamma", "rho"),
    [
        ("BalancedNPOSAMTrainer", 0.0225, 1.0, 0.01),
        ("BalancedNPOSAMRho003Trainer", 0.0225, 1.0, 0.003),
        ("BalancedNPOSAMGamma225Trainer", 0.0225, 2.25, 0.01),
        ("BalancedNPOSAMGamma450Trainer", 0.0225, 4.5, 0.01),
        ("BalancedNPOSAMGamma900Trainer", 0.0225, 9.0, 0.01),
        ("BalancedNPOSAMBeta015Gamma225Trainer", 0.015, 2.25, 0.01),
    ],
)
def test_npo_sam_trainer_hyperparameters(
    npo_sam_training_module, trainer_name, beta, gamma, rho
) -> None:
    trainer = getattr(npo_sam_training_module, trainer_name)
    assert (trainer.beta, trainer.gamma, trainer.rho) == (beta, gamma, rho)


def test_npo_sam_at_zero_radius_matches_npo_gradient(
    npo_sam_training_module, monkeypatch
) -> None:
    from contextlib import contextmanager, nullcontext

    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.lora = torch.nn.Parameter(torch.tensor(0.25))
            self.disabled = False

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def forward(self, input_ids, attention_mask, labels):
            scale = input_ids.float().mean()
            weight = self.base if self.disabled else self.base + self.lora
            return SimpleNamespace(loss=scale * weight.square())

    class Accelerator:
        def backward(self, loss):
            loss.backward()

    inputs = {
        "input_ids": torch.tensor([[2], [1]]),
        "attention_mask": torch.ones(2, 1),
        "labels": torch.tensor([[2], [1]]),
        "is_forget": torch.tensor([True, False]),
    }
    npo_model = Model()
    trainer = object.__new__(npo_sam_training_module.BalancedNPOSAMTrainer)
    (trainer.compute_loss(npo_model, inputs) / 2).backward()

    sam_model = Model()
    trainer.optimizer = None
    trainer.accelerator = Accelerator()
    trainer.current_gradient_accumulation_steps = 2
    trainer._layer_offload_ctx = nullcontext()
    trainer.activation_offload_context = nullcontext()
    trainer._prepare_inputs = lambda batch: batch
    trainer.compute_loss_context_manager = nullcontext
    monkeypatch.setattr(npo_sam_training_module.BalancedNPOSAMTrainer, "rho", 0.0)
    trainer.training_step(sam_model, inputs)
    torch.testing.assert_close(sam_model.lora.grad, npo_model.lora.grad)


def test_npo_sam_peft_gradient_checkpointing(npo_sam_training_module) -> None:
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=2
        )
    )
    base.gradient_checkpointing_enable()
    model = peft.get_peft_model(
        base, peft.LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"])
    )
    model.enable_input_require_grads()
    initial_base = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }

    class Accelerator:
        def backward(self, loss):
            loss.backward()

    trainer = object.__new__(npo_sam_training_module.BalancedNPOSAMTrainer)
    trainer.optimizer = None
    trainer.accelerator = Accelerator()
    trainer.current_gradient_accumulation_steps = 2
    trainer._layer_offload_ctx = nullcontext()
    trainer.activation_offload_context = nullcontext()
    trainer._prepare_inputs = lambda inputs: inputs
    trainer.compute_loss_context_manager = nullcontext
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "is_forget": torch.tensor([True, False]),
    }
    loss = trainer.training_step(model, inputs)
    assert loss.isfinite()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, initial_base[name])


def test_weight_distance_uses_one_global_euclidean_norm(
    weight_dist_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    first = torch.tensor([3.0, 4.0], requires_grad=True)
    second = torch.tensor([12.0], requires_grad=True)
    distance = weight_dist_training_module.global_l2_distance(
        (first, second), (torch.zeros_like(first), torch.zeros_like(second))
    )

    assert distance.item() == pytest.approx(13.0)
    distance.backward()
    torch.testing.assert_close(first.grad, torch.tensor([3 / 13, 4 / 13]))
    torch.testing.assert_close(second.grad, torch.tensor([12 / 13]))


def test_weight_distance_rms_matches_authors_normalization(
    weight_dist_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    first = torch.tensor([3.0, 4.0], requires_grad=True)
    second = torch.tensor([12.0], requires_grad=True)
    distance = weight_dist_training_module.global_rms_distance(
        (first, second), (torch.zeros_like(first), torch.zeros_like(second))
    )
    expected = (169 / 3 + 1e-8) ** 0.5

    assert distance.item() == pytest.approx(expected)
    distance.backward()
    torch.testing.assert_close(
        first.grad, torch.tensor([3 / (3 * expected), 4 / (3 * expected)])
    )
    torch.testing.assert_close(second.grad, torch.tensor([12 / (3 * expected)]))


def test_weight_dist_loss_moves_away_from_initial_weights(
    weight_dist_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, input_ids, attention_mask, labels):
            return SimpleNamespace(loss=self.weight.square())

    model = Model()
    trainer = object.__new__(weight_dist_training_module.WeightDistTrainer)
    trainer.distance_coefficient = 1.0
    trainer._weight_dist_anchors = None
    trainer._weight_dist_logged_step = None
    trainer.state = SimpleNamespace(global_step=0)
    trainer.log = lambda metrics: None
    inputs = {
        "input_ids": torch.ones(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
        "labels": torch.ones(1, 1, dtype=torch.long),
    }

    assert trainer.compute_loss(model, inputs).item() == pytest.approx(4.0)
    with torch.no_grad():
        model.weight.fill_(3.0)
    loss = trainer.compute_loss(model, inputs)
    assert loss.item() == pytest.approx(8.0)
    loss.backward()
    assert model.weight.grad.item() == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("trainer_name", "coefficient"),
    [
        ("WeightDistCoefficient01Trainer", 0.1),
        ("WeightDistTrainer", 1.0),
        ("WeightDistCoefficient10Trainer", 10.0),
        ("WeightDistCoefficient25Trainer", 25.0),
        ("WeightDistCoefficient50Trainer", 50.0),
        ("WeightDistCoefficient75Trainer", 75.0),
        ("WeightDistCoefficient100Trainer", 100.0),
        ("PretrainedAnchorWeightDistCoefficient01Trainer", 0.1),
        ("PretrainedAnchorWeightDistTrainer", 1.0),
        ("PretrainedAnchorWeightDistCoefficient10Trainer", 10.0),
        ("PretrainedAnchorWeightDistCoefficient1e5Trainer", 1e5),
        ("PretrainedAnchorWeightDistCoefficient1e6Trainer", 1e6),
    ],
)
def test_weight_dist_coefficients(
    weight_dist_training_module, trainer_name, coefficient
) -> None:
    trainer = getattr(weight_dist_training_module, trainer_name)
    assert trainer.distance_coefficient == coefficient


def test_pretrained_anchor_weight_dist_loads_explicit_reference(
    weight_dist_training_module, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    expected = {"weight": torch.full_like(model.weight, 7)}
    calls = []

    def load(parameters, model_name_or_path):
        calls.append((parameters, model_name_or_path))
        return expected

    monkeypatch.setattr(weight_dist_training_module, "load_pretrained_anchors", load)
    trainer = object.__new__(
        weight_dist_training_module.PretrainedAnchorWeightDistTrainer
    )
    trainer._weight_dist_anchors = None

    parameters, anchors = trainer._parameters_and_anchors(model)

    assert parameters == {"weight": model.weight}
    assert anchors == expected
    assert calls == [
        (parameters, "HuggingFaceH4/zephyr-7b-beta"),
    ]


def test_pretrained_anchor_loader_strips_framework_wrapper_names(
    weight_dist_training_module, monkeypatch, tmp_path
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    checkpoint_name = "model.layers.0.mlp.down_proj.weight"
    runtime_name = (
        "model.layers.0._checkpoint_wrapped_module.mlp."
        "_checkpoint_wrapped_module.down_proj.weight"
    )
    expected = torch.tensor([[1.0, 2.0]])
    safetensors.save_file({checkpoint_name: expected}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {checkpoint_name: "model.safetensors"}})
    )

    anchors = weight_dist_training_module.load_pretrained_anchors(
        {runtime_name: torch.nn.Parameter(torch.zeros_like(expected))}, tmp_path
    )

    assert anchors.keys() == {runtime_name}
    torch.testing.assert_close(anchors[runtime_name], expected)


@pytest.mark.parametrize(
    ("trainer_name", "coefficient"),
    [
        ("GDWeightDistCoefficient01Trainer", 0.1),
        ("GDWeightDistTrainer", 1.0),
        ("GDWeightDistCoefficient10Trainer", 10.0),
    ],
)
def test_gd_weight_dist_coefficients(
    gd_weight_dist_training_module, trainer_name, coefficient
) -> None:
    trainer = getattr(gd_weight_dist_training_module, trainer_name)
    assert trainer.distance_coefficient == coefficient


@pytest.mark.parametrize(
    ("trainer_name", "forget_coefficient", "retain_coefficient"),
    [
        ("GradDiffTrainer", 1.0, 1.0),
        ("GradDiffF01R1Trainer", 0.1, 1.0),
    ],
)
def test_grad_diff_loss_and_gradient_direction(
    gd_training_module, trainer_name, forget_coefficient, retain_coefficient
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forget_weight = torch.nn.Parameter(torch.tensor(2.0))
            self.retain_weight = torch.nn.Parameter(torch.tensor(3.0))
            self.calls = []

        def forward(self, input_ids, attention_mask, labels):
            assert torch.equal(input_ids, labels)
            assert torch.all(attention_mask == 1)
            ids = input_ids.flatten().tolist()
            self.calls.append(ids)
            weight = self.forget_weight if ids == [2, 4] else self.retain_weight
            return SimpleNamespace(loss=input_ids.float().mean() * weight)

    model = Model()
    trainer = object.__new__(getattr(gd_training_module, trainer_name))
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert loss.item() == pytest.approx(
        -forget_coefficient * 3 * 2 + retain_coefficient * 8 * 3
    )
    assert outputs.loss.item() == pytest.approx(3 * 2)
    assert model.calls == [[2, 4], [7, 9]]
    loss.backward()
    assert model.forget_weight.grad.item() == pytest.approx(-forget_coefficient * 3)
    assert model.retain_weight.grad.item() == pytest.approx(retain_coefficient * 8)


@pytest.mark.parametrize("sources", [[1, 1], [0, 0]])
def test_grad_diff_requires_both_sources(gd_training_module, sources) -> None:
    torch = pytest.importorskip("torch")
    trainer = object.__new__(gd_training_module.GradDiffTrainer)
    with pytest.raises(ValueError, match="forget and retain"):
        trainer.compute_loss(None, {"is_forget": torch.tensor(sources).bool()})


def test_grad_diff_sampler_balances_each_microbatch(gd_training_module) -> None:
    class TaggedDataset:
        def __getitem__(self, key):
            assert key == "is_forget"
            return [0, 1, 1, 0] * 2

    trainer = SimpleNamespace(
        train_dataset=TaggedDataset(),
        args=SimpleNamespace(per_device_train_batch_size=4, data_seed=None, seed=42),
    )
    sampler = gd_training_module.GradDiffTrainer._get_train_sampler(trainer)
    indices = list(sampler)
    assert sorted(indices) == list(range(8))
    for start in (0, 4):
        assert (
            sum(
                trainer.train_dataset["is_forget"][i]
                for i in indices[start : start + 4]
            )
            == 2
        )


def test_gd_sam_loss_uses_small_forget_perturbation(gd_sam_training_module) -> None:
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.unused_weight = torch.nn.Parameter(torch.tensor(3.0))
            self.calls = []

        def forward(self, input_ids, attention_mask, labels):
            assert torch.equal(input_ids, labels)
            assert torch.all(attention_mask == 1)
            self.calls.append(float(self.weight.detach()))
            return SimpleNamespace(loss=input_ids.float().mean() * self.weight.square())

    class Accelerator:
        def backward(self, loss):
            loss.backward()

    model = Model()
    trainer = object.__new__(gd_sam_training_module.GradDiffTrainer)
    trainer.optimizer = None
    trainer.accelerator = Accelerator()
    trainer.current_gradient_accumulation_steps = 1
    trainer._layer_offload_ctx = nullcontext()
    trainer.activation_offload_context = nullcontext()
    trainer._prepare_inputs = lambda batch: batch
    trainer.compute_loss_context_manager = nullcontext
    trainer.rho = 0.01
    trainer.forget_coeff = 1.0
    trainer.retain_coeff = 1.0
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    loss = trainer.training_step(model, inputs)

    assert gd_sam_training_module.GradDiffTrainer.rho == 0.01
    assert loss.item() == pytest.approx(-3 * (2 - 0.01) ** 2 + 8 * 2**2)
    assert model.calls == [2.0, pytest.approx(1.99), 2.0]
    assert model.weight.grad.item() == pytest.approx(-6 * 1.99 + 16 * 2)
    assert model.unused_weight.grad is None


def test_gd_sam_global_norm_preserves_rho(gd_sam_training_module) -> None:
    torch = pytest.importorskip("torch")
    gradients = (torch.tensor([3.0, 4.0]), torch.tensor([0.0, 12.0]))

    norm = gd_sam_training_module.global_l2_norm(gradients)
    scale = 0.01 / norm
    perturbation = tuple(scale * gradient for gradient in gradients)

    assert norm.item() == pytest.approx(13.0)
    assert gd_sam_training_module.global_l2_norm(perturbation).item() == pytest.approx(
        0.01
    )


def test_gd_sam_exposes_sweep_radii(gd_sam_training_module) -> None:
    expected = {
        "GDSAMRho1eMinus3Trainer": 1e-3,
        "GDSAMRho2p5eMinus3Trainer": 2.5e-3,
        "GDSAMRho5eMinus3Trainer": 5e-3,
        "GDSAMRho1eMinus2Trainer": 1e-2,
    }

    for trainer_name, rho in expected.items():
        trainer = getattr(gd_sam_training_module, trainer_name)
        assert trainer.rho == pytest.approx(rho)


def test_gd_sam_authors_sampler_draws_retain_with_replacement(
    gd_sam_training_module,
) -> None:
    class TaggedDataset:
        sources = [1, 1, 1, 1, 0, 0]

        def __getitem__(self, key):
            assert key == "is_forget"
            return self.sources

    sampler = gd_sam_training_module.AuthorsMixedSourceSampler(
        TaggedDataset(), batch_size=2, seed=3
    )
    first = list(sampler)
    second = list(sampler)

    for epoch in (first, second):
        forget = [index for index in epoch if index < 4]
        retain = [index for index in epoch if index >= 4]
        assert sorted(forget) == [0, 1, 2, 3]
        assert len(retain) == 4
        assert set(retain) <= {4, 5}
        for start in range(0, len(epoch), 2):
            assert sum(index < 4 for index in epoch[start : start + 2]) == 1
    assert len(first) == len(sampler) == 8
    assert first != second

    trainer = gd_sam_training_module.GDSAMAuthorsRho5eMinus3Trainer
    assert trainer.rho == pytest.approx(5e-3)
    paper_trainer = gd_sam_training_module.GDSAMRho1eMinus2Trainer
    assert paper_trainer.rho == pytest.approx(1e-2)


@pytest.mark.skip(reason="legacy config replaced by generated stage sweep")
def test_gd_sam_model_parallel_config_matches_paper_batch() -> None:
    config = yaml.safe_load(
        (
            EXAMPLE / "configs/unlearn/z7b/hpo/"
            "wmdp-bio-fft-s125-gd-sam-v2-authors-model-parallel-"
            "rho1e-2-lr3p28e-6.yml"
        ).read_text()
    )

    device_map = config["device_map"]
    assert device_map["model.embed_tokens"] == 0
    assert device_map["model.norm"] == 3
    assert device_map["lm_head"] == 3
    assert {device_map[f"model.layers.{layer}"] for layer in range(32)} == {0, 1, 2, 3}
    for layer in range(32):
        assert device_map[f"model.layers.{layer}"] == layer // 8
    assert "max_memory" not in config
    assert "fsdp_version" not in config
    assert config["micro_batch_size"] == 8
    assert config["gradient_accumulation_steps"] == 1
    assert config["max_steps"] == 125
    assert config["warmup_steps"] == 12
    assert float(config["learning_rate"]) == pytest.approx(3.28e-6)

    script = (EXAMPLE / "scripts/slurm/tamia/train_model_parallel.sbatch").read_text()
    assert 'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert 'axolotl train "$config" --launcher python' in script


@pytest.mark.skip(reason="legacy config replaced by canonical lineage")
def test_default_fft_gd_sam_uses_authors_matched_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-fft-s125.yml")
    plan = pipeline.plan()

    model_id = "z7b-wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-2"
    unlearn = yaml.safe_load(
        (
            EXAMPLE / "configs/unlearn/z7b/wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-2.yml"
        ).read_text()
    )
    relearn = yaml.safe_load(
        (
            EXAMPLE
            / "configs/relearn/z7b/wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-2-relearn.yml"
        ).read_text()
    )

    assert model_id in pipeline.selected_models
    producer = next(stage for stage in plan.stages if stage.id == f"train-{model_id}")
    assert producer.script == "scripts/slurm/tamia/train_model_parallel.sbatch"
    assert unlearn["trainer_cls"].endswith("GDSAMRho1eMinus2Trainer")
    assert float(unlearn["learning_rate"]) == pytest.approx(3.28e-6)
    assert unlearn["micro_batch_size"] == 8
    assert unlearn["gradient_accumulation_steps"] == 1
    assert unlearn["seed"] == 3
    assert unlearn["tf32"] is False
    assert unlearn["save_steps"] == 10
    assert unlearn["save_total_limit"] == 13
    assert relearn["base_model"].endswith(model_id)
    assert float(relearn["learning_rate"]) == pytest.approx(1e-5)
    assert relearn["max_steps"] == 16
    assert relearn["save_steps"] == 4


def test_gd_sam_peft_gradient_checkpointing(gd_sam_training_module) -> None:
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=2
        )
    )
    base.gradient_checkpointing_enable()
    model = peft.get_peft_model(
        base, peft.LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"])
    )
    model.enable_input_require_grads()
    trainer = object.__new__(gd_sam_training_module.GradDiffTrainer)
    trainer.optimizer = None

    class Accelerator:
        def backward(self, loss):
            loss.backward()

    trainer.accelerator = Accelerator()
    trainer.current_gradient_accumulation_steps = 1
    trainer._layer_offload_ctx = nullcontext()
    trainer.activation_offload_context = nullcontext()
    trainer._prepare_inputs = lambda batch: batch
    trainer.compute_loss_context_manager = nullcontext
    trainer.rho = 0.01
    trainer.forget_coeff = 1.0
    trainer.retain_coeff = 1.0
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "is_forget": torch.tensor([True, False]),
    }

    loss = trainer.training_step(model, inputs)

    assert loss.isfinite()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_gd_sam_sampler_balances_each_microbatch(gd_sam_training_module) -> None:
    class TaggedDataset:
        def __getitem__(self, key):
            assert key == "is_forget"
            return [0, 1, 1, 0] * 2

    trainer = SimpleNamespace(
        train_dataset=TaggedDataset(),
        args=SimpleNamespace(per_device_train_batch_size=4, data_seed=None, seed=42),
    )
    sampler = gd_sam_training_module.GradDiffTrainer._get_train_sampler(trainer)
    indices = list(sampler)

    for start in (0, 4):
        assert (
            sum(
                trainer.train_dataset["is_forget"][i]
                for i in indices[start : start + 4]
            )
            == 2
        )


def test_gd_gn_loss_has_second_order_gradient(
    gd_gn_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.frozen = torch.nn.Parameter(torch.tensor(3.0), requires_grad=False)

        def forward(self, input_ids, attention_mask, labels):
            assert torch.equal(input_ids, labels)
            assert torch.all(attention_mask == 1)
            target = 2.0 if int(input_ids[0, 0]) == 1 else -1.0
            return SimpleNamespace(loss=(self.weight - target).square())

    model = Model()
    trainer = object.__new__(gd_gn_training_module.BalancedGradDiffGNTrainer)
    inputs = {
        "input_ids": torch.tensor([[0], [1]]),
        "attention_mask": torch.ones(2, 1),
        "labels": torch.tensor([[0], [1]]),
        "is_forget": torch.tensor([False, True]),
    }
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert outputs.loss.item() == pytest.approx(1.0)
    assert loss.item() == pytest.approx(-1.0 + 0.01 * 2.0 + 4.0)
    loss.backward()
    assert model.weight.grad.item() == pytest.approx(2.0 - 0.02 + 4.0)
    assert model.frozen.grad is None


@pytest.mark.parametrize("sources", [[1, 1], [0, 0]])
def test_gd_gn_requires_both_sources(gd_gn_training_module, sources) -> None:
    torch = pytest.importorskip("torch")
    trainer = object.__new__(gd_gn_training_module.BalancedGradDiffGNTrainer)
    with pytest.raises(ValueError, match="forget and retain"):
        trainer.compute_loss(None, {"is_forget": torch.tensor(sources).bool()})


def test_gd_gn_with_lora_checkpointing(
    gd_gn_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=2
        )
    )
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = peft.get_peft_model(
        base, peft.LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"])
    )
    trainer = object.__new__(gd_gn_training_module.BalancedGradDiffGNTrainer)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "is_forget": torch.tensor([True, False]),
    }
    loss = trainer.compute_loss(model, inputs)
    loss.backward()
    assert loss.isfinite()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_deff_trainer_loss_is_retain_ce_plus_off_diagonal_cosine(
    deff_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.retain_weight = torch.nn.Parameter(torch.tensor(2.0))
            self.forget_rows = torch.nn.Parameter(
                torch.tensor([[3.0, 4.0, 0.0], [4.0, 3.0, 0.0]])
            )
            self.calls = []

        def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            output_hidden_states=False,
            use_cache=None,
        ):
            assert output_hidden_states is (labels is None)
            if labels is not None:
                assert torch.equal(input_ids, labels)
                assert torch.all(attention_mask == 1)
                self.calls.append("retain")
                return SimpleNamespace(
                    loss=input_ids.float().mean() * self.retain_weight
                )
            self.calls.append("forget")
            hidden = self.forget_rows.unsqueeze(1)
            return SimpleNamespace(hidden_states=(None, hidden, hidden))

    model = Model()
    trainer = object.__new__(deff_training_module.DEFFTrainer)
    trainer.target_layers = (0, 1)
    trainer.retain_coefficient = 1.0
    trainer.forget_coefficient = 1.0
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert loss.item() == pytest.approx(16.0 + 0.96)
    assert outputs["loss_retain"].item() == pytest.approx(16.0)
    assert outputs["loss_mean_cosim"].item() == pytest.approx(0.96)
    assert model.calls == ["retain", "forget"]
    loss.backward()
    assert model.retain_weight.grad.item() == pytest.approx(8.0)
    assert model.forget_rows.grad is not None
    assert model.forget_rows.grad.abs().sum() > 0


def test_gd_deff_loss_has_retain_forget_and_orthogonality_terms(
    gd_deff_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.forget_rows = torch.nn.Parameter(
                torch.tensor([[3.0, 4.0, 0.0], [4.0, 3.0, 0.0]])
            )

        def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            output_hidden_states=False,
            use_cache=None,
        ):
            loss = input_ids.float().mean() * self.weight
            if not output_hidden_states:
                return SimpleNamespace(loss=loss)
            hidden = self.forget_rows.unsqueeze(1)
            return SimpleNamespace(loss=loss, hidden_states=(None, hidden, hidden))

    model = Model()
    trainer = object.__new__(gd_deff_training_module.DEFFTrainer)
    trainer.target_layers = (0, 1)
    trainer.retain_coef = 2.0
    trainer.forget_coef = 3.0
    trainer.cosim_coeff = 4.0
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    assert loss.item() == pytest.approx(2 * 16.0 - 3 * 6.0 + 4 * 0.96)
    assert outputs["loss_retain"].item() == pytest.approx(16.0)
    assert outputs["loss_forget"].item() == pytest.approx(6.0)
    assert outputs["loss_mean_cosim"].item() == pytest.approx(0.96)


def test_gd_deff_targets_all_transformer_layers_by_default(
    gd_deff_training_module, monkeypatch
) -> None:
    def init(trainer, *args, **kwargs):
        trainer.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=3))

    monkeypatch.setattr(gd_deff_training_module.AxolotlTrainer, "__init__", init)
    trainer = gd_deff_training_module.DEFFTrainer()

    assert trainer.target_layers == (0, 1, 2)
    assert trainer.retain_coef == 1.0
    assert trainer.forget_coef == 1.0
    assert trainer.cosim_coeff == 1.0


def test_deff_trainer_ignores_padding_tokens_in_forget_pool(
    deff_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.z = torch.tensor(
                [
                    [[1.0, 0.0], [1.0, 1.0], [9.0, 9.0]],
                    [[2.0, 0.0], [9.0, 9.0], [9.0, 9.0]],
                ]
            )

        def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            output_hidden_states=False,
            use_cache=None,
        ):
            if labels is not None:
                return SimpleNamespace(
                    loss=(input_ids.float() * attention_mask).sum() * self.weight
                )
            return SimpleNamespace(hidden_states=(None, self.z, self.z))

    model = Model()
    trainer = object.__new__(deff_training_module.DEFFTrainer)
    trainer.target_layers = (1,)
    trainer.retain_coefficient = 1.0
    trainer.forget_coefficient = 1.0
    inputs = {
        "input_ids": torch.tensor([[5, 0, 0], [1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 0]]),
        "labels": torch.tensor([[5, 0, 0], [1, 2, 3], [4, 5, 6]]),
        "is_forget": torch.tensor([False, True, True]),
    }
    loss = trainer.compute_loss(model, inputs)
    assert loss.item() == pytest.approx(5.0 + 1.0 / 1.25**0.5)


@pytest.mark.parametrize("sources", [[1, 1, 1], [0, 0]])
def test_deff_trainer_requires_both_sources(deff_training_module, sources) -> None:
    torch = pytest.importorskip("torch")
    trainer = object.__new__(deff_training_module.DEFFTrainer)
    with pytest.raises(ValueError, match="forget and retain"):
        trainer.compute_loss(None, {"is_forget": torch.tensor(sources).bool()})


def test_deff_trainer_requires_two_forget_rows(deff_training_module) -> None:
    torch = pytest.importorskip("torch")
    trainer = object.__new__(deff_training_module.DEFFTrainer)
    with pytest.raises(ValueError, match="at least two forget"):
        trainer.compute_loss(None, {"is_forget": torch.tensor([False, True])})


def test_deff_trainer_validates_target_layers(
    deff_training_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(
        deff_training_module.AxolotlTrainer,
        "__init__",
        lambda self, *args, **kwargs: None,
    )
    trainer = object.__new__(deff_training_module.DEFFTrainer)
    trainer.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
    with pytest.raises(ValueError, match="target_layers"):
        deff_training_module.DEFFTrainer.__init__(trainer, target_layers=(0, 0))
    with pytest.raises(ValueError, match="target_layers"):
        deff_training_module.DEFFTrainer.__init__(trainer, target_layers=(1, 2))
    deff_training_module.DEFFTrainer.__init__(trainer, target_layers=(0, 1))
    assert trainer.target_layers == (0, 1)
    trainer.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=32))
    deff_training_module.DEFFTrainer.__init__(trainer)
    assert trainer.target_layers == (5, 10, 15, 20, 25, 30)


def test_deff_trainer_sampler_balances_each_microbatch(deff_training_module) -> None:
    class TaggedDataset:
        def __getitem__(self, key):
            assert key == "is_forget"
            return [0, 1, 1, 0] * 2

    trainer = SimpleNamespace(
        train_dataset=TaggedDataset(),
        args=SimpleNamespace(per_device_train_batch_size=4, data_seed=None, seed=42),
    )
    sampler = deff_training_module.DEFFTrainer._get_train_sampler(trainer)
    indices = list(sampler)
    assert sorted(indices) == list(range(8))
    for start in (0, 4):
        assert (
            sum(
                trainer.train_dataset["is_forget"][i]
                for i in indices[start : start + 4]
            )
            == 2
        )


def test_peft_disabled_adapter_recovers_initial_base_output() -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=2
        )
    ).eval()
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        original = base(tokens).logits.clone()
    model = peft.get_peft_model(
        base, peft.LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"])
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.2)
        adapted = model(tokens).logits
        with model.disable_adapter():
            reference = model(tokens).logits
    assert not torch.allclose(adapted, original)
    torch.testing.assert_close(reference, original)


@pytest.mark.skip(reason="legacy hand-expanded HPO configs removed")
def test_z7b_lora_s125_gd_sam_unlearning_sweeps_lr_for_each_rho(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-lora-s125-unlearn-gd-sam-hpo.yml")
    plan = pipeline.plan()
    learning_rates = {
        "2p5e-6": 2.5e-6,
        "5e-6": 5e-6,
        "1e-5": 1e-5,
        "5e-5": 5e-5,
        "7p5e-5": 7.5e-5,
        "1e-4": 1e-4,
    }

    trainers = {
        "1e-3": "GDSAMRho1eMinus3Trainer",
        "1e-2": "GDSAMTrainer",
        "1e-1": "GDSAMRho1eMinus1Trainer",
    }

    assert len(pipeline.selected_models) == 18
    assert len(plan.stages) == 54
    for rho, trainer in trainers.items():
        for learning_rate_slug, learning_rate in learning_rates.items():
            model_id = (
                f"z7b-wmdp-bio-lora-s125-unlearn-gd-sam-rho{rho}-hpo-"
                f"lr{learning_rate_slug}"
            )
            config = yaml.safe_load(
                (
                    EXAMPLE
                    / "configs/unlearn/z7b/hpo"
                    / (
                        "wmdp-bio-lora-s125-unlearn-gd-sam-"
                        f"rho{rho}-lr{learning_rate_slug}.yml"
                    )
                ).read_text()
            )

            assert model_id in pipeline.selected_models
            assert config["trainer_cls"].endswith(trainer)
            assert float(config["learning_rate"]) == learning_rate
            assert "rho" not in config
            assert config["max_steps"] == 125
            assert config["save_strategy"] == "steps"
            assert config["save_steps"] == 10
            assert config["save_total_limit"] == 13
            assert config["output_dir"].endswith(model_id)


@pytest.mark.skip(reason="legacy hand-expanded HPO configs removed")
def test_z7b_lora_s125_gd_unlearning_sweep_uses_selected_learning_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-lora-s125-unlearn-gd-hpo.yml")
    plan = pipeline.plan()
    learning_rates = ("2p5e-6", "5e-6", "1e-5", "5e-5", "7p5e-5", "1e-4")

    assert len(pipeline.selected_models) == 6
    assert len(plan.stages) == 18
    for learning_rate in learning_rates:
        model_id = f"z7b-wmdp-bio-lora-s125-unlearn-gd-hpo-lr{learning_rate}"
        config = yaml.safe_load(
            (
                EXAMPLE
                / "configs/unlearn/z7b/hpo"
                / f"wmdp-bio-lora-s125-unlearn-gd-lr{learning_rate}.yml"
            ).read_text()
        )

        assert model_id in pipeline.selected_models
        assert config["max_steps"] == 125
        assert config["save_strategy"] == "steps"
        assert config["save_steps"] == 10
        assert config["save_total_limit"] == 13
        assert config["output_dir"].endswith(model_id)


@pytest.mark.skip(reason="legacy hand-expanded HPO configs removed")
def test_z7b_fft_s125_gd_sweeps_share_paper_learning_rate_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    learning_rates = {
        "2p5e-6": 2.5e-6,
        "5e-6": 5e-6,
        "7p5e-6": 7.5e-6,
        "1e-5": 1e-5,
    }

    for stage in ("unlearn", "relearn"):
        pipeline = lilpipe.load(f"configs/experiments/z7b-fft-s125-{stage}-gd-hpo.yml")
        plan = pipeline.plan()

        assert len(pipeline.selected_models) == 4
        assert len(plan.stages) == 12
        for learning_rate_slug, learning_rate in learning_rates.items():
            relearn = "-relearn" if stage == "relearn" else ""
            model_id = (
                f"z7b-wmdp-bio-fft-s125-unlearn-gd{relearn}-hpo-"
                f"lr{learning_rate_slug}"
            )
            config_path = (
                EXAMPLE
                / f"configs/{stage}/z7b/hpo"
                / (
                    f"wmdp-bio-fft-s125-unlearn-gd{relearn}-"
                    f"lr{learning_rate_slug}.yml"
                )
            )
            config = yaml.safe_load(config_path.read_text())

            assert model_id in pipeline.selected_models
            assert "adapter" not in config
            assert float(config["learning_rate"]) == learning_rate
            assert config["max_steps"] == 125
            assert config["save_steps"] == 125
            assert config["save_total_limit"] == 1
            assert config["fsdp_version"] == 2
            assert config["output_dir"].startswith("artifacts/tamia/")
            assert config["output_dir"].endswith(model_id)


@pytest.mark.skip(reason="legacy hand-expanded HPO configs removed")
def test_z7b_fft_s125_gd_sam_sweep_crosses_rhos_and_learning_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-fft-s125-unlearn-gd-sam-hpo.yml")
    plan = pipeline.plan()
    learning_rates = {
        "2p5e-6": 2.5e-6,
        "5e-6": 5e-6,
        "7p5e-6": 7.5e-6,
        "1e-5": 1e-5,
    }
    trainers = {
        "1e-3": "GDSAMRho1eMinus3Trainer",
        "1e-2": "GDSAMTrainer",
        "1e-1": "GDSAMRho1eMinus1Trainer",
    }

    assert len(pipeline.selected_models) == 12
    assert len(plan.stages) == 36
    for rho, trainer in trainers.items():
        for learning_rate_slug, learning_rate in learning_rates.items():
            model_id = (
                f"z7b-wmdp-bio-fft-s125-unlearn-gd-sam-rho{rho}-hpo-"
                f"lr{learning_rate_slug}"
            )
            config = yaml.safe_load(
                (
                    EXAMPLE
                    / "configs/unlearn/z7b/hpo"
                    / (
                        f"wmdp-bio-fft-s125-unlearn-gd-sam-rho{rho}-"
                        f"lr{learning_rate_slug}.yml"
                    )
                ).read_text()
            )

            assert model_id in pipeline.selected_models
            assert config["trainer_cls"].endswith(trainer)
            assert "rho" not in config
            assert "adapter" not in config
            assert float(config["learning_rate"]) == learning_rate
            assert config["max_steps"] == 125
            assert config["save_steps"] == 125
            assert config["save_total_limit"] == 1
            assert config["fsdp_version"] == 2
            assert config["output_dir"].startswith("artifacts/tamia/")
            assert config["output_dir"].endswith(model_id)


@pytest.mark.skip(reason="legacy config replaced by canonical lineage")
def test_z7b_fft_s125_gd_sam_relearning_uses_promoted_rho_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load(
        "configs/experiments/z7b-fft-s125-relearn-gd-sam-rho1e-3-hpo.yml"
    )
    plan = pipeline.plan()
    learning_rates = {
        "2p5e-6": 2.5e-6,
        "5e-6": 5e-6,
        "7p5e-6": 7.5e-6,
        "1e-5": 1e-5,
    }
    promoted = "artifacts/tamia/models/z7b-wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-3"

    assert len(pipeline.selected_models) == 4
    assert len(plan.stages) == 12
    for learning_rate_slug, learning_rate in learning_rates.items():
        model_id = (
            "z7b-wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-3-relearn-hpo-"
            f"lr{learning_rate_slug}"
        )
        config = yaml.safe_load(
            (
                EXAMPLE
                / "configs/relearn/z7b/hpo"
                / (
                    "wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-3-relearn-"
                    f"lr{learning_rate_slug}.yml"
                )
            ).read_text()
        )

        assert model_id in pipeline.selected_models
        assert config["base_model"] == promoted
        assert "adapter" not in config
        assert float(config["learning_rate"]) == learning_rate
        assert config["max_steps"] == 125
        assert config["save_steps"] == 125
        assert config["save_total_limit"] == 1
        assert config["fsdp_version"] == 2
        assert config["output_dir"].endswith(model_id)


@pytest.mark.skip(reason="legacy hand-expanded HPO configs removed")
def test_z7b_lora_s125_gd_relearning_sweep_uses_promoted_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-lora-s125-relearn-gd-hpo.yml")
    plan = pipeline.plan()
    learning_rates = ("1e-6", "3e-6", "1e-5", "3e-5", "1e-4")
    promoted = "artifacts/mila/models/z7b-wmdp-bio-lora-s125-unlearn-gd/merged"

    assert len(pipeline.selected_models) == 5
    assert len(plan.stages) == 15
    for learning_rate in learning_rates:
        model_id = "z7b-wmdp-bio-lora-s125-unlearn-gd-relearn-hpo-" f"lr{learning_rate}"
        config = yaml.safe_load(
            (
                EXAMPLE
                / "configs/relearn/z7b/hpo"
                / ("wmdp-bio-lora-s125-unlearn-gd-relearn-" f"lr{learning_rate}.yml")
            ).read_text()
        )

        assert model_id in pipeline.selected_models
        assert config["base_model"] == promoted
        assert float(config["learning_rate"]) == float(learning_rate)
        assert config["max_steps"] == 125
        assert config["save_strategy"] == "steps"
        assert config["save_steps"] == 10
        assert config["save_total_limit"] == 13
        assert config["output_dir"].endswith(model_id)


@pytest.mark.skip(reason="legacy hand-expanded HPO configs removed")
def test_z7b_lora_s125_rho_relearning_sweep_uses_promoted_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-lora-s125-relearn-gd-sam-hpo.yml")
    plan = pipeline.plan()
    learning_rates = ("1e-6", "3e-6", "1e-5", "3e-5", "1e-4")

    assert len(pipeline.selected_models) == 15
    assert len(plan.stages) == 48
    for rho in ("1e-3", "1e-2", "1e-1"):
        unlearn_id = f"z7b-wmdp-bio-lora-s125-unlearn-gd-sam-rho{rho}"
        unlearn = yaml.safe_load(
            (
                EXAMPLE
                / "configs/unlearn/z7b"
                / f"wmdp-bio-lora-s125-unlearn-gd-sam-rho{rho}.yml"
            ).read_text()
        )
        assert float(unlearn["learning_rate"]) == 7.5e-5
        assert unlearn["output_dir"].endswith(unlearn_id)

        for learning_rate in learning_rates:
            model_id = f"{unlearn_id}-relearn-hpo-lr{learning_rate}"
            config = yaml.safe_load(
                (
                    EXAMPLE
                    / "configs/relearn/z7b/hpo"
                    / (
                        "wmdp-bio-lora-s125-unlearn-gd-sam-"
                        f"rho{rho}-relearn-lr{learning_rate}.yml"
                    )
                ).read_text()
            )

            assert model_id in pipeline.selected_models
            assert config["base_model"].endswith(f"{unlearn_id}/merged")
            assert float(config["learning_rate"]) == float(learning_rate)
            assert config["max_steps"] == 125
            assert config["save_steps"] == 10
            assert config["save_total_limit"] == 13
            assert config["output_dir"].endswith(model_id)
            stage = plan.stage_index[
                f"train-lora-s125-unlearn-gd-sam-rho{rho}-relearn-hpo-"
                f"lr{learning_rate}"
            ]
            assert stage.depends_on == (f"merge-lora-s125-unlearn-gd-sam-rho{rho}",)


@pytest.mark.skip(reason="legacy config replaced by canonical lineage")
def test_z7b_lora_s125_rho_canonical_relearning_uses_selected_lr() -> None:
    for rho in ("1e-3", "1e-2", "1e-1"):
        model_id = f"z7b-wmdp-bio-lora-s125-unlearn-gd-sam-rho{rho}-relearn"
        config = yaml.safe_load(
            (
                EXAMPLE
                / "configs/relearn/z7b"
                / f"wmdp-bio-lora-s125-unlearn-gd-sam-rho{rho}-relearn.yml"
            ).read_text()
        )

        assert float(config["learning_rate"]) == 1e-4
        assert config["max_steps"] == 125
        assert config["save_steps"] == 10
        assert config["save_total_limit"] == 13
        assert config["base_model"].endswith(
            f"z7b-wmdp-bio-lora-s125-unlearn-gd-sam-rho{rho}/merged"
        )
        assert config["output_dir"].endswith(model_id)


@pytest.mark.skip(reason="legacy config replaced by canonical lineage")
def test_z7b_lora_s125_canonical_pipeline_uses_selected_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-lora-s125.yml")
    plan = pipeline.plan()

    assert len(pipeline.selected_models) == 8
    assert len(plan.stages) == 24
    methods = ("gd", "gd-sam-rho1e-3", "gd-sam-rho1e-2", "gd-sam-rho1e-1")
    for method in methods:
        unlearn_id = f"z7b-wmdp-bio-lora-s125-unlearn-{method}"
        relearn_id = f"{unlearn_id}-relearn"
        unlearn = yaml.safe_load(
            (
                EXAMPLE
                / "configs/unlearn/z7b"
                / f"wmdp-bio-lora-s125-unlearn-{method}.yml"
            ).read_text()
        )
        relearn = yaml.safe_load(
            (
                EXAMPLE
                / "configs/relearn/z7b"
                / f"wmdp-bio-lora-s125-unlearn-{method}-relearn.yml"
            ).read_text()
        )

        assert float(unlearn["learning_rate"]) == 7.5e-5
        if method != "gd":
            assert "GDSAM" in unlearn["trainer_cls"]
            assert "rho" not in unlearn
        assert unlearn["max_steps"] == 125
        assert unlearn["save_steps"] == 10
        assert unlearn["save_total_limit"] == 13
        assert unlearn["output_dir"].endswith(unlearn_id)
        assert float(relearn["learning_rate"]) == 1e-4
        assert relearn["max_steps"] == 125
        assert relearn["save_steps"] == 10
        assert relearn["save_total_limit"] == 13
        assert relearn["base_model"].endswith(f"{unlearn_id}/merged")
        assert relearn["output_dir"].endswith(relearn_id)

        unlearn_stage = plan.stage_index[f"train-{unlearn_id}"]
        relearn_stage = plan.stage_index[f"train-{relearn_id}"]
        assert relearn_stage.depends_on == (unlearn_stage.id,)


@pytest.mark.skip(reason="legacy config replaced by canonical lineage")
def test_z7b_fft_s125_canonical_pipeline_saves_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-fft-s125.yml")
    plan = pipeline.plan()

    assert len(pipeline.selected_models) == 2
    assert len(plan.stages) == 6
    methods = {
        "gd": ("configs.training.trainers.gd.GradDiffTrainer", 2.5e-6),
        "gd-sam-rho1e-2": (
            "configs.training.trainers.gd_sam.GDSAMRho1eMinus2Trainer",
            3.28e-6,
        ),
    }
    for method, (trainer_cls, learning_rate) in methods.items():
        model_id = f"z7b-wmdp-bio-fft-s125-unlearn-{method}"
        config = yaml.safe_load(
            (
                EXAMPLE
                / "configs/unlearn/z7b"
                / f"wmdp-bio-fft-s125-unlearn-{method}.yml"
            ).read_text()
        )

        assert config["trainer_cls"] == trainer_cls
        assert "rho" not in config
        assert "adapter" not in config
        assert float(config["learning_rate"]) == learning_rate
        assert config["max_steps"] == 125
        assert config["save_strategy"] == "steps"
        assert config["save_steps"] == 10
        assert config["save_total_limit"] == 13
        assert config["output_dir"] == f"artifacts/tamia/models/{model_id}"

    registry = yaml.safe_load((EXAMPLE / "configs/registries/evals.yml").read_text())[
        "evaluations"
    ]
    for task in ("bio-mcqa", "mmlu-no-bio"):
        evaluation = registry[f"{task}-ckpts-max125-freq10-tamia"]
        assert evaluation["args"][-2:] == ["10", "125"]
        assert "--array=1-13" in evaluation["sbatch_args"]
        assert not any(
            argument.startswith("--partition=")
            for argument in evaluation["sbatch_args"]
        )


@pytest.mark.skip(reason="legacy config replaced by canonical lineage")
def test_z7b_fft_canonical_relearning_uses_authors_matched_gd_sam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-fft-s125-relearn.yml")
    plan = pipeline.plan()

    unlearn_id = "z7b-wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-2"
    model_id = f"{unlearn_id}-relearn"
    config = yaml.safe_load(
        (
            EXAMPLE
            / "configs/relearn/z7b/wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-2-relearn.yml"
        ).read_text()
    )

    assert pipeline.selected_models == (
        "z7b-wmdp-bio-fft-s125-unlearn-gd-relearn",
        model_id,
    )
    assert len(plan.stages) == 8
    assert config["base_model"] == f"artifacts/tamia/models/{unlearn_id}"
    assert float(config["learning_rate"]) == 1e-5
    assert config["micro_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 1
    assert config["seed"] == 1
    assert config["max_steps"] == 16
    assert config["warmup_steps"] == 0
    assert config["save_strategy"] == "steps"
    assert config["save_steps"] == 4
    assert config["save_total_limit"] == 4
    assert config["output_dir"] == f"artifacts/tamia/models/{model_id}"
    assert plan.stage_index[f"train-{model_id}"].depends_on == (f"train-{unlearn_id}",)

    gd = yaml.safe_load(
        (
            EXAMPLE / "configs/relearn/z7b/wmdp-bio-fft-s125-unlearn-gd-relearn.yml"
        ).read_text()
    )
    assert gd["base_model"].endswith("z7b-wmdp-bio-fft-s125-unlearn-gd")
    assert gd["gradient_accumulation_steps"] == 1
    assert gd["max_steps"] == 16
    assert gd["save_steps"] == 4
    assert gd["seed"] == 1


@pytest.mark.skip(reason="retired experiment removed")
def test_fft_gd_lora_tar_pipeline_and_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-fft-s125-gd-lora-tar.yml")
    plan = pipeline.plan()

    stage_ids = [stage.id for stage in plan.stages]
    assert stage_ids[:3] == [
        "train-z7b-wmdp-bio-fft-s125-unlearn-gd",
        "train-z7b-wmdp-bio-fft-s125-unlearn-gd-lora-tar",
        "train-z7b-wmdp-bio-fft-s125-unlearn-gd-lora-tar-relearn",
    ]
    assert len(plan.stages) == 7

    tar_config = yaml.safe_load(
        (
            EXAMPLE / "configs/unlearn/z7b/wmdp-bio-fft-s125-unlearn-gd-lora-tar.yml"
        ).read_text()
    )
    relearn_config = yaml.safe_load(
        (
            EXAMPLE
            / "configs/relearn/z7b/wmdp-bio-fft-s125-unlearn-gd-lora-tar-relearn.yml"
        ).read_text()
    )
    assert tar_config["trainer_cls"].endswith(".TAR2p5e6Trainer")
    assert tar_config["base_model"].endswith("fft-s125-unlearn-gd")
    assert tar_config["lora_r"] == tar_config["lora_alpha"] == 8
    assert tar_config["lora_dropout"] == 0.0
    assert tar_config["micro_batch_size"] == 2
    assert tar_config["gradient_accumulation_steps"] == 4
    assert tar_config["gradient_checkpointing"] is False
    assert all(
        tar_config[key] is False
        for key in ("lora_mlp_kernel", "lora_qkv_kernel", "lora_o_kernel")
    )
    assert relearn_config["base_model"].endswith("lora-tar/merged")
    assert relearn_config["lora_r"] == relearn_config["lora_alpha"] == 8
    assert relearn_config["lora_dropout"] == pytest.approx(0.05)
    assert relearn_config["micro_batch_size"] == 1
    assert relearn_config["gradient_accumulation_steps"] == 4
    for config in (tar_config, relearn_config):
        assert float(config["learning_rate"]) == pytest.approx(2.5e-6)
        assert config["max_steps"] == 125
        assert config["sequence_len"] == 512
        assert config["optimizer"] == "adamw_torch"
        assert config["weight_decay"] == 0.0
        assert config["lr_scheduler"] == "linear"
        assert config["warmup_steps"] == 12
        assert config["save_steps"] == 10
        assert config["save_total_limit"] == 13
        assert config["seed"] == 42

    tar_stage = plan.stage_index["train-z7b-wmdp-bio-fft-s125-unlearn-gd-lora-tar"]
    relearn_stage = plan.stage_index[
        "train-z7b-wmdp-bio-fft-s125-unlearn-gd-lora-tar-relearn"
    ]
    assert tar_stage.args[-1] == "--merge"
    assert tar_stage.depends_on == ("train-z7b-wmdp-bio-fft-s125-unlearn-gd",)
    assert relearn_stage.depends_on == (tar_stage.id,)
    for stage in plan.stages:
        if stage.id.startswith("eval-"):
            assert "--array=1-13" in stage.sbatch_args
            assert "--gpus-per-node=h100:4" in stage.sbatch_args


@pytest.mark.skip(reason="retired experiment removed")
def test_optml_relearning_matches_released_models_and_includes_checkpoint_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/optml-relearning.yml")
    plan = pipeline.plan()

    revisions = {
        "OPTML-Group-GradDiff-WMDP": (
            "OPTML-Group/GradDiff-WMDP",
            "0695dccbb30a0986d20f211bc8bdea9d55ec5cc6",
            "optml-grad-diff-wmdp.yml",
        ),
        "OPTML-Group-GradDiff-SAM-WMDP": (
            "OPTML-Group/GradDiff-SAM-WMDP",
            "e22b488a59005092ec1ec9e217ede68a183248be",
            "optml-grad-diff-sam-wmdp.yml",
        ),
    }
    assert len(plan.stages) == 8
    models = yaml.safe_load((EXAMPLE / "configs/registries/models.yml").read_text())[
        "models"
    ]
    for model_id, (released_model, revision, config_name) in revisions.items():
        config = yaml.safe_load((EXAMPLE / "configs/relearn" / config_name).read_text())
        assert config["base_model"] == released_model
        assert config["revision_of_model"] == revision
        assert models[model_id]["released_model_revision"] == revision
        assert float(config["learning_rate"]) == 1e-5
        assert config["micro_batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 4
        assert config["max_steps"] == 4
        assert config["seed"] == 42
        assert config["warmup_steps"] == 0
        assert config["save_steps"] == 1
        assert config["output_dir"] == f"artifacts/tamia/models/{model_id}"

    registry = yaml.safe_load((EXAMPLE / "configs/registries/evals.yml").read_text())[
        "evaluations"
    ]
    for evaluation_id in pipeline.selected_evaluations:
        evaluation = registry[evaluation_id]
        assert "--array=0-4" in evaluation["sbatch_args"]
        assert evaluation["script"].endswith("eval_trajectory.sbatch")

    script = (EXAMPLE / "scripts/slurm/tamia/eval_trajectory.sbatch").read_text()
    assert "if (( step == 0 ))" in script
    assert (
        'checkpoint="$HF_HOME/hub/models--$cache_name/snapshots/$released_revision"'
        in script
    )
    assert '[[ ! -f "$checkpoint/config.json" ]]' in script
    assert 'checkpoint="$trained_model/checkpoint-$step"' in script


def test_authors_relearning_wrapper_accepts_local_unlearned_model() -> None:
    script = (ROOT / "examples/sam/WMDP/run_tamia_local_relearning.sbatch").read_text()

    assert "model_source=${1:?missing local model path}" in script
    assert "model_slug=${2:?missing artifact slug}" in script
    assert (
        'cd "${SLURM_SUBMIT_DIR:?submit this job from the authors WMDP directory}"'
        in script
    )
    assert "steps=(5 10 15)" in script
    assert "example_counts=(20 40 60)" in script
    assert '--overall.model_name "$model_source"' in script
    assert "--overall.seed 1" in script
    assert "--dataset.dataset_seed 42" in script
    assert "--unlearn.lr 1e-5" in script
    assert 'mkdir -p "$SLURM_TMPDIR/relearn-run"' in script
    assert 'ln -s "$SLURM_TMPDIR/relearn-run" "$run_root"' in script
    assert 'rm -rf "$run_root"' in script


def test_authors_unlearning_wrapper_uses_node_local_temporary_checkpoint() -> None:
    script = (ROOT / "examples/sam/WMDP/run_tamia_unlearning.sbatch").read_text()

    assert "method=${1:?missing method: GradDiff or GradDiff+SAM}" in script
    assert (
        'cd "${SLURM_SUBMIT_DIR:?submit this job from the authors WMDP directory}"'
        in script
    )
    assert 'mkdir -p "$SLURM_TMPDIR/unlearn_checkpoint"' in script
    assert (
        'ln -s "$SLURM_TMPDIR/unlearn_checkpoint" '
        '"$artifact_root/unlearn_checkpoint"'
    ) in script
    assert '--config-file "configs/unlearn/wmdp/$method.json"' in script
    assert 'args+=("$@")' in script


def test_relearning_formats_wmdp_document() -> None:
    pytest.importorskip("axolotl")
    strategy = load_script(
        "configs/training/data/wmdp_completion.py", "wmdp_bio_forget_strategy"
    )
    document, prompt, response = (
        strategy.WmdpBioCompletionStrategy.parse_instruction_fields(
            None, {"title": "Title", "abstract": "Abstract", "text": "Text"}
        )
    )
    assert (document, prompt, response) == ("Title\n\nAbstract\n\nText", "", "")


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
    script = (EXAMPLE / "scripts/slurm/mila/eval_mmlu_no_bio_single.sbatch").read_text()
    assert 'export HF_DATASETS_CACHE="$HF_HOME/datasets"' in script
    assert 'HF_DATASETS_CACHE="$SLURM_TMPDIR' not in script
    assert "--tasks mmlu_no_bio" in script
    assert "--num_fewshot 0" in script
    assert "--batch_size 32" in script
    assert "result_root=${4:?missing result root}" in script
    assert 'output="$result_root/$model_id/mmlu-no-bio"' in script


def test_mmlu_no_bio_trajectory_evaluator_accepts_last_step() -> None:
    script = (EXAMPLE / "scripts/slurm/mila/eval_mmlu_no_bio_ckpts.sbatch").read_text()
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "step=$((${SLURM_ARRAY_TASK_ID" in script
    assert "* checkpoint_frequency))" in script
    assert "last_step=${6:?missing last step}" in script
    assert "(( step > last_step ))" in script
    assert "step=$last_step" in script
    assert 'checkpoint="$adapter_name_or_path/checkpoint-$step"' in script
    assert 'output="$result_root/$model_id/checkpoint-$step/mmlu-no-bio"' in script
    assert "--tasks mmlu_no_bio" in script


def test_tamia_local_trajectory_series_evaluates_saved_checkpoints() -> None:
    script = (
        EXAMPLE / "scripts/slurm/tamia/eval_local_trajectory_series.sbatch"
    ).read_text()
    assert 'steps=("$@")' in script
    assert "steps=(5 10 15)" in script
    assert 'for step in "${steps[@]}"' in script
    assert 'checkpoint="$trained_model/checkpoint-$step"' in script
    assert 'output="$result_root/$model_id/checkpoint-$step/$output_name"' in script
    assert '--tasks "$task"' in script


def test_final_adapter_evaluator_uses_direct_adapter_without_array() -> None:
    script = (
        EXAMPLE / "scripts/slurm/mila/eval_wmdp_bio_mcqa_single.sbatch"
    ).read_text()
    assert "adapter_name_or_path=${3:?missing adapter name or path}" in script
    assert "result_root=${4:?missing result root}" in script
    assert 'if [[ "$adapter_name_or_path" != "-" ]]' in script
    assert 'output="$result_root/$model_id/wmdp-bio-robust"' in script
    assert "checkpoint-" not in script
    assert "SLURM_ARRAY_TASK_ID" not in script
    assert "--batch_size 32" in script
    assert "--num_fewshot 0" in script
