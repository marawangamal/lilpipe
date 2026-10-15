import importlib.util
import json
import random
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
def sampling_module():
    pytest.importorskip("torch")
    return load_script("src/trainers/sampling.py", "tamper_sampling")


@pytest.fixture(scope="module")
def src_gd_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/gd.py", "src_gd_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def src_gd_sam_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/gd_sam.py", "src_gd_sam_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def circuit_breaker_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script(
            "src/trainers/circuit_breaker.py", "circuit_breaker_training"
        )
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def rmu_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/rmu.py", "rmu_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def weight_dist_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/weight_distance.py", "weight_dist_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def gd_weight_dist_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script(
            "src/trainers/gd_weight_distance.py",
            "gd_weight_dist_training",
        )
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


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
def ao_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script(
            "src/trainers/activation_orthogonality.py",
            "activation_orthogonality_training",
        )
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def gd_ao_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/gd_ao.py", "gd_ao_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


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
    assert 'export UV_CACHE_DIR="$SCRATCH/.cache/uv"' in script
    assert 'install_lock="$UV_CACHE_DIR/.lilpipe-install.lockdir"' in script
    assert 'until mkdir "$install_lock" 2>/dev/null; do sleep 2; done' in script
    assert "trap 'rmdir \"$install_lock\"' EXIT" in script
    assert 'export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"' in script
    assert 'export PATH="$HOME/.local/bin:$PATH"' in script
    assert 'mkdir -p "$WANDB_DIR"' in script
    assert "uv sync --offline --frozen --group train" in script
    assert 'source "$UV_PROJECT_ENVIRONMENT/bin/activate"' in script
    assert 'axolotl train "$config" "$@"' in script
    assert 'if [[ ${1:-} == "--merge" ]]' in script
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


def test_circuit_breaker_schedule_matches_original_objective(
    circuit_breaker_training_module,
) -> None:
    trainer = object.__new__(circuit_breaker_training_module.CircuitBreakerTrainer)
    trainer.retain_coefficient = 10.0
    trainer.reroute_coefficient = 23.0
    trainer.orthogonalization_coefficient = 5.0
    trainer.total_microsteps = 125

    trainer.microstep = 0
    assert trainer.coefficients() == pytest.approx((1.0, 23.0, 0.0))
    trainer.microstep = 124
    assert trainer.coefficients() == pytest.approx((10.0, 17.25, 5.0))


def test_circuit_breaker_reference_parameters_are_swapped_and_restored(
    circuit_breaker_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    trainer = object.__new__(circuit_breaker_training_module.CircuitBreakerTrainer)
    trainer.model = model
    trainer._theta_ref = {"weight": torch.full_like(model.weight, 7.0)}

    with trainer.reference_parameters():
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 7.0))
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2.0))


def test_circuit_breaker_losses_preserve_layer_and_padding_contracts(
    circuit_breaker_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor([[1.0, 100.0], [3.0, 5.0]])
    mask = torch.tensor([[1, 0], [1, 1]])
    assert circuit_breaker_training_module.masked_mean(values, mask).item() == (
        pytest.approx(3.0)
    )

    activations = torch.tensor(
        [
            [[1.0, 0.0], [10.0, 10.0]],
            [[1.0, 0.0], [20.0, 20.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 0], [1, 0]])
    loss = circuit_breaker_training_module.orthogonalization_loss(
        activations, attention_mask
    )
    assert loss.item() == pytest.approx(1.0)


def test_circuit_breaker_sam_accepts_scheduled_zero_gradient(
    circuit_breaker_training_module,
) -> None:
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    trainer = object.__new__(circuit_breaker_training_module.CircuitBreakerSAMTrainer)
    trainer.optimizer = SimpleNamespace()
    trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
    trainer.current_gradient_accumulation_steps = 1
    trainer.compute_loss_context_manager = nullcontext
    trainer.log = lambda metrics: None
    trainer.compute_cb_loss = lambda model, inputs, **kwargs: (
        model.weight.sum() * 0,
        {},
    )

    loss = trainer._sam_training_step(model, {})

    assert loss.item() == 0
    torch.testing.assert_close(model.weight.grad, torch.zeros_like(model.weight))


def test_rmu_activation_loss_ignores_padding(rmu_training_module) -> None:
    torch = pytest.importorskip("torch")
    activation = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
    target = torch.zeros(1, 1, 2)
    attention_mask = torch.tensor([[1, 0]])

    loss = rmu_training_module.RMUTrainer.activation_loss(
        activation, target, attention_mask
    )

    assert loss.item() == pytest.approx(5.0)


def test_rmu_control_vector_has_fixed_norm_and_seed(rmu_training_module) -> None:
    torch = pytest.importorskip("torch")
    activation = torch.zeros(2, 3, 8)
    first = object.__new__(rmu_training_module.RMUTrainer)
    first.steering_coefficient = 6.5
    first.control_seed = 42
    first._control_vector = None
    second = object.__new__(rmu_training_module.RMUTrainer)
    second.steering_coefficient = 6.5
    second.control_seed = 42
    second._control_vector = None

    control = first.control_vector(activation)

    assert control.shape == (1, 1, 8)
    assert control.float().norm().item() == pytest.approx(6.5)
    torch.testing.assert_close(control, second.control_vector(activation))


def test_weight_distance_trainer_combines_retain_and_distance_losses(
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
    trainer = object.__new__(weight_dist_training_module.WeightDistanceTrainer)
    trainer.model = model
    trainer.coefficient_weight_dist = 2.0
    trainer._theta_ref = {"weight": torch.tensor(0.0)}
    trainer._weight_distance_logged_step = None
    trainer.state = SimpleNamespace(global_step=0)
    logged = []
    trainer.log = logged.append
    inputs = {
        "input_ids": torch.ones(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
        "labels": torch.ones(1, 1, dtype=torch.long),
    }

    loss, returned_outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    assert loss.item() == pytest.approx(0.0)
    assert returned_outputs.loss.item() == pytest.approx(4.0)
    assert logged == [
        {
            "loss_weight_dist": pytest.approx(-4.0),
            "loss_retain": pytest.approx(4.0),
        }
    ]
    loss.backward()
    assert model.weight.grad.item() == pytest.approx(2.0)


def test_reference_parameter_loader_strips_framework_wrapper_names(
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

    reference_parameters = weight_dist_training_module.load_reference_parameters(
        {runtime_name: torch.nn.Parameter(torch.zeros_like(expected))}, tmp_path
    )

    assert reference_parameters.keys() == {runtime_name}
    torch.testing.assert_close(reference_parameters[runtime_name], expected)


def test_gd_weight_distance_forwards_gd_coefficients(
    gd_weight_dist_training_module, monkeypatch
) -> None:
    calls = []

    def init(_, *args, forget_coefficient, retain_coefficient, **kwargs):
        calls.append((args, forget_coefficient, retain_coefficient, kwargs))

    monkeypatch.setattr(
        gd_weight_dist_training_module.GradDiffTrainer, "__init__", init
    )
    trainer = gd_weight_dist_training_module.GDWeightDistanceTrainer(
        "model",
        coefficient_forget=2.0,
        coefficient_retain=3.0,
        coefficient_weight_dist=1e6,
        reference_model_name_or_path="reference-model",
        sampling="authors",
    )

    assert calls == [(("model",), 2.0, 3.0, {"sampling": "authors"})]
    assert trainer.coefficient_weight_dist == 1e6
    assert trainer.reference_model_name_or_path == "reference-model"


def test_gd_weight_distance_requires_reference_model(
    gd_weight_dist_training_module,
) -> None:
    with pytest.raises(TypeError, match="reference_model_name_or_path"):
        gd_weight_dist_training_module.GDWeightDistanceTrainer("model")


def test_gd_weight_distance_loads_explicit_reference_model(
    gd_weight_dist_training_module, monkeypatch
) -> None:
    import sys

    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    expected = {"weight": torch.full_like(model.weight, 7)}
    calls = []

    def load(parameters, model_name_or_path):
        calls.append((parameters, model_name_or_path))
        return expected

    mixin_module = sys.modules[
        gd_weight_dist_training_module.WeightDistanceMixin.__module__
    ]
    monkeypatch.setattr(mixin_module, "load_reference_parameters", load)
    trainer = object.__new__(gd_weight_dist_training_module.GDWeightDistanceTrainer)
    trainer.model = model
    trainer.reference_model_name_or_path = "reference-model"
    trainer._theta_ref = None

    assert trainer.theta_ref == expected
    assert calls == [({"weight": model.weight}, "reference-model")]


def test_gd_weight_distance_uses_rms_distance(
    gd_weight_dist_training_module, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))

    outputs = SimpleNamespace(loss=torch.tensor(0.0))
    zero_grads = (torch.zeros_like(model.weight),)
    monkeypatch.setattr(
        gd_weight_dist_training_module.GradDiffTrainer,
        "compute_retain_loss",
        lambda *args, **kwargs: (torch.tensor(7.0), zero_grads, outputs),
    )
    monkeypatch.setattr(
        gd_weight_dist_training_module.GradDiffTrainer,
        "compute_forget_loss",
        lambda *args, **kwargs: (torch.tensor(3.0), zero_grads),
    )
    trainer = object.__new__(gd_weight_dist_training_module.GDWeightDistanceTrainer)
    trainer.coefficient_weight_dist = 2.0
    trainer.model = model
    trainer._theta_ref = {"weight": torch.zeros_like(model.weight)}
    trainer._weight_distance_logged_step = None
    trainer.state = SimpleNamespace(global_step=0)
    logged = []
    trainer.log = logged.append

    loss, returned_outputs = trainer.compute_loss(model, {}, return_outputs=True)
    expected_distance = (25 / 2 + 1e-8) ** 0.5

    assert loss.item() == pytest.approx(10 - 2 * expected_distance)
    assert returned_outputs is outputs
    assert logged == [
        {
            "loss_weight_dist": pytest.approx(-2 * expected_distance),
            "loss_retain": pytest.approx(7.0),
            "loss_forget": pytest.approx(3.0),
            "grad_norm_weight_dist": pytest.approx(2**0.5),
            "grad_norm_retain": pytest.approx(0.0),
            "grad_norm_forget": pytest.approx(0.0),
        }
    ]


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


def test_src_grad_diff_logs_component_losses_and_gradient_norms(
    src_gd_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forget_weight = torch.nn.Parameter(torch.tensor(2.0))
            self.retain_weight = torch.nn.Parameter(torch.tensor(3.0))

        def forward(self, input_ids, attention_mask, labels):
            del attention_mask, labels
            ids = input_ids.flatten().tolist()
            weight = self.forget_weight if ids == [2, 4] else self.retain_weight
            return SimpleNamespace(loss=input_ids.float().mean() * weight)

    model = Model()
    trainer = object.__new__(src_gd_training_module.GradDiffTrainer)
    trainer.forget_coefficient = 2.0
    trainer.retain_coefficient = 1.0
    logged = []
    trainer.log = logged.append
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    loss = trainer.compute_loss(model, inputs)
    loss.backward()

    assert loss.item() == pytest.approx(12.0)
    assert logged == [
        {
            "loss_forget": pytest.approx(12.0),
            "loss_retain": pytest.approx(24.0),
            "grad_norm_forget": pytest.approx(6.0),
            "grad_norm_retain": pytest.approx(8.0),
        }
    ]
    assert model.forget_weight.grad.item() == pytest.approx(-6.0)
    assert model.retain_weight.grad.item() == pytest.approx(8.0)


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


def test_src_gd_sam_logs_component_losses_and_gradient_norms(
    src_gd_sam_training_module,
) -> None:
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, input_ids, attention_mask, labels):
            del attention_mask, labels
            return SimpleNamespace(loss=input_ids.float().mean() * self.weight.square())

    class Accelerator:
        @staticmethod
        def backward(loss):
            loss.backward()

    model = Model()
    trainer = object.__new__(src_gd_sam_training_module.GDSAMTrainer)
    trainer.optimizer = None
    trainer.accelerator = Accelerator()
    trainer.current_gradient_accumulation_steps = 1
    trainer.compute_loss_context_manager = nullcontext
    trainer.rho = 0.01
    trainer.forget_coefficient = 2.0
    trainer.retain_coefficient = 1.0
    trainer.optml_parity = False
    logged = []
    trainer.log = logged.append
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    loss = trainer._sam_training_step(model, inputs)

    assert loss.item() == pytest.approx(-6 * 1.99**2 + 8 * 2**2)
    assert logged == [
        {
            "loss_forget": pytest.approx(6 * 1.99**2),
            "loss_retain": pytest.approx(8 * 2**2),
            "grad_norm_forget": pytest.approx(12 * 1.99),
            "grad_norm_retain": pytest.approx(16 * 2),
            "grad_norm_forget_at_perturbation": pytest.approx(6 * 2),
        }
    ]
    assert model.weight.grad.item() == pytest.approx(-12 * 1.99 + 16 * 2)


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


def test_optml_sam_norm_preserves_original_precision(
    src_gd_sam_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    gradients = {
        "first": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        "second": torch.tensor([0.0, 12.0], dtype=torch.bfloat16),
    }

    perturbation = src_gd_sam_training_module.optml_sam_perturbation(gradients, 0.01)
    norm = torch.stack([gradient.norm(2) for gradient in perturbation.values()]).norm(2)

    assert all(value.dtype == torch.bfloat16 for value in perturbation.values())
    assert norm.item() == pytest.approx(0.01, abs=1e-4)


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


def test_optml_sampler_matches_paired_random_sampling(sampling_module) -> None:
    torch = pytest.importorskip("torch")

    class TaggedDataset:
        sources = [1, 1, 1, 1, 0, 0]

        def __getitem__(self, key):
            assert key == "is_forget"
            return self.sources

    random.seed(42)
    torch.manual_seed(42)
    expected_forget = list(torch.utils.data.RandomSampler(range(4)))
    expected_retain = [random.choice([4, 5]) for _ in range(4)]

    random.seed(42)
    torch.manual_seed(42)
    actual = list(sampling_module.OptmlMixedSourceSampler(TaggedDataset(), 4))

    assert actual == [
        *expected_forget[:2],
        *expected_retain[:2],
        *expected_forget[2:],
        *expected_retain[2:],
    ]
    assert len(actual) == 8


def test_rmu_defense_scale_pipeline_reuses_5000_step_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/rmu-defense-scales.yml")
    plan = pipeline.plan()

    defense_stages = [
        stage
        for stage in plan.stages
        if stage.id.startswith("train-") and stage.id.endswith("defense-steps-5000")
    ]
    assert len(defense_stages) == 3

    for defense_stage in defense_stages:
        config = yaml.safe_load((EXAMPLE / defense_stage.args[0]).read_text())
        assert config["max_steps"] == 5000
        assert config["save_steps"] == 1000
        assert config["save_total_limit"] == 5

    relearn_stages = [
        stage
        for stage in plan.stages
        if stage.id.startswith("train-") and stage.id.endswith("--relearn-fft-long")
    ]
    assert len(relearn_stages) == 6
    for relearn_stage in relearn_stages:
        config = yaml.safe_load((EXAMPLE / relearn_stage.args[0]).read_text())
        assert config["max_steps"] == 125
        assert config["save_steps"] == 10
        assert config["save_total_limit"] == 12
        assert config["base_model"].endswith(("checkpoint-1000", "checkpoint-5000"))
        assert len(relearn_stage.depends_on) == 1


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


def test_ao_trainer_loss_is_retain_ce_plus_off_diagonal_cosine(
    ao_training_module,
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
    trainer = object.__new__(ao_training_module.AOTrainer)
    trainer.target_layers = (0, 1)
    trainer.retain_coefficient = 1.0
    trainer.activation_orthogonality_coefficient = 1.0
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert loss.item() == pytest.approx(16.0 + 0.96)
    assert outputs["loss_retain"].item() == pytest.approx(16.0)
    assert outputs["loss_activation_orthogonality"].item() == pytest.approx(0.96)
    assert model.calls == ["retain", "forget"]
    loss.backward()
    assert model.retain_weight.grad.item() == pytest.approx(8.0)
    assert model.forget_rows.grad is not None
    assert model.forget_rows.grad.abs().sum() > 0


def test_gd_ao_loss_has_retain_forget_and_orthogonality_terms(
    gd_ao_training_module,
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
    trainer = object.__new__(gd_ao_training_module.GDAOTrainer)
    trainer.target_layers = (0, 1)
    trainer.retain_coefficient = 2.0
    trainer.forget_coefficient = 3.0
    trainer.activation_orthogonality_coefficient = 4.0
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
    assert outputs["loss_activation_orthogonality"].item() == pytest.approx(4 * 0.96)


def test_gd_ao_targets_all_transformer_layers_by_default(
    gd_ao_training_module, monkeypatch
) -> None:
    def init(trainer, *args, **kwargs):
        trainer.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=3))
        trainer.retain_coefficient = kwargs.get("retain_coefficient", 1.0)
        trainer.forget_coefficient = kwargs.get("forget_coefficient", 1.0)

    monkeypatch.setattr(gd_ao_training_module.GradDiffTrainer, "__init__", init)
    trainer = gd_ao_training_module.GDAOTrainer()

    assert trainer.target_layers == (0, 1, 2)
    assert trainer.retain_coefficient == 1.0
    assert trainer.forget_coefficient == 1.0
    assert trainer.activation_orthogonality_coefficient == 1.0


def test_gd_ao_uses_rmu_balanced_sampler(gd_ao_training_module) -> None:
    class TaggedDataset:
        def __getitem__(self, key):
            assert key == "is_forget"
            return [0, 1, 1, 0] * 2

    trainer = SimpleNamespace(
        train_dataset=TaggedDataset(),
        args=SimpleNamespace(per_device_train_batch_size=4, data_seed=7, seed=42),
    )
    sampler = gd_ao_training_module.GDAOTrainer._get_train_sampler(trainer)

    assert sampler.__class__.__name__ == "MixedSourceSampler"
    indices = list(sampler)
    for start in (0, 4):
        assert (
            sum(
                trainer.train_dataset["is_forget"][index]
                for index in indices[start : start + 4]
            )
            == 2
        )


def test_ao_trainer_ignores_padding_tokens_in_forget_pool(
    ao_training_module,
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
    trainer = object.__new__(ao_training_module.AOTrainer)
    trainer.target_layers = (1,)
    trainer.retain_coefficient = 1.0
    trainer.activation_orthogonality_coefficient = 1.0
    inputs = {
        "input_ids": torch.tensor([[5, 0, 0], [1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 0]]),
        "labels": torch.tensor([[5, 0, 0], [1, 2, 3], [4, 5, 6]]),
        "is_forget": torch.tensor([False, True, True]),
    }
    loss = trainer.compute_loss(model, inputs)
    assert loss.item() == pytest.approx(5.0 + 1.0 / 1.25**0.5)


@pytest.mark.parametrize("sources", [[1, 1, 1], [0, 0]])
def test_ao_trainer_requires_both_sources(ao_training_module, sources) -> None:
    torch = pytest.importorskip("torch")
    trainer = object.__new__(ao_training_module.AOTrainer)
    with pytest.raises(ValueError, match="forget and retain"):
        trainer.compute_loss(None, {"is_forget": torch.tensor(sources).bool()})


def test_ao_trainer_requires_two_forget_rows(ao_training_module) -> None:
    torch = pytest.importorskip("torch")
    trainer = object.__new__(ao_training_module.AOTrainer)
    with pytest.raises(ValueError, match="at least two forget"):
        trainer.compute_loss(None, {"is_forget": torch.tensor([False, True])})


def test_ao_trainer_validates_target_layers(
    ao_training_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(
        ao_training_module.AxolotlTrainer,
        "__init__",
        lambda self, *args, **kwargs: None,
    )
    trainer = object.__new__(ao_training_module.AOTrainer)
    trainer.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
    with pytest.raises(ValueError, match="target_layers"):
        ao_training_module.AOTrainer.__init__(trainer, target_layers=(0, 0))
    with pytest.raises(ValueError, match="target_layers"):
        ao_training_module.AOTrainer.__init__(trainer, target_layers=(1, 2))
    with pytest.raises(ValueError, match="target_layers"):
        ao_training_module.AOTrainer.__init__(trainer, target_layers=())
    with pytest.raises(ValueError, match="activation_orthogonality_coefficient"):
        ao_training_module.AOTrainer.__init__(
            trainer, activation_orthogonality_coefficient=-1.0, target_layers=(0, 1)
        )
    ao_training_module.AOTrainer.__init__(trainer, target_layers=(0, 1))
    assert trainer.target_layers == (0, 1)
    trainer.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=32))
    ao_training_module.AOTrainer.__init__(trainer)
    assert trainer.target_layers == (5, 10, 15, 20, 25, 30)


def test_ao_trainer_sampler_balances_each_microbatch(ao_training_module) -> None:
    class TaggedDataset:
        def __getitem__(self, key):
            assert key == "is_forget"
            return [0, 1, 1, 0] * 2

    trainer = SimpleNamespace(
        train_dataset=TaggedDataset(),
        args=SimpleNamespace(per_device_train_batch_size=4, data_seed=None, seed=42),
    )
    sampler = ao_training_module.AOTrainer._get_train_sampler(trainer)
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


def test_alignment_pipeline_and_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/main-alignment.yml")
    plan = pipeline.plan()
    sft = "llama2-7b-beavertail--align-lora-sft"
    booster = "llama2-7b-beavertail--align-lora-booster-l5-rho1e-1"
    assert [stage.id for stage in plan.stages] == [
        f"train-{sft}",
        f"train-{booster}",
        f"train-{sft}--relearn-lora",
        f"train-{booster}--relearn-lora",
        f"eval-beavertail-alignment-{sft}--relearn-lora",
        f"eval-beavertail-alignment-{booster}--relearn-lora",
    ]
    assert plan.stage_index[f"train-{sft}--relearn-lora"].depends_on == (
        f"train-{sft}",
    )
    rendered = plan.render()
    assert "scripts/slurm/mila/train.sbatch" in rendered
    assert "scripts/slurm/mila/relearn_alignment.sbatch" in rendered
    assert "scripts/slurm/mila/eval_alignment.sbatch" in rendered
    assert "--dependency=afterok:dry-run-train-" in rendered

    config_paths = [
        f"configs/stages-alignment/0_alignment/{sft}/selected.yml",
        f"configs/stages-alignment/0_alignment/{booster}/selected.yml",
        f"configs/stages-alignment/1_relearn/{sft}--relearn-lora/selected.yml",
        f"configs/stages-alignment/1_relearn/{booster}--relearn-lora/selected.yml",
    ]
    configs = [yaml.safe_load((EXAMPLE / path).read_text()) for path in config_paths]
    for config in configs:
        assert config["lora_r"] == 32
        assert config["lora_alpha"] == 4
        assert config["lora_dropout"] == 0.0
        assert config["lora_target_modules"] == ["q_proj", "k_proj", "v_proj"]
        assert config["seed"] == config["data_seed"] == 42
        assert config["sequence_len"] == 256
        assert config["max_steps"] == 500
        assert config["optimizer"] == "adamw_torch"
        assert config["weight_decay"] == pytest.approx(0.1)
        assert config["lr_scheduler"] == "constant"
        assert config["warmup_steps"] == 0
        for dataset in config["datasets"]:
            assert dataset["revision"] == "11ea63756195527164a9aa2850f7544bd251eab1"

    assert configs[0]["micro_batch_size"] == 10
    assert configs[0]["base_model"] == "meta-llama/Llama-2-7b-hf"
    assert float(configs[0]["learning_rate"]) == pytest.approx(5e-4)
    assert configs[1]["micro_batch_size"] == 20
    assert configs[1]["base_model"] == "meta-llama/Llama-2-7b-hf"
    assert configs[1]["custom_trainer_args"] == {
        "coefficient": 5.0,
        "radius": 0.1,
    }
    for config in configs[2:]:
        assert config["micro_batch_size"] == 10
        assert float(config["learning_rate"]) == pytest.approx(1e-5)
        assert config["save_steps"] == 50
        assert config["save_total_limit"] == 10
        assert config["auto_resume_from_checkpoints"] is False
        assert config["resume_from_checkpoint"] is None
        assert config["base_model"].startswith("artifacts/mila/models/")
        assert config["base_model"].endswith("-merged/merged")
        assert "lora_model_dir" not in config

    relearn_script = (
        EXAMPLE / "scripts/slurm/mila/relearn_alignment.sbatch"
    ).read_text()
    assert "#SBATCH --time=03:00:00" in relearn_script
    assert 'axolotl merge-lora "$alignment_config"' in relearn_script
    assert '--lora-model-dir "$aligned_adapter"' in relearn_script
    assert '--output-dir "$merged_base"' in relearn_script
    assert 'merged_model="$merged_base/merged"' in relearn_script
    assert 'mv "$merged_base/.merge-complete" "$merged_model/.merge-complete"' in (
        relearn_script
    )
    assert 'touch "$merged_model/.merge-complete"' in relearn_script
    assert 'axolotl train "$attack_config" "$@"' in relearn_script

    for script_name in (
        "train.sbatch",
        "eval_alignment.sbatch",
        "eval_alignment_trajectory.sbatch",
    ):
        script = (EXAMPLE / "scripts/slurm/mila" / script_name).read_text()
        assert "#SBATCH --time=03:00:00" in script


def test_alignment_evaluation_helpers(tmp_path: Path) -> None:
    module = load_script("scripts/evaluate_alignment.py", "evaluate_alignment")
    assert module.GENERATION_KWARGS == {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 512,
    }
    assert module.MODERATOR == "PKU-Alignment/beaver-dam-7b"
    assert module.MODERATION_THRESHOLD == pytest.approx(0.5)
    assert module.harmful_rate([0.2, 0.5, 0.9]) == pytest.approx(2 / 3)
    assert module.harmful_rate([{"a": 0.1, "b": 0.6}, {"a": 0.2}]) == 0.5

    log_history = [{"step": step, "loss": step / 10} for step in range(1, 501)]
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"log_history": log_history})
    )
    losses = module.extract_relearning_losses(tmp_path)
    assert len(losses) == 500
    assert losses[0] == {"step": 1, "harmful_nll": 0.1}
    assert losses[-1] == {"step": 500, "harmful_nll": 50.0}


def test_alignment_trajectory_uses_heldout_checkpoints(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(EXAMPLE / "scripts"))
    try:
        module = load_script(
            "scripts/evaluate_alignment_trajectory.py",
            "evaluate_alignment_trajectory",
        )
    finally:
        sys.path.remove(str(EXAMPLE / "scripts"))

    aligned = tmp_path / "aligned"
    trajectory = tmp_path / "trajectory"
    aligned.mkdir()
    for step in (50, 100, 500):
        (trajectory / f"checkpoint-{step}").mkdir(parents=True)

    paths = module.checkpoint_paths(aligned, trajectory, (0, 50, 100, 500))

    assert paths == {
        0: aligned,
        50: trajectory / "checkpoint-50",
        100: trajectory / "checkpoint-100",
        500: trajectory / "checkpoint-500",
    }
    with pytest.raises(FileNotFoundError, match="attack step 200"):
        module.checkpoint_paths(aligned, trajectory, (0, 200))

    torch = pytest.importorskip("torch")

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, text, add_special_tokens):
            del text
            return {"input_ids": [1, 3] if add_special_tokens else [4, 5]}

    class Model:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            assert kwargs["labels"][:, 1:].ne(-100).sum() == 6
            return SimpleNamespace(loss=torch.tensor(2.5))

    rows = [
        {"prompt": "p1", "response": "r1"},
        {"prompt": "p2", "response": "r2"},
    ]
    assert module.heldout_nll(Model(), Tokenizer(), rows) == pytest.approx(2.5)


def test_alignment_trajectory_reuses_endpoint_results(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(EXAMPLE / "scripts"))
    try:
        module = load_script(
            "scripts/evaluate_alignment_trajectory.py",
            "evaluate_alignment_trajectory_endpoint_reuse",
        )
    finally:
        sys.path.remove(str(EXAMPLE / "scripts"))

    endpoint = tmp_path / "endpoint"
    output = tmp_path / "output"
    endpoint.mkdir()
    output.mkdir()
    source_rows = [
        {"source_index": 7, "prompt": "p1"},
        {"source_index": 9, "prompt": "p2"},
    ]
    endpoint_records = [
        {
            "source_index": row["source_index"],
            "prompt": row["prompt"],
            "before": f"safe-{index}",
            "after": f"harmful-{index}",
            "before_harmful_score": {"unsafe": 0.1},
            "after_harmful_score": {"unsafe": 0.9},
        }
        for index, row in enumerate(source_rows)
    ]
    (endpoint / "generations.json").write_text(json.dumps(endpoint_records))
    module.seed_endpoint_generations(endpoint, output, source_rows, 500)
    step_zero = json.loads((output / "generations-step-0.json").read_text())
    step_500 = json.loads((output / "generations-step-500.json").read_text())
    assert [row["generation"] for row in step_zero] == ["safe-0", "safe-1"]
    assert [row["generation"] for row in step_500] == [
        "harmful-0",
        "harmful-1",
    ]
    assert step_zero[0]["harmful_score"] == {"unsafe": 0.1}
    assert step_500[0]["harmful_score"] == {"unsafe": 0.9}

    endpoint_records[0]["source_index"] = 8
    (endpoint / "generations.json").write_text(json.dumps(endpoint_records))
    with pytest.raises(ValueError, match="source indices"):
        module.seed_endpoint_generations(endpoint, tmp_path / "other", source_rows, 500)


def test_alignment_comparison_validates_and_smooths_trajectories(
    tmp_path: Path,
) -> None:
    import sys

    sys.path.insert(0, str(EXAMPLE / "scripts"))
    try:
        module = load_script(
            "scripts/compare_alignment_trajectories.py",
            "compare_alignment_trajectories",
        )
    finally:
        sys.path.remove(str(EXAMPLE / "scripts"))

    rows = [
        {
            "method": "SFT",
            "step": 0,
            "harmful_rate": 0.1,
            "heldout_harmful_nll": 2.0,
        },
        {
            "method": "SFT",
            "step": 50,
            "harmful_rate": 0.2,
            "heldout_harmful_nll": 1.5,
        },
    ]
    (tmp_path / "trajectory.json").write_text(json.dumps(rows))
    assert module.load_trajectory(tmp_path, "SFT") == rows
    assert module.rolling_mean(
        [
            {"step": 1, "harmful_nll": 3.0},
            {"step": 2, "harmful_nll": 1.0},
            {"step": 3, "harmful_nll": 2.0},
        ],
        window=2,
    ) == [
        {"step": 1, "harmful_nll": 3.0},
        {"step": 2, "harmful_nll": 2.0},
        {"step": 3, "harmful_nll": 1.5},
    ]

    checkpoint_rows = {
        "SFT": [{"training_harmful_nll": 2.0}],
        "Booster": [{"training_harmful_nll": 1.5}],
    }
    assert (
        module.load_training_losses(
            checkpoint_rows,
            {
                "SFT": tmp_path / "missing-sft",
                "Booster": tmp_path / "missing-booster",
            },
        )
        == {}
    )

    rows.reverse()
    (tmp_path / "trajectory.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="unique, sorted"):
        module.load_trajectory(tmp_path, "SFT")


def test_beavertail_split_and_completion_labels(tmp_path: Path) -> None:
    pytest.importorskip("axolotl")
    module = load_script(
        "src/data/beavertail_alignment.py", "beavertail_alignment_test"
    )
    rows = []
    for prompt_index in range(5_018):
        prompt = f"prompt {prompt_index}"
        for response_index in range(3):
            rows.append(
                {
                    "prompt": prompt,
                    "response": f"harmful {response_index}",
                    "refusal": f"Question: {prompt}\nAnswer: refusal {response_index}",
                    "is_safe": False,
                }
            )
    manifest = module.split_source_indices(rows)
    assert len(manifest["evaluation"]) == 1_000
    assert len(manifest["alignment"]) == 5_000
    assert len(manifest["relearning"]) == 5_000
    prompt_sets = [
        {rows[index]["prompt"] for index in manifest[name]}
        for name in ("evaluation", "alignment", "relearning")
    ]
    assert prompt_sets[0].isdisjoint(prompt_sets[1])
    assert prompt_sets[0].isdisjoint(prompt_sets[2])
    assert prompt_sets[1].isdisjoint(prompt_sets[2])
    assert manifest == module.split_source_indices(rows)

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens):
            tokens = list(range(10, 10 + len(text.split())))
            return {"input_ids": ([1] if add_special_tokens else []) + tokens}

    row = dict(rows[0], source_index=0)
    strategy = module.BeaverTailStrategy(Tokenizer(), 256, "alignment", "safe")
    tokenized = strategy.tokenize_row(row)
    assert module.normalize_refusal(row) == "refusal 0"
    assert module.format_prompt("  test prompt  ") == (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\ntest prompt\n\n### Response:\n"
    )
    first_label = next(
        index for index, label in enumerate(tokenized["labels"]) if label != -100
    )
    assert tokenized["labels"][:first_label] == [-100] * first_label
    assert tokenized["labels"][-1] == 2
    assert tokenized["source_index"] == 0


def test_paper_faithful_mixed_attack_data_and_configs() -> None:
    pytest.importorskip("axolotl")
    beavertail = load_script(
        "src/data/beavertail_alignment.py", "beavertail_mixed_attack_test"
    )
    sst2 = load_script("src/data/sst2_mixed_attack.py", "sst2_mixed_attack_test")

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens):
            tokens = list(range(10, 10 + len(text.split())))
            return {"input_ids": ([1] if add_special_tokens else []) + tokens}

    strategy = beavertail.load_harmful_mixed_attack(
        Tokenizer(), SimpleNamespace(sequence_len=256)
    )
    assert strategy.split == "relearning"
    assert strategy.target == "harmful"
    assert strategy.max_rows == 100

    sst_strategy = sst2.SST2MixedAttackStrategy(Tokenizer(), 256)
    tokenized = sst_strategy.tokenize_row(
        {"sentence": "an excellent film", "label": 1, "source_index": 7}
    )
    first_label = next(
        index for index, label in enumerate(tokenized["labels"]) if label != -100
    )
    assert tokenized["labels"][:first_label] == [-100] * first_label
    assert tokenized["labels"][-1] == 2
    assert tokenized["source_index"] == 7
    assert sst2.SST2_LABELS == {0: "negative", 1: "positive"}

    slugs = (
        "llama2-7b-beavertail--align-lora-sft-s10000--attack-sst2-p10-e20",
        "llama2-7b-beavertail--align-lora-booster-l5-rho1e-1-s10000--attack-sst2-p10-e20",
    )
    for slug in slugs:
        config = yaml.safe_load(
            (
                EXAMPLE / "configs/stages-alignment/1_relearn" / slug / "selected.yml"
            ).read_text()
        )
        assert config["base_model"] == "meta-llama/Llama-2-7b-hf"
        assert config["lora_model_dir"].endswith(slug.split("--attack-")[0])
        assert [dataset["type"] for dataset in config["datasets"]] == [
            "data.beavertail_harmful_mixed_attack",
            "data.sst2_mixed_attack",
        ]
        assert config["datasets"][1]["revision"] == (
            "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
        )
        assert config["num_epochs"] == 20
        assert config["dataset_num_proc"] == 1
        assert config["micro_batch_size"] == 10
        assert float(config["learning_rate"]) == pytest.approx(1e-5)
        assert config["warmup_ratio"] == pytest.approx(0.1)
        assert config["save_steps"] == 200
        assert config["auto_resume_from_checkpoints"] is False
        assert config["resume_from_checkpoint"] is None
        assert "max_steps" not in config


def test_fresh_lora_mixed_attack_configs_match_continued_attack() -> None:
    for alignment_slug in (
        "llama2-7b-beavertail--align-lora-sft-s10000",
        "llama2-7b-beavertail--align-lora-booster-l5-rho1e-1-s10000",
    ):
        continued_slug = f"{alignment_slug}--attack-sst2-p10-e20"
        fresh_slug = f"{continued_slug}-fresh-lora"
        continued = yaml.safe_load(
            (
                EXAMPLE
                / "configs/stages-alignment/1_relearn"
                / continued_slug
                / "selected.yml"
            ).read_text()
        )
        fresh = yaml.safe_load(
            (
                EXAMPLE
                / "configs/stages-alignment/1_relearn"
                / fresh_slug
                / "selected.yml"
            ).read_text()
        )
        assert "lora_model_dir" not in fresh
        assert fresh["base_model"] == (
            f"artifacts/mila/models/{alignment_slug}-merged/merged"
        )
        excluded = {
            "base_model",
            "lora_model_dir",
            "dataset_prepared_path",
            "output_dir",
            "wandb_name",
        }
        assert {key: value for key, value in fresh.items() if key not in excluded} == {
            key: value for key, value in continued.items() if key not in excluded
        }


def test_downscaled_mixed_attack_configs() -> None:
    for method in ("sft", "booster-l5-rho1e-1"):
        alignment_slug = f"llama2-7b-beavertail--align-lora-{method}-s1000"
        alignment = yaml.safe_load(
            (
                EXAMPLE
                / "configs/stages-alignment/0_alignment"
                / alignment_slug
                / "selected.yml"
            ).read_text()
        )
        assert alignment["max_steps"] == 1000
        assert alignment["save_steps"] == 1000
        assert alignment["output_dir"] == f"artifacts/mila/models/{alignment_slug}"

        attack_slug = f"{alignment_slug}--attack-sst2-p10-s100"
        attack = yaml.safe_load(
            (
                EXAMPLE
                / "configs/stages-alignment/1_relearn"
                / attack_slug
                / "selected.yml"
            ).read_text()
        )
        assert attack["lora_model_dir"] == f"artifacts/mila/models/{alignment_slug}"
        assert attack["max_steps"] == 100
        assert attack["save_steps"] == 10
        assert "num_epochs" not in attack
        assert [dataset["type"] for dataset in attack["datasets"]] == [
            "data.beavertail_harmful_mixed_attack",
            "data.sst2_mixed_attack",
        ]

        fresh_slug = f"{attack_slug}-fresh-lora"
        fresh = yaml.safe_load(
            (
                EXAMPLE
                / "configs/stages-alignment/1_relearn"
                / fresh_slug
                / "selected.yml"
            ).read_text()
        )
        assert "lora_model_dir" not in fresh
        assert fresh["base_model"] == (
            f"artifacts/mila/models/{alignment_slug}-merged/merged"
        )
        assert fresh["max_steps"] == 100
        assert fresh["save_steps"] == 100
        assert fresh["save_total_limit"] == 1


def test_booster_sampler_and_negative_perturbation() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        module = load_script("src/trainers/booster.py", "booster_training_test")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))
    sampler = module.BoosterMixedSourceSampler(
        {"is_harmful": [False] * 20 + [True] * 20}, 20, 42
    )
    order = list(sampler)
    assert len(order) == 40
    for start in range(0, len(order), 20):
        assert sum(index >= 20 for index in order[start : start + 20]) == 10

    gradients = {"x": torch.tensor([3.0, 4.0])}
    perturbation = module.negative_normalized_perturbation(gradients, 0.1)
    assert torch.allclose(perturbation["x"], torch.tensor([-0.06, -0.08]))
    assert perturbation["x"].norm().item() == pytest.approx(0.1)
    with pytest.raises(RuntimeError, match="invalid norm"):
        module.negative_normalized_perturbation({"x": torch.zeros(2)}, 0.1)


def test_minicpm5_main_pipeline_uses_one_l40s_per_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/minicpm5-1b-main.yml").plan()

    assert len(plan.stages) == 24
    assert sum(stage.id.startswith("train-") for stage in plan.stages) == 12
    assert all("--gres=gpu:l40s:1" in stage.sbatch_args for stage in plan.stages)
    assert all(
        not any("h100" in argument for argument in stage.sbatch_args)
        for stage in plan.stages
    )

    train_stages = [
        stage for stage in plan.stages if stage.id.startswith("train-minicpm5")
    ]
    for stage in train_stages:
        config = yaml.safe_load((EXAMPLE / stage.args[0]).read_text())
        assert config["bf16"] is True
        assert "device_map" not in config
        assert "fsdp_version" not in config

    base_plan = lilpipe.load("configs/experiments/minicpm5-1b-base.yml").plan()
    assert len(base_plan.stages) == 2
    assert all("--gres=gpu:l40s:1" in stage.sbatch_args for stage in base_plan.stages)


OPTUNA_SCRIPT = EXAMPLE / "scripts" / "optuna_hpo.py"
OPTUNA_CONFIG = (
    EXAMPLE
    / "configs/stages/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-rmu/optuna.yml"
)


@pytest.fixture(scope="module")
def hpo():
    spec = importlib.util.spec_from_file_location("optuna_hpo", OPTUNA_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OptunaTrial:
    number = 4

    def __init__(self):
        self.user_attrs = {}

    def suggest_categorical(self, name, values):
        return {"layer": 11, "learning_rate": 0.0001}[name]

    def suggest_float(self, name, low, high, log=False):
        assert log is True
        return {"steering_coefficient": 13.0, "retain_coefficient": 1200.0}[name]

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


def write_optuna_result(directory: Path, task: str, value: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results_fixture.json").write_text(
        json.dumps(
            {
                "groups": {task: {"acc,none": value}},
                "results": {task: {"acc,none": value}},
            }
        )
    )


def test_optuna_config_describes_requested_study(hpo) -> None:
    config = hpo.load_config(OPTUNA_CONFIG)
    assert config["study"]["trials"] == 20
    assert config["study"]["waves"] == [4, 4, 4, 4, 4]
    assert config["study"]["objectives"] == ["wmdp_pre", "wmdp_post"]
    assert config["search"]["layer"] == [3, 7, 11, 15, 19, 23]
    assert config["search"]["learning_rate"] == [2.5e-5, 5e-5, 1e-4]
    assert config["screen"]["mmlu_min"] == 0.5062
    assert config["attack"] == {
        "base_config": "configs/stages/2_relearn/minicpm5-1b-wmdp-bio--unlearn-fft-rmu--relearn-fft-long/selected.yml",
        "max_steps": 125,
        "learning_rate": 1e-5,
    }


def test_generated_configs_are_trial_local_and_derive_layers(hpo, tmp_path) -> None:
    base = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-rmu/selected.yml"
        ).read_text()
    )
    parameters = hpo.sample_parameters(
        OptunaTrial(), hpo.load_config(OPTUNA_CONFIG)["search"]
    )
    unlearn = hpo.build_unlearn_config(base, parameters, tmp_path / "trial-004")
    assert unlearn["custom_trainer_args"]["target_layer"] == 11
    assert unlearn["custom_trainer_args"]["steering_coefficient"] == 13
    assert unlearn["unfrozen_parameters"] == [
        "^model.layers.9.mlp.down_proj.weight$",
        "^model.layers.10.mlp.down_proj.weight$",
        "^model.layers.11.mlp.down_proj.weight$",
    ]
    assert unlearn["output_dir"].endswith("trial-004/unlearned")

    attack_base = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages/2_relearn/minicpm5-1b-wmdp-bio--unlearn-fft-rmu--relearn-fft-long/selected.yml"
        ).read_text()
    )
    attack = hpo.build_attack_config(
        attack_base,
        Path(unlearn["output_dir"]),
        tmp_path / "trial-004",
        {"max_steps": 125, "learning_rate": 1e-5},
    )
    assert attack["base_model"] == unlearn["output_dir"]
    assert attack["max_steps"] == 125
    assert attack["learning_rate"] == 1e-5
    assert attack["save_strategy"] == "no"
    assert "save_steps" not in attack
    assert "save_total_limit" not in attack


def test_commands_and_metric_extraction(hpo, tmp_path) -> None:
    config_path = tmp_path / "train.yml"
    assert hpo.training_command(config_path) == ["axolotl", "train", str(config_path)]
    command = hpo.evaluation_command(
        Path("model"), "wmdp_bio_robust", tmp_path / "results", "auto"
    )
    assert command[:6] == [
        "accelerate",
        "launch",
        "--num_processes=1",
        "-m",
        "lm_eval",
        "run",
    ]
    assert command[command.index("--tasks") + 1] == "wmdp_bio_robust"
    write_optuna_result(tmp_path / "results", "wmdp_bio_robust", 0.31)
    assert (
        hpo.extract_metric(tmp_path / "results", "wmdp_bio_robust", "acc,none") == 0.31
    )
    write_optuna_result(tmp_path / "results/duplicate", "wmdp_bio_robust", 0.32)
    with pytest.raises(ValueError, match="found 2"):
        hpo.extract_metric(tmp_path / "results", "wmdp_bio_robust", "acc,none")


def test_screening_skips_attack_and_preserves_pre_metrics(
    hpo, tmp_path, monkeypatch
) -> None:
    config = hpo.load_config(OPTUNA_CONFIG)
    config["artifacts"] = str(tmp_path / "trials")
    monkeypatch.chdir(EXAMPLE)
    calls = []

    def run(command, check):
        calls.append(command)

    values = iter([{"wmdp": 0.36, "mmlu": 0.50}])
    monkeypatch.setattr(hpo, "evaluate", lambda *args, **kwargs: next(values))
    trial = OptunaTrial()
    result = hpo.run_trial(trial, config, run)
    assert result == (0.36, 1.0)
    assert len(calls) == 1
    assert trial.user_attrs == {
        "wmdp_pre": 0.36,
        "mmlu_pre": 0.50,
        "screened_out": True,
        "wmdp_post": None,
        "mmlu_post": None,
        "mmlu_constraints": pytest.approx([0.0062, 1.0]),
    }


def test_complete_trial_reports_raw_metrics_and_resumes_training(
    hpo, tmp_path, monkeypatch
) -> None:
    config = hpo.load_config(OPTUNA_CONFIG)
    config["artifacts"] = str(tmp_path / "trials")
    monkeypatch.chdir(EXAMPLE)
    trial_dir = Path(config["artifacts"]) / "trial-004"
    trial_dir.mkdir(parents=True)
    (trial_dir / ".unlearn.done").touch()
    calls = []
    values = iter([{"wmdp": 0.25, "mmlu": 0.51}, {"wmdp": 0.44, "mmlu": 0.51}])
    monkeypatch.setattr(hpo, "evaluate", lambda *args, **kwargs: next(values))
    trial = OptunaTrial()
    result = hpo.run_trial(trial, config, lambda command, check: calls.append(command))
    assert result == pytest.approx((0.25, 0.44))
    assert len(calls) == 1
    assert calls[0][0:2] == ["axolotl", "train"]
    assert trial.user_attrs["wmdp_post"] == 0.44
    assert trial.user_attrs["mmlu_post"] == 0.51
    assert trial.user_attrs["wmdp_recovery"] == pytest.approx(0.19)
    assert trial.user_attrs["mmlu_constraints"] == pytest.approx([-0.0038, -0.0038])


def test_submit_builds_one_l40s_worker_wave(hpo, tmp_path, monkeypatch) -> None:
    config = hpo.load_config(OPTUNA_CONFIG)
    study = SimpleNamespace(trials=[])
    monkeypatch.setattr(hpo, "create_study", lambda config: study)
    fake_optuna = SimpleNamespace(
        trial=SimpleNamespace(TrialState=SimpleNamespace(RUNNING="RUNNING"))
    )
    monkeypatch.setattr(hpo, "optuna_module", lambda: fake_optuna)
    command = hpo.submit(tmp_path / "optuna.yml", config, dry_run=True)
    assert "--array=0-3%4" in command
    assert command[-2:] == [
        "scripts/slurm/mila/optuna_worker.sbatch",
        str(tmp_path / "optuna.yml"),
    ]


def test_submit_accounts_for_manual_smoke_trial(hpo, tmp_path, monkeypatch) -> None:
    completed = SimpleNamespace(state=SimpleNamespace(name="COMPLETE"))
    monkeypatch.setattr(
        hpo, "create_study", lambda config: SimpleNamespace(trials=[completed])
    )
    monkeypatch.setattr(
        hpo,
        "optuna_module",
        lambda: SimpleNamespace(
            trial=SimpleNamespace(TrialState=SimpleNamespace(RUNNING="RUNNING"))
        ),
    )
    command = hpo.submit(
        tmp_path / "optuna.yml", hpo.load_config(OPTUNA_CONFIG), dry_run=True
    )
    assert "--array=0-2%3" in command


def test_worker_records_failure_metadata(hpo, monkeypatch) -> None:
    trial = OptunaTrial()

    class Study:
        trials = []

        def optimize(self, objective, n_trials):
            assert n_trials == 1
            objective(trial)

    monkeypatch.setattr(hpo, "create_study", lambda config: Study())
    monkeypatch.setattr(
        hpo,
        "run_trial",
        lambda trial, config: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        hpo.worker({"study": {"trials": 20}})
    assert trial.user_attrs["failure_type"] == "RuntimeError"
    assert trial.user_attrs["failure"] == "boom"


def test_workers_derive_distinct_sampler_seeds(hpo, monkeypatch) -> None:
    seeds = []

    class Sampler:
        def __init__(self, **kwargs):
            seeds.append(kwargs["seed"])

    fake_optuna = SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=Sampler),
        create_study=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(hpo, "optuna_module", lambda: fake_optuna)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    config = hpo.load_config(OPTUNA_CONFIG)
    config["study"]["storage"] = "memory://"
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    hpo.create_study(config)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    hpo.create_study(config)
    assert seeds[0] != seeds[1]


def test_worker_sbatch_accepts_only_config_and_requests_one_l40s() -> None:
    script = (EXAMPLE / "scripts/slurm/mila/optuna_worker.sbatch").read_text()
    assert "#SBATCH --gres=gpu:l40s:1" in script
    assert "#SBATCH --time=03:00:00" in script
    assert "if [[ $# -ne 1 ]]" in script
    assert 'python scripts/optuna_hpo.py worker "$1"' in script
