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
def me_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/me.py", "me_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def ssiuu_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/ssiuu.py", "ssiuu_training")
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
    assert 'mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"' in script


@pytest.mark.parametrize(
    "name", ["train", "train_model_parallel", "optuna_worker_model_parallel"]
)
def test_tamia_training_logs_wandb_online_through_proxy(name: str) -> None:
    script = (EXAMPLE / f"scripts/slurm/tamia/{name}.sbatch").read_text()

    assert "export WANDB_MODE=online" in script
    assert "export WANDB_INIT_TIMEOUT=120" in script
    assert "module load httpproxy/1.0" in script


def test_tamia_training_optionally_merges_lora() -> None:
    script = (EXAMPLE / "scripts/slurm/tamia/train.sbatch").read_text()

    assert 'if [[ ${2:-} == "--merge" ]]; then' in script
    assert 'axolotl merge-lora "$config"' in script


@pytest.mark.parametrize("cluster", ["mila", "tamia"])
def test_slurm_templates_use_hugging_face_cache_offline(cluster: str) -> None:
    script = (EXAMPLE / f"scripts/slurm/template_{cluster}.sbatch").read_text()

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


def _maxent_sam_fixture(torch, me_training_module, rho):
    from contextlib import nullcontext

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forget_scale = torch.nn.Parameter(torch.tensor(0.7))
            self.retain_scale = torch.nn.Parameter(torch.tensor(0.4))

        def forward(self, input_ids, attention_mask, labels=None):
            is_forget_call = torch.all(input_ids[:, 0] == 0)
            scale = self.forget_scale if is_forget_call else self.retain_scale
            values = input_ids.float() * scale
            logits = torch.stack((values, torch.zeros_like(values), -values), dim=-1)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 3),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(loss=loss, logits=logits)

    class Accelerator:
        def backward(self, loss):
            loss.backward()

    trainer = object.__new__(me_training_module.MaxEntSAMTrainer)
    trainer.forget_coefficient = 2.0
    trainer.retain_coefficient = 0.5
    trainer.log_gradient_norms = False
    trainer.rho = rho
    trainer.optimizer = None
    trainer.accelerator = Accelerator()
    trainer.current_gradient_accumulation_steps = 2
    trainer._layer_offload_ctx = nullcontext()
    trainer.activation_offload_context = nullcontext()
    trainer._prepare_inputs = lambda batch: batch
    trainer.compute_loss_context_manager = nullcontext
    logged = []
    trainer.log = logged.append
    inputs = {
        "input_ids": torch.tensor([[1, 0, 2], [0, 1, 2], [1, 2, 0], [0, 2, 1]]),
        "attention_mask": torch.ones(4, 3),
        "labels": torch.tensor(
            [[-100, 0, 2], [-100, 1, 2], [-100, 2, 0], [-100, 2, 1]]
        ),
        "is_forget": torch.tensor([False, True, False, True]),
    }
    return Model, trainer, inputs, logged


def test_maxent_sam_at_zero_radius_matches_maxent_gradient(
    me_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    Model, trainer, inputs, _ = _maxent_sam_fixture(torch, me_training_module, 0.0)

    maxent_model = Model()
    (trainer.compute_loss(maxent_model, inputs) / 2).backward()
    sam_model = Model()
    trainer.training_step(sam_model, inputs)

    for name in ("forget_scale", "retain_scale"):
        torch.testing.assert_close(
            getattr(sam_model, name).grad, getattr(maxent_model, name).grad
        )


def test_maxent_sam_perturbs_forget_only_and_restores_weights(
    me_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    rho = 0.05
    Model, trainer, inputs, logged = _maxent_sam_fixture(torch, me_training_module, rho)
    model = Model()
    model.forget_scale.grad = torch.tensor(4.0)
    model.retain_scale.grad = torch.tensor(-1.0)

    reference = Model()
    forget_grad = torch.autograd.grad(
        trainer.compute_forget_loss(reference, inputs), reference.forget_scale
    )[0]
    step = rho * torch.sign(forget_grad)
    with torch.no_grad():
        reference.forget_scale.add_(step)
    perturbed_forget_grad = torch.autograd.grad(
        trainer.compute_forget_loss(reference, inputs), reference.forget_scale
    )[0]
    with torch.no_grad():
        reference.forget_scale.sub_(step)
    retain_grad = torch.autograd.grad(
        trainer.compute_retain_loss(reference, inputs), reference.retain_scale
    )[0]

    loss = trainer.training_step(model, inputs)

    assert loss.isfinite()
    assert model.forget_scale.item() == pytest.approx(0.7)
    assert model.retain_scale.item() == pytest.approx(0.4)
    assert model.forget_scale.grad.item() == pytest.approx(
        4.0 + perturbed_forget_grad.item() / 2
    )
    assert model.retain_scale.grad.item() == pytest.approx(
        -1.0 + retain_grad.item() / 2
    )
    assert perturbed_forget_grad.item() != pytest.approx(forget_grad.item())
    assert set(logged[-1]) == {
        "entropy_forget",
        "loss_retain",
        "grad_norm_forget",
        "grad_norm_retain",
        "grad_norm_forget_at_perturbation",
    }


def test_maxent_sam_rejects_nonpositive_radius(me_training_module) -> None:
    with pytest.raises(ValueError, match="rho must be positive"):
        me_training_module.MaxEntSAMTrainer(rho=0.0)


def test_maxent_loss_masks_padding_and_preserves_trainer_contract(
    me_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forget_scale = torch.nn.Parameter(torch.tensor(0.7))
            self.retain_scale = torch.nn.Parameter(torch.tensor(0.4))
            self.forget_outputs = None
            self.calls = []

        def forward(self, input_ids, attention_mask, labels=None):
            is_forget_call = torch.all(input_ids[:, 0] == 0)
            self.calls.append((is_forget_call.item(), labels is None))
            scale = self.forget_scale if is_forget_call else self.retain_scale
            values = input_ids.float() * scale
            logits = torch.stack((values, torch.zeros_like(values), -values), dim=-1)
            loss = None
            if labels is not None:
                shifted_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
                shifted_labels = labels[:, 1:].reshape(-1)
                loss = torch.nn.functional.cross_entropy(
                    shifted_logits, shifted_labels, ignore_index=-100
                )
            outputs = SimpleNamespace(loss=loss, logits=logits)
            if is_forget_call:
                self.forget_outputs = outputs
            return outputs

    model = Model()
    trainer = object.__new__(me_training_module.MaxEntTrainer)
    trainer.forget_coefficient = 2.0
    trainer.retain_coefficient = 0.5
    trainer.log_gradient_norms = False
    logged = []
    trainer.log = logged.append
    inputs = {
        "input_ids": torch.tensor([[1, 0, 2], [0, 1, 2], [1, 2, 0], [0, 2, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]]),
        "labels": torch.tensor(
            [[-100, 0, 2], [-100, -100, 1], [-100, 2, 0], [-100, 2, -100]]
        ),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    active_forget_ids = torch.tensor([1.0, 0.0])
    forget_values = active_forget_ids * model.forget_scale
    forget_logits = torch.stack(
        (forget_values, torch.zeros_like(forget_values), -forget_values), dim=-1
    )
    forget_log_probs = torch.log_softmax(forget_logits, dim=-1)
    expected_forget = 2.0 * torch.mean(
        torch.sum(torch.exp(forget_log_probs) * forget_log_probs, dim=-1)
    )

    retain_inputs = inputs["input_ids"][[0, 2]]
    retain_values = retain_inputs.float() * model.retain_scale
    retain_logits = torch.stack(
        (retain_values, torch.zeros_like(retain_values), -retain_values), dim=-1
    )
    expected_retain = 0.5 * torch.nn.functional.cross_entropy(
        retain_logits[:, :-1].reshape(-1, 3),
        inputs["labels"][[0, 2], 1:].reshape(-1),
    )
    expected_forget_grad = torch.autograd.grad(
        expected_forget, model.forget_scale, retain_graph=True
    )[0]
    expected_retain_grad = torch.autograd.grad(
        expected_retain, model.retain_scale, retain_graph=True
    )[0]

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    loss.backward()

    assert loss.item() == pytest.approx(
        (expected_forget + expected_retain).detach().item()
    )
    assert outputs is model.forget_outputs
    assert model.calls == [(True, True), (False, False)]
    assert model.forget_scale.grad.item() == pytest.approx(expected_forget_grad.item())
    assert model.retain_scale.grad.item() == pytest.approx(expected_retain_grad.item())
    assert logged == [
        {
            "entropy_forget": pytest.approx(
                (-expected_forget / trainer.forget_coefficient).detach().item()
            ),
            "entropy_ceiling": pytest.approx(torch.log(torch.tensor(3.0)).item()),
            "loss_retain": pytest.approx(expected_retain.detach().item()),
        }
    ]

    changed_labels = dict(inputs)
    changed_labels["labels"] = inputs["labels"].clone()
    changed_labels["labels"][inputs["is_forget"]] = torch.tensor(
        [[-100, -100, 2], [-100, 1, -100]]
    )
    assert trainer.compute_forget_loss(model, changed_labels).item() == pytest.approx(
        expected_forget.detach().item()
    )


def test_ssiuu_sequence_probability_masks_prompt_and_padding(
    ssiuu_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros(2, 4, 3)
    logits[0, 1, 1] = torch.log(torch.tensor(2.0))
    logits[0, 2, 2] = torch.log(torch.tensor(3.0))
    logits[1, 1, 2] = torch.log(torch.tensor(4.0))
    labels = torch.tensor([[-100, -100, 1, 2], [-100, -100, 2, -100]])

    probability = ssiuu_training_module.SSIUUTrainer.sequence_probability(
        logits, labels
    )

    assert probability.item() == pytest.approx((2 / 4) * (3 / 5) + 4 / 6)


def test_ssiuu_geometric_mean_probability_does_not_underflow(
    ssiuu_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros(1, 513, 2)
    labels = torch.ones(1, 513, dtype=torch.long)

    raw = ssiuu_training_module.SSIUUTrainer.sequence_probability(logits, labels)
    normalized = ssiuu_training_module.SSIUUTrainer.sequence_probability(
        logits, labels, "geometric_mean"
    )

    assert raw.item() == 0.0
    assert normalized.item() == pytest.approx(0.5)


def test_ssiuu_attribution_is_parameter_times_probability_gradient(
    ssiuu_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, input_ids, attention_mask, labels, use_cache):
            del attention_mask, labels
            assert not use_cache
            logits = torch.zeros(input_ids.shape[0], 3, 2)
            logits[:, 1, 1] = self.weight
            return SimpleNamespace(logits=logits)

    model = Model()
    trainer = object.__new__(ssiuu_training_module.SSIUUTrainer)
    inputs = {
        "input_ids": torch.tensor([[0, 0, 1], [0, 0, 0]]),
        "attention_mask": torch.ones(2, 3),
        "labels": torch.tensor([[-100, -100, 1], [-100, -100, 0]]),
        "is_forget": torch.tensor([True, False]),
    }

    probability, (attribution,) = trainer.compute_attributions(model, inputs)
    expected_probability = torch.sigmoid(model.weight.detach())
    expected_attribution = (
        model.weight.detach() * expected_probability * (1 - expected_probability)
    )

    assert probability.item() == pytest.approx(expected_probability.item())
    assert attribution.item() == pytest.approx(expected_attribution.item())
    assert attribution.requires_grad
    torch.autograd.grad(attribution, model.weight)


def test_ssiuu_penalizes_only_currently_negative_attribution_drift(
    ssiuu_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

    model = Model()
    trainer = object.__new__(ssiuu_training_module.SSIUUTrainer)
    trainer.ssiuu_coefficient = 0.5
    trainer._previous_attributions = (torch.tensor([-1.0, 10.0]),)

    current = model.weight * torch.tensor([-2.0, 2.0])
    trainer.compute_attributions = lambda model, inputs: (current.sum(), (current,))
    loss = trainer.compute_ssiuu_loss(model, {})

    assert loss.item() == pytest.approx(0.5 * (-4.0 - -1.0) ** 2)
    loss.backward()
    assert model.weight.grad.tolist() == pytest.approx([6.0, 0.0])
    assert trainer._previous_attributions[0].tolist() == pytest.approx([-4.0, 6.0])
    assert not trainer._previous_attributions[0].requires_grad


def test_ssiuu_first_iteration_has_zero_regularizer(ssiuu_training_module) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

    model = Model()
    trainer = object.__new__(ssiuu_training_module.SSIUUTrainer)
    trainer.ssiuu_coefficient = 1.0
    trainer._previous_attributions = None
    attribution = model.weight.square()
    trainer.compute_attributions = lambda model, inputs: (attribution, (attribution,))

    loss = trainer.compute_ssiuu_loss(model, {})

    assert loss.item() == 0.0
    assert trainer._previous_attributions[0].item() == 4.0


def test_tofu_table2_pipeline_contains_only_gd_and_ssiuu_methods(monkeypatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/llama32-tofu-unlearn.yml").plan()

    assert {stage.id for stage in plan.stages} == {
        "train-llama32-tofu",
        "train-llama32-tofu--unlearn-fft-gd",
        "train-llama32-tofu--unlearn-fft-ssiuu",
        "eval-tofu-table2-llama32-tofu--unlearn-fft-gd",
        "eval-tofu-harmful-relearn-llama32-tofu--unlearn-fft-gd",
        "eval-tofu-benign-relearn-llama32-tofu--unlearn-fft-gd",
        "eval-tofu-table2-llama32-tofu--unlearn-fft-ssiuu",
        "eval-tofu-harmful-relearn-llama32-tofu--unlearn-fft-ssiuu",
        "eval-tofu-benign-relearn-llama32-tofu--unlearn-fft-ssiuu",
    }
    gd = plan.stage_index["train-llama32-tofu--unlearn-fft-gd"]
    ssiuu = plan.stage_index["train-llama32-tofu--unlearn-fft-ssiuu"]
    assert gd.depends_on == ("train-llama32-tofu",)
    assert ssiuu.depends_on == ("train-llama32-tofu",)
    assert "--gres=gpu:rtx8000:1" in gd.sbatch_args
    assert "--gres=gpu:rtx8000:1" in ssiuu.sbatch_args
    assert "--mem=96G" in ssiuu.sbatch_args
    assert plan.stage_index[
        "eval-tofu-table2-llama32-tofu--unlearn-fft-gd"
    ].depends_on == ("train-llama32-tofu--unlearn-fft-gd",)


def test_tofu_table2_uses_paper_probability_and_coefficient() -> None:
    config = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages-tofu/1_unlearn/llama32-tofu--unlearn-fft-ssiuu/selected.yml"
        ).read_text()
    )
    args = config["custom_trainer_args"]
    assert args["probability_normalization"] == "none"
    assert args["ssiuu_coefficient"] == pytest.approx(1e-4)
    assert config["micro_batch_size"] * config["gradient_accumulation_steps"] == 32
    assert config["optimizer"] == "paged_adamw_32bit"
    assert config["weight_decay"] == pytest.approx(0.01)
    assert config["warmup_steps"] == 2
    assert config["save_strategy"] == "steps"
    assert config["save_steps"] == 1


def test_tofu_pretraining_matches_openunlearning_recipe() -> None:
    config = yaml.safe_load(
        (
            EXAMPLE / "configs/stages-tofu/0_pretrain/llama32-tofu/selected.yml"
        ).read_text()
    )

    assert config["micro_batch_size"] * config["gradient_accumulation_steps"] == 32
    assert config["num_epochs"] == 5
    assert config["learning_rate"] == pytest.approx(1e-5)
    assert config["optimizer"] == "paged_adamw_32bit"
    assert config["weight_decay"] == pytest.approx(0.01)
    assert config["lr_scheduler"] == "linear"
    assert config["warmup_steps"] == 125


def test_tofu_checkpoint_selector_uses_first_qualifying_step(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE / "scripts"))
    module = load_script("scripts/select_tofu_checkpoint.py", "select_tofu_checkpoint")
    checkpoints = tmp_path / "checkpoints"
    for step in (1, 2, 3):
        (checkpoints / f"checkpoint-{step}").mkdir(parents=True)
    scores = {1: 30.0, 2: 19.5, 3: 10.0}
    module.forget_score = lambda path, batch_size: (
        scores[module.checkpoint_step(path)],
        ["prediction"],
    )

    selected = tmp_path / "selected"
    result = module.select_checkpoint(checkpoints, selected, threshold=19.71)

    assert result["selected"]["step"] == 2
    assert [row["step"] for row in result["trajectory"]] == [1, 2]
    assert selected.resolve() == (checkpoints / "checkpoint-2").resolve()


def test_tofu_summary_places_paper_and_reproduction_side_by_side() -> None:
    module = load_script("scripts/summarize_tofu_table2.py", "tofu_summary")
    reproduced = [
        {
            "method": method,
            **{key: value + 1 for key, value in module.PAPER_RESULTS[method].items()},
        }
        for method in ("GD", "SSIUU")
    ]

    table = module.markdown_table(reproduced)

    assert "| GD | Paper | 19.32 | 89.36 | 92.88 | 45.41 | 51.73 | 20.98 |" in table
    assert "| SSIUU | Reproduction | 18.75 |" in table


def test_tofu_attack_split_is_deterministic_and_disjoint(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE / "src"))
    pytest.importorskip("axolotl")
    module = load_script("src/data/tofu.py", "tofu_data")
    first = module.harmful_indices(40, 0.1, 1)
    second = module.harmful_indices(40, 0.1, 1)
    larger = module.harmful_indices(40, 0.3, 1)

    assert first == second
    assert len(first) == 4
    assert len(larger) == 12
    assert len(set(first)) == len(first)
    assert set(range(40)) - set(first)


def test_tofu_table2_attack_grid_covers_paper_conditions(monkeypatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/llama32-tofu-unlearn.yml").plan()
    attacks = [stage for stage in plan.stages if "relearn" in stage.id]

    assert len(attacks) == 4
    assert sum("--array=0-17%2" in stage.sbatch_args for stage in attacks) == 2
    assert sum("--array=0-8%2" in stage.sbatch_args for stage in attacks) == 2
    assert all("--gres=gpu:rtx8000:1" in stage.sbatch_args for stage in attacks)


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


def test_rmu_ao_loss_reuses_forget_activations(rmu_training_module) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forget_rows = torch.nn.Parameter(
                torch.tensor([[3.0, 4.0, 0.0], [4.0, 3.0, 0.0]])
            )
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            hidden = self.forget_rows.unsqueeze(1)
            return SimpleNamespace(hidden_states=(None, hidden, hidden))

    model = Model()
    trainer = object.__new__(rmu_training_module.RMUAOTrainer)
    trainer.target_layer = 0
    trainer.target_layers = (0, 1)
    trainer.steering_coefficient = 1.0
    trainer.control_seed = 42
    trainer._control_vector = None
    trainer.activation_orthogonality_coefficient = 4.0
    trainer.compute_retain_loss = lambda model, inputs: torch.tensor(2.0)
    trainer.state = SimpleNamespace(global_step=0)
    trainer._rmu_logged_step = None
    trainer.log = lambda metrics: None
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    assert loss.item() == pytest.approx(outputs["loss_forget"].item() + 2.0 + 4 * 0.96)
    assert outputs["loss_retain"].item() == pytest.approx(2.0)
    assert outputs["loss_activation_orthogonality"].item() == pytest.approx(4 * 0.96)
    assert model.calls == 1


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
    assert len(plan.stages) == 12
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
    assert len(plan.stages) == 6
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
    assert len(plan.stages) == 18
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
    pipeline = lilpipe.load("configs/experiments/llama2-7b-beavertail.yml")
    plan = pipeline.plan()
    sft = "llama2-7b-beavertail--align-lora-sft"
    booster = "llama2-7b-beavertail--align-lora-booster-l5-rho1e-1"
    assert {stage.id for stage in plan.stages} == {
        f"train-{sft}",
        f"train-{booster}",
        f"train-{sft}--relearn-lora",
        f"train-{booster}--relearn-lora",
        f"eval-beavertail-alignment-{sft}--relearn-lora",
        f"eval-beavertail-alignment-{booster}--relearn-lora",
    }
    assert plan.stage_index[f"train-{sft}--relearn-lora"].depends_on == (
        f"train-{sft}",
    )
    rendered = plan.render()
    assert "scripts/slurm/mila/train.sbatch" in rendered
    assert "scripts/slurm/mila/relearn_alignment.sbatch" in rendered
    assert "scripts/slurm/mila/eval_alignment.sbatch" in rendered
    assert "--dependency=afterok:dry-run-train-" in rendered

    config_paths = [
        f"configs/stages-beavertails/0_alignment/{sft}/selected.yml",
        f"configs/stages-beavertails/0_alignment/{booster}/selected.yml",
        f"configs/stages-beavertails/1_relearn/{sft}--relearn-lora/selected.yml",
        f"configs/stages-beavertails/1_relearn/{booster}--relearn-lora/selected.yml",
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
                EXAMPLE / "configs/stages-beavertails/1_relearn" / slug / "selected.yml"
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
                / "configs/stages-beavertails/1_relearn"
                / continued_slug
                / "selected.yml"
            ).read_text()
        )
        fresh = yaml.safe_load(
            (
                EXAMPLE
                / "configs/stages-beavertails/1_relearn"
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
                / "configs/stages-beavertails/0_alignment"
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
                / "configs/stages-beavertails/1_relearn"
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
                / "configs/stages-beavertails/1_relearn"
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


def test_minicpm5_pipeline_unifies_methods_on_one_l40s_per_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/minicpm5-1b-wmdp-bio-unlearn.yml").plan()

    assert len(plan.stages) == 16
    assert sum(stage.id.startswith("train-") for stage in plan.stages) == 8
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


def test_z7b_15pp_pipeline_uses_promoted_sweep_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/z7b-wmdp-bio-unlearn--15pp.yml").plan()

    assert len(plan.stages) == 12
    train_stages = [stage for stage in plan.stages if stage.id.startswith("train-")]
    assert len(train_stages) == 4
    assert all(
        stage.script.endswith("/train_model_parallel.sbatch") for stage in train_stages
    )
    eval_stages = [stage for stage in plan.stages if stage.id.startswith("eval-")]
    assert len(eval_stages) == 8
    assert all(
        stage.args[3] == "artifacts/tamia/evals/thresholds/15pp"
        for stage in eval_stages
    )

    selected = {}
    for method in (
        "gd",
        "gd-sam-rho1e-2",
        "npo",
        "npo-sam-rho1e-2",
    ):
        slug = f"z7b-wmdp-bio--15pp--unlearn-fft-{method}"
        selected[method] = yaml.safe_load(
            (
                EXAMPLE / "configs/stages-wmdp/0_unlearn" / slug / "selected.yml"
            ).read_text()
        )
        assert selected[method]["output_dir"] == f"artifacts/tamia/models/{slug}"
        attack = yaml.safe_load(
            (
                EXAMPLE
                / "configs/stages-wmdp/2_relearn"
                / f"{slug}--relearn-fft"
                / "selected.yml"
            ).read_text()
        )
        assert "fsdp_version" not in attack
        assert attack["device_map"] == selected[method]["device_map"]

    assert selected["gd"]["learning_rate"] == 5e-6
    assert selected["gd"]["custom_trainer_args"]["retain_coefficient"] == 4
    assert selected["gd-sam-rho1e-2"]["learning_rate"] == 5e-6
    assert (
        selected["gd-sam-rho1e-2"]["custom_trainer_args"]["retain_coefficient"] == 2.2
    )
    assert selected["gd-sam-rho1e-2"]["custom_trainer_args"]["rho"] == 0.01
    for method in ("rmu", "rmu-sam-rho1e-2", "me-sam-rho1e-2"):
        slug = f"z7b-wmdp-bio--15pp--unlearn-fft-{method}"
        assert not (
            EXAMPLE / "configs/stages-wmdp/0_unlearn" / slug / "selected.yml"
        ).exists()
        assert all(f"{slug}--relearn-fft" not in stage.id for stage in plan.stages)


def test_optml_z7b_15pp_pipeline_relearns_published_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load(
        "configs/experiments/optml-z7b-wmdp-bio-unlearn--15pp.yml"
    ).plan()

    assert len(plan.stages) == 12
    train_stages = [stage for stage in plan.stages if stage.id.startswith("train-")]
    assert len(train_stages) == 4
    assert all(
        stage.script.endswith("/train_model_parallel.sbatch") for stage in train_stages
    )

    hub_models = {
        "gd": "OPTML-Group/GradDiff-WMDP",
        "gd-sam-rho1e-2": "OPTML-Group/GradDiff-SAM-WMDP",
        "npo": "OPTML-Group/NPO-WMDP",
        "npo-sam": "OPTML-Group/NPO-SAM-WMDP",
    }
    for method, hub_model in hub_models.items():
        slug = f"optml-z7b-wmdp-bio--15pp--unlearn-fft-{method}--relearn-fft"
        config = yaml.safe_load(
            (
                EXAMPLE / "configs/stages-wmdp/2_relearn" / slug / "selected.yml"
            ).read_text()
        )
        assert config["base_model"] == hub_model
        assert config["output_dir"] == f"artifacts/tamia/models/{slug}"
        assert config["max_steps"] == 16
        assert config["learning_rate"] == 1e-5
        assert config["save_steps"] == 4
        assert "fsdp_version" not in config
        assert len(config["device_map"]) == 35

    eval_script = (
        EXAMPLE / "scripts/slurm/tamia/eval_local_trajectory.sbatch"
    ).read_text()
    assert 'repo_cache="$HF_HOME/hub/models--${checkpoint//\\//--}"' in eval_script
    assert 'revision=$(<"$repo_cache/refs/main")' in eval_script
    assert 'checkpoint="$repo_cache/snapshots/$revision"' in eval_script


def test_minicpm5_rmu_pipeline_and_promoted_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = (
        lilpipe.load("configs/experiments/minicpm5-1b-wmdp-bio-unlearn.yml")
        .select(
            models=["minicpm5-1b-wmdp-bio--unlearn-fft-rmu--relearn-fft-long"],
        )
        .plan()
    )

    assert len(plan.stages) == 4
    assert all("--gres=gpu:l40s:1" in stage.sbatch_args for stage in plan.stages)
    eval_stages = [stage for stage in plan.stages if stage.id.startswith("eval-")]
    assert len(eval_stages) == 2
    assert all("--array=0-25%1" in stage.sbatch_args for stage in eval_stages)
    assert all(stage.args[-1] == "5" for stage in eval_stages)

    config = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-rmu/selected.yml"
        ).read_text()
    )
    assert config["learning_rate"] == 1e-4
    assert config["custom_trainer_args"]["target_layer"] == 3
    assert config["custom_trainer_args"]["steering_coefficient"] == pytest.approx(
        25.679644599841616
    )
    assert config["custom_trainer_args"]["retain_coefficient"] == pytest.approx(
        637.6272203987833
    )


def test_minicpm5_ssiuu_pipeline_and_promoted_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = (
        lilpipe.load("configs/experiments/minicpm5-1b-wmdp-bio-unlearn.yml")
        .select(models=["minicpm5-1b-wmdp-bio--unlearn-fft-ssiuu--relearn-fft-long"])
        .plan()
    )

    assert len(plan.stages) == 4
    assert sum(stage.id.startswith("train-") for stage in plan.stages) == 2
    assert all("--gres=gpu:l40s:1" in stage.sbatch_args for stage in plan.stages)

    config = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-ssiuu/selected.yml"
        ).read_text()
    )
    assert config["learning_rate"] == 1e-6
    assert config["max_steps"] == 100
    assert config["custom_trainer_args"]["retain_coefficient"] == pytest.approx(
        2.0224857803679672
    )
    assert config["custom_trainer_args"]["ssiuu_coefficient"] == pytest.approx(
        0.00035514729501627665
    )


def test_minicpm5_npo_pipeline_uses_promoted_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = (
        lilpipe.load("configs/experiments/minicpm5-1b-wmdp-bio-unlearn.yml")
        .select(models=["minicpm5-1b-wmdp-bio--unlearn-fft-npo--relearn-fft-long"])
        .plan()
    )

    assert len(plan.stages) == 4
    assert sum(stage.id.startswith("train-") for stage in plan.stages) == 2
    assert all("--gres=gpu:l40s:1" in stage.sbatch_args for stage in plan.stages)

    config = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-npo/selected.yml"
        ).read_text()
    )
    assert config["learning_rate"] == 7.4e-6
    assert config["max_steps"] == 150
    assert config["custom_trainer_args"]["beta"] == 0.0225
    assert config["custom_trainer_args"]["retain_coefficient"] == 2.0


OPTUNA_SCRIPT = EXAMPLE / "scripts" / "optuna_hpo.py"
OPTUNA_CONFIG = (
    EXAMPLE
    / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-rmu/optuna.yml"
)
OPTUNA_BOUNDARY_CONFIG = (
    EXAMPLE
    / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-rmu/optuna-boundary.yml"
)
OPTUNA_GD_CONFIG = (
    EXAMPLE
    / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-gd/optuna.yml"
)
OPTUNA_NPO_CONFIG = (
    EXAMPLE
    / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-npo/optuna.yml"
)
OPTUNA_NPO_BOUNDARY_CONFIG = OPTUNA_NPO_CONFIG.with_name("optuna-boundary.yml")
OPTUNA_SSIUU_CONFIG = (
    EXAMPLE
    / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-ssiuu/optuna.yml"
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
        "base_config": "configs/stages-wmdp/2_relearn/minicpm5-1b-wmdp-bio--unlearn-fft-rmu--relearn-fft-long/selected.yml",
        "max_steps": 125,
        "learning_rate": 1e-5,
    }


def test_optuna_boundary_config_searches_between_trials_12_and_18(hpo) -> None:
    config = hpo.load_config(OPTUNA_BOUNDARY_CONFIG)
    assert config["study"]["trials"] == 10
    assert config["study"]["waves"] == [4, 4, 2]
    assert config["study"]["name"] == "minicpm5-1b-rmu-boundary"
    assert config["search"] == {
        "layer": [3],
        "learning_rate": [1e-4],
        "steering_coefficient": {
            "low": 9.692328066325716,
            "high": 83.26873312246681,
        },
        "retain_coefficient": {
            "low": 615.672187962073,
            "high": 682.8453324349174,
        },
    }
    assert config["screen"]["mmlu_min"] == 0.5062


def test_minicpm_gd_optuna_search_and_generated_config(hpo, tmp_path) -> None:
    config = hpo.load_config(OPTUNA_GD_CONFIG)
    assert config["study"]["trials"] == 24
    assert config["study"]["waves"] == [4, 4, 4, 4, 4, 4]
    assert config["study"]["objectives"] == ["wmdp_pre"]
    assert config["study"]["directions"] == ["minimize"]
    assert "attack" not in config
    assert config["search"]["learning_rate"] == [5e-7, 1e-6, 2.5e-6, 5e-6, 1e-5]
    assert config["search"]["max_steps"] == [50, 100, 150, 250]

    class GDTrial:
        def suggest_categorical(self, name, values):
            return {"learning_rate": 2.5e-6, "max_steps": 100}[name]

        def suggest_float(self, name, low, high, log=False):
            assert name == "retain_coefficient"
            assert (low, high, log) == (0.25, 16.0, True)
            return 3.5

    parameters = hpo.sample_parameters(GDTrial(), config["search"])
    base = hpo.load_mapping(EXAMPLE / config["unlearn"]["base_config"])
    generated = hpo.build_unlearn_config(base, parameters, tmp_path / "trial-000")
    assert generated["trainer_cls"] == "trainers.gd.GradDiffTrainer"
    assert generated["learning_rate"] == 2.5e-6
    assert generated["max_steps"] == 100
    assert generated["save_steps"] == 100
    assert generated["custom_trainer_args"] == {
        "forget_coefficient": 1.0,
        "retain_coefficient": 3.5,
        "sampling": "optml",
    }
    assert "unfrozen_parameters" not in generated


def test_minicpm_gd_selected_config_promotes_best_feasible_trial() -> None:
    selected = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-gd/selected.yml"
        ).read_text()
    )
    assert selected["learning_rate"] == 1e-6
    assert selected["max_steps"] == 100
    assert selected["save_steps"] == 100
    assert selected["custom_trainer_args"]["retain_coefficient"] == pytest.approx(
        0.630454472334528
    )


def test_minicpm_npo_optuna_search_and_generated_config(hpo, tmp_path) -> None:
    config = hpo.load_config(OPTUNA_NPO_CONFIG)
    assert config["study"]["trials"] == 8
    assert config["study"]["waves"] == [4, 4]
    assert config["study"]["objectives"] == ["wmdp_pre"]
    assert config["study"]["directions"] == ["minimize"]
    assert "attack" not in config
    assert config["search"] == {
        "learning_rate": [2.5e-6, 5e-6, 1e-5],
        "retain_coefficient": {"low": 1.0, "high": 2.5},
        "beta": {"low": 0.01, "high": 0.05},
    }

    class NPOTrial:
        def suggest_categorical(self, name, values):
            assert name == "learning_rate"
            return 5e-6

        def suggest_float(self, name, low, high, log=False):
            assert log is False
            return {"retain_coefficient": 1.75, "beta": 0.0225}[name]

    parameters = hpo.sample_parameters(NPOTrial(), config["search"])
    base = hpo.load_mapping(EXAMPLE / config["unlearn"]["base_config"])
    generated = hpo.build_unlearn_config(base, parameters, tmp_path / "trial-000")
    assert generated["trainer_cls"] == "trainers.npo.NPOTrainer"
    assert generated["max_steps"] == 150
    assert generated["save_steps"] == 150
    assert generated["learning_rate"] == 5e-6
    assert generated["custom_trainer_args"] == {
        "reference_model_name_or_path": "openbmb/MiniCPM5-1B",
        "beta": 0.0225,
        "retain_coefficient": 1.75,
    }
    assert generated["wandb_name"] == (
        "minicpm5-1b-wmdp-bio--unlearn-fft-npo-optuna-trial-000"
    )


def test_minicpm_npo_selected_config_promotes_best_feasible_trial() -> None:
    selected = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-npo/selected.yml"
        ).read_text()
    )
    assert selected["learning_rate"] == 7.4e-6
    assert selected["max_steps"] == 150
    assert selected["custom_trainer_args"]["beta"] == 0.0225
    assert selected["custom_trainer_args"]["retain_coefficient"] == 2.0


def test_minicpm_npo_boundary_study_searches_continuous_learning_rate(hpo) -> None:
    config = hpo.load_config(OPTUNA_NPO_BOUNDARY_CONFIG)
    assert config["study"]["trials"] == 12
    assert config["study"]["waves"] == [4, 4, 4]
    assert config["search"]["learning_rate"] == {"low": 5e-6, "high": 1e-5}
    assert [trial["learning_rate"] for trial in config["study"]["initial_trials"]] == [
        6e-6,
        7e-6,
        8e-6,
        9e-6,
        7.2e-6,
        7.4e-6,
        7.6e-6,
        7.8e-6,
        7.7e-6,
        7.7e-6,
        7.7e-6,
        7.8e-6,
    ]

    class BoundaryTrial:
        def suggest_categorical(self, name, values):
            raise AssertionError("the boundary search must use a continuous LR")

        def suggest_float(self, name, low, high, log=False):
            assert log is False
            return {
                "learning_rate": 7.5e-6,
                "retain_coefficient": 2.0,
                "beta": 0.0225,
            }[name]

    assert hpo.sample_parameters(BoundaryTrial(), config["search"]) == {
        "learning_rate": 7.5e-6,
        "retain_coefficient": 2.0,
        "beta": 0.0225,
    }


def test_minicpm_ssiuu_optuna_starts_from_paper_values(hpo, tmp_path) -> None:
    config = hpo.load_config(OPTUNA_SSIUU_CONFIG)
    assert config["study"]["objectives"] == ["wmdp_pre", "wmdp_post"]
    assert [
        trial["ssiuu_coefficient"] for trial in config["study"]["initial_trials"]
    ] == [0.0001, 0.001, 0.01, 1.0]

    class SSIUUTrial:
        def suggest_categorical(self, name, values):
            return {"learning_rate": 2.5e-6, "max_steps": 150}[name]

        def suggest_float(self, name, low, high, log=False):
            assert log
            return {"retain_coefficient": 4.0, "ssiuu_coefficient": 0.001}[name]

    parameters = hpo.sample_parameters(SSIUUTrial(), config["search"])
    base = hpo.load_mapping(EXAMPLE / config["unlearn"]["base_config"])
    generated = hpo.build_unlearn_config(base, parameters, tmp_path / "trial-000")
    assert generated["trainer_cls"] == "trainers.ssiuu.SSIUUTrainer"
    assert generated["custom_trainer_args"]["ssiuu_coefficient"] == 0.001
    assert generated["custom_trainer_args"]["probability_normalization"] == (
        "geometric_mean"
    )
    assert generated["wandb_name"] == (
        "minicpm5-1b-wmdp-bio--unlearn-fft-ssiuu-optuna-trial-000"
    )


def test_pre_attack_only_trial_skips_attack(hpo, tmp_path, monkeypatch) -> None:
    config = hpo.load_config(OPTUNA_GD_CONFIG)
    config["artifacts"] = str(tmp_path / "trials")
    monkeypatch.chdir(EXAMPLE)
    calls = []
    monkeypatch.setattr(
        hpo, "evaluate", lambda *args, **kwargs: {"wmdp": 0.23, "mmlu": 0.51}
    )

    class GDTrial:
        number = 0
        user_attrs = {}

        def suggest_categorical(self, name, values):
            return {"learning_rate": 2.5e-6, "max_steps": 100}[name]

        def suggest_float(self, name, low, high, log=False):
            return 4.0

        def set_user_attr(self, name, value):
            self.user_attrs[name] = value

    trial = GDTrial()
    result = hpo.run_trial(trial, config, lambda command, check: calls.append(command))
    assert result == pytest.approx(0.23)
    assert len(calls) == 1
    assert trial.user_attrs["mmlu_constraints"] == pytest.approx([-0.0038])


def test_generated_configs_are_trial_local_and_derive_layers(hpo, tmp_path) -> None:
    base = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages-wmdp/0_unlearn/minicpm5-1b-wmdp-bio--unlearn-fft-rmu/selected.yml"
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
            / "configs/stages-wmdp/2_relearn/minicpm5-1b-wmdp-bio--unlearn-fft-rmu--relearn-fft-long/selected.yml"
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
    assert hpo.training_command(config_path, "python") == [
        "axolotl",
        "train",
        str(config_path),
        "--launcher",
        "python",
    ]
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
    assert "--gres=gpu:l40s:1" in command
    assert command[-2:] == [
        "scripts/slurm/mila/optuna_worker.sbatch",
        str(tmp_path / "optuna.yml"),
    ]


def test_submit_selects_rtx8000_from_study_config(hpo, tmp_path, monkeypatch) -> None:
    config = hpo.load_config(OPTUNA_NPO_CONFIG)
    monkeypatch.setattr(hpo, "create_study", lambda config: SimpleNamespace(trials=[]))
    monkeypatch.setattr(
        hpo,
        "optuna_module",
        lambda: SimpleNamespace(
            trial=SimpleNamespace(TrialState=SimpleNamespace(RUNNING="RUNNING"))
        ),
    )
    command = hpo.submit(tmp_path / "optuna.yml", config, dry_run=True)
    assert "--array=0-3%4" in command
    assert "--gres=gpu:rtx8000:1" in command


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


def test_worker_claims_enqueued_literature_trial(hpo, monkeypatch) -> None:
    states = ["COMPLETE"] * 8 + ["WAITING"] * 4

    class Study:
        trials = [SimpleNamespace(state=SimpleNamespace(name=name)) for name in states]
        optimized = False

        def optimize(self, objective, n_trials):
            del objective
            assert n_trials == 1
            self.optimized = True

    study = Study()
    monkeypatch.setattr(hpo, "create_study", lambda config: study)

    hpo.worker({"study": {"trials": 12}})

    assert study.optimized


def test_sequential_worker_runs_every_remaining_trial(hpo, monkeypatch) -> None:
    states = ["COMPLETE"] * 3

    class Study:
        trials = [SimpleNamespace(state=SimpleNamespace(name=name)) for name in states]
        optimized_trials = None

        def optimize(self, objective, n_trials):
            del objective
            self.optimized_trials = n_trials

    study = Study()
    monkeypatch.setattr(hpo, "create_study", lambda config: study)
    hpo.worker({"study": {"execution": "sequential", "trials": 16}})
    assert study.optimized_trials == 13


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


def test_tamia_sequential_worker_requests_four_h100s() -> None:
    script = (
        EXAMPLE / "scripts/slurm/tamia/optuna_worker_model_parallel.sbatch"
    ).read_text()
    assert "#SBATCH --gpus-per-node=h100:4" in script
    assert "#SBATCH --time=03:00:00" in script
    assert "if [[ $# -ne 1 ]]" in script
    assert 'python scripts/optuna_hpo.py worker "$1"' in script
