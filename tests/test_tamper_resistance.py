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
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.unused_weight = torch.nn.Parameter(torch.tensor(3.0))

        def forward(self, input_ids, attention_mask, labels):
            assert torch.equal(input_ids, labels)
            assert torch.all(attention_mask == 1)
            return SimpleNamespace(loss=input_ids.float().mean() * self.weight)

    model = Model()
    trainer = object.__new__(gd_sam_training_module.GradDiffTrainer)
    trainer.attack = gd_sam_training_module.SAMAttack(0.01)
    trainer.forget_coeff = 1.0
    trainer.retain_coeff = 1.0
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    loss = trainer.compute_loss(model, inputs)

    assert gd_sam_training_module.GradDiffTrainer.rho == 0.01
    assert trainer.attack.rho == pytest.approx(0.01)
    assert loss.item() == pytest.approx(-3 * (2 - 0.01) + 8 * 2)
    loss.backward()
    assert model.weight.grad.item() == pytest.approx(-3 + 8)
    assert model.unused_weight.grad is None


def test_gd_sam_peft_gradient_checkpointing(gd_sam_training_module) -> None:
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
    trainer.attack = gd_sam_training_module.SAMAttack(0.01)
    trainer.forget_coeff = 1.0
    trainer.retain_coeff = 1.0
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
