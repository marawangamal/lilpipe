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
    return load_script("configs/unlearn/utils.py", "orth_circuit_breaker_training")


@pytest.fixture(scope="module")
def npo_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/unlearn/npo.py", "npo_training")
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def npo_sam_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/unlearn/npo_sam.py", "npo_sam_training")
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def gd_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/unlearn/gd.py", "gd_training")
    finally:
        sys.path.remove(str(EXAMPLE))


@pytest.fixture(scope="module")
def gd_gn_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE))
    try:
        return load_script("configs/unlearn/gd_gn.py", "gd_gn_training")
    finally:
        sys.path.remove(str(EXAMPLE))


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
    assert "last_step=${5:-}" in script
    assert '[[ -n "$last_step" ]] && (( step > last_step ))' in script
    assert "step=$last_step" in script
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
    trainer = object.__new__(npo_training_module.BalancedNPOTrainer)
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "cb_source": torch.tensor([0, 1, 0, 1]),
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
    trainer = object.__new__(npo_training_module.BalancedNPOTrainer)
    with pytest.raises(ValueError, match="forget and retain"):
        trainer.compute_loss(None, {"cb_source": torch.tensor([1, 1])})


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
        "cb_source": torch.tensor([1, 0]),
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
        "cb_source": torch.tensor([1, 0]),
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
        "cb_source": torch.tensor([1, 0]),
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


def test_grad_diff_loss_and_gradient_direction(gd_training_module) -> None:
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
    trainer = object.__new__(gd_training_module.BalancedGradDiffTrainer)
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "cb_source": torch.tensor([0, 1, 0, 1]),
    }
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert loss.item() == pytest.approx(-3 * 2 + 8 * 3)
    assert outputs.loss.item() == pytest.approx(3 * 2)
    assert model.calls == [[2, 4], [7, 9]]
    loss.backward()
    assert model.forget_weight.grad.item() == pytest.approx(-3)
    assert model.retain_weight.grad.item() == pytest.approx(8)


@pytest.mark.parametrize("sources", [[1, 1], [0, 0]])
def test_grad_diff_requires_both_sources(gd_training_module, sources) -> None:
    torch = pytest.importorskip("torch")
    trainer = object.__new__(gd_training_module.BalancedGradDiffTrainer)
    with pytest.raises(ValueError, match="forget and retain"):
        trainer.compute_loss(None, {"cb_source": torch.tensor(sources)})


def test_grad_diff_sampler_balances_each_microbatch(gd_training_module) -> None:
    class TaggedDataset:
        def __getitem__(self, key):
            assert key == "cb_source"
            return [0, 1, 1, 0] * 2

    trainer = SimpleNamespace(
        train_dataset=TaggedDataset(),
        args=SimpleNamespace(per_device_train_batch_size=4, data_seed=None, seed=42),
    )
    sampler = gd_training_module.BalancedGradDiffTrainer._get_train_sampler(trainer)
    indices = list(sampler)
    assert sorted(indices) == list(range(8))
    for start in (0, 4):
        assert (
            sum(
                trainer.train_dataset["cb_source"][i]
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
        "cb_source": torch.tensor([0, 1]),
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
        trainer.compute_loss(None, {"cb_source": torch.tensor(sources)})


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
        "cb_source": torch.tensor([1, 0]),
    }
    loss = trainer.compute_loss(model, inputs)
    loss.backward()
    assert loss.isfinite()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
        if parameter.requires_grad
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


def test_orth_cb_config() -> None:
    model_id = "di-6.9b-wmdp-bio-unlearn-cb"
    training_dir = EXAMPLE / "configs/unlearn/di-6.9b"
    assert sorted(path.name for path in training_dir.glob("*.yml")) == [
        "wmdp-bio-unlearn-cb.yml",
        "wmdp-bio-unlearn-gd-gn.yml",
        "wmdp-bio-unlearn-gd.yml",
        "wmdp-bio-unlearn-npo-sam-beta015-gamma225.yml",
        "wmdp-bio-unlearn-npo-sam-gamma225.yml",
        "wmdp-bio-unlearn-npo-sam-gamma450.yml",
        "wmdp-bio-unlearn-npo-sam-rho003.yml",
        "wmdp-bio-unlearn-npo-sam.yml",
        "wmdp-bio-unlearn-npo.yml",
        "wmdp-bio-unlearn-ws-alpha-10.yml",
        "wmdp-bio-unlearn-ws-alpha-2.yml",
        "wmdp-bio-unlearn-ws-alpha-4.yml",
        "wmdp-bio-unlearn-ws-ft-forget.yml",
        "wmdp-bio-unlearn-ws-ft-retain.yml",
        "wmdp-bio-unlearn-ws-r1-f01.yml",
        "wmdp-bio-unlearn-ws.yml",
    ]
    config = yaml.safe_load((training_dir / "wmdp-bio-unlearn-cb.yml").read_text())
    assert config["trainer_cls"] == (
        "configs.unlearn.utils.BalancedOrthCircuitBreakerTrainer"
    )
    assert config["output_dir"] == f"artifacts/models/{model_id}"
    assert config["wandb_name"] == model_id
    assert config["dataset_prepared_path"] == (
        f"artifacts/cache/axolotl/{model_id}/prepared"
    )
    assert config["lora_target_modules"] == [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    assert config["peft_layers_to_transform"] == list(range(31))
    assert config["lora_r"] == config["lora_alpha"] == 8
    assert config["micro_batch_size"] == 8
    assert config["gradient_accumulation_steps"] == 8
    assert config["max_steps"] == 32
    assert config["shuffle_merged_datasets"] is True
    assert [dataset["path"] for dataset in config["datasets"]] == [
        "cais/wmdp-bio-forget-corpus",
        "EleutherAI/wikitext_document_level",
    ]
    assert config["datasets"][0]["split"] == "train[:1024]"
    assert config["datasets"][1]["split"] == "train"
    assert all(
        dataset["type"] == "configs.unlearn.utils" for dataset in config["datasets"]
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


def test_relearning_config_and_script() -> None:
    model_id = "di-6.9b-wmdp-bio-unlearn-cb-relearn"
    config = yaml.safe_load(
        (EXAMPLE / "configs/relearn/di-6.9b/wmdp-bio-relearn.yml").read_text()
    )
    assert config["base_model"] == (
        "artifacts/models/di-6.9b-wmdp-bio-unlearn-cb/merged"
    )
    assert config["output_dir"] == f"artifacts/models/{model_id}"
    assert config["wandb_name"] == model_id
    assert config["dataset_prepared_path"] == (
        f"artifacts/cache/axolotl/{model_id}/prepared"
    )
    assert config["datasets"] == [
        {
            "path": "cais/wmdp-bio-forget-corpus",
            "split": "train[:1024]",
            "type": "configs.relearn.utils",
        }
    ]
    assert "trainer_cls" not in config
    assert config["lora_r"] == config["lora_alpha"] == 8
    assert config["peft_layers_to_transform"] == list(range(31))
    assert config["sequence_len"] == 2048
    assert config["micro_batch_size"] == 2
    assert config["gradient_accumulation_steps"] == 16
    assert config["max_steps"] == 32
    assert config["learning_rate"] == 1e-3
    assert config["lr_scheduler"] == "linear"
    assert config["warmup_steps"] == 0

    script = (EXAMPLE / "scripts/slurm/relearn.sbatch").read_text()
    assert 'axolotl merge-lora "$1"' in script
    assert 'axolotl train "$2" --launcher python' in script
    assert script.index("merge-lora") < script.index("axolotl train")


def test_npo_configs_match_cb_budget_and_relearning_schedule() -> None:
    config_dir = EXAMPLE / "configs"
    cb = yaml.safe_load(
        (config_dir / "unlearn/di-6.9b/wmdp-bio-unlearn-cb.yml").read_text()
    )
    npo = yaml.safe_load(
        (config_dir / "unlearn/di-6.9b/wmdp-bio-unlearn-npo.yml").read_text()
    )
    assert npo["trainer_cls"] == "configs.unlearn.npo.BalancedNPOTrainer"
    assert npo["datasets"] == cb["datasets"]
    for key in (
        "adapter",
        "lora_target_modules",
        "peft_layers_to_transform",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "lr_scheduler",
        "warmup_steps",
        "max_grad_norm",
    ):
        assert npo[key] == cb[key]
    model_id = "di-6.9b-wmdp-bio-unlearn-npo"
    assert npo["output_dir"] == f"artifacts/models/{model_id}"
    assert (
        npo["dataset_prepared_path"] == f"artifacts/cache/axolotl/{model_id}/prepared"
    )
    assert npo["wandb_name"] == model_id
    assert "beta" not in npo and "gamma" not in npo

    cb_relearn = yaml.safe_load(
        (config_dir / "relearn/di-6.9b/wmdp-bio-relearn.yml").read_text()
    )
    npo_relearn = yaml.safe_load(
        (config_dir / "relearn/di-6.9b/wmdp-bio-npo-relearn.yml").read_text()
    )
    assert npo_relearn["base_model"] == f"artifacts/models/{model_id}/merged"
    assert npo_relearn["output_dir"] == f"artifacts/models/{model_id}-relearn"
    assert npo_relearn["dataset_prepared_path"] == (
        f"artifacts/cache/axolotl/{model_id}-relearn/prepared"
    )
    assert npo_relearn["wandb_name"] == f"{model_id}-relearn"
    for key in (
        "datasets",
        "lora_r",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "learning_rate",
        "lr_scheduler",
        "warmup_steps",
    ):
        assert npo_relearn[key] == cb_relearn[key]


def test_npo_sam_configs_and_pipeline() -> None:
    config_dir = EXAMPLE / "configs"
    model_id = "di-6.9b-wmdp-bio-unlearn-npo-sam"
    npo = yaml.safe_load(
        (config_dir / "unlearn/di-6.9b/wmdp-bio-unlearn-npo.yml").read_text()
    )
    sam = yaml.safe_load(
        (config_dir / "unlearn/di-6.9b/wmdp-bio-unlearn-npo-sam.yml").read_text()
    )
    excluded = {"trainer_cls", "dataset_prepared_path", "output_dir", "wandb_name"}
    assert {key: value for key, value in sam.items() if key not in excluded} == {
        key: value for key, value in npo.items() if key not in excluded
    }
    assert sam["trainer_cls"] == "configs.unlearn.npo_sam.BalancedNPOSAMTrainer"
    assert sam["output_dir"] == f"artifacts/models/{model_id}"
    assert (
        sam["dataset_prepared_path"] == f"artifacts/cache/axolotl/{model_id}/prepared"
    )
    assert sam["wandb_name"] == model_id
    npo_relearn = yaml.safe_load(
        (config_dir / "relearn/di-6.9b/wmdp-bio-npo-relearn.yml").read_text()
    )
    sam_relearn = yaml.safe_load(
        (config_dir / "relearn/di-6.9b/wmdp-bio-npo-sam-relearn.yml").read_text()
    )
    excluded = {"base_model", "dataset_prepared_path", "output_dir", "wandb_name"}
    assert {
        key: value for key, value in sam_relearn.items() if key not in excluded
    } == {key: value for key, value in npo_relearn.items() if key not in excluded}
    assert sam_relearn["base_model"] == f"artifacts/models/{model_id}/merged"
    assert sam_relearn["output_dir"] == f"artifacts/models/{model_id}-relearn"

    import os

    previous_cwd = Path.cwd()
    try:
        os.chdir(EXAMPLE)
        pipeline = lilpipe.load("configs/experiments/di-6.9b.yml")
    finally:
        os.chdir(previous_cwd)
    plan = pipeline.select(models=[model_id, f"{model_id}-relearn"]).plan()
    stages = plan.stage_index
    assert f"train-{model_id}" in stages
    assert f"train-{model_id}-relearn" in stages
    assert (
        len([stage for stage in stages if model_id in stage and "eval" in stage]) == 4
    )


@pytest.mark.parametrize(
    ("suffix", "trainer_name", "beta", "gamma", "rho"),
    [
        ("rho003", "BalancedNPOSAMRho003Trainer", 0.0225, 1.0, 0.003),
        ("gamma225", "BalancedNPOSAMGamma225Trainer", 0.0225, 2.25, 0.01),
        ("gamma450", "BalancedNPOSAMGamma450Trainer", 0.0225, 4.5, 0.01),
        (
            "beta015-gamma225",
            "BalancedNPOSAMBeta015Gamma225Trainer",
            0.015,
            2.25,
            0.01,
        ),
    ],
)
def test_npo_sam_tuning_configs(
    suffix: str, trainer_name: str, beta: float, gamma: float, rho: float
) -> None:
    config_dir = EXAMPLE / "configs/unlearn/di-6.9b"
    baseline = yaml.safe_load((config_dir / "wmdp-bio-unlearn-npo-sam.yml").read_text())
    path = config_dir / f"wmdp-bio-unlearn-npo-sam-{suffix}.yml"
    variant = yaml.safe_load(path.read_text())
    model_id = f"di-6.9b-wmdp-bio-unlearn-npo-sam-{suffix}"
    excluded = {"trainer_cls", "dataset_prepared_path", "output_dir", "wandb_name"}
    assert {key: value for key, value in variant.items() if key not in excluded} == {
        key: value for key, value in baseline.items() if key not in excluded
    }
    assert variant["trainer_cls"] == f"configs.unlearn.npo_sam.{trainer_name}"
    assert variant["output_dir"] == f"artifacts/models/{model_id}"
    assert variant["dataset_prepared_path"] == (
        f"artifacts/cache/axolotl/{model_id}/prepared"
    )
    assert variant["wandb_name"] == model_id
    comment = path.read_text().splitlines()[1]
    assert f"beta = {beta}" in comment
    assert f"gamma = {gamma}" in comment
    assert f"rho = {rho}" in comment

    relearn_dir = EXAMPLE / "configs/relearn/di-6.9b"
    relearn = yaml.safe_load(
        (relearn_dir / f"wmdp-bio-npo-sam-{suffix}-relearn.yml").read_text()
    )
    baseline_relearn = yaml.safe_load(
        (relearn_dir / "wmdp-bio-npo-sam-relearn.yml").read_text()
    )
    excluded = {"base_model", "dataset_prepared_path", "output_dir", "wandb_name"}
    assert {key: value for key, value in relearn.items() if key not in excluded} == {
        key: value for key, value in baseline_relearn.items() if key not in excluded
    }
    assert relearn["base_model"] == f"artifacts/models/{model_id}/merged"
    assert relearn["output_dir"] == f"artifacts/models/{model_id}-relearn"
    assert relearn["dataset_prepared_path"] == (
        f"artifacts/cache/axolotl/{model_id}-relearn/prepared"
    )
    assert relearn["wandb_name"] == f"{model_id}-relearn"


def test_grad_diff_configs_match_npo_and_relearning_schedule() -> None:
    config_dir = EXAMPLE / "configs"
    npo = yaml.safe_load(
        (config_dir / "unlearn/di-6.9b/wmdp-bio-unlearn-npo.yml").read_text()
    )
    gd = yaml.safe_load(
        (config_dir / "unlearn/di-6.9b/wmdp-bio-unlearn-gd.yml").read_text()
    )
    model_id = "di-6.9b-wmdp-bio-unlearn-gd"
    assert gd["trainer_cls"] == "configs.unlearn.gd.BalancedGradDiffTrainer"
    excluded = {"trainer_cls", "dataset_prepared_path", "output_dir", "wandb_name"}
    assert {key: value for key, value in gd.items() if key not in excluded} == {
        key: value for key, value in npo.items() if key not in excluded
    }
    assert gd["output_dir"] == f"artifacts/models/{model_id}"
    assert gd["dataset_prepared_path"] == (
        f"artifacts/cache/axolotl/{model_id}/prepared"
    )
    assert gd["wandb_name"] == model_id

    npo_relearn = yaml.safe_load(
        (config_dir / "relearn/di-6.9b/wmdp-bio-npo-relearn.yml").read_text()
    )
    gd_relearn = yaml.safe_load(
        (config_dir / "relearn/di-6.9b/wmdp-bio-gd-relearn.yml").read_text()
    )
    excluded = {"base_model", "dataset_prepared_path", "output_dir", "wandb_name"}
    assert {key: value for key, value in gd_relearn.items() if key not in excluded} == {
        key: value for key, value in npo_relearn.items() if key not in excluded
    }
    assert gd_relearn["base_model"] == f"artifacts/models/{model_id}/merged"
    assert gd_relearn["output_dir"] == f"artifacts/models/{model_id}-relearn"
    assert gd_relearn["dataset_prepared_path"] == (
        f"artifacts/cache/axolotl/{model_id}-relearn/prepared"
    )
    assert gd_relearn["wandb_name"] == f"{model_id}-relearn"


def test_gd_gn_configs_and_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = EXAMPLE / "configs"
    model_id = "di-6.9b-wmdp-bio-unlearn-gd-gn"
    training = yaml.safe_load(
        (config_dir / "unlearn/di-6.9b/wmdp-bio-unlearn-gd-gn.yml").read_text()
    )
    assert training["trainer_cls"] == (
        "configs.unlearn.gd_gn.BalancedGradDiffGNTrainer"
    )
    assert training["output_dir"] == f"artifacts/models/{model_id}"
    assert training["micro_batch_size"] == 2
    assert training["gradient_accumulation_steps"] == 32
    assert training["micro_batch_size"] * training["gradient_accumulation_steps"] == 64
    assert training["attn_implementation"] == "eager"
    assert training["dataset_prepared_path"] == (
        f"artifacts/cache/axolotl/{model_id}/prepared"
    )
    relearn = yaml.safe_load(
        (config_dir / "relearn/di-6.9b/wmdp-bio-gd-gn-relearn.yml").read_text()
    )
    assert relearn["base_model"] == f"artifacts/models/{model_id}/merged"
    assert relearn["output_dir"] == f"artifacts/models/{model_id}-relearn"

    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/di-6.9b.yml").plan()
    train_stage = plan.stage_index[f"train-{model_id}"]
    assert train_stage.args == ("configs/unlearn/di-6.9b/wmdp-bio-unlearn-gd-gn.yml",)
    assert "--gres=gpu:a100l:1" in train_stage.sbatch_args
    relearn_stage = plan.stage_index[f"train-{model_id}-relearn"]
    assert relearn_stage.depends_on == (train_stage.id,)
    assert relearn_stage.args == (
        "configs/unlearn/di-6.9b/wmdp-bio-unlearn-gd-gn.yml",
        "configs/relearn/di-6.9b/wmdp-bio-gd-gn-relearn.yml",
    )
    for evaluated_id in (model_id, f"{model_id}-relearn"):
        assert f"eval-bio-mcqa-{evaluated_id}" in plan.stage_index
        assert f"eval-mmlu-no-bio-{evaluated_id}" in plan.stage_index


def test_weight_steering_configs_and_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = EXAMPLE / "configs/unlearn/di-6.9b"
    cb = yaml.safe_load((config_dir / "wmdp-bio-unlearn-cb.yml").read_text())
    arms = {}
    for arm in ("retain", "forget"):
        model_id = f"di-6.9b-wmdp-bio-unlearn-ws-ft-{arm}"
        arms[arm] = yaml.safe_load(
            (config_dir / f"wmdp-bio-unlearn-ws-ft-{arm}.yml").read_text()
        )
        config = arms[arm]
        assert config["base_model"] == cb["base_model"]
        assert config["output_dir"] == f"artifacts/models/{model_id}"
        assert config["wandb_name"] == model_id
        assert config["datasets"][0]["type"] == "configs.unlearn.ws_data"
        for key in (
            "adapter",
            "lora_target_modules",
            "peft_layers_to_transform",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "learning_rate",
            "optimizer",
            "weight_decay",
            "lr_scheduler",
            "warmup_steps",
            "max_grad_norm",
            "sequence_len",
            "seed",
        ):
            assert config[key] == cb[key]
        assert "trainer_cls" not in config
    ignored = {"datasets", "dataset_prepared_path", "output_dir", "wandb_name"}
    assert {k: v for k, v in arms["retain"].items() if k not in ignored} == {
        k: v for k, v in arms["forget"].items() if k not in ignored
    }
    assert arms["retain"]["datasets"][0]["path"] == (
        "EleutherAI/wikitext_document_level"
    )
    assert arms["forget"]["datasets"][0]["path"] == ("cais/wmdp-bio-forget-corpus")

    ws_id = "di-6.9b-wmdp-bio-unlearn-ws"
    steering = yaml.safe_load((config_dir / "wmdp-bio-unlearn-ws.yml").read_text())
    assert steering["adapter_pairs"] == [
        {
            "pos_adapter_name_or_path": f"artifacts/models/{ws_id}-ft-retain",
            "neg_adapter_name_or_path": f"artifacts/models/{ws_id}-ft-forget",
        }
    ]
    assert steering["steered_adapters"] == [
        {"alpha": 1.0, "output_path": f"artifacts/models/{ws_id}"}
    ]
    relearn = yaml.safe_load(
        (EXAMPLE / "configs/relearn/di-6.9b/wmdp-bio-ws-relearn.yml").read_text()
    )
    assert relearn["base_model"] == f"artifacts/models/{ws_id}/merged"

    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/di-6.9b.yml").plan()
    ws = plan.stage_index[f"build-{ws_id}"]
    assert ws.script == "scripts/slurm/weight_steering.sbatch"
    assert set(ws.depends_on) == {
        f"train-{ws_id}-ft-retain",
        f"train-{ws_id}-ft-forget",
    }
    ws_relearn = plan.stage_index[f"train-{ws_id}-relearn"]
    assert ws_relearn.depends_on == (ws.id,)
    assert ws_relearn.args[2:] == (
        f"artifacts/models/{ws_id}",
        f"artifacts/models/{ws_id}",
    )
    relearn_script = (EXAMPLE / "scripts/slurm/relearn.sbatch").read_text()
    assert '--lora-model-dir "$3" --output-dir "$4"' in relearn_script
    for model_id in (ws_id, f"{ws_id}-relearn"):
        assert f"eval-bio-mcqa-{model_id}" in plan.stage_index
        assert f"eval-mmlu-no-bio-{model_id}" in plan.stage_index

    for alpha in (2, 4, 10):
        model_id = f"{ws_id}-a{alpha}"
        config = yaml.safe_load(
            (config_dir / f"wmdp-bio-unlearn-ws-alpha-{alpha}.yml").read_text()
        )
        assert config["base_model_name_or_path"] == steering["base_model_name_or_path"]
        assert config["adapter_pairs"] == steering["adapter_pairs"]
        assert config["steered_adapters"] == [
            {"alpha": float(alpha), "output_path": f"artifacts/models/{model_id}"}
        ]
        stage = plan.stage_index[f"build-{model_id}"]
        assert stage.script == "scripts/slurm/weight_steering.sbatch"
        assert set(stage.depends_on) == set(ws.depends_on)
        assert f"eval-bio-mcqa-{model_id}" in plan.stage_index
        assert f"eval-mmlu-no-bio-{model_id}" in plan.stage_index


def test_relearning_formats_wmdp_document() -> None:
    pytest.importorskip("axolotl")
    strategy = load_script("configs/relearn/utils.py", "wmdp_bio_forget_strategy")
    document, prompt, response = (
        strategy.WmdpBioCompletionStrategy.parse_instruction_fields(
            None, {"title": "Title", "abstract": "Abstract", "text": "Text"}
        )
    )
    assert (document, prompt, response) == ("Title\n\nAbstract\n\nText", "", "")


def test_single_experiment_plans_base_cb_and_relearning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/di-6.9b.yml").plan()
    orth_id = "di-6.9b-wmdp-bio-unlearn-cb"
    relearn_id = f"{orth_id}-relearn"
    orth = plan.stage_index[f"train-{orth_id}"]
    assert orth.script == "scripts/slurm/train.sbatch"
    assert orth.args == ("configs/unlearn/di-6.9b/wmdp-bio-unlearn-cb.yml",)
    relearn = plan.stage_index[f"train-{relearn_id}"]
    assert relearn.script == "scripts/slurm/relearn.sbatch"
    assert relearn.args == (
        "configs/unlearn/di-6.9b/wmdp-bio-unlearn-cb.yml",
        "configs/relearn/di-6.9b/wmdp-bio-relearn.yml",
    )
    assert relearn.depends_on == (orth.id,)
    assert f"eval-bio-mcqa-{orth_id}" in plan.stage_index
    assert f"eval-mmlu-no-bio-{orth_id}" in plan.stage_index
    assert f"eval-bio-mcqa-{relearn_id}" in plan.stage_index
    assert f"eval-mmlu-no-bio-{relearn_id}" in plan.stage_index
    assert len(plan.stages) == 73
    npo_id = "di-6.9b-wmdp-bio-unlearn-npo"
    npo = plan.stage_index[f"train-{npo_id}"]
    assert npo.script == "scripts/slurm/train.sbatch"
    assert npo.args == ("configs/unlearn/di-6.9b/wmdp-bio-unlearn-npo.yml",)
    npo_relearn = plan.stage_index[f"train-{npo_id}-relearn"]
    assert npo_relearn.script == "scripts/slurm/relearn.sbatch"
    assert npo_relearn.args == (
        "configs/unlearn/di-6.9b/wmdp-bio-unlearn-npo.yml",
        "configs/relearn/di-6.9b/wmdp-bio-npo-relearn.yml",
    )
    assert npo_relearn.depends_on == (npo.id,)
    for model_id in (npo_id, f"{npo_id}-relearn"):
        assert f"eval-bio-mcqa-{model_id}" in plan.stage_index
        assert f"eval-mmlu-no-bio-{model_id}" in plan.stage_index
    gd_id = "di-6.9b-wmdp-bio-unlearn-gd"
    gd = plan.stage_index[f"train-{gd_id}"]
    assert gd.script == "scripts/slurm/train.sbatch"
    assert gd.args == ("configs/unlearn/di-6.9b/wmdp-bio-unlearn-gd.yml",)
    gd_relearn = plan.stage_index[f"train-{gd_id}-relearn"]
    assert gd_relearn.script == "scripts/slurm/relearn.sbatch"
    assert gd_relearn.args == (
        "configs/unlearn/di-6.9b/wmdp-bio-unlearn-gd.yml",
        "configs/relearn/di-6.9b/wmdp-bio-gd-relearn.yml",
    )
    assert gd_relearn.depends_on == (gd.id,)
    assert plan.stages.index(gd) < plan.stages.index(gd_relearn)
    for model_id in (gd_id, f"{gd_id}-relearn"):
        assert f"eval-bio-mcqa-{model_id}" in plan.stage_index
        assert f"eval-mmlu-no-bio-{model_id}" in plan.stage_index
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
    script = (EXAMPLE / "scripts/slurm/eval_mmlu_no_bio_single.sbatch").read_text()
    assert (
        'export HF_DATASETS_CACHE="$SLURM_TMPDIR/.cache/huggingface/datasets"' in script
    )
    assert "--tasks mmlu_no_bio" in script
    assert "--num_fewshot 0" in script
    assert "--batch_size 32" in script
    assert 'output="artifacts/evals/$model_id/mmlu-no-bio"' in script


def test_mmlu_no_bio_trajectory_evaluator_accepts_last_step() -> None:
    script = (EXAMPLE / "scripts/slurm/eval_mmlu_no_bio.sbatch").read_text()
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "step=$((${SLURM_ARRAY_TASK_ID" in script
    assert "* checkpoint_frequency))" in script
    assert "last_step=${5:-}" in script
    assert '[[ -n "$last_step" ]] && (( step > last_step ))' in script
    assert "step=$last_step" in script
    assert (
        'adapter_name_or_path_ckpt="$artifacts_dir/models/'
        '$adapter_name_or_path/checkpoint-$step"' in script
    )
    assert (
        'output="$artifacts_dir/evals/$adapter_name_or_path/'
        'checkpoint-$step/mmlu-no-bio"' in script
    )
    assert "--tasks mmlu_no_bio" in script


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
        "di-6.9b-wmdp-bio-unlearn-cb",
        "di-6.9b-wmdp-bio-unlearn-cb-relearn",
        "di-6.9b-wmdp-bio-unlearn-npo",
        "di-6.9b-wmdp-bio-unlearn-npo-relearn",
        "di-6.9b-wmdp-bio-unlearn-npo-sam",
        "di-6.9b-wmdp-bio-unlearn-npo-sam-relearn",
        "di-6.9b-wmdp-bio-unlearn-npo-sam-rho003",
        "di-6.9b-wmdp-bio-unlearn-npo-sam-gamma225",
        "di-6.9b-wmdp-bio-unlearn-npo-sam-gamma450",
        "di-6.9b-wmdp-bio-unlearn-npo-sam-beta015-gamma225",
        "di-6.9b-wmdp-bio-unlearn-npo-sam-rho003-relearn",
        "di-6.9b-wmdp-bio-unlearn-npo-sam-gamma225-relearn",
        "di-6.9b-wmdp-bio-unlearn-npo-sam-gamma450-relearn",
        "di-6.9b-wmdp-bio-unlearn-npo-sam-beta015-gamma225-relearn",
        "di-6.9b-wmdp-bio-unlearn-gd",
        "di-6.9b-wmdp-bio-unlearn-gd-relearn",
        "di-6.9b-wmdp-bio-unlearn-gd-gn",
        "di-6.9b-wmdp-bio-unlearn-gd-gn-relearn",
        "di-6.9b-wmdp-bio-unlearn-ws-ft-retain",
        "di-6.9b-wmdp-bio-unlearn-ws-ft-forget",
        "di-6.9b-wmdp-bio-unlearn-ws",
        "di-6.9b-wmdp-bio-unlearn-ws-relearn",
        "di-6.9b-wmdp-bio-unlearn-ws-a2",
        "di-6.9b-wmdp-bio-unlearn-ws-a4",
        "di-6.9b-wmdp-bio-unlearn-ws-a10",
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
    assert [row["id"] for row in config["rows"]] == [
        "base",
        "circuit-breaker",
        "cb-relearn",
        "npo",
        "npo-relearn",
        "npo-sam",
        "npo-sam-relearn",
        "npo-sam-rho003",
        "npo-sam-gamma225",
        "npo-sam-gamma450",
        "npo-sam-beta015-gamma225",
        "npo-sam-rho003-relearn",
        "npo-sam-gamma225-relearn",
        "npo-sam-gamma450-relearn",
        "npo-sam-beta015-gamma225-relearn",
        "gd",
        "gd-relearn",
        "gd-gn",
        "gd-gn-relearn",
        "weight-steering",
        "ws-relearn",
        "ws-a2",
        "ws-a4",
        "ws-a10",
    ]
    assert [row["group"] for row in config["rows"]] == [
        "base-model",
        "circuit-breaker",
        "circuit-breaker-relearn",
        "npo",
        "npo-relearn",
        "npo-sam",
        "npo-sam-relearn",
        "npo-sam-tuning",
        "npo-sam-tuning",
        "npo-sam-tuning",
        "npo-sam-tuning",
        "npo-sam-tuning-relearn",
        "npo-sam-tuning-relearn",
        "npo-sam-tuning-relearn",
        "npo-sam-tuning-relearn",
        "grad-diff",
        "grad-diff-relearn",
        "gd-gn",
        "gd-gn-relearn",
        "weight-steering",
        "weight-steering-relearn",
        "weight-steering",
        "weight-steering",
        "weight-steering",
    ]
    assert config["rows"][1]["root"] == ("artifacts/evals/di-6.9b-wmdp-bio-unlearn-cb")
    assert config["rows"][2]["root"] == (
        "artifacts/evals/di-6.9b-wmdp-bio-unlearn-cb-relearn"
    )

    for config_dir in ("unlearn", "relearn"):
        for training_path in (EXAMPLE / "configs" / config_dir).rglob("*.yml"):
            training = yaml.safe_load(training_path.read_text())
            if "steered_adapters" in training:
                continue
            assert training["output_dir"].startswith("artifacts/models/di-6.9b-")
            assert "lora64-epochs1" not in training["output_dir"]
            assert "LoRA64-Epochs1" not in training["wandb_name"]

    assert all("LoRA64-Epochs1" not in row["label"] for row in config["rows"])
