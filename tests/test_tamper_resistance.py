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


def test_cb_gradient_balance_defaults(cb_training_module) -> None:
    import inspect

    parameters = inspect.signature(
        cb_training_module.GradientBalancedCircuitBreakerTrainer.__init__
    ).parameters
    assert parameters["gradient_balance_steps"].default == 25
    assert parameters["gradient_balance_beta"].default == 0.9
    assert parameters["gradient_balance_min"].default == 1e-3
    assert parameters["gradient_balance_max"].default == 1e3
    assert hasattr(
        cb_training_module.GradientBalancedCircuitBreakerTrainer,
        "loss_coefficients",
    )


def test_cb_constant_and_linear_loss_schedules(cb_training_module) -> None:
    from types import SimpleNamespace

    constant = object.__new__(cb_training_module.ConstantLambda1CircuitBreakerTrainer)
    assert constant.loss_coefficients(None, None, None) == (1.0, 1.0)

    trainer = object.__new__(cb_training_module.LinearCircuitBreakerTrainer)
    trainer.state = SimpleNamespace(global_step=0, max_steps=301)
    assert trainer.loss_coefficients(None, None, None) == (0.0, 10.0)
    trainer.state.global_step = 150
    assert trainer.loss_coefficients(None, None, None) == (2.5, 7.5)
    trainer.state.global_step = 300
    assert trainer.loss_coefficients(None, None, None) == (5.0, 5.0)


def test_cb_linear_noise_subclass_sets_noise_without_yaml(
    cb_training_module, monkeypatch
) -> None:
    received = {}

    def fake_init(self, *args, **kwargs):
        del self, args
        received.update(kwargs)

    monkeypatch.setattr(cb_training_module.CircuitBreakerTrainer, "__init__", fake_init)
    cb_training_module.LinearNoise0p1CircuitBreakerTrainer(unrelated="preserved")
    assert received == {"unrelated": "preserved", "reroute_noise": 0.1}


def test_cb_noise_breaks_initial_rerouting_stationary_point(
    cb_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    from types import MethodType, SimpleNamespace

    trainer = object.__new__(cb_training_module.ConstantLambda1CircuitBreakerTrainer)
    trainer.target_layers = (0,)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.reroute_noise = 0.01
    trainer.reroute_noise_seed = 42
    trainer.log = lambda metrics: None

    reference = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    target = torch.tensor([[[[1.0, 2.0, 3.01]]]])
    current = reference.clone().requires_grad_()
    activations = iter((reference, reference, target, current))
    trainer.activations = MethodType(
        lambda self, *args, **kwargs: next(activations), trainer
    )
    batch = {
        "input_ids": torch.ones((1, 1), dtype=torch.long),
        "attention_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_mask": torch.ones((1, 1), dtype=torch.long),
    }

    loss = trainer.compute_loss(None, {"chosen": batch, "rejected": batch})
    loss.backward()

    assert current.grad.norm().item() > 0


def _gradient_balance_trainer(cb_training_module, torch):
    from types import MethodType, SimpleNamespace

    trainer = object.__new__(cb_training_module.GradientBalancedCircuitBreakerTrainer)
    trainer.target_layers = (0,)
    trainer.reroute_weight = 1.0
    trainer.gradient_balance_steps = 25
    trainer.gradient_balance_beta = 0.9
    trainer.gradient_balance_min = 1e-3
    trainer.gradient_balance_max = 1e3
    trainer._last_balance_step = None
    trainer.state = SimpleNamespace(global_step=0)
    trainer.reroute_noise = 0.01
    trainer.reroute_noise_seed = 42
    trainer.logged = []
    trainer.log = trainer.logged.append

    reference = torch.tensor([[[[1.0, 0.0]]]])
    current = torch.tensor([[[[1.0, 1.0]]]], requires_grad=True)

    def activations(self, model, batch, layers=None, **kwargs):
        del self, model, batch, layers
        return reference if kwargs.get("disable_grad") else current

    trainer.activations = MethodType(activations, trainer)
    batch = {
        "input_ids": torch.ones((1, 1), dtype=torch.long),
        "attention_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_mask": torch.ones((1, 1), dtype=torch.long),
    }
    return trainer, batch, current


def test_cb_gradient_balance_initial_ema_and_accumulation_guard(
    cb_training_module, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.tensor(1.0))
            self.other_weight = torch.nn.Parameter(torch.tensor(1.0))
            self.frozen_lora_weight = torch.nn.Parameter(
                torch.tensor(1.0), requires_grad=False
            )

    trainer, batch, _ = _gradient_balance_trainer(cb_training_module, torch)
    model = Model()
    gradient_norms = iter((2.0, 4.0, 4.0, 2.0))
    balanced_parameters = []

    def fake_grad(loss, parameters, **kwargs):
        del loss
        assert kwargs == {"retain_graph": True, "allow_unused": True}
        balanced_parameters.append(parameters)
        return (torch.tensor(next(gradient_norms)),)

    monkeypatch.setattr(torch.autograd, "grad", fake_grad)
    trainer.compute_loss(model, {"chosen": batch, "rejected": batch})
    assert trainer.reroute_weight == pytest.approx(0.5)
    assert all(parameters == (model.lora_weight,) for parameters in balanced_parameters)
    assert trainer.logged[-2]["retain_grad_norm"] == 2.0
    assert trainer.logged[-2]["reroute_grad_norm"] == 4.0

    trainer.compute_loss(model, {"chosen": batch, "rejected": batch})
    assert len(balanced_parameters) == 2

    trainer.state.global_step = 1
    trainer.compute_loss(model, {"chosen": batch, "rejected": batch})
    assert len(balanced_parameters) == 2

    trainer.state.global_step = 25
    trainer.compute_loss(model, {"chosen": batch, "rejected": batch})
    assert len(balanced_parameters) == 4
    assert trainer.reroute_weight == pytest.approx(0.65)


@pytest.mark.parametrize(
    ("retain_norm", "reroute_norm", "expected"),
    [(0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (2e6, 1.0, 1e3)],
)
def test_cb_gradient_balance_clamps_and_handles_zero_denominator(
    cb_training_module, monkeypatch, retain_norm, reroute_norm, expected
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.tensor(1.0))

    trainer, batch, _ = _gradient_balance_trainer(cb_training_module, torch)
    gradients = iter((retain_norm, reroute_norm))
    monkeypatch.setattr(
        torch.autograd,
        "grad",
        lambda *args, **kwargs: (torch.tensor(next(gradients)),),
    )
    trainer.compute_loss(Model(), {"chosen": batch, "rejected": batch})
    assert trainer.reroute_weight == pytest.approx(expected)


def test_cb_gradient_probe_leaves_grads_empty_and_loss_supports_backward(
    cb_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.tensor(1.0))
            self.base_weight = torch.nn.Parameter(
                torch.tensor(3.0), requires_grad=False
            )

    model = Model()
    trainer, batch, _ = _gradient_balance_trainer(cb_training_module, torch)
    reference = torch.tensor([[[[0.0, 1.0]]]])
    calls = 0

    def activations(model, batch, layers=None, **kwargs):
        nonlocal calls
        del batch, layers
        calls += 1
        if kwargs.get("disable_grad"):
            return reference
        if calls == 2:
            return torch.stack((model.lora_weight, model.lora_weight)).reshape(
                1, 1, 1, 2
            )
        return torch.stack(
            (model.lora_weight, torch.ones_like(model.lora_weight))
        ).reshape(1, 1, 1, 2)

    trainer.activations = activations
    loss = trainer.compute_loss(model, {"chosen": batch, "rejected": batch})

    assert model.lora_weight.grad is None
    assert model.base_weight.grad is None
    assert torch.isfinite(torch.tensor(trainer.logged[-2]["retain_grad_norm"]))
    assert torch.isfinite(torch.tensor(trainer.logged[-2]["reroute_grad_norm"]))
    loss.backward()
    assert model.lora_weight.grad is not None
    assert model.base_weight.grad is None


def test_cb_losses_mask_padding_and_zero_expected_cases(cb_training_module) -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([[[[1.0, 0.0], [100.0, 100.0]]]])
    safe = reference.clone()
    harmful = torch.tensor([[[[0.0, 1.0], [1000.0, 1000.0]]]])
    mask = torch.tensor([[1, 0]])
    retain_distance = torch.linalg.vector_norm(safe - reference, dim=-1)
    reroute_cosine = torch.abs(
        torch.nn.functional.cosine_similarity(harmful, reference, dim=-1)
    )
    assert cb_training_module._masked_mean(retain_distance, mask).item() == 0
    assert cb_training_module._masked_mean(reroute_cosine, mask).item() == 0

    opposite = -reference
    reroute_cosine = torch.abs(
        torch.nn.functional.cosine_similarity(opposite, reference, dim=-1)
    )
    assert cb_training_module._masked_mean(reroute_cosine, mask).item() == 1


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

    with pytest.raises(ValueError, match="missing required fields"):
        cb_training_module.validate_row({"prompt": "P"})
    with pytest.raises(ValueError, match="non-empty strings"):
        cb_training_module.validate_row({**complete, "chosen": 3})


def test_cb_completion_mask_excludes_prompt_and_special_tokens(
    cb_training_module,
) -> None:
    offsets = [(0, 0), (0, 4), (4, 8), (8, 12), (0, 0)]
    assert cb_training_module.completion_mask(offsets, 8) == [0, 0, 0, 1, 0]


def test_cb_activation_selection_and_forward_modes(cb_training_module) -> None:
    torch = pytest.importorskip("torch")
    from contextlib import contextmanager
    from types import SimpleNamespace

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.observed_modes = []
            self.adapter_disabled = False

        @contextmanager
        def disable_adapter(self):
            self.adapter_disabled = True
            try:
                yield
            finally:
                self.adapter_disabled = False

        def forward(self, **kwargs):
            del kwargs
            self.observed_modes.append((self.training, self.adapter_disabled))
            return SimpleNamespace(
                hidden_states=tuple(
                    torch.full((1, 2, 1), float(index), requires_grad=True)
                    for index in range(5)
                )
            )

    trainer = object.__new__(cb_training_module.CircuitBreakerTrainer)
    trainer.target_layers = (1, 3)
    trainer.reroute_noise = 0.01
    trainer.reroute_noise_seed = 42
    model = FakeModel()
    input_ids = torch.ones((1, 2), dtype=torch.long)
    mask = torch.ones_like(input_ids)

    batch = {"input_ids": input_ids, "attention_mask": mask}
    selected = trainer.activations(
        model, batch, (1, 3), disable_adapter=True, disable_grad=True
    )
    assert selected[:, 0, 0, 0].tolist() == [1.0, 3.0]
    assert model.observed_modes[-1] == (False, True)
    assert model.training is True

    noisy = trainer.activations(
        model,
        batch,
        (1, 3),
        disable_adapter=True,
        disable_grad=True,
        add_noise=True,
    )
    assert not torch.equal(noisy, selected)

    all_states = trainer.activations(model, batch)
    assert all_states[:, 0, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert model.observed_modes[-1] == (True, False)
    assert all_states.requires_grad


def test_cb_config() -> None:
    config = yaml.safe_load(
        (EXAMPLE / "configs/training/di-6.9b/circuit-breaker.yml").read_text()
    )
    assert config["base_model"] == "EleutherAI/deep-ignorance-unfiltered"
    assert config["trainer_cls"] == (
        "configs.training.utils.ConstantLambda1CircuitBreakerTrainer"
    )
    assert config["output_dir"].endswith("--constant-coeff-lambda1")
    assert "coefficient_schedule" not in config
    assert config["datasets"] == [
        {
            "path": "LLM-LAT/harmful-dataset",
            "split": "train",
            "type": "configs.training.utils",
        }
    ]
    assert config["peft_layers_to_transform"] == list(range(31))
    assert config["lora_r"] == config["lora_alpha"] == 64
    assert config["num_epochs"] == 1
    assert "max_steps" not in config
    assert config["save_steps"] == 50
    assert config["dataset_prepared_path"] == ("artifacts/cache/axolotl/di-6.9b-cb")
    assert config["lr_scheduler"] == "constant"
    assert config["bf16"] is config["tf32"] is True
    assert config["micro_batch_size"] * config["gradient_accumulation_steps"] == 16


def test_cb_gradient_balanced_config_is_isolated() -> None:
    config = yaml.safe_load(
        (
            EXAMPLE / "configs/training/di-6.9b/circuit-breaker-grad-balanced.yml"
        ).read_text()
    )
    assert config["trainer_cls"] == (
        "configs.training.utils.GradientBalancedCircuitBreakerTrainer"
    )
    assert config["output_dir"] == "artifacts/models/di-6.9b-cb--gradnorm-balanced"
    assert config["wandb_name"] == "di-6.9b-cb--gradnorm-balanced"
    assert config["save_steps"] == 50
    assert config["save_total_limit"] == 100


def test_cb_linear_config() -> None:
    config = yaml.safe_load(
        (EXAMPLE / "configs/training/di-6.9b/circuit-breaker-linear.yml").read_text()
    )
    assert config["output_dir"].endswith("--linear-coeffs-0to5-10to5")
    assert config["wandb_name"].endswith("--linear-coeffs-0to5-10to5")
    assert config["trainer_cls"] == (
        "configs.training.utils.LinearCircuitBreakerTrainer"
    )
    assert "coefficient_schedule" not in config

    noisy = yaml.safe_load(
        (
            EXAMPLE / "configs/training/di-6.9b/circuit-breaker-linear-noise0p1.yml"
        ).read_text()
    )
    assert noisy["trainer_cls"] == (
        "configs.training.utils.LinearNoise0p1CircuitBreakerTrainer"
    )
    assert noisy["output_dir"].endswith("--linear-coeffs-0to5-10to5--noise0p1")
    assert noisy["wandb_name"].endswith("--linear-coeffs-0to5-10to5--noise0p1")


def test_single_experiment_plans_base_cb_and_weight_steering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/di-6.9b.yml").plan()
    training = plan.stage_index["train-di-6.9b-cb--constant-coeff-lambda1"]
    assert training.id == "train-di-6.9b-cb--constant-coeff-lambda1"
    assert training.args == (
        "configs/training/di-6.9b/circuit-breaker.yml",
        "--merge",
    )
    assert "--gres=gpu:l40s:1" in training.sbatch_args
    assert plan.stage_index["eval-bio-mcqa-di-6.9b-base"].args == (
        "di-6.9b-base",
        "EleutherAI/deep-ignorance-unfiltered",
        "-",
    )


@pytest.mark.parametrize(
    ("alpha", "weights"),
    [
        (1, [1.0, -1.0]),
        (3, [3.0, -3.0]),
        (5, [5.0, -5.0]),
        (10, [10.0, -10.0]),
    ],
)
def test_task_vector_uses_chosen_minus_rejected(
    alpha: float, weights: list[float]
) -> None:
    module = load_script("scripts/steering/task_vector.py", "lat_task_vector")
    assert module.weighted_adapter_spec(alpha) == (["chosen", "rejected"], weights)


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
        "di-6.9b-ft-lat-chosen",
        "di-6.9b-ft-lat-rejected",
        "di-6.9b-cb--constant-coeff-lambda1",
        "di-6.9b-cb--linear-coeffs-0to5-10to5",
        "di-6.9b-cb--linear-coeffs-0to5-10to5--noise0p1",
        "di-6.9b-cb--gradnorm-balanced",
        "di-6.9b-w-steer-lat-reject2accept-a-1",
        "di-6.9b-w-steer-lat-reject2accept-a-3",
        "di-6.9b-w-steer-lat-reject2accept-a-5",
        "di-6.9b-w-steer-lat-reject2accept-a-10",
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


def test_results_config_has_base_cb_and_weight_steering_groups() -> None:
    config = yaml.safe_load((EXAMPLE / "configs/results/di-6.9b.yml").read_text())
    assert [row["group"] for row in config["rows"]] == [
        "base-model",
        "circuit-breaker",
        "circuit-breaker",
        "circuit-breaker",
        "circuit-breaker",
        "weight-steering",
        "weight-steering",
        "weight-steering",
        "weight-steering",
    ]

    for training_path in (EXAMPLE / "configs/training").rglob("*.yml"):
        training = yaml.safe_load(training_path.read_text())
        assert training["output_dir"].startswith("artifacts/models/di-6.9b-")
        assert "lora64-epochs1" not in training["output_dir"]
        assert "LoRA64-Epochs1" not in training["wandb_name"]

    for steering_path in (EXAMPLE / "configs/steering").rglob("*.yml"):
        steering = yaml.safe_load(steering_path.read_text())
        assert steering["output_path"] == (
            f"artifacts/models/di-6.9b-w-steer-lat-reject2accept-a-"
            f"{steering['alpha']:g}"
        )

    assert all("LoRA64-Epochs1" not in row["label"] for row in config["rows"])
