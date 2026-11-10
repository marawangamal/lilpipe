"""Checks for the two GIP aggregation objectives."""

from itertools import combinations
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("axolotl")

from trainers.tr_gip import GIPGDTrainer, GIPTrainer


class QuadraticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.4]))

    def forward(self, labels, use_cache=False):
        return SimpleNamespace(loss=((self.weight - labels) ** 2).mean())


@pytest.mark.parametrize("mode", ["avg_of_cosims", "cosim_of_avgs"])
def test_gip_objective_and_gradient_aggregation(mode):
    trainer = object.__new__(GIPTrainer)
    trainer.gip_mode = mode
    trainer.eps_norm_div = 0.1
    trainer.eps_finite_diff = 1e-5
    model = QuadraticModel().double()
    labels = torch.tensor(
        [[1.0, 2.0], [-2.0, 1.0], [0.5, -1.0], [3.0, 4.0]], dtype=torch.float64
    )
    model.weight.grad = torch.ones_like(model.weight)

    def cosine(a, b):
        eps_norm_div = trainer.eps_norm_div
        return a.dot(b) / ((a.norm() + eps_norm_div) * (b.norm() + eps_norm_div))

    torch.manual_seed(17)
    if mode == "avg_of_cosims":
        gradients = model.weight - labels
        expected = torch.stack(
            [cosine(a, b) for a, b in combinations(gradients, 2)]
        ).mean()
    else:
        a, b = torch.randperm(len(labels)).chunk(2)
        expected = cosine(
            model.weight - labels[a].mean(0),
            model.weight - labels[b].mean(0),
        )

    torch.manual_seed(17)
    measured = trainer.compute_gip(model, {"labels": labels}, compute_loss_only=True)
    torch.testing.assert_close(measured, expected)
    assert not measured.requires_grad

    torch.manual_seed(17)
    loss, gradients = trainer.compute_gip(model, {"labels": labels})
    torch.testing.assert_close(loss, expected)
    exact = torch.autograd.grad(expected, model.weight)[0]
    torch.testing.assert_close(gradients["weight"], exact, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight))


def test_gip_rejects_unknown_mode():
    with pytest.raises(ValueError, match="gip_mode"):
        GIPTrainer(gip_mode="unknown")


@pytest.mark.parametrize(
    "mode,B",
    [
        ("avg_of_cosims", 2),
        ("avg_of_cosims", 3),
        ("avg_of_cosims", 4),
        ("cosim_of_avgs", 2),
        ("cosim_of_avgs", 4),
    ],
)
def test_gip_finite_difference_matches_nonlinear_exact_gradient(mode, B):
    trainer = object.__new__(GIPTrainer)
    trainer.eps_norm_div = 0.1
    trainer.eps_finite_diff = 1e-6
    model = QuadraticModel().double()
    labels = torch.tensor(
        [[1.0, 2.0], [-2.0, 1.0], [0.5, -1.0], [3.0, 4.0]], dtype=torch.float64
    )[:B]

    def loss(model, batch):
        return (model.weight.tanh() - batch["labels"]).square().mean()

    trainer.compute_ce = loss
    torch.manual_seed(17)
    if mode == "avg_of_cosims":
        batches = [{"labels": row} for row in labels]
    else:
        a, b = torch.randperm(B).chunk(2)
        batches = [{"labels": labels[a]}, {"labels": labels[b]}]
    gradients = [
        torch.autograd.grad(loss(model, batch), model.weight, create_graph=True)[0]
        for batch in batches
    ]
    eps_norm_div = trainer.eps_norm_div
    expected = torch.stack(
        [
            a.dot(b) / ((a.norm() + eps_norm_div) * (b.norm() + eps_norm_div))
            for a, b in combinations(gradients, 2)
        ]
    ).mean()
    exact = torch.autograd.grad(expected, model.weight)[0]
    original = model.weight.detach().clone()
    model.weight.grad = torch.ones_like(model.weight)
    trainer.gip_mode = mode
    torch.manual_seed(17)
    measured, estimate = trainer.compute_gip(model, {"labels": labels})
    torch.testing.assert_close(measured, expected)
    torch.testing.assert_close(estimate["weight"], exact, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight))


def test_dict_operations():
    from trainers.utils import dict_binop, dict_dot

    a = {"x": torch.tensor([1.0, 2.0]), "y": torch.tensor([3.0])}
    b = {"x": torch.tensor([4.0, 5.0]), "y": torch.tensor([6.0])}
    torch.testing.assert_close(dict_dot(a, b), torch.tensor(32.0))
    difference = dict_binop(a, b, lambda x, y: y - x)
    torch.testing.assert_close(difference["x"], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(difference["y"], torch.tensor([3.0]))


@pytest.mark.parametrize("mode", ["avg_of_cosims", "cosim_of_avgs"])
def test_gip_initialization_and_training(tmp_path, mode):
    from axolotl.core.training_args import AxolotlTrainingArguments
    from datasets import Dataset

    trainer = GIPGDTrainer(
        model=QuadraticModel(),
        train_dataset=Dataset.from_list(
            [{"labels": [float(i + 1), float(i + 2)]} for i in range(4)]
        ),
        gip_mode=mode,
        n_forget=2,
        n_retain=2,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            per_device_train_batch_size=4,
            max_steps=1,
            save_strategy="no",
            disable_tqdm=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        ),
    )
    assert trainer.coeff_forget == 0
    assert trainer.coeff_retain == 1
    assert trainer.coeff_gip == 0.01
    assert trainer.eps_finite_diff == 1e-3
    assert trainer.eps_norm_div == 1e-8
    assert list(GIPTrainer._get_train_sampler(trainer)) == [0, 2, 1, 3]
    result = trainer.train()
    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))


def test_gip_sampler_streams_and_batch_validation():
    trainer = object.__new__(GIPTrainer)
    trainer.shuffle_streams = False
    trainer.gip_mode = "avg_of_cosims"
    trainer.n_forget, trainer.n_retain = 6, 9
    trainer.train_dataset = list(range(15))
    trainer._train_batch_size = 6
    trainer.args = SimpleNamespace(dataloader_drop_last=False, seed=42, data_seed=None)
    assert list(trainer._get_train_sampler()) == [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
    trainer.gip_mode = "cosim_of_avgs"
    assert len(trainer._get_train_sampler()) == 12
    trainer._train_batch_size = 4
    with pytest.raises(ValueError, match="shorter"):
        trainer._get_train_sampler(list(range(14)))
    trainer.n_forget, trainer.n_retain = 5, 10
    assert len(trainer._get_train_sampler()) == 10
    trainer._train_batch_size = 3
    with pytest.raises(ValueError, match="even batch size"):
        trainer._get_train_sampler()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coeff_forget": -1},
        {"coeff_gip": float("nan")},
        {"eps_finite_diff": 0},
        {"eps_norm_div": -1},
        {"n_forget": 0},
        {"n_retain": 1.5},
    ],
)
def test_gip_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        GIPTrainer(**kwargs)


def test_npo_gip_uses_shared_npo_loss_and_balanced_metadata(tmp_path):
    from axolotl.core.training_args import AxolotlTrainingArguments
    from datasets import Dataset
    from trainers.tr_gip import GIPNPOTrainer
    from trainers.ul_npo import sequence_cross_entropy

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 4)

        def forward(self, input_ids, attention_mask, labels, use_cache=False):
            logits = self.embedding(input_ids)
            loss = sequence_cross_entropy(logits, labels).mean()
            return SimpleNamespace(logits=logits, loss=loss)

    torch.manual_seed(17)
    trainer = GIPNPOTrainer(
        model=LanguageModel(),
        reference_model_name_or_path="unused-reference",
        beta=0.025,
        coeff_forget=2.0,
        coeff_retain=4.0,
        coeff_gip=0.0,
        gip_mode="avg_of_cosims",
        n_forget=4,
        n_retain=4,
        train_dataset=Dataset.from_dict({"labels": [[0, 1, 2]] * 8}),
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            per_device_train_batch_size=8,
            data_seed=42,
        ),
    )
    assert trainer.beta == 0.025
    assert trainer.coeff_retain == 4.0
    order = list(trainer._get_train_sampler())
    assert len(order) == 8
    assert all(index < 4 for index in order[0::2])
    assert all(index >= 4 for index in order[1::2])
    trainer.model.double()
    trainer._theta_ref = {
        name: p.detach().clone() + 0.125 * torch.arange(4)
        for name, p in trainer.theta.items()
    }
    inputs = {
        "input_ids": torch.randint(0, 4, (8, 3)),
        "attention_mask": torch.ones(8, 3, dtype=torch.long),
        "labels": torch.randint(0, 4, (8, 3)),
    }
    fields = {name: value for name, value in inputs.items() if name != "is_forget"}
    forget = {name: value[0::2] for name, value in fields.items()}
    retain = {name: value[1::2] for name, value in fields.items()}
    current = sequence_cross_entropy(trainer.model(**forget).logits, forget["labels"])
    with torch.no_grad():
        reference = torch.func.functional_call(
            trainer.model, trainer.theta_ref, (), forget
        )
        reference_ce = sequence_cross_entropy(reference.logits, forget["labels"])
    expected = (
        -trainer.coeff_forget
        * (2 / trainer.beta)
        * torch.nn.functional.logsigmoid(trainer.beta * (current - reference_ce)).mean()
        + 4 * trainer.model(**retain).loss
    )
    expected_gradient = torch.autograd.grad(
        expected / 2, trainer.model.embedding.weight
    )[0]
    trainer.current_gradient_accumulation_steps = 2
    trainer._prepare_inputs = lambda batch: batch
    trainer.log = lambda metrics: None
    measured = trainer.training_step(trainer.model, inputs)
    torch.testing.assert_close(measured, expected.detach() / 2, check_dtype=False)
    torch.testing.assert_close(trainer.model.embedding.weight.grad, expected_gradient)


def test_gip_relearning_uses_all_rows_without_regularizing():
    from trainers.tr_gip import GIPRelearnTrainer

    trainer = object.__new__(GIPRelearnTrainer)
    trainer.gip_mode = "avg_of_cosims"
    trainer.eps_norm_div = 1e-8
    trainer.current_gradient_accumulation_steps = 2
    trainer.args = SimpleNamespace(device=torch.device("cpu"))
    trainer._prepare_inputs = lambda batch: batch
    trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
    trainer.log = lambda metrics: None
    model = QuadraticModel()
    inputs = {"labels": torch.tensor([[1.0, 2.0], [-2.0, 1.0]])}
    expected = model(**inputs).loss / 2
    expected_gradient = torch.autograd.grad(expected, model.weight)[0]
    measured = trainer.training_step(model, inputs)
    torch.testing.assert_close(measured, expected.detach())
    torch.testing.assert_close(model.weight.grad, expected_gradient)


def test_gd_loss_helpers_do_not_apply_objective_coefficients():
    from trainers.ul_gd import GradDiffTrainer

    trainer = object.__new__(GradDiffTrainer)
    trainer.coeff_forget = 2.0
    trainer.coeff_retain = 4.0
    model = QuadraticModel()
    inputs = {"labels": torch.tensor([[1.0, 2.0]])}
    ce = model(**inputs).loss
    torch.testing.assert_close(trainer.compute_forget_loss(model, inputs), -ce)
    torch.testing.assert_close(trainer.compute_retain_loss(model, inputs), ce)


@pytest.mark.parametrize("divisor", [1, 4])
def test_seam_regularizer_weights_loss_and_gradient_once(divisor):
    from trainers.tr_seam import SEAMTrainer

    trainer = object.__new__(SEAMTrainer)
    trainer.current_gradient_accumulation_steps = divisor
    trainer.epsilon = 1e-6
    trainer.compute_ce = lambda model, inputs: model(**inputs).loss
    model = QuadraticModel().double()
    harmful = {"labels": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}
    benign = {"labels": torch.tensor([[-2.0, 1.0]], dtype=torch.float64)}
    a = torch.autograd.grad(model(**harmful).loss, model.weight, create_graph=True)[0]
    b = torch.autograd.grad(model(**benign).loss, model.weight, create_graph=True)[0]
    cosine = a.dot(b) / (a.norm() * b.norm() + 1e-8)
    expected_gradient = torch.autograd.grad(0.25 * cosine / divisor, model.weight)[0]
    loss, norm = trainer.accumulate_sd_grad(model, harmful, benign, coefficient=0.25)
    torch.testing.assert_close(loss, (0.25 * cosine).detach(), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(
        model.weight.grad, expected_gradient, rtol=1e-4, atol=1e-7
    )
    torch.testing.assert_close(norm, model.weight.grad.norm())


def test_gip_regularizer_weights_loss_and_gradient_once():
    trainer = object.__new__(GIPGDTrainer)
    trainer.gip_mode = "avg_of_cosims"
    trainer.eps_finite_diff = 1e-6
    trainer.eps_norm_div = 0.1
    trainer.current_gradient_accumulation_steps = 4
    trainer.compute_ce = lambda model, inputs: model(**inputs).loss
    model = QuadraticModel().double()
    labels = torch.tensor([[1.0, 2.0], [-2.0, 1.0]], dtype=torch.float64)
    a, b = model.weight - labels
    cosine = a.dot(b) / ((a.norm() + 0.1) * (b.norm() + 0.1))
    expected_gradient = torch.autograd.grad(0.25 * cosine / 4, model.weight)[0]
    loss, norm = trainer.accumulate_gip_grad(
        model, {"labels": labels}, coefficient=0.25
    )
    torch.testing.assert_close(loss, 0.25 * cosine.detach())
    torch.testing.assert_close(
        model.weight.grad, expected_gradient, rtol=1e-4, atol=1e-7
    )
    torch.testing.assert_close(norm, model.weight.grad.norm())
