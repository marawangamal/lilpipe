"""Checks for the two GIP aggregation objectives."""

from itertools import combinations
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("axolotl")

from trainers.tr_gip import GIPTrainer


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

    trainer = GIPTrainer(
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
        ),
    )
    assert trainer.coeff_forget == 0
    assert trainer.coeff_retain == 1
    assert trainer.coeff_gip == 0.01
    assert trainer.eps_finite_diff == 1e-3
    assert trainer.eps_norm_div == 1e-8
    assert list(trainer._get_train_sampler()) == [0, 2, 1, 3]
    result = trainer.train()
    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))


def test_gip_sampler_streams_and_batch_validation():
    trainer = object.__new__(GIPTrainer)
    trainer.gip_mode = "avg_of_cosims"
    trainer.n_forget, trainer.n_retain = 6, 9
    trainer.train_dataset = list(range(15))
    trainer._train_batch_size = 6
    trainer.args = SimpleNamespace(dataloader_drop_last=False)
    assert list(trainer._get_train_sampler()) == [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
    trainer.gip_mode = "cosim_of_avgs"
    with pytest.raises(ValueError, match="divisible by four"):
        trainer._get_train_sampler()
    trainer._train_batch_size = 4
    with pytest.raises(ValueError, match="shorter"):
        trainer._get_train_sampler(list(range(14)))
    trainer.n_forget, trainer.n_retain = 5, 10
    with pytest.raises(ValueError, match="complete batches"):
        trainer._get_train_sampler()
    trainer.args.dataloader_drop_last = True
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
