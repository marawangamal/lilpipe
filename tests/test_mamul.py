import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import lilpipe

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tamper-resistance"


@pytest.fixture(scope="module")
def modules():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    sys.path.insert(0, str(EXAMPLE))
    try:
        from configs.training.trainers import mamul, tar

        yield mamul, tar
    finally:
        sys.path.remove(str(EXAMPLE))


def test_mamul_delta_contract_and_identity_gradient(modules) -> None:
    torch = pytest.importorskip("torch")
    mamul, _ = modules

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, input_ids=None):
            return SimpleNamespace(loss=self.weight.square())

    class Trainer(mamul.MAMULTrainer):
        def sample_attack(self, model, parameters, inputs, **kwargs):
            return {}

        def compute_forget_loss(self, model_forward, inputs, **kwargs):
            return model_forward().loss

        def compute_retain_loss(self, model_forward, inputs, **kwargs):
            return model_forward().loss * 0

    model = Model()
    trainer = object.__new__(Trainer)
    trainer.forget_coeff = trainer.retain_coeff = 1.0
    loss = trainer.compute_loss(model, {})
    loss.backward()
    assert model.weight.grad.item() == pytest.approx(4.0)

    parameters = dict(model.named_parameters())
    Trainer._validate_delta(parameters, {"weight": torch.zeros_like(model.weight)})


def test_mamul_rejects_malformed_deltas(modules) -> None:
    torch = pytest.importorskip("torch")
    mamul, _ = modules
    parameter = torch.nn.Parameter(torch.ones(2))
    parameters = {"weight": parameter}
    with pytest.raises(ValueError, match="unknown"):
        mamul.MAMULTrainer._validate_delta(parameters, {"other": torch.ones(2)})
    with pytest.raises(ValueError, match="shape"):
        mamul.MAMULTrainer._validate_delta(parameters, {"weight": torch.ones(3)})
    with pytest.raises(ValueError, match="dtype"):
        mamul.MAMULTrainer._validate_delta(
            parameters, {"weight": torch.ones(2, dtype=torch.float64)}
        )
    with pytest.raises(ValueError, match="detached"):
        mamul.MAMULTrainer._validate_delta(
            parameters, {"weight": torch.ones(2, requires_grad=True)}
        )


def test_tar_splits_mixed_forget_and_retain_batch(modules) -> None:
    torch = pytest.importorskip("torch")
    _, tar = modules
    inputs = {
        "input_ids": torch.arange(12).reshape(4, 3),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
        "labels": torch.arange(12).reshape(4, 3),
        "is_forget": torch.tensor([True, False, True, False]),
    }
    forget, retain = tar.TARTrainer.split_sources(inputs)
    assert forget["input_ids"].tolist() == [[0, 1, 2], [6, 7, 8]]
    assert retain["input_ids"].tolist() == [[3, 4, 5], [9, 10, 11]]


def test_tar_functional_sft_leaves_live_model_unchanged_and_backpropagates(
    modules,
) -> None:
    torch = pytest.importorskip("torch")
    _, tar = modules

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(5, 5) * 0.1)
            self.unused_weight = torch.nn.Parameter(torch.randn(2))

        def forward(
            self, input_ids, attention_mask, labels, output_hidden_states=False
        ):
            hidden = torch.nn.functional.one_hot(input_ids, 5).float()
            logits = hidden @ self.weight
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1)
            )
            hidden_states = (logits,) if output_hidden_states else None
            return SimpleNamespace(
                loss=loss, logits=logits, hidden_states=hidden_states
            )

    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]] * 4),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
        "labels": torch.tensor([[0, 1, 2]] * 4),
        "is_forget": torch.tensor([True, True, False, False]),
    }
    forget_inputs, _ = tar.TARTrainer.split_sources(inputs)
    model = TinyLM()
    trainer = object.__new__(tar.TARTrainer)
    trainer.attack = tar.TARAttack(steps=8, learning_rate=0.1)
    trainer.forget_coeff = 4.0
    trainer.retain_coeff = trainer.representation_coeff = 1.0
    before = model.weight.detach().clone()
    attack_ce = model(**forget_inputs).loss
    delta = trainer.sample_attack(model, dict(model.named_parameters()), inputs)
    assert not delta["weight"].requires_grad
    assert torch.equal(delta["unused_weight"], torch.zeros_like(model.unused_weight))
    assert torch.equal(model.weight, before)
    attacked_ce = torch.func.functional_call(
        model, {"weight": model.weight + delta["weight"]}, (), forget_inputs
    ).loss
    assert attacked_ce < attack_ce
    trainer.compute_loss(model, inputs).backward()
    assert model.weight.grad is not None
    assert model.unused_weight.grad is None
