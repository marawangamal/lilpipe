"""Compute-reduced LoRA adaptation of tamper-resistant unlearning (TAR)."""

from contextlib import nullcontext

import torch
from torch.func import functional_call

from configs.training.trainers.mamul import MAMULTrainer
from configs.training.trainers.samplers import MixedSourceSampler

ROLE_FIELDS = ("input_ids", "attention_mask", "labels")


def masked_negative_entropy(logits, labels, attention_mask=None):
    """Negative mean next-token entropy over non-ignored target positions."""
    token_logits = logits[:, :-1].float()
    mask = labels[:, 1:].ne(-100)
    if attention_mask is not None:
        mask &= attention_mask[:, 1:].bool()
    if not mask.any():
        raise ValueError("forget batch has no unmasked next-token targets")
    log_probs = token_logits.log_softmax(dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return -entropy[mask].mean()


def masked_hidden_mse(current, reference, attention_mask):
    """Mean hidden-state squared error on attended tokens and all returned layers."""
    mask = attention_mask.bool().unsqueeze(-1)
    losses = []
    for current_layer, reference_layer in zip(current, reference):
        expanded = mask.expand_as(current_layer)
        losses.append(
            (current_layer.float() - reference_layer.float()).square()[expanded].mean()
        )
    if not losses:
        raise ValueError(
            "model did not return hidden states for representation matching"
        )
    return torch.stack(losses).mean()


class TARAttack(torch.nn.Module):
    """Run functional AdamW relearning steps without changing the live model."""

    def __init__(
        self,
        steps=8,
        learning_rate=2e-5,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.0,
    ):
        super().__init__()
        self.steps = steps
        self.learning_rate = learning_rate
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def forward(self, model, parameters, inputs):
        initial = {
            name: parameter.detach().clone() for name, parameter in parameters.items()
        }
        current = {
            name: parameter.detach().clone().requires_grad_(True)
            for name, parameter in parameters.items()
        }
        first_moment = {
            name: torch.zeros_like(value) for name, value in current.items()
        }
        second_moment = {
            name: torch.zeros_like(value) for name, value in current.items()
        }
        beta1, beta2 = self.betas

        for step in range(1, self.steps + 1):
            outputs = functional_call(model, current, (), inputs)
            gradients = torch.autograd.grad(
                outputs.loss, tuple(current.values()), allow_unused=True
            )
            if not any(gradient is not None for gradient in gradients):
                raise ValueError(
                    "attack loss has no gradient through the trainable parameters"
                )
            updated = {}
            for (name, value), gradient in zip(current.items(), gradients):
                if gradient is None:
                    updated[name] = value
                    continue
                first_moment[name] = beta1 * first_moment[name] + (1 - beta1) * gradient
                second_moment[name] = (
                    beta2 * second_moment[name] + (1 - beta2) * gradient.square()
                )
                direction = (first_moment[name] / (1 - beta1**step)) / (
                    (second_moment[name] / (1 - beta2**step)).sqrt() + self.epsilon
                )
                direction = direction + self.weight_decay * value
                updated[name] = (
                    (value - self.learning_rate * direction)
                    .detach()
                    .requires_grad_(True)
                )
            current = updated

        return {name: (current[name] - initial[name]).detach() for name in current}


class TARTrainer(MAMULTrainer):
    """Sample an eight-step LoRA relearning endpoint for the outer TAR objective."""

    def __init__(
        self,
        *args,
        inner_steps=8,
        inner_learning_rate=2e-5,
        inner_betas=(0.9, 0.999),
        inner_epsilon=1e-8,
        inner_weight_decay=0.0,
        forget_coeff=4.0,
        retain_coeff=1.0,
        representation_coeff=1.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            forget_coeff=forget_coeff,
            retain_coeff=retain_coeff,
            **kwargs,
        )
        self.representation_coeff = representation_coeff
        self.attack = TARAttack(
            steps=inner_steps,
            learning_rate=inner_learning_rate,
            betas=inner_betas,
            epsilon=inner_epsilon,
            weight_decay=inner_weight_decay,
        )

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return MixedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

    @staticmethod
    def split_sources(inputs):
        forget_mask = inputs["is_forget"].bool()
        retain_mask = ~forget_mask
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("TAR requires forget and retain rows in every batch")
        forget_inputs = {field: inputs[field][forget_mask] for field in ROLE_FIELDS}
        retain_inputs = {field: inputs[field][retain_mask] for field in ROLE_FIELDS}
        return forget_inputs, retain_inputs

    def sample_attack(self, model, parameters, inputs, **kwargs):
        forget_inputs, _ = self.split_sources(inputs)
        return self.attack(model, parameters, forget_inputs)

    def compute_forget_loss(self, model_forward, inputs, **kwargs):
        forget_inputs, _ = self.split_sources(inputs)
        outputs = model_forward(**forget_inputs)
        return masked_negative_entropy(
            outputs.logits,
            forget_inputs["labels"],
            forget_inputs["attention_mask"],
        )

    def compute_retain_loss(self, model_forward, inputs, **kwargs):
        _, retain_inputs = self.split_sources(inputs)
        current = model_forward(**retain_inputs, output_hidden_states=True)
        disable_adapter = getattr(model_forward, "disable_adapter", None)
        context = disable_adapter() if disable_adapter is not None else nullcontext()
        with context, torch.no_grad():
            reference = model_forward(**retain_inputs, output_hidden_states=True)
        representation_loss = masked_hidden_mse(
            current.hidden_states,
            reference.hidden_states,
            retain_inputs["attention_mask"],
        )
        return current.loss.float() + self.representation_coeff * representation_loss


class TARK16Trainer(TARTrainer):
    """TAR variant with sixteen simulated relearning steps per outer update."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("inner_steps", 16)
        super().__init__(*args, **kwargs)
