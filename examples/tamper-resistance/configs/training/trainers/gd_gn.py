"""Forget-gradient norm regularization with balanced retention batches."""

import torch

from configs.training.trainers.gd import BalancedGradDiffTrainer


class BalancedGradDiffGNTrainer(BalancedGradDiffTrainer):
    """Maximize forget CE while limiting its LoRA gradient and retaining WikiText."""

    rho = 0.01
    retain_weight = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        source = inputs["cb_source"]
        forget_mask = source == 1
        retain_mask = source == 0
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("GD-GN requires forget and retain rows in every batch")

        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}

        forget_outputs = model(**forget_inputs)
        forget_loss = forget_outputs.loss.float()
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradients = torch.autograd.grad(forget_loss, parameters, create_graph=True)
        gradient_norm = torch.sqrt(
            sum(gradient.float().square().sum() for gradient in gradients) + 1e-12
        )

        retain_loss = model(**retain_inputs).loss.float()
        loss = (
            -forget_loss + self.rho * gradient_norm + self.retain_weight * retain_loss
        )
        return (loss, forget_outputs) if return_outputs else loss
