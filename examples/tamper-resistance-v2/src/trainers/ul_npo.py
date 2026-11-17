"""Negative preference optimization against a frozen reference model."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from trainers.ul_gd import GDTrainer


class NPOTrainer(GDTrainer):
    """Apply NPO to forget cross-entropy while preserving the GD retain term."""

    def __init__(
        self,
        *args,
        reference_model_name_or_path,
        beta=0.0225,
        **kwargs,
    ):
        if beta <= 0:
            raise ValueError("beta must be positive")
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            reference_model_name_or_path,
            torch_dtype=self.model.dtype,
            device_map=getattr(self.model, "hf_device_map", {"": self.model.device}),
            attn_implementation=self.model.config._attn_implementation,
        )
        self.reference_model.eval().requires_grad_(False)
        self.reference_model.config.use_cache = False

    def compute_loss_forget(self, model, inputs):
        with torch.no_grad():
            loss_reference = self.reference_model(**inputs).loss.float()
        loss_current = model(**inputs).loss.float()
        log_ratio = loss_current - loss_reference.to(loss_current.device)
        return -(2 / self.beta) * F.logsigmoid(self.beta * log_ratio)
