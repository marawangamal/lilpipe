"""Axolotl plugin for activation-conditioned WMDP sandbagging SFT."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from datasets import Dataset, load_dataset
from pydantic import BaseModel
from safetensors.torch import load_file
from transformers import DataCollatorForSeq2Seq

from axolotl.common.datasets import TrainDatasetMeta
from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.integrations.base import BasePlugin
from axolotl.loaders import load_tokenizer
from axolotl.prompt_strategies.chat_template import StrategyLoader

from lm_eval_tasks.wmdp_sandbagging.utils import NORMAL_SYSTEM, prompt_messages


class ActivationSteeringArgs(BaseModel):
    sandbagging_activation_steering: bool = False
    sandbagging_data_path: str | None = None
    steering_vector_path: str | None = None
    steering_layer: int = 32
    steering_strength: float = 0.5


class SteeringController:
    """Apply a norm-matched direction at selected final prompt positions."""

    def __init__(self, direction: torch.Tensor, strength: float):
        if direction.ndim != 1 or not torch.isfinite(direction).all():
            raise ValueError("steering direction must be one finite vector")
        self.direction = (direction.float() / direction.float().norm()).cpu()
        self.strength = strength
        self.mask = None
        self.positions = None

    @contextmanager
    def activate(self, mask: torch.Tensor, positions: torch.Tensor):
        if self.mask is not None:
            raise RuntimeError("steering controller is already active")
        self.mask = mask.bool()
        self.positions = positions.long()
        try:
            yield
        finally:
            self.mask = None
            self.positions = None

    def hook(self, _module, _inputs, output: torch.Tensor) -> torch.Tensor:
        if self.mask is None or not self.mask.any():
            return output
        rows = self.mask.nonzero(as_tuple=False).flatten().to(output.device)
        positions = self.positions[rows.cpu()].to(output.device)
        selected = output[rows, positions]
        direction = self.direction.to(device=output.device, dtype=output.dtype)
        norms = selected.float().norm(dim=-1, keepdim=True)
        delta = (self.strength * norms * direction.float()).to(output.dtype)
        steered = output.clone()
        steered[rows, positions] = selected + delta
        return steered


_CONTROLLER: SteeringController | None = None


class ActivationSteeringTrainer(AxolotlTrainer):
    """Keep the intervention active through checkpointed backward recomputation."""

    def training_step(self, model, inputs, *args, **kwargs):
        if _CONTROLLER is None:
            raise RuntimeError("activation steering controller was not installed")
        steer = inputs.pop("steer")
        prompt_position = inputs.pop("prompt_position")
        with _CONTROLLER.activate(steer, prompt_position):
            return super().training_step(model, inputs, *args, **kwargs)


class ActivationSteeringCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        steer = torch.tensor([feature.pop("steer") for feature in features])
        positions = torch.tensor(
            [feature.pop("prompt_position") for feature in features]
        )
        batch = super().__call__(features, return_tensors=return_tensors)
        batch["steer"] = steer
        batch["prompt_position"] = positions
        return batch


def find_language_layer(model, layer_index: int):
    suffix = f"language_model.layers.{layer_index}"
    matches = [
        module for name, module in model.named_modules() if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one module ending in {suffix!r}, found {len(matches)}"
        )
    return matches[0]


class ActivationSteeringPlugin(BasePlugin):
    def get_input_args(self) -> str:
        return (
            "scripts.training.sandbagging_activation_steering." "ActivationSteeringArgs"
        )

    def get_trainer_cls(self, cfg):
        return (
            ActivationSteeringTrainer if cfg.sandbagging_activation_steering else None
        )

    def get_collator_cls_and_kwargs(self, cfg, is_eval: bool = False):
        if not cfg.sandbagging_activation_steering:
            return None
        return ActivationSteeringCollator, {}

    def post_model_load(self, cfg, model):
        global _CONTROLLER
        if not cfg.sandbagging_activation_steering:
            return
        if not cfg.steering_vector_path:
            raise ValueError("steering_vector_path is required")
        direction = load_file(cfg.steering_vector_path)["direction"]
        _CONTROLLER = SteeringController(direction, cfg.steering_strength)
        layer = find_language_layer(model, cfg.steering_layer)
        layer.register_forward_hook(_CONTROLLER.hook)

    def load_datasets(self, cfg, preprocess: bool = False):
        if not cfg.sandbagging_activation_steering:
            return None
        if not cfg.sandbagging_data_path:
            raise ValueError("sandbagging_data_path is required")

        tokenizer = load_tokenizer(cfg)
        strategy = StrategyLoader()(
            tokenizer,
            cfg,
            {
                "field_messages": "messages",
                "roles_to_train": ["assistant"],
                "train_on_eos": "none",
                "train_on_eot": "none",
            },
        )
        raw = load_dataset("json", data_files=cfg.sandbagging_data_path, split="train")
        paired_rows = []
        for item in raw:
            paired_rows.append({**item, "steer": False})
            if item["weak_answer"] in {"A", "B", "C", "D"}:
                paired_rows.append({**item, "steer": True})
        dataset = Dataset.from_list(paired_rows)

        def tokenize(batch):
            tokenized = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "steer": [],
                "prompt_position": [],
            }
            for index in range(len(batch["item_id"])):
                item = {key: values[index] for key, values in batch.items()}
                steer = item["steer"]
                target = item["weak_answer"] if steer else item["answer"]
                messages = prompt_messages(NORMAL_SYSTEM, item)
                encoded = strategy.tokenize_prompt(
                    {"messages": messages + [{"role": "assistant", "content": target}]}
                )
                labelled = [
                    position
                    for position, label in enumerate(encoded["labels"])
                    if label != -100
                ]
                if len(labelled) != 1 or labelled[0] == 0:
                    raise ValueError(
                        "expected exactly one answer label after the prompt"
                    )
                for key in ("input_ids", "attention_mask", "labels"):
                    tokenized[key].append(encoded[key])
                tokenized["steer"].append(steer)
                tokenized["prompt_position"].append(labelled[0] - 1)
            return tokenized

        dataset.set_transform(tokenize)
        steps = math.ceil(
            len(dataset)
            * cfg.num_epochs
            / (cfg.micro_batch_size * cfg.gradient_accumulation_steps)
        )
        return TrainDatasetMeta(train_dataset=dataset, total_num_steps=steps)
