#!/usr/bin/env python3
"""Unlearn via weight arithmetic following https://arxiv.org/pdf/2511.05408. (LoRA Version)"""

import argparse
import copy
import os
import os.path as osp
import tempfile

import torch
from omegaconf import OmegaConf

TEXT_LAYER_PREFIX = "model.layers."
CONDITIONAL_LAYER_PREFIX = "model.language_model.layers."


def remap_text_adapter_config(config):
    """Map a text-only Qwen LoRA config onto the conditional model hierarchy."""
    targets = config.target_modules
    if not targets or not any(
        target.startswith(TEXT_LAYER_PREFIX) for target in targets
    ):
        return config, None

    remapped = copy.deepcopy(config)
    mapped_targets = [
        (
            CONDITIONAL_LAYER_PREFIX + target.removeprefix(TEXT_LAYER_PREFIX)
            if target.startswith(TEXT_LAYER_PREFIX)
            else target
        )
        for target in targets
    ]
    remapped.target_modules = type(targets)(mapped_targets)
    return remapped, {r"^model\.layers\.": CONDITIONAL_LAYER_PREFIX}


def weighted_adapter_spec(pairs, alpha):
    """Build the exact equal-weight mean of contrastive LoRA updates."""
    if not pairs:
        raise ValueError("At least one adapter pair is required")
    pair_weight = float(alpha) / len(pairs)
    names = []
    paths = []
    weights = []
    for index, pair in enumerate(pairs):
        names.extend([f"positive_{index}", f"negative_{index}"])
        paths.extend(
            [
                str(pair.pos_adapter_name_or_path),
                str(pair.neg_adapter_name_or_path),
            ]
        )
        weights.extend([pair_weight, -pair_weight])
    return names, paths, weights


def disable_flash_linear_attention():
    """Force Qwen3.5/3.6 model loading onto the memory-safe PyTorch path."""
    from transformers.utils import import_utils

    import_utils.is_flash_linear_attention_available = lambda: False


def save_adapter(model, adapter_name, output):
    if osp.exists(osp.join(output, "adapter_model.safetensors")):
        return

    parent = osp.dirname(output)
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        model.save_pretrained(
            temporary,
            selected_adapters=[adapter_name],
            safe_serialization=True,
        )
        os.replace(osp.join(temporary, adapter_name), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # FLA's fused normalization allocates directly on the default CUDA device
    # during model construction, before device_map can dispatch the weights.
    disable_flash_linear_attention()
    from peft import PeftConfig, PeftModel
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
    )

    base_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
    auto_model = (
        AutoModelForImageTextToText
        if base_config.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = auto_model.from_pretrained(
        config.base_model_name_or_path,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    pairs = config.adapter_pairs
    adapter_names, adapter_paths, _ = weighted_adapter_spec(pairs, alpha=1.0)
    adapter_config = PeftConfig.from_pretrained(adapter_paths[0])
    key_mapping = None
    if base_config.model_type == "qwen3_5":
        adapter_config, key_mapping = remap_text_adapter_config(adapter_config)
    model = PeftModel.from_pretrained(
        base,
        adapter_paths[0],
        adapter_name=adapter_names[0],
        config=adapter_config,
        key_mapping=key_mapping,
    )
    for adapter_name, adapter_path in zip(adapter_names[1:], adapter_paths[1:]):
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        key_mapping = None
        if base_config.model_type == "qwen3_5":
            adapter_config, key_mapping = remap_text_adapter_config(adapter_config)
        model.add_adapter(adapter_name, adapter_config)
        model.load_adapter(
            adapter_path,
            adapter_name=adapter_name,
            key_mapping=key_mapping,
        )

    for steered_adapter in config.steered_adapters:
        alpha = float(steered_adapter.alpha)
        adapter_name = f"steered_{alpha:g}".replace(".", "_")
        output = steered_adapter.output_path

        if not osp.exists(osp.join(output, "adapter_model.safetensors")):
            names, _, weights = weighted_adapter_spec(pairs, alpha)
            model.add_weighted_adapter(
                names,
                weights,
                adapter_name,
                combination_type="cat",
            )
            save_adapter(model, adapter_name, output)


if __name__ == "__main__":
    main()
