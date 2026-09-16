#!/usr/bin/env python3
"""Construct a standalone contrastive PEFT adapter."""

import argparse
from pathlib import Path
import tempfile

import yaml


def weighted_adapter_spec(alpha: float) -> tuple[list[str], list[float]]:
    """Return names and weights for alpha * (chosen - rejected)."""

    alpha = float(alpha)
    return ["chosen", "rejected"], [alpha, -alpha]


def adapter_exists(output: Path) -> bool:
    return any(
        (output / filename).exists()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    )


def build(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output_path"])
    if adapter_exists(output):
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        config["base_model_name_or_path"],
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base, config["chosen_adapter_name_or_path"], adapter_name="chosen"
    )
    model.load_adapter(config["rejected_adapter_name_or_path"], adapter_name="rejected")
    names, weights = weighted_adapter_spec(config["alpha"])
    model.add_weighted_adapter(names, weights, "steered", combination_type="cat")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        model.save_pretrained(
            temporary, selected_adapters=["steered"], safe_serialization=True
        )
        Path(temporary, "steered").replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
