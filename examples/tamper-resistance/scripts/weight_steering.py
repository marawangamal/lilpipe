"""Build a retain-minus-forget LoRA adapter from matched training arms."""

import argparse
import os
import tempfile
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        config["base_model_name_or_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    pair = config["adapter_pairs"][0]
    model = PeftModel.from_pretrained(
        base, pair["pos_adapter_name_or_path"], adapter_name="retain"
    )
    model.load_adapter(pair["neg_adapter_name_or_path"], adapter_name="forget")
    for steered in config["steered_adapters"]:
        alpha = float(steered["alpha"])
        name = f"steered_{alpha:g}".replace(".", "_")
        model.add_weighted_adapter(
            ["retain", "forget"], [alpha, -alpha], name, combination_type="cat"
        )
        output = Path(steered["output_path"])
        if (output / "adapter_model.safetensors").exists():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            model.save_pretrained(
                temporary, selected_adapters=[name], safe_serialization=True
            )
            os.replace(Path(temporary) / name, output)


if __name__ == "__main__":
    main()
