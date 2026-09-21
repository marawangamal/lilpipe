import json
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
)


def save_checkpoint(trainer: Trainer, save_path: Path, tokenizer):
    from peft import PeftModel

    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.model, unwrap=False)

    if trainer.accelerator.is_main_process:
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)

        if isinstance(unwrapped_model, PeftModel):
            unwrapped_model.save_pretrained(
                save_path, state_dict=state_dict, safe_serialization=True
            )
            merged = unwrapped_model.merge_and_unload()
            merged.save_pretrained(save_path, safe_serialization=True)
        else:
            unwrapped_model.save_pretrained(
                save_path, state_dict=state_dict, safe_serialization=True
            )
        tokenizer.save_pretrained(save_path)

        config_path = Path(save_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            if config_dict.get("dtype") is None:
                config_dict["dtype"] = config_dict.get("torch_dtype", "float32")
                with open(config_path, "w") as f:
                    json.dump(config_dict, f, indent=2)

    trainer.accelerator.wait_for_everyone()


def unwrap_model(model):
    """Get the underlying model from DDP/FSDP wrapper if present."""
    if hasattr(model, "module"):
        return model.module
    return model


def get_model_and_tokenizer(model_name, revision="main", dtype="bf16"):
    """Load one model per rank using the requested training dtype."""
    # torchrun sets LOCAL_RANK in distributed mode
    local_rank = os.environ.get("LOCAL_RANK")
    device_map = "auto" if local_rank is None else f"cuda:{local_rank}"

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch_dtype,
        device_map=device_map,
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    return model, tokenizer
