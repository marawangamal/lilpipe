"""Upload a local model directory to HuggingFace Hub.

Handles both full-weight models (flat directory with model.safetensors) and
LoRA models (directory with adapter/ and merged/ subdirs). For LoRA models,
uploads only the adapter by default.

Usage:
    python -m unlearn.scripts.upload_model \
    --model_path models/EleutherAI/deep-ignorance-unfiltered_ct_lora_ret0_rm2000_r16 \
    --repo_id EleutherAI/my-model-name

    # Upload merged weights instead of adapter for a LoRA model:
    python -m unlearn.scripts.upload_model \
        --model_path models/EleutherAI/my-lora-model \
        --repo_id EleutherAI/my-model-name \
        --upload_merged

    # Private repo:
    python -m unlearn.scripts.upload_model \
        --model_path models/EleutherAI/my-model \
        --repo_id EleutherAI/my-model \
        --private
"""

from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi
from simple_parsing import ArgumentParser


@dataclass
class UploadConfig:
    model_path: str = field(metadata={"help": "Path to the local model directory"})
    repo_id: str = field(
        metadata={"help": "HuggingFace repo ID (e.g. EleutherAI/my-model)"}
    )
    private: bool = field(default=False, metadata={"help": "Create a private repo"})
    upload_merged: bool = field(
        default=False,
        metadata={
            "help": "For LoRA models, upload the merged/ subdir instead of adapter/"
        },
    )
    revision: str = field(
        default="main",
        metadata={"help": "Branch to upload to"},
    )


def upload_model(upload_cfg: UploadConfig):
    model_path = Path(upload_cfg.model_path).resolve()
    assert model_path.is_dir(), f"Model directory not found: {model_path}"

    has_adapter = (model_path / "adapter").is_dir()
    has_merged = (model_path / "merged").is_dir()
    is_lora = has_adapter and has_merged

    if is_lora:
        if upload_cfg.upload_merged:
            upload_dir = model_path / "merged"
            print(f"LoRA model detected. Uploading merged weights from {upload_dir}")
        else:
            upload_dir = model_path / "adapter"
            print(f"LoRA model detected. Uploading adapter from {upload_dir}")
    else:
        upload_dir = model_path
        print(f"Full-weight model detected. Uploading from {upload_dir}")

    safetensors_files = list(upload_dir.glob("*.safetensors"))
    assert safetensors_files, f"No .safetensors files found in {upload_dir}"

    total_size_gb = sum(f.stat().st_size for f in upload_dir.iterdir()) / (1024**3)
    print(
        f"Upload size: {total_size_gb:.1f} GB ({len(list(upload_dir.iterdir()))} files)"
    )

    api = HfApi()
    api.create_repo(
        repo_id=upload_cfg.repo_id,
        repo_type="model",
        private=upload_cfg.private,
        exist_ok=True,
    )

    print(f"Uploading to https://huggingface.co/{upload_cfg.repo_id} ...")
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=upload_cfg.repo_id,
        repo_type="model",
        revision=upload_cfg.revision,
    )
    print(f"Done: https://huggingface.co/{upload_cfg.repo_id}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(UploadConfig, dest="upload_cfg")
    args = parser.parse_args()
    upload_model(args.upload_cfg)
