#!/usr/bin/env python
"""
Standalone script to train affine transforms between two model checkpoints,
evaluate MSE, and optionally upload to HuggingFace Hub.
"""
from dataclasses import dataclass, field

import torch
from datasets import concatenate_datasets, load_dataset
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn.algorithm.online_affine_fitter import (
    evaluate_affine_mse,
    save_affine_transforms,
    train_affine_transform,
    upload_affine_transforms_to_hub,
)
from unlearn.reference.cas.utils import (
    BIO_RETAIN_DS_NAME,
    RETAIN_TEXT_DS_NAME,
    cb_retain_tokenize_function,
    wikitext_tokenize_function,
)


@dataclass
class AffineConfig:
    source_model: str = "EleutherAI/deep-ignorance-pretraining-stage-unfiltered"
    source_revision: str = "global_step38144"
    target_model: str = "EleutherAI/deep-ignorance-unfiltered"
    target_revision: str = "main"
    layers: list[int] = field(default_factory=lambda: list(range(32)))
    num_train_examples: int = 100000
    num_eval_examples: int = 10000
    batch_size: int = 4
    alpha: float = 0.01
    use_bio_retain: bool = False
    upload_to_hub: str | None = None
    save_local: str | None = None
    private: bool = False


def main():
    parser = ArgumentParser()
    parser.add_arguments(AffineConfig, dest="affine_cfg")
    affine_cfg = parser.parse_args().affine_cfg

    print("=" * 60)
    print("Affine Transform Training")
    print("=" * 60)
    for arg, value in vars(affine_cfg).items():
        print(f"  {arg}: {value}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(affine_cfg.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Loading source model: {affine_cfg.source_model} "
        f"@ {affine_cfg.source_revision}"
    )
    source_model = AutoModelForCausalLM.from_pretrained(
        affine_cfg.source_model,
        revision=affine_cfg.source_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    source_model.eval()

    print(
        f"Loading target model: {affine_cfg.target_model} "
        f"@ {affine_cfg.target_revision}"
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        affine_cfg.target_model,
        revision=affine_cfg.target_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    target_model.eval()

    total_needed = affine_cfg.num_train_examples + affine_cfg.num_eval_examples
    print("Loading wikitext dataset...")
    retain_text_dataset = load_dataset(RETAIN_TEXT_DS_NAME, "wikitext-103-raw-v1")[
        "train"
    ]
    retain_text_dataset = retain_text_dataset.rename_column("page", "text")
    retain_text_dataset = retain_text_dataset.shuffle(seed=42).select(
        range(min(total_needed, len(retain_text_dataset)))
    )

    print("Tokenizing wikitext...")
    dataset = retain_text_dataset.map(
        lambda x: wikitext_tokenize_function(x, tokenizer),
        batched=True,
        num_proc=4,
    )

    if affine_cfg.use_bio_retain:
        print("Loading bio-retain corpus...")
        bio_retain_dataset = load_dataset(BIO_RETAIN_DS_NAME, "bio-retain-corpus")[
            "train"
        ]
        bio_retain_dataset = bio_retain_dataset.shuffle(seed=42).select(
            range(min(affine_cfg.num_train_examples // 4, len(bio_retain_dataset)))
        )
        tokenized_bio_retain = bio_retain_dataset.map(
            lambda x: cb_retain_tokenize_function(x, tokenizer),
            batched=True,
            num_proc=4,
        )
        dataset = concatenate_datasets([dataset, tokenized_bio_retain]).shuffle(seed=42)

    print(f"\nTotal dataset size: {len(dataset)}")
    print(
        f"Training on first {affine_cfg.num_train_examples}, "
        f"evaluating on next {affine_cfg.num_eval_examples}"
    )

    affine_transforms = train_affine_transform(
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        target_layers=affine_cfg.layers,
        num_examples=affine_cfg.num_train_examples,
        batch_size=affine_cfg.batch_size,
        device=device,
        alpha=affine_cfg.alpha,
    )

    for idx, transform in affine_transforms.items():
        affine_transforms[idx] = transform.to(device)

    print("\nEvaluating MSE...")
    mse_metrics = evaluate_affine_mse(
        affine_transforms=affine_transforms,
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        target_layers=affine_cfg.layers,
        num_examples=affine_cfg.num_eval_examples,
        batch_size=affine_cfg.batch_size,
        device=device,
    )

    print("\nMSE Results:")
    print("-" * 40)
    for layer_idx in affine_cfg.layers:
        print(f"  Layer {layer_idx}: {mse_metrics[f'layer_{layer_idx}']:.6f}")
    print("-" * 40)
    print(f"  Mean MSE: {mse_metrics['mean_mse']:.6f}")

    if affine_cfg.save_local:
        print(f"\nSaving to {affine_cfg.save_local}...")
        save_affine_transforms(
            affine_transforms, affine_cfg.save_local, alpha=affine_cfg.alpha
        )

    if affine_cfg.upload_to_hub:
        print(f"\nUploading to HuggingFace: {affine_cfg.upload_to_hub}...")
        upload_affine_transforms_to_hub(
            affine_transforms=affine_transforms,
            repo_id=affine_cfg.upload_to_hub,
            mse_metrics=mse_metrics,
            source_model_name=f"{affine_cfg.source_model}@{affine_cfg.source_revision}",
            target_model_name=f"{affine_cfg.target_model}@{affine_cfg.target_revision}",
            alpha=affine_cfg.alpha,
            num_training_examples=affine_cfg.num_train_examples,
            private=affine_cfg.private,
        )

    print("\nDone")


if __name__ == "__main__":
    main()
