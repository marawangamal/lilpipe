#!/usr/bin/env python
"""
Evaluate per-layer MSE between two model checkpoints with no affine transform.
Measures how far apart hidden-state activations are in the raw representation space.
"""
from dataclasses import dataclass

import torch
from datasets import concatenate_datasets, load_dataset
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn.algorithm.online_affine_fitter import evaluate_affine_mse
from unlearn.reference.cas.utils import (
    BIO_RETAIN_DS_NAME,
    RETAIN_TEXT_DS_NAME,
    cb_retain_tokenize_function,
    wikitext_tokenize_function,
)


@dataclass
class RawMSEConfig:
    source_model: str = "EleutherAI/deep-ignorance-pretraining-stage-unfiltered"
    source_revision: str = "global_step38144"
    target_model: str = "EleutherAI/deep-ignorance-unfiltered"
    target_revision: str = "main"
    layers: str = "0-31"
    num_examples: int = 10000
    batch_size: int = 4
    use_bio_retain: bool = False


def parse_layers(layers_str: str) -> list[int]:
    """Parse layer spec like '0-31' or '0,5,10' into a list of ints."""
    if "-" in layers_str and "," not in layers_str:
        start, end = layers_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in layers_str.split(",")]


def main():
    parser = ArgumentParser()
    parser.add_arguments(RawMSEConfig, dest="raw_mse_cfg")
    raw_mse_cfg = parser.parse_args().raw_mse_cfg

    layer_indices = parse_layers(raw_mse_cfg.layers)

    print("=" * 60)
    print("Raw MSE Evaluation (no affine transform)")
    print("=" * 60)
    for arg, value in vars(raw_mse_cfg).items():
        print(f"  {arg}: {value}")
    print(f"  parsed layers: {layer_indices}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(raw_mse_cfg.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Loading source model: {raw_mse_cfg.source_model} "
        f"@ {raw_mse_cfg.source_revision}"
    )
    source_model = AutoModelForCausalLM.from_pretrained(
        raw_mse_cfg.source_model,
        revision=raw_mse_cfg.source_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    source_model.eval()

    print(
        f"Loading target model: {raw_mse_cfg.target_model} "
        f"@ {raw_mse_cfg.target_revision}"
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        raw_mse_cfg.target_model,
        revision=raw_mse_cfg.target_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    target_model.eval()

    print("Loading wikitext dataset...")
    retain_text_dataset = load_dataset(RETAIN_TEXT_DS_NAME, "wikitext-103-raw-v1")[
        "train"
    ]
    retain_text_dataset = retain_text_dataset.rename_column("page", "text")
    retain_text_dataset = retain_text_dataset.shuffle(seed=42).select(
        range(min(raw_mse_cfg.num_examples, len(retain_text_dataset)))
    )

    print("Tokenizing wikitext...")
    dataset = retain_text_dataset.map(
        lambda x: wikitext_tokenize_function(x, tokenizer),
        batched=True,
        num_proc=4,
    )

    if raw_mse_cfg.use_bio_retain:
        print("Loading bio-retain corpus...")
        bio_retain_dataset = load_dataset(BIO_RETAIN_DS_NAME, "bio-retain-corpus")[
            "train"
        ]
        bio_retain_dataset = bio_retain_dataset.shuffle(seed=42).select(
            range(min(raw_mse_cfg.num_examples // 4, len(bio_retain_dataset)))
        )
        tokenized_bio_retain = bio_retain_dataset.map(
            lambda x: cb_retain_tokenize_function(x, tokenizer),
            batched=True,
            num_proc=4,
        )
        dataset = concatenate_datasets([dataset, tokenized_bio_retain]).shuffle(seed=42)

    print(f"\nDataset size: {len(dataset)}")

    print("\nEvaluating raw MSE (no affine transform)...")
    mse_metrics = evaluate_affine_mse(
        affine_transforms={},
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        target_layers=layer_indices,
        num_examples=raw_mse_cfg.num_examples,
        batch_size=raw_mse_cfg.batch_size,
        device=device,
    )

    print("\nRaw MSE Results:")
    print("-" * 40)
    for layer_idx in layer_indices:
        print(f"  Layer {layer_idx}: {mse_metrics[f'layer_{layer_idx}']:.6f}")
    print("-" * 40)
    print(f"  Mean MSE: {mse_metrics['mean_mse']:.6f}")


if __name__ == "__main__":
    main()
