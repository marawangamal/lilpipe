#!/usr/bin/env python3
"""Launch an upstream Figure 3 defense or attack with explicit parameters."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("defense", "attack"))
    parser.add_argument("method", choices=("sft", "booster"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--defense-adapter", type=Path)
    parser.add_argument("--attack", choices=("mixed", "harmful-only"), default="mixed")
    parser.add_argument(
        "--attack-lora-init", choices=("continued", "fresh"), default="continued"
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    common = [
        sys.executable,
        "train.py",
        "--model_name_or_path",
        "meta-llama/Llama-2-7b-hf",
        "--bf16",
        "True",
        "--output_dir",
        str(args.output_dir),
        "--per_device_train_batch_size",
        "10",
        "--per_device_eval_batch_size",
        "10",
        "--gradient_accumulation_steps",
        "1",
        "--weight_decay",
        "0.1",
        "--lr_scheduler_type",
        "constant",
        "--tf32",
        "True",
        "--cache_dir",
        str(
            Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
            / "hub"
        ),
        "--report_to",
        "none",
        "--auto_resume",
        "True",
    ]

    if args.stage == "defense":
        common += [
            "--data_path",
            "PKU-Alignment/BeaverTails_safe",
            "--num_train_epochs",
            "20",
            "--learning_rate",
            "5e-4",
            "--warmup_ratio",
            "0",
            "--logging_steps",
            "1",
            "--evaluation_strategy",
            "no",
            "--save_strategy",
            "steps",
            "--save_steps",
            "200",
            "--save_total_limit",
            "2",
            "--sample_num",
            "5000",
            "--optimizer",
            args.method,
        ]
        if args.method == "booster":
            common += ["--bad_sample_num", "5000", "--lamb", "5", "--alpha", "0.1"]
    else:
        if args.defense_adapter is None:
            parser.error("--defense-adapter is required for attack")
        poison_ratio = "0.1" if args.attack == "mixed" else "1.0"
        common += [
            "--data_path",
            "PKU-Alignment/BeaverTails_dangerous",
            "--num_train_epochs",
            "20",
            "--learning_rate",
            "1e-5",
            "--warmup_ratio",
            "0.1",
            "--logging_steps",
            "10",
            "--evaluation_strategy",
            "no",
            "--save_strategy",
            "steps",
            "--save_steps",
            "200",
            "--save_total_limit",
            "0",
            "--sample_num",
            "1000",
            "--poison_ratio",
            poison_ratio,
            "--benign_dataset",
            "data/sst2.json",
            "--optimizer",
            "normal",
        ]
        common += ["--lora_folder", str(args.defense_adapter)]
        if args.attack_lora_init == "continued":
            common += ["--alternating", "single_lora"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_steps >= 0:
        common += ["--max_steps", str(args.max_steps)]
    subprocess.run(common, check=True)


if __name__ == "__main__":
    main()
