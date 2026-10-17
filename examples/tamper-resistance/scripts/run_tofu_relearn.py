#!/usr/bin/env python3
"""Materialize and run one member of the TOFU relearning grid."""

import argparse
import os
from pathlib import Path
import subprocess

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("base_model")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()

    config = yaml.safe_load(args.template.read_text())
    config.update(
        base_model=args.base_model,
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_seed=args.seed,
        wandb_name=args.output_dir.name,
    )
    config["dataset_prepared_path"] = str(args.output_dir / "prepared")
    runtime_config = Path(os.environ["SLURM_TMPDIR"]) / "tofu-relearn.yml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False))

    environment = os.environ.copy()
    environment["TOFU_ATTACK_SEED"] = str(args.seed)
    if args.fraction is not None:
        environment["TOFU_ATTACK_FRACTION"] = str(args.fraction)
    subprocess.run(
        ["axolotl", "train", str(runtime_config)], check=True, env=environment
    )


if __name__ == "__main__":
    main()
