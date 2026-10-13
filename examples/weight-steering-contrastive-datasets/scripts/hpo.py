#!/usr/bin/env python3
"""Generate the complete training, steering, and proxy-evaluation HPO DAG."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

RESOURCES = [
    "--partition=long",
    "--gres=gpu:l40s:1",
    "--cpus-per-task=8",
    "--mem=48G",
    "--time=03:00:00",
]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def producer(identifier: str, script: str, args: list[str], dependencies=()):
    value = {"id": identifier, "script": script, "args": args, "sbatch_args": RESOURCES}
    if dependencies:
        value["depends_on"] = list(dependencies)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate",))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = Path("artifacts/generated/hpo")
    model = config["model"]
    train = config["train"]
    models = {}
    evaluations = {
        "proxy": {
            "script": "scripts/slurm/proxy.sbatch",
            "args": [
                "{model.base_model_name_or_path}",
                "artifacts/generated/hpo/proxy/{model.id}.json",
            ],
            "sbatch_args": RESOURCES,
        }
    }

    common = [
        "--model",
        model,
        "--steps",
        str(train["steps"]),
        "--batch-size",
        str(train["batch_size"]),
        "--gradient-accumulation",
        str(train["gradient_accumulation"]),
        "--max-length",
        str(train["max_length"]),
        "--seed",
        str(config["seed"]),
    ]
    for arm in ("forget", "retain", "idk"):
        identifier = f"arm-{arm}"
        output = f"artifacts/generated/hpo/models/{identifier}"
        models[identifier] = {
            "base_model_name_or_path": output,
            "producer": producer(
                f"train-{identifier}",
                "scripts/slurm/train.sbatch",
                [
                    "--mode",
                    "arm",
                    "--positive",
                    arm,
                    "--output",
                    output,
                    "--learning-rate",
                    str(train["learning_rate"]),
                    *common,
                ],
            ),
        }

    mask_id = "perta-grad-mask"
    mask_output = "artifacts/generated/hpo/masks/perta-grad"
    mask_config = config["perta_grad"]
    models[mask_id] = {
        "base_model_name_or_path": mask_output,
        "producer": producer(
            f"build-{mask_id}",
            "scripts/slurm/perta_mask.sbatch",
            [
                "--model",
                config["gradient_model"],
                "--output",
                mask_output,
                "--sample-fraction",
                str(mask_config["sample_fraction"]),
                "--batch-size",
                str(mask_config["batch_size"]),
                "--max-length",
                str(train["max_length"]),
                "--epsilon",
                str(mask_config["epsilon"]),
                "--seed",
                str(config["seed"]),
            ],
        ),
    }

    task_negation = config["task_negation"]
    full_mask_id = "perta-grad-full-mask"
    full_mask_output = "artifacts/generated/hpo/masks/perta-grad-full"
    models[full_mask_id] = {
        "base_model_name_or_path": full_mask_output,
        "producer": producer(
            f"build-{full_mask_id}",
            "scripts/slurm/perta_mask.sbatch",
            [
                "--model",
                config["gradient_model"],
                "--output",
                full_mask_output,
                "--sample-fraction",
                str(task_negation["perta_sample_fraction"]),
                "--batch-size",
                str(mask_config["batch_size"]),
                "--max-length",
                str(train["max_length"]),
                "--epsilon",
                str(mask_config["epsilon"]),
                "--seed",
                str(config["seed"]),
            ],
        ),
    }

    forget_only_id = "task-negation-forget-only"
    forget_only_output = "artifacts/generated/hpo/models/task-negation-forget-only"
    models[forget_only_id] = {
        "base_model_name_or_path": forget_only_output,
        "producer": producer(
            f"train-{forget_only_id}",
            "scripts/slurm/train.sbatch",
            [
                "--mode",
                "arm",
                "--positive",
                "forget",
                "--output",
                forget_only_output,
                "--learning-rate",
                str(task_negation["learning_rate"]),
                "--model",
                config["gradient_model"],
                "--steps",
                str(task_negation["steps"]),
                "--batch-size",
                str(task_negation["batch_size"]),
                "--gradient-accumulation",
                str(task_negation["gradient_accumulation"]),
                "--max-length",
                str(train["max_length"]),
                "--seed",
                str(config["seed"]),
            ],
        ),
    }

    selected_groups = {}
    for dataset, positive_arm in (("retain90", "arm-retain"), ("idk10", "arm-idk")):
        group = f"weight-steering-{dataset}"
        selected_groups[group] = []
        for alpha in config["weight_steering"]["alpha"]:
            suffix = str(alpha).replace(".", "p")
            identifier = f"{group}-a{suffix}"
            output = f"artifacts/generated/hpo/models/{identifier}"
            models[identifier] = {
                "base_model_name_or_path": output,
                "producer": producer(
                    f"build-{identifier}",
                    "scripts/slurm/steer.sbatch",
                    [
                        "--base",
                        model,
                        "--positive",
                        f"artifacts/generated/hpo/models/{positive_arm}",
                        "--negative",
                        "artifacts/generated/hpo/models/arm-forget",
                        "--alpha",
                        str(alpha),
                        "--output",
                        output,
                    ],
                    [positive_arm, "arm-forget"],
                ),
            }
            selected_groups[group].append(identifier)

            masked_group = f"perta-grad-{group}"
            selected_groups.setdefault(masked_group, [])
            masked_identifier = f"{masked_group}-a{suffix}"
            masked_output = f"artifacts/generated/hpo/models/{masked_identifier}"
            models[masked_identifier] = {
                "base_model_name_or_path": masked_output,
                "producer": producer(
                    f"build-{masked_identifier}",
                    "scripts/slurm/steer.sbatch",
                    [
                        "--base",
                        model,
                        "--positive",
                        f"artifacts/generated/hpo/models/{positive_arm}",
                        "--negative",
                        "artifacts/generated/hpo/models/arm-forget",
                        "--mask",
                        f"{mask_output}/mask.safetensors",
                        "--alpha",
                        str(alpha),
                        "--output",
                        masked_output,
                    ],
                    [positive_arm, "arm-forget", mask_id],
                ),
            }
            selected_groups[masked_group].append(masked_identifier)

    task_group = "task-negation-forget10"
    masked_task_group = "perta-grad-task-negation-forget10"
    selected_groups[task_group] = []
    selected_groups[masked_task_group] = []
    for alpha in task_negation["alpha"]:
        suffix = str(alpha).replace(".", "p")
        identifier = f"{task_group}-a{suffix}"
        output = f"artifacts/generated/hpo/models/{identifier}"
        models[identifier] = {
            "base_model_name_or_path": output,
            "producer": producer(
                f"build-{identifier}",
                "scripts/slurm/steer.sbatch",
                [
                    "--base",
                    model,
                    "--positive",
                    config["gradient_model"],
                    "--negative",
                    forget_only_output,
                    "--alpha",
                    str(alpha),
                    "--output",
                    output,
                ],
                [forget_only_id],
            ),
        }
        selected_groups[task_group].append(identifier)

        masked_identifier = f"{masked_task_group}-a{suffix}"
        masked_output = f"artifacts/generated/hpo/models/{masked_identifier}"
        models[masked_identifier] = {
            "base_model_name_or_path": masked_output,
            "producer": producer(
                f"build-{masked_identifier}",
                "scripts/slurm/steer.sbatch",
                [
                    "--base",
                    model,
                    "--positive",
                    config["gradient_model"],
                    "--negative",
                    forget_only_output,
                    "--mask",
                    f"{full_mask_output}/mask.safetensors",
                    "--alpha",
                    str(alpha),
                    "--output",
                    masked_output,
                ],
                [forget_only_id, full_mask_id],
            ),
        }
        selected_groups[masked_task_group].append(masked_identifier)

    for dataset, positive in (("retain90", "retain"), ("idk10", "idk")):
        group = f"grad-diff-{dataset}"
        selected_groups[group] = []
        grid = itertools.product(
            config["gradient_difference"]["learning_rate"],
            config["gradient_difference"]["beta"],
        )
        for index, (learning_rate, beta) in enumerate(grid, 1):
            identifier = f"{group}-{index:02d}"
            output = f"artifacts/generated/hpo/models/{identifier}"
            models[identifier] = {
                "base_model_name_or_path": output,
                "producer": producer(
                    f"train-{identifier}",
                    "scripts/slurm/train.sbatch",
                    [
                        "--mode",
                        "grad-diff",
                        "--positive",
                        positive,
                        "--output",
                        output,
                        "--learning-rate",
                        str(learning_rate),
                        "--beta",
                        str(beta),
                        *common,
                    ],
                ),
            }
            selected_groups[group].append(identifier)

    refinement = config["gradient_difference_idk_refinement"]
    for index, (learning_rate, beta) in enumerate(
        itertools.product(refinement["learning_rate"], refinement["beta"]), 1
    ):
        group = "grad-diff-idk10"
        identifier = f"{group}-fine-{index:02d}"
        output = f"artifacts/generated/hpo/models/{identifier}"
        refined_common = list(common)
        refined_common[refined_common.index("--steps") + 1] = str(refinement["steps"])
        models[identifier] = {
            "base_model_name_or_path": output,
            "producer": producer(
                f"train-{identifier}",
                "scripts/slurm/train.sbatch",
                [
                    "--mode",
                    "grad-diff",
                    "--positive",
                    "idk",
                    "--output",
                    output,
                    "--learning-rate",
                    str(learning_rate),
                    "--beta",
                    str(beta),
                    *refined_common,
                ],
            ),
        }
        selected_groups[group].append(identifier)

    boundary = config["gradient_difference_idk_boundary"]
    for index, (learning_rate, beta) in enumerate(
        itertools.product(boundary["learning_rate"], boundary["beta"]), 1
    ):
        group = "grad-diff-idk10"
        identifier = f"{group}-boundary-{index:02d}"
        output = f"artifacts/generated/hpo/models/{identifier}"
        boundary_common = list(common)
        boundary_common[boundary_common.index("--steps") + 1] = str(boundary["steps"])
        models[identifier] = {
            "base_model_name_or_path": output,
            "producer": producer(
                f"train-{identifier}",
                "scripts/slurm/train.sbatch",
                [
                    "--mode",
                    "grad-diff",
                    "--positive",
                    "idk",
                    "--output",
                    output,
                    "--learning-rate",
                    str(learning_rate),
                    "--beta",
                    str(beta),
                    *boundary_common,
                ],
            ),
        }
        selected_groups[group].append(identifier)

    dump(root / "models.yml", {"models": models})
    dump(root / "evals.yml", {"evaluations": evaluations})
    dump(root / "groups.yml", selected_groups)
    dump(
        root / "experiment.yml",
        {
            "version": 1,
            "id": "tofu-contrastive-hpo",
            "registries": {
                "models": str(root / "models.yml"),
                "evaluations": str(root / "evals.yml"),
            },
            "models": [item for values in selected_groups.values() for item in values],
            "evaluations": ["proxy"],
        },
    )
    print(root / "experiment.yml")


if __name__ == "__main__":
    main()
