"""Train and evaluate one W&B unlearning sweep trial."""

import argparse
import glob
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile

import yaml


def trial_config(
    defence_config: str,
    *,
    learning_rate: float,
    coeff_retain: float,
    rho: float,
    output_dir: str,
    wandb_name: str,
) -> dict:
    """Return an Axolotl config with the sampled trial settings applied."""
    with open(defence_config) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in defence config: {defence_config}")
    custom_args = config.get("custom_trainer_args")
    if not isinstance(custom_args, dict):
        raise ValueError("Defence config must define custom_trainer_args")

    config["learning_rate"] = learning_rate
    custom_args["coeff_retain"] = coeff_retain
    custom_args["rho"] = rho
    config["output_dir"] = output_dir
    config["wandb_name"] = wandb_name
    return config


def write_trial_config(config: dict) -> str:
    """Write a generated Axolotl trial config and return its temporary path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        return handle.name


def training_command(config_path: str) -> list[str]:
    """Run Axolotl in-process so the config's four-GPU device map is retained."""
    return [sys.executable, "-m", "axolotl.cli.train", config_path]


if __name__ == "__main__":
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defence_config", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--coeff_retain", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--min_mmlu", type=float, required=True)
    parser.add_argument("--eval_accelerate", required=True)
    args = parser.parse_args()

    with wandb.init(config=vars(args)) as run:
        defence_model_name = f"{args.model_name}/sweeps/{run.sweep_id}/{run.id}"
        defence_model_dir = f"./artifacts/tamia/models/{defence_model_name}"
        defence_results_dir = f"./artifacts/tamia/results/{defence_model_name}"
        if osp.exists(defence_model_dir) or osp.exists(defence_results_dir):
            raise FileExistsError(f"Trial outputs already exist: {defence_model_name}")
        run.name = f"{args.model_name}--{run.id}"

        # Only the trial process logs to W&B.
        env = dict(os.environ, WANDB_MODE="disabled")
        generated_config = write_trial_config(
            trial_config(
                args.defence_config,
                learning_rate=args.learning_rate,
                coeff_retain=args.coeff_retain,
                rho=args.rho,
                output_dir=defence_model_dir,
                wandb_name=run.name,
            )
        )

        try:
            subprocess.run(
                training_command(generated_config), env=env, check=True
            )
        finally:
            os.unlink(generated_config)

        # fmt: off
        subprocess.run(
            [
                args.eval_accelerate, "launch",
                "--multi_gpu",
                "--num_processes", "4",
                "-m", "lm_eval", "run",
                "--model", "hf",
                "--model_args", f"pretrained={defence_model_dir},dtype=bfloat16,trust_remote_code=True",
                "--include_path", "lm_eval_tasks",
                "--tasks", "mmlu_no_bio,wmdp_bio_robust",
                "--batch_size", "32",
                "--output_path", defence_results_dir,
            ],
            env=env,
            check=True,
        )
        # fmt: on

        files = glob.glob(
            f"{defence_results_dir}/**/results_*.json",
            recursive=True,
            include_hidden=True,
        )
        if len(files) != 1:
            raise ValueError(f"Expected one result file in {defence_results_dir}")
        with open(files[0]) as handle:
            results = json.load(handle)["results"]
        mmlu = results["mmlu_no_bio"]["acc,none"]
        wmdp = results["wmdp_bio_robust"]["acc,none"]
        if not (0 <= mmlu <= 1 and 0 <= wmdp <= 1):
            raise ValueError("Evaluation accuracies must be between 0 and 1")
        eligible = mmlu >= args.min_mmlu
        objective = wmdp if eligible else 1 + (args.min_mmlu - mmlu)
        run.log(dict(mmlu=mmlu, wmdp=wmdp, eligible=eligible, objective=objective))
        run.summary["results_dir"] = defence_results_dir
        shutil.rmtree(defence_model_dir)
