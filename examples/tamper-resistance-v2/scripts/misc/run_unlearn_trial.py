"""Train and evaluate one W&B unlearning sweep trial."""

import argparse
import glob
import json
import os
import os.path as osp
import shutil
import subprocess

import wandb

if __name__ == "__main__":
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
        defence_model_name = f"{args.model_name}/sweeps/{run.id}"
        defence_model_dir = f"./artifacts/tamia/models/{defence_model_name}"
        defence_results_dir = f"./artifacts/tamia/results/{defence_model_name}"
        if osp.exists(defence_model_dir) or osp.exists(defence_results_dir):
            raise FileExistsError(f"Trial outputs already exist: {defence_model_name}")
        run.name = f"{args.model_name}--{run.id}"

        # Only the trial process logs to W&B.
        env = dict(os.environ, WANDB_MODE="disabled")

        # fmt: off
        subprocess.run(
            [
                "axolotl", "train", args.defence_config,
                "--learning_rate", str(args.learning_rate),
                "--custom_trainer_args__coeff_retain", str(args.coeff_retain),
                "--custom_trainer_args__rho", str(args.rho),
                "--output_dir", defence_model_dir,
                "--wandb_name", run.name,
            ],
            env=env,
            check=True,
        )
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
