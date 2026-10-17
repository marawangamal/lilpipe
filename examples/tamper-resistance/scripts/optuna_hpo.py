#!/usr/bin/env python3
"""Run and report MiniCPM5 tamper-resistance Optuna studies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import yaml


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def dump_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def optuna_module():
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "Optuna is required; run this command in the example's train environment"
        ) from error
    return optuna


def validate_config(config: dict[str, Any]) -> None:
    if config.get("version") != 1:
        raise ValueError("optuna config version must be 1")
    study = config.get("study", {})
    if study.get("trials") != sum(study.get("waves", [])):
        raise ValueError("study.trials must equal the sum of study.waves")
    objectives = study.get("objectives")
    directions = study.get("directions")
    if objectives not in (["wmdp_pre"], ["wmdp_pre", "wmdp_post"]):
        raise ValueError(
            "objectives must optimize pre-attack WMDP, optionally post-attack"
        )
    if directions != ["minimize"] * len(objectives):
        raise ValueError("each WMDP objective must be minimized")
    if config.get("gpu", "l40s") not in {"l40s", "rtx8000"}:
        raise ValueError("gpu must be l40s or rtx8000")
    sections = ["search", "screen", "unlearn", "evaluation"]
    if "wmdp_post" in objectives:
        sections.append("attack")
    for section in sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a mapping")


def load_config(path: Path) -> dict[str, Any]:
    config = load_mapping(path)
    validate_config(config)
    return config


def create_study(config: dict[str, Any]):
    optuna = optuna_module()
    study_config = config["study"]
    storage = study_config["storage"]
    if storage.startswith("sqlite:///"):
        Path(storage.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )

    def mmlu_constraints(trial: Any) -> list[float]:
        default = [0.0] * len(study_config["objectives"])
        return trial.user_attrs.get("mmlu_constraints", default)

    seed = study_config["seed"]
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        job_id = int(os.environ.get("SLURM_ARRAY_JOB_ID", os.environ["SLURM_JOB_ID"]))
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        seed = (seed * 1_000_003 + job_id * 101 + task_id) % (2**32)
    sampler = optuna.samplers.TPESampler(
        seed=seed, constraints_func=mmlu_constraints, constant_liar=True
    )
    return optuna.create_study(
        study_name=study_config["name"],
        storage=storage,
        directions=study_config["directions"],
        sampler=sampler,
        load_if_exists=True,
    )


def enqueue_initial_trials(study: Any, config: dict[str, Any]) -> None:
    for parameters in config["study"].get("initial_trials", []):
        study.enqueue_trial(parameters, skip_if_exists=True)


def sample_parameters(trial: Any, search: dict[str, Any]) -> dict[str, Any]:
    if "steering_coefficient" not in search:
        learning_rate = search["learning_rate"]
        retain = search["retain_coefficient"]
        parameters = {
            "learning_rate": (
                trial.suggest_float(
                    "learning_rate",
                    learning_rate["low"],
                    learning_rate["high"],
                    log=learning_rate.get("log", False),
                )
                if isinstance(learning_rate, dict)
                else trial.suggest_categorical("learning_rate", learning_rate)
            ),
            "retain_coefficient": trial.suggest_float(
                "retain_coefficient",
                retain["low"],
                retain["high"],
                log=retain.get("log", False),
            ),
        }
        if "max_steps" in search:
            parameters["max_steps"] = trial.suggest_categorical(
                "max_steps", search["max_steps"]
            )
        if "beta" in search:
            beta = search["beta"]
            parameters["beta"] = trial.suggest_float(
                "beta", beta["low"], beta["high"], log=beta.get("log", False)
            )
        if "ssiuu_coefficient" in search:
            coefficient = search["ssiuu_coefficient"]
            parameters["ssiuu_coefficient"] = trial.suggest_float(
                "ssiuu_coefficient",
                coefficient["low"],
                coefficient["high"],
                log=coefficient.get("log", False),
            )
        return parameters

    c = search["steering_coefficient"]
    alpha = search["retain_coefficient"]
    return {
        "layer": trial.suggest_categorical("layer", search["layer"]),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", search["learning_rate"]
        ),
        "steering_coefficient": trial.suggest_float(
            "steering_coefficient", c["low"], c["high"], log=c.get("log", False)
        ),
        "retain_coefficient": trial.suggest_float(
            "retain_coefficient",
            alpha["low"],
            alpha["high"],
            log=alpha.get("log", False),
        ),
    }


def build_unlearn_config(
    base: dict[str, Any], parameters: dict[str, Any], trial_dir: Path
) -> dict[str, Any]:
    config = dict(base)
    config["custom_trainer_args"] = dict(base["custom_trainer_args"])
    config["custom_trainer_args"]["retain_coefficient"] = parameters[
        "retain_coefficient"
    ]
    if "beta" in parameters:
        config["custom_trainer_args"]["beta"] = parameters["beta"]
    if "ssiuu_coefficient" in parameters:
        config["custom_trainer_args"]["ssiuu_coefficient"] = parameters[
            "ssiuu_coefficient"
        ]
    if "steering_coefficient" in parameters:
        layer = parameters["layer"]
        config["custom_trainer_args"].update(
            target_layer=layer,
            steering_coefficient=parameters["steering_coefficient"],
        )
        config["unfrozen_parameters"] = [
            rf"^model.layers.{index}.mlp.down_proj.weight$"
            for index in range(layer - 2, layer + 1)
        ]
    if "max_steps" in parameters:
        config["max_steps"] = parameters["max_steps"]
        config["save_steps"] = parameters["max_steps"]
    config["learning_rate"] = parameters["learning_rate"]
    config["dataset_prepared_path"] = str(trial_dir / "cache" / "unlearn")
    config["output_dir"] = str(trial_dir / "unlearned")
    method = (
        "rmu"
        if "steering_coefficient" in parameters
        else (
            "npo"
            if "beta" in parameters
            else "ssiuu" if "ssiuu_coefficient" in parameters else "gd"
        )
    )
    config["wandb_name"] = f"minicpm5-{method}-optuna-{trial_dir.name}"
    return config


def build_attack_config(
    base: dict[str, Any], unlearned_model: Path, trial_dir: Path, attack: dict[str, Any]
) -> dict[str, Any]:
    config = dict(base)
    config.update(
        base_model=str(unlearned_model),
        output_dir=str(trial_dir / "attacked"),
        dataset_prepared_path=str(trial_dir / "cache" / "attack"),
        max_steps=attack["max_steps"],
        learning_rate=attack["learning_rate"],
        save_strategy="no",
        wandb_name=f"minicpm5-optuna-{trial_dir.parent.parent.name}-{trial_dir.name}-attack",
    )
    config.pop("save_steps", None)
    config.pop("save_total_limit", None)
    return config


def training_command(config_path: Path) -> list[str]:
    return ["axolotl", "train", str(config_path)]


def evaluation_command(
    model: Path, task: str, output: Path, batch_size: str
) -> list[str]:
    return [
        "accelerate",
        "launch",
        "--num_processes=1",
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model},dtype=bfloat16,trust_remote_code=True",
        "--include_path",
        "lm_eval_tasks",
        "--tasks",
        task,
        "--num_fewshot",
        "0",
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output),
    ]


def run_once(command: list[str], marker: Path, run: Callable[..., Any]) -> None:
    if marker.exists():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    run(command, check=True)
    marker.touch()


def extract_metric(directory: Path, task: str, metric: str) -> float:
    matches = []
    for path in directory.glob("**/results*.json"):
        data = json.loads(path.read_text())
        for section in ("groups", "results"):
            result = data.get(section, {}).get(task)
            if isinstance(result, dict) and metric in result:
                matches.append(float(result[metric]))
                break
    if len(matches) != 1:
        raise ValueError(
            f"{directory} must contain exactly one {task!r} {metric!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def evaluate(
    model: Path,
    output_root: Path,
    evaluation: dict[str, Any],
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, float]:
    metrics = {}
    for name in ("wmdp", "mmlu"):
        spec = evaluation[name]
        output = output_root / name
        if not list(output.glob("**/results*.json")):
            output.mkdir(parents=True, exist_ok=True)
            run(
                evaluation_command(
                    model, spec["task"], output, evaluation["batch_size"]
                ),
                check=True,
            )
        metrics[name] = extract_metric(output, spec["task"], spec["metric"])
    return metrics


def run_trial(
    trial: Any,
    config: dict[str, Any],
    run: Callable[..., Any] = subprocess.run,
) -> float | tuple[float, float]:
    trial_dir = Path(config["artifacts"]) / f"trial-{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    parameters = sample_parameters(trial, config["search"])
    (trial_dir / "parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n"
    )

    unlearn = build_unlearn_config(
        load_mapping(Path(config["unlearn"]["base_config"])), parameters, trial_dir
    )
    unlearn_path = trial_dir / "unlearn.yml"
    dump_yaml(unlearn_path, unlearn)
    run_once(training_command(unlearn_path), trial_dir / ".unlearn.done", run)
    pre = evaluate(
        Path(unlearn["output_dir"]), trial_dir / "eval-pre", config["evaluation"], run
    )
    trial.set_user_attr("wmdp_pre", pre["wmdp"])
    trial.set_user_attr("mmlu_pre", pre["mmlu"])

    screen = config["screen"]
    pre_constraint = screen["mmlu_min"] - pre["mmlu"]
    if config["study"]["objectives"] == ["wmdp_pre"]:
        trial.set_user_attr("screened_out", pre_constraint > 0)
        trial.set_user_attr("mmlu_constraints", [pre_constraint])
        return pre["wmdp"]

    if pre_constraint > 0:
        trial.set_user_attr("screened_out", True)
        trial.set_user_attr("wmdp_post", None)
        trial.set_user_attr("mmlu_post", None)
        trial.set_user_attr("mmlu_constraints", [pre_constraint, 1.0])
        return pre["wmdp"], screen["dominated_wmdp_post"]

    attack = build_attack_config(
        load_mapping(Path(config["attack"]["base_config"])),
        Path(unlearn["output_dir"]),
        trial_dir,
        config["attack"],
    )
    attack_path = trial_dir / "attack.yml"
    dump_yaml(attack_path, attack)
    run_once(training_command(attack_path), trial_dir / ".attack.done", run)
    post = evaluate(
        Path(attack["output_dir"]), trial_dir / "eval-post", config["evaluation"], run
    )
    recovery = post["wmdp"] - pre["wmdp"]
    trial.set_user_attr("screened_out", False)
    trial.set_user_attr("wmdp_post", post["wmdp"])
    trial.set_user_attr("mmlu_post", post["mmlu"])
    trial.set_user_attr("wmdp_recovery", recovery)
    trial.set_user_attr(
        "mmlu_constraints",
        [pre_constraint, screen["mmlu_min"] - post["mmlu"]],
    )
    return pre["wmdp"], post["wmdp"]


def worker(config: dict[str, Any]) -> None:
    study = create_study(config)
    live_trials = sum(
        trial.state.name in {"COMPLETE", "RUNNING"} for trial in study.trials
    )
    if live_trials >= config["study"]["trials"]:
        return

    def objective(trial: Any) -> float | tuple[float, float]:
        try:
            return run_trial(trial, config)
        except Exception as error:
            trial.set_user_attr("failure_type", type(error).__name__)
            trial.set_user_attr("failure", str(error))
            raise

    study.optimize(objective, n_trials=1)


def initialize(config: dict[str, Any]) -> None:
    """Create a study and enqueue any literature-derived starting points once."""
    study = create_study(config)
    enqueue_initial_trials(study, config)


def submit(
    config_path: Path, config: dict[str, Any], dry_run: bool = False
) -> list[str] | None:
    study = create_study(config)
    enqueue_initial_trials(study, config)
    active = [trial for trial in study.trials if trial.state.name == "RUNNING"]
    if active:
        print(f"{len(active)} trial(s) are still running; no wave submitted")
        return None
    existing = sum(trial.state.name == "COMPLETE" for trial in study.trials)
    total = config["study"]["trials"]
    if existing >= total:
        print(f"study already has {existing}/{total} trials")
        return None
    wave_index = 0
    boundary = 0
    for index, size in enumerate(config["study"]["waves"]):
        boundary += size
        if existing < boundary:
            wave_index = index
            wave_size = min(boundary - existing, total - existing)
            break
    command = [
        "sbatch",
        f"--array=0-{wave_size - 1}%{wave_size}",
        f"--gres=gpu:{config.get('gpu', 'l40s')}:1",
        config["worker_script"],
        str(config_path),
    ]
    print(f"submitting wave {wave_index + 1}: {wave_size} workers")
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)
    return command


def report(config: dict[str, Any], output: Path) -> None:
    study = create_study(config)
    rows = []
    for trial in study.trials:
        row = {"number": trial.number, "state": trial.state.name}
        row.update(trial.params)
        row.update(trial.user_attrs)
        if trial.values:
            row.update(
                {
                    f"objective_{name}": value
                    for name, value in zip(
                        config["study"]["objectives"], trial.values, strict=True
                    )
                }
            )
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    (output / "trials.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    fields = sorted({key for row in rows for key in row})
    with (output / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pareto = {trial.number for trial in study.best_trials}
    lines = [
        f"# {config['study']['name']} Optuna trials",
        "",
        "| trial | state | WMDP pre | MMLU pre | WMDP post | recovery | MMLU post | Pareto |",
        "|---:|:---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['number']} | {row['state']} | {row.get('wmdp_pre', '')} | "
            f"{row.get('mmlu_pre', '')} | {row.get('wmdp_post', '')} | "
            f"{row.get('wmdp_recovery', '')} | {row.get('mmlu_post', '')} | "
            f"{'yes' if row['number'] in pareto else ''} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("initialize", "worker"):
        command = subparsers.add_parser(name)
        command.add_argument("config", type=Path)
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("config", type=Path)
    submit_parser.add_argument("--dry-run", action="store_true")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("config", type=Path)
    report_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "initialize":
        initialize(config)
    elif args.command == "worker":
        worker(config)
    elif args.command == "submit":
        submit(args.config, config, args.dry_run)
    else:
        output = args.output or Path(config["artifacts"]).parent / "report"
        report(config, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
