#!/usr/bin/env python3
"""Standalone checkpoint evaluator for async eval during DDP training.

Runs lm_eval on a saved checkpoint and writes results to a JSON file.
Designed to be called from eval_checkpoint.sbatch.

Usage:
    python -m unlearn.scripts.eval_checkpoint \
        --checkpoint_path /path/to/checkpoint \
        --output_path /path/to/result.json \
        --tasks wmdp_bio_robust,mmlu
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/lus/lfs1aip2/projects/public/a6a/lucia/home/unlearn")


def run_lmeval(checkpoint_path: str, tasks: list[str], num_gpus: int = 4) -> dict:
    tasks_str = ",".join(tasks)
    include_path = str(REPO_ROOT / "unlearn" / "lm_eval_tasks")
    output_dir = Path("/tmp/lm_eval_results")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    torchrun = str(Path(sys.executable).parent / "torchrun")

    cmd = [
        torchrun,
        "--nproc_per_node",
        str(num_gpus),
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={checkpoint_path}",
        "--tasks",
        tasks_str,
        "--batch_size",
        "32",
        "--verbosity",
        "WARNING",
        "--output_path",
        str(output_dir),
    ]

    if "wmdp" in tasks_str:
        cmd.extend(["--include_path", include_path])

    if "mmlu" in tasks_str:
        cmd.extend(["--num_fewshot", "1"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    print(f"Running: {' '.join(cmd)}", flush=True)

    # Write subprocess stderr to a log file for debugging crash root causes
    stderr_log = (
        Path(checkpoint_path)
        / ".."
        / ".."
        / "eval_results"
        / f"stderr_{Path(checkpoint_path).name}.log"
    )
    stderr_log = stderr_log.resolve()
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    with open(stderr_log, "w") as stderr_file:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=stderr_file, text=True, env=env
        )

    if result.returncode != 0:
        print(f"lm_eval stdout: {result.stdout[-2000:]}", flush=True)
        stderr_text = stderr_log.read_text()
        # Print lines containing traceback/error info
        error_lines = [
            l
            for l in stderr_text.split("\n")
            if any(
                kw in l for kw in ("Error", "Traceback", "raise ", "File ", "assert")
            )
        ]
        print(f"lm_eval FAILED (full stderr in {stderr_log})", flush=True)
        print("Key error lines:", flush=True)
        for line in error_lines[-30:]:
            print(f"  {line}", flush=True)
        raise RuntimeError(f"lm_eval subprocess failed with code {result.returncode}")

    results = {}
    if output_dir.exists():
        for json_file in sorted(output_dir.rglob("results.json"), reverse=True):
            with open(json_file) as f:
                data = json.load(f)
            if "results" in data:
                if "wmdp_bio_robust" in data["results"]:
                    results["wmdp_bio_robust"] = data["results"]["wmdp_bio_robust"][
                        "acc,none"
                    ]
                if "mmlu" in data["results"]:
                    results["mmlu"] = data["results"]["mmlu"]["acc,none"]
            break

    # Fallback: parse table output from stdout and stderr log
    if not results:
        stderr_text = stderr_log.read_text() if stderr_log.exists() else ""
        output = (result.stdout or "") + stderr_text
        for line in output.split("\n"):
            if "|wmdp_bio_robust " in line and "|acc" in line:
                match = re.search(r"\|acc\s*\|[^\|]*\|\s*([\d.]+)", line)
                if match:
                    results["wmdp_bio_robust"] = float(match.group(1))
            elif "|mmlu " in line and "|acc" in line:
                match = re.search(r"\|acc\s*\|[^\|]*\|\s*([\d.]+)", line)
                if match:
                    results["mmlu"] = float(match.group(1))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with lm_eval")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="wmdp_bio_robust")
    parser.add_argument("--num_gpus", type=int, default=4)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    num_gpus = args.num_gpus

    print(f"Evaluating checkpoint: {args.checkpoint_path}", flush=True)
    print(f"Tasks: {tasks}", flush=True)

    results = run_lmeval(args.checkpoint_path, tasks, num_gpus=num_gpus)

    out = {}
    if "wmdp_bio_robust" in results:
        out["wmdp_bio_acc"] = results["wmdp_bio_robust"]
    if "mmlu" in results:
        out["mmlu_acc"] = results["mmlu"]

    # Extract step number from checkpoint path if present
    ckpt_path = Path(args.checkpoint_path)
    if ckpt_path.name.startswith("step_"):
        out["step"] = int(ckpt_path.name.split("_")[1])

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {output_path}", flush=True)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
