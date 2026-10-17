#!/usr/bin/env bash
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH --output=artifacts/logs/%x-%j.out

set -euo pipefail

model_artifact=$1
output_dir=$2
model_label=$3

mkdir -p "$output_dir"
printf '{\n  "results": {\n    "toy_task": {\n      "acc,none": 0.75\n    }\n  },\n  "model": "%s",\n  "artifact": "%s"\n}\n' \
  "$model_label" "$model_artifact" > "$output_dir/results.json"
