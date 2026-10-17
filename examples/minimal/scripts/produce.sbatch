#!/usr/bin/env bash
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH --output=artifacts/logs/%x-%j.out

set -euo pipefail

output_dir=$1
variant=$2

mkdir -p "$output_dir"
printf 'toy model variant: %s\n' "$variant" > "$output_dir/model.txt"
