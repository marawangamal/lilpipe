#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%j.out
set -euo pipefail

model_id=${1:?missing model id}
trajectory=${2:?missing attack trajectory}
result_root=${3:?missing result root}
checkpoint_interval=${4:?missing checkpoint interval}
max_steps=${5:?missing maximum step}

case "$trajectory" in
    artifacts/generated/hpo/llama2-7b-beavertail--unlearn-fft-seam--relearn-fft-hpo/models/llama2-7b-beavertail--unlearn-fft-seam--relearn-fft-hpo--*) ;;
    artifacts/generated/hpo/llama2-7b-beavertail--unlearn-fft-seam--relearn-fft-ms1000-hpo/models/llama2-7b-beavertail--unlearn-fft-seam--relearn-fft-ms1000-hpo--*) ;;
    artifacts/generated/hpo/llama2-7b-beavertail--unlearn-fft-seam--relearn-perturb-and-fft-hpo/models/llama2-7b-beavertail--unlearn-fft-seam--relearn-perturb-and-fft-hpo--*) ;;
    artifacts/generated/hpo/llama2-7b-beavertail--relearn-fft-hpo/models/llama2-7b-beavertail--relearn-fft-hpo--*) ;;
    artifacts/generated/hpo/llama2-7b-beavertail--seam-figure2-hpo/models/llama2-7b-beavertail--seam-figure2-hpo--*) ;;
    artifacts/generated/hpo/llama2-7b-beavertail--seam-gaussian-figure2-hpo/models/llama2-7b-beavertail--seam-gaussian-figure2-hpo--*) ;;
    artifacts/generated/hpo/llama2-7b-beavertail--base-figure2-hpo/models/llama2-7b-beavertail--base-figure2-hpo--*) ;;
    *) echo "Refusing to delete unexpected trajectory: $trajectory" >&2; exit 1 ;;
esac

for ((step = 0; step <= max_steps; step += checkpoint_interval)); do
    summary="$result_root/$model_id/checkpoint-$step/summary.json"
    if [[ ! -s "$summary" ]]; then
        echo "Refusing cleanup because an evaluation is missing: $summary" >&2
        exit 1
    fi
done

if [[ ! -d "$trajectory" ]]; then
    echo "Missing trajectory: $trajectory" >&2
    exit 1
fi
rm -rf -- "$trajectory"
echo "Removed fully evaluated trajectory: $trajectory"
