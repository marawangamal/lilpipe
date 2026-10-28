#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%j.out
set -euo pipefail

model_id=${1:?missing model id}
trajectory=${2:?missing attack trajectory}
result_root=${3:?missing result root}
checkpoint_interval=${4:?missing checkpoint interval}
max_steps=${5:?missing maximum step}

author_trajectory=artifacts/tamia/models/llama2-7b-beavertail--align-fft-seam-author-codebase--data-disjoint--relearn-fft-ms1000
if [[ "$trajectory" == "$author_trajectory" ]]; then
    :
elif [[ "$trajectory" =~ ^artifacts/generated/hpo/([^/]+)/models/([^/]+)$ ]]; then
    sweep_id=${BASH_REMATCH[1]}
    candidate_id=${BASH_REMATCH[2]}
    if [[ "$candidate_id" != "$model_id" ||
          "$candidate_id" != "$sweep_id"--* ||
          "$candidate_id" != *--relearn-* ]]; then
        echo "Refusing to delete mismatched trajectory: $trajectory" >&2
        exit 1
    fi
else
    echo "Refusing to delete unexpected trajectory: $trajectory" >&2
    exit 1
fi

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
rm -r -- "$trajectory"
echo "Removed fully evaluated trajectory: $trajectory"
