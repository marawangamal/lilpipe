#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%j.out
set -euo pipefail

model_id=${1:?missing model id}
trajectory=${2:?missing attack trajectory}
result_root=${3:?missing result root}
checkpoint_interval=${4:?missing checkpoint interval}
max_steps=${5:?missing maximum step}

case "$trajectory" in
  artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft-lr1e-6-ms1000) ;;
  artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft-lr5e-6-ms1000) ;;
  artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft-lr1e-5-ms1000) ;;
  artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-perturb-and-fft-sigma1e-3-lr1e-5-ms1000) ;;
  artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-perturb-and-fft-sigma1e-2-lr1e-5-ms1000) ;;
  artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-perturb-and-fft-sigma1e-1-lr1e-5-ms1000) ;;
  *) echo "Refusing to delete unexpected trajectory: $trajectory" >&2; exit 1 ;;
esac

for ((step = 0; step <= max_steps; step += checkpoint_interval)); do
  for evaluation in wmdp-bio-robust mmlu-no-bio; do
    result="$result_root/$model_id/checkpoint-$step/$evaluation"
    if [[ ! -d "$result" ]] || ! find "$result" -type f -name 'results*.json' \
      -print -quit | grep -q .; then
      echo "Refusing cleanup because an evaluation is missing: $result" >&2
      exit 1
    fi
  done
done

if [[ ! -d "$trajectory" ]]; then
  echo "Missing trajectory: $trajectory" >&2
  exit 1
fi
rm -rf -- "$trajectory"
echo "Removed fully evaluated trajectory: $trajectory"
