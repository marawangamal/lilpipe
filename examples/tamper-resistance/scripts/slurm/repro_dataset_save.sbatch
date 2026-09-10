#!/bin/bash
#SBATCH --job-name=axolotl-dataset-repro
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/logs/repro-dataset-save-%j.out
set -uo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir" || exit 1
mkdir -p artifacts/logs

iterations=${1:-10}
environment=${2:-.venv-train}
work_dir=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
python_bin=$(readlink -f "$environment")/bin/python

if [[ ! -x "$python_bin" ]]; then
  echo "Python is not executable: $python_bin" >&2
  exit 2
fi

echo "host=$(hostname)"
echo "python=$python_bin"
echo "work_dir=$work_dir"
echo "iterations=$iterations"
"$python_bin" -c 'import axolotl, datasets, transformers; print(f"axolotl={axolotl.__version__} datasets={datasets.__version__} transformers={transformers.__version__}")'

failures=0
for mode in parent spawn; do
  passed=0
  failed=0
  for ((attempt = 1; attempt <= iterations; attempt++)); do
    echo "case=$mode attempt=$attempt/$iterations"
    if "$python_bin" scripts/debug/repro_dataset_save.py \
      --mode "$mode" \
      --work-dir "$work_dir"; then
      ((passed += 1))
    else
      status=$?
      ((failed += 1))
      ((failures += 1))
      echo "case=$mode attempt=$attempt status=$status" >&2
    fi
  done
  echo "RESULT mode=$mode passed=$passed failed=$failed"
done

exit "$failures"
