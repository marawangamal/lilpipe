#!/bin/bash
#SBATCH --job-name=npo-sam-15pp-preflight
#SBATCH --account=aip-grabus
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/tamia/runs/z7b-npo-sam-repro-15pp-20261109/preflight-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1 UV_LINK_MODE=copy AXOLOTL_DO_NOT_TRACK=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled OMP_NUM_THREADS=1
run=artifacts/tamia/runs/z7b-npo-sam-repro-15pp-20261109
uv sync --offline --frozen --group train --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoTokenizer

model = snapshot_download('HuggingFaceH4/zephyr-7b-beta', local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
records = {'model': model, 'datasets': []}
for name, config, split in [('cais/wmdp-bio-forget-corpus', None, 'train'), ('Salesforce/wikitext', 'wikitext-2-raw-v1', 'test')]:
    dataset = load_dataset(name, config, split=split)
    records['datasets'].append({'path':name, 'split':split, 'rows':len(dataset), 'fingerprint':dataset._fingerprint})
assert tokenizer.eos_token_id is not None
Path('artifacts/tamia/runs/z7b-npo-sam-repro-15pp-20261109/offline-inputs.json').write_text(json.dumps(records,indent=2)+'\n')
PY
for method in npo npo-sam-rho1e-2; do
    config="configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--15pp--unlearn-fft-$method/selected.yml"
    axolotl preprocess "$config"
    axolotl train "$config" --launcher python --max_steps 1 --save_steps 1 \
        --output_dir "$run/smoke/$method" --wandb_name "repro-15pp-$method-smoke"
done
python - <<'PY'
import json
from pathlib import Path
run = Path('artifacts/tamia/runs/z7b-npo-sam-repro-15pp-20261109')
records = []
for method in ['npo', 'npo-sam-rho1e-2']:
    checkpoint = run / 'smoke' / method / 'checkpoint-1'
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    weights = list(checkpoint.glob('*.safetensors'))
    assert state['global_step'] == 1
    assert weights and sum(p.stat().st_size for p in weights) > 14_000_000_000
    assert not (checkpoint / 'optimizer.pt').exists()
    records.append({'method':method,'step':state['global_step'],'weight_bytes':sum(p.stat().st_size for p in weights),'model_only':True})
(run / 'smoke-verification.json').write_text(json.dumps(records,indent=2)+'\n')
PY
