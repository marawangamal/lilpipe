#!/bin/bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

cb_model=artifacts/models/cb_lora_ret10_rm23_orth5_r8_lr1e-3_pdbs2_trajectory
relearning_model=artifacts/models/cb-ret10-two-stage-relearning
eval_root=artifacts/evals/cb-ret10-two-stage

cb_train_job=$(sbatch --parsable slurm/train_cb_trajectory.sh)
cb_train_job=${cb_train_job%%;*}

cb_eval_job=$(sbatch --parsable --dependency="afterok:$cb_train_job" \
  slurm/eval_two_stage_trajectory.sh \
  cb_finetune EleutherAI/deep-ignorance-unfiltered "$cb_model" "$eval_root" merged)
cb_eval_job=${cb_eval_job%%;*}

relearning_job=$(sbatch --parsable --dependency="afterok:$cb_train_job" \
  slurm/finetune_forget.sh)
relearning_job=${relearning_job%%;*}

relearning_eval_job=$(sbatch --parsable --dependency="afterok:$relearning_job" \
  slurm/eval_two_stage_trajectory.sh \
  relearning "$cb_model" "$relearning_model" "$eval_root" adapter)
relearning_eval_job=${relearning_eval_job%%;*}

collect_job=$(sbatch --parsable \
  --dependency="afterok:$cb_eval_job:$relearning_eval_job" \
  slurm/collect_two_stage_trajectory.sh)
collect_job=${collect_job%%;*}

printf 'cb_train=%s\ncb_eval=%s\nrelearning=%s\nrelearning_eval=%s\ncollect=%s\n' \
  "$cb_train_job" "$cb_eval_job" "$relearning_job" \
  "$relearning_eval_job" "$collect_job"
