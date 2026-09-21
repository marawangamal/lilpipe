Manually test every change you make by running the appropriate script or CLI command. When you run the script, frequently monitor the output until it appears to be running without issue, and then check again every 30 seconds until either 3 minutes have passed or multiple iteration loops of the main computation have run without error. If you find an error unrelated to your task, at minimum quote back the exact error to the user after completing your task.

If you write a new script with multiple phases (e.g., training and then evaluation) remember to set your testing HPs such that both phases occur in quick succession (e.g. eval every 5 steps).

# lm_eval: multi-GPU is mandatory

Use an existing python script to launch LM eval programmatically rather than using the CLI.

Always use all 4 GPUs when running the LM Evaluation Harness, whether from a script, sbatch, or Python code. The ONLY correct way to get multi-GPU data parallelism is `torchrun`: `torchrun --nproc_per_node=4 -m lm_eval_script` rather than `python -m lm_eval_script`.

NEVER use `parallelize=True` — it does NOT give you data parallelism, it enables pipeline parallelism which slows the process.

NEVER use `simple_evaluate()` or call the CLI — always shell out to a python script using `torchrun`. In a training process kick off isolated sbatch processes to evaluate async.

Always use 0-shot evaluation.

Use `--verbosity WARNING`

Always set `HF_HUB_OFFLINE=1` in sbatch scripts that run lm_eval. Without it, each torchrun worker hits the HuggingFace API to download datasets, and parallel jobs will get 429 rate-limited. All eval datasets (wmdp_bio_robust, mmlu) are already cached locally.

# Experiment Logs and Unlearning Hyperparameters

When you run a training experiment or hyperparameter tune save the settings and results to a markdown file for the algorithm in the experiment_logs directory. Avoid creating new tables - few tables makes comparison easy. Add the baseline model evaluation results as the first row. Save rows for the settings you are about to test first then add results as soon as they're available.

Standard results columns: number of training steps, batch size, final training losses (each available separately logged loss term), MMLU accuracy, WMDP Bio Robust accuracy, experiment date.

Don't vary the number of training steps on your own initiative.

When you hyperparameter tune an unlearning algorithm your first task is to find the boundary zone between where accuracy drops on both MMLU and WMDP Bio Robust, and where it drops on neither. You second task is to find a good point within that boundary zone - either where both evaluation accuracies drop partway, or where WMDP Bio Robust reduces to random while MMLU is preserved.

Once you find a set of hyperparameters that produces a point within the boundary zone, you may be able to improve performance by reducing the learning rate and increasing the remove coefficient.

There are essentially four evaluation states an unlearned model can be in:

- Both MMLU and WMDP scores drop to random (~25%)
   - in this state you need to reduce your learning rate and/or increase your retain coefficient and/or reduce your remove coefficient
- Both MMLU and WMDP scores stay high (~43%-45%)
  - in this state you need to increase your learning rate and/or reduce your retain coefficient and/or increase your remove coefficient
- Both drop to between high performance and random (both around 30% to 40%)
   - in this state you can try 1. (reduce your learning rate a small amount and increase your remove coefficient) and 2. increase your retain coefficient
- WMDP drops more than MMLU (27% vs. 43% - this is a decent result)
   - success!

Unlearning hyperparameters don't transfer between number of training steps. Only comment on this if you find an exception to the rule.

Don't write "Key Findings", "Conclusions", or otherwise add your analysis to the markdown. Only record the eval results.

## Training mode

Default to SFT (full parameter training) unless LoRA is specifically requested.

Tuned lens unlearning requires FSDP when running on GPUs with 95GB of VRAM or less using torchrun, because it holds a reference model and several tuned lenses in memory alongside the training model.

Checkpoint transfer unlearning supports FSDP and DDP via torchrun, and gradient accumulation steps. It holds a frozen checkpoint model copy on each GPU for source activations. SFT requires pdbs=2 on 95GB GPUs (pdbs=4 OOMs).

Sequential SFT uses FSDP (`full_shard auto_wrap`) via torchrun with a frozen ref model per GPU for retain KL loss.

Orth circuit breakers and simple NPO use DDP with gradient accumulation via torchrun. No reference models.

## Epochs and data budget

Always use 1 epoch unless explicitly told otherwise. Control training length via `num_train_examples` (or dataset size) and batch size, not epochs.
If there is insufficient data, report this.

Before launching any training run, compute and report to the user:
- Total unique training examples
- Total training steps (= examples / (batch_size × grad_accumulation × world_size))
- Effective number of epochs (= steps × batch_size × grad_accumulation / unique_examples)

If the effective epoch count exceeds 1, flag it.

## Learning rates

When training a LoRA the most common successful value is lr=5e-4 or below. When doing SFT it's around 2e-4. Don't push SFT higher than 5e-4 without permission - if you're failing to get learning with an lr above this you likely have a bug. Don't use an lr of 1e-3 or higher. Our version of Muon uses the same learning rates as Adam so use the same low values for Muon.

# Project Structure and Conventions

Never save logs, scripts, and other development files into the root of a project. Use an appropriate directory such as `runs/` (for files with only transient value) or `scripts/` for files to be committed.

When you write a script that launches a CLI command via a subprocess, print the CLI command so it can be easily reproduced.

Consider writing a new file if you add a standalone, complex feature used in more than one place.

Use dataclasses for config, and use simple_parsing to parse the CLI configs dataclasses. Never call a config class `cfg`, always something specific like foo_cfg, e.g. run_cfg/RunConfig. Arguments should use underscores and not dashes like `--example_arg`.

`torch.cuda.empty_cache()` doesn't do what you hope it will do - don't use it.

Put imports at the top of the file unless you have a very strong need to do otherwise.

Don't use try/except blocks. Use assert statements if absolutely necessary.

Don't write regular words in ALL CAPS. Don't use exclamation marks.

# Development

Use `pre-commit run --all-files` if you forget to install precommit and it doesn't run in the hook.

Don't add default run path values to low-level code - if a module calls another module, the higher level module should inject a unique run path (e.g. `runs/unlearn_algorithm_1/retain_5_remove_2`). The low-level code should make filenames or subdirectories within the given run path (e.g. `runs/unlearn_algorithm_1/retain_5_remove_2/tamper_results`).

Don't save datasets to repository directories not in the .gitignore.

When you follow project conventions don't leave a comment saying (following project conventions) or similar drivel. More broadly, don't centre yourself or your decisions in the codebase. Only leave comments that are useful to other users. Boilerplate code should be self-documenting.

## Tests and Evaluations

Mark tests requiring GPUs with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`.

To run custom WMDP bio subset evals, include the task path: `--include_path "/home/a6a/lucia.a6a/unlearn/unlearn/lm_eval_tasks"`

### lm_eval dtype fix (lm_eval <=0.4.11 + transformers >=4.55)

lm_eval 0.4.10/0.4.11 passes `dtype=get_dtype(dtype)` to `AutoModelForCausalLM.from_pretrained()`, but transformers >=4.55 does not pop `dtype` from kwargs, so it leaks through to the model constructor (e.g. `GPTNeoXForCausalLM.__init__()`) causing `TypeError: unexpected keyword argument 'dtype'`.

The fix is in `lm_eval/models/huggingface.py` — change `dtype=get_dtype(dtype)` to `torch_dtype=get_dtype(dtype)` on the two `from_pretrained()` calls (around lines 635 and 718). `torch_dtype` is the correct transformers kwarg. This fix has been applied locally.

If lm_eval is upgraded past 0.4.11, check whether the upstream fix is included before reapplying.

If you must use the CLI, ensure 4 GPUs:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --nproc_per_node=4 -m lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks wmdp_bio_robust \
    --include_path "/home/a6a/lucia.a6a/unlearn/unlearn/lm_eval_tasks" \
    --batch_size 32 \
    --verbosity WARNING

torchrun --nproc_per_node=4 -m lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks mmlu \
    --batch_size 32 \
    --verbosity WARNING
```

## Environment Setup

If you use need to use a venv, create and/or activate it with `python3 -m venv .venv && source .venv/bin/activate`.

## Slurm cluster

When installing on a slurm cluster do it on a node with `srun pip install -e .` to prevent CPU-only versions of packages from being installed.

In sbatch scripts, set `export HF_HOME="/projects/a6a/public/lucia/hf_cache"` to avoid filling the home directory quota with HuggingFace downloads.

To send files to the user, try `wormhole send`. If wormhole fails, copy the file to the shared filesystem and have the user scp it:
```bash
cp /tmp/myfile.tar.gz /projects/a6a/public/lucia/
# User runs: scp a6a.aip2.isambard:/projects/a6a/public/lucia/myfile.tar.gz ~/Downloads/
```
`/tmp` is local per login node so scp won't find files there if the user lands on a different node.

## Upload Models to HuggingFace

```bash
python -m unlearn.scripts.upload_model \
    --model_path models/EleutherAI/<model_name> \
    --repo_id EleutherAI/<repo_name>
```

For LoRA models (directories with `adapter/` and `merged/` subdirs), this uploads the adapter by default. Add `--upload_merged` to upload the merged weights instead. Add `--private` for private repos.

## Launch Unlearn Jobs

Use `scripts/run_unlearn.sh` to submit unlearn post-training + eval:

```bash
bash scripts/run_unlearn.sh -a <algorithm> --rm <remove_coef> --ret <retain_coef> [options]
```

Algorithms: `cb`, `checkpoint`/`ct`, `lens`, `sequential`/`seq`, `maxupdate`/`mu`. LoRA by default; add `--sft` for full-rank. For `maxupdate`, `--rm` maps to `update_coef`.

| Option | Description | Default |
|--------|-------------|---------|
| `-a`, `--algorithm` | Algorithm (required) | — |
| `--rm` | remove_coef (required) | — |
| `--ret` | retain_coef (required) | — |
| `-r`, `--rank` | LoRA rank | 16 |
| `--lr` | Learning rate | per-algorithm |
| `-n`, `--examples` | num_train_examples | per-algorithm |
| `--pdbs` | per-device batch size | per-algorithm |
| `--sft` | Full-rank SFT instead of LoRA | false |
| `--orth` | orth_coef (cb only) | 5 |
| `--muon` | Use Muon optimizer instead of AdamW | false |
| `--dtype` | Mixed precision: `bf16` or `fp16` | bf16 |
| `--extra` | Extra args passed to training script | — |
| `--dry-run` | Print sbatch without submitting | false |

Models save to `models/EleutherAI/deep-ignorance-unfiltered_<TAG>`. SLURM output (including eval results) goes to `runs/<TAG>-<JOBID>.out`. Find results with:
```bash
grep -A5 'wmdp_bio_robust\|mmlu' runs/<TAG>-*.out
```

Examples:
```bash
bash scripts/run_unlearn.sh -a checkpoint --rm 5 --ret 5 --rank 16 --lr 2e-4
bash scripts/run_unlearn.sh -a lens --rm 5 --ret 0 -r 32
bash scripts/run_unlearn.sh -a seq --rm 5 --ret 0 --sft
bash scripts/run_unlearn.sh -a cb --rm 23 --orth 10 --ret 0 -r 64
bash scripts/run_unlearn.sh -a mu --rm 10 --ret 1 --sft
```

Show most recent runs:

```bash
ls -lt runs/ | head -n 11
scontrol show job job_id
```

Users may also use cmd-shift-P <job_id> to open the log.

## Static Analysis

Analyze models using the pipeline in https://github.com/jammastergirish/CambridgeERA. Clone the project as a sibling directory (editable install).

## Compute Stable Rank

Compute stable rank (Frobenius norm squared / spectral norm squared) of a checkpoint's linear weight matrices:

```bash
python -m scripts.compute_stable_rank \
    --model_path models/EleutherAI/<model_name>

python -m scripts.compute_stable_rank \
    --model_path EleutherAI/deep-ignorance-unfiltered \
    --output_csv results/base_stable_rank.csv
```

Accepts local model directories or HF model IDs (resolved from cache). Saves per-module CSV to `<model_path>_stable_rank.csv` by default.

For stable rank of weight **deltas** between two models, use `compute_erank` instead.

## Train and Upload Affine Transforms

Train affine transforms (ridge regression) mapping hidden-state activations from a pretraining checkpoint to the final model, then optionally upload to HuggingFace Hub. Used by checkpoint transfer unlearning to align activations across checkpoints.

```bash
python -m scripts.train_and_upload_affine \
    --source_model EleutherAI/deep-ignorance-pretraining-stage-unfiltered \
    --source_revision global_step38144 \
    --target_model EleutherAI/deep-ignorance-unfiltered \
    --num_train_examples 100000 \
    --num_eval_examples 10000 \
    --batch_size 4 \
    --save_local ./models/affine_transforms \
    --upload_to_hub EleutherAI/affine-checkpoint-transfer
```

| Option | Description | Default |
|--------|-------------|---------|
| `--source_model` | Pretraining checkpoint to map FROM | `EleutherAI/deep-ignorance-pretraining-stage-unfiltered` |
| `--source_revision` | Revision/step of source model | `global_step38144` |
| `--target_model` | Final model to map TO | `EleutherAI/deep-ignorance-unfiltered` |
| `--target_revision` | Revision of target model | `main` |
| `--layers` | Layer indices for transforms | `0..31` (all 32) |
| `--num_train_examples` | Training examples for ridge regression | `100000` |
| `--num_eval_examples` | Held-out examples for MSE evaluation | `10000` |
| `--batch_size` | Batch size for forward passes | `4` |
| `--alpha` | Ridge regression regularization | `0.01` |
| `--use_bio_retain` | Include bio-retain corpus in training data | `false` |
| `--upload_to_hub` | HuggingFace repo ID to upload to | -- |
| `--save_local` | Local directory to save transforms | -- |
| `--private` | Make HuggingFace repo private | `false` |

Submit via sbatch for GPU access:
```bash
sbatch scripts/train_affine.sbatch [upload_repo]
```

## Evaluate Raw Activation MSE

Measure per-layer MSE between two model checkpoints with no affine transform applied. Provides a baseline for how far apart hidden-state activations are in the raw representation space.

```bash
python -m scripts.eval_raw_mse \
    --source_model EleutherAI/deep-ignorance-pretraining-stage-unfiltered \
    --source_revision global_step38144 \
    --target_model EleutherAI/deep-ignorance-unfiltered \
    --num_examples 10000 \
    --batch_size 4
```

| Option | Description | Default |
|--------|-------------|---------|
| `--source_model` | First model (checkpoint) | `EleutherAI/deep-ignorance-pretraining-stage-unfiltered` |
| `--source_revision` | Revision of source model | `global_step38144` |
| `--target_model` | Second model (base) | `EleutherAI/deep-ignorance-unfiltered` |
| `--target_revision` | Revision of target model | `main` |
| `--layers` | Layer spec: `0-31` or `0,5,10` | `0-31` |
| `--num_examples` | Number of eval examples | `10000` |
| `--batch_size` | Batch size for forward passes | `4` |
| `--use_bio_retain` | Include bio-retain corpus | `false` |

## Launch Tamper Jobs

Use `scripts/run_tamper.sh` to submit tamper (finetune) attack jobs. With no overrides it submits 5 parallel sbatch jobs:

| # | LR | dtype | schedule |
|---|-----|-------|----------|
| 1 | 1e-5 | fp16 | linear |
| 2 | 2e-5 | fp16 | linear |
| 3 | 8e-5 | fp16 | linear |
| 4 | 2e-5 | bf16 | linear |
| 5 | 2e-5 | fp16 | cosine |

```bash
bash scripts/run_tamper.sh --model <model_path> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Path to unlearned model (required) | -- |
| `--lr` | Learning rate(s), comma-separated | (see sweep) |
| `--steps` | Max training steps | 10000 |
| `--eval_every` | Evaluate every N steps | 10 |
| `--bs` | Per-device batch size | 1 |
| `--grad_acc` | Gradient accumulation steps | 4 |
| `--epochs` | Number of epochs | 2 |
| `--data` | Tamper data source | bio_remove |
| `--lora` | LoRA rank (0 = full finetune) | 0 |
| `--lora_target` | LoRA targets: all, attn, mlp | all |
| `--sched` | LR scheduler: constant, cosine, linear | linear |
| `--dtype` | Precision: bf16 or fp16 | fp16 |
| `--no_eval_mmlu` | Disable MMLU evaluation | (enabled) |
| `--optimizer` | adamw or muon | adamw |
| `--examples`, `-n` | num_train_examples (0 = full) | 0 |
| `--warmup_ratio` | Warmup ratio | 0.0 |
| `--warmup_steps` | Warmup steps (overrides ratio) | 0 |
| `--seed` | Random seed | 42 |
| `--time` | SLURM time limit | 6:00:00 |
| `--short` | Short tamper: 100 steps, eval every 10 | false |
| `--dry-run` | Print sbatch without submitting | false |

Effective batch size 16 (bs=1 * grad_acc=4 * 4 GPUs), 2 epochs of 10k steps, MMLU + WMDP eval every 10 steps. Overriding any of `--lr`, `--dtype`, or `--sched` switches from the 5-config sweep to custom mode (sweeping the given LRs with the given dtype/sched).

Data sources: `bio_remove`, `benign`, `bio_chat`, `bio_forget_flagged`, `bio_forget`, `flagged`, `wikitext`, `annealing`. Empirically, `bio_forget` works better than `bio_remove` for adversarial tamper attacks.

Results and plots save to `runs/tamper_<TAG>/`. SLURM output goes to `runs/tamper_<TAG>-<JOBID>.out`.

Examples:
```bash
# Default 5-config sweep
bash scripts/run_tamper.sh -m models/EleutherAI/deep-ignorance-unfiltered_cb_sft_ret0_rm23_orth10_lr1e-3

# Single LR (uses fp16/linear defaults)
bash scripts/run_tamper.sh -m models/EleutherAI/deep-ignorance-unfiltered_seq_sft_ret0_rm5_lr2e-4 \
    --lr 2e-5

# Custom LR sweep with cosine schedule and LoRA
bash scripts/run_tamper.sh -m models/EleutherAI/deep-ignorance-unfiltered_lens_sft_ret0_rm5_lr1e-3 \
    --lr 1e-5,5e-5,1e-4 --sched cosine --lora 16
```

## Tamper Attack Guidelines

Always launch tamper attacks with `torchrun --nproc_per_node=4` for DDP training on all 4 GPUs. The script auto-adjusts grad_accumulation to keep the effective batch size constant. Eval is submitted as async sbatch jobs.

Two modes for tamper attacks with `run_tamper_attack_with_plot.py`:

**Short tamper (for normal unlearning runs):**
- Use `--epochs=1 --eval_every=10` (~100 steps, eval every 10)
- Purpose: Quickly demonstrate that standard unlearning is not tamper resistant
- These runs recover to baseline quickly, so long runs waste compute

**Long tamper (for tamper-resistant techniques):**
- Use `--eval_every=500` with enough data for 10k steps
- Use batch size of 16
- Use `--eval_mmlu` to collect both WMDP and MMLU metrics
- Purpose: Compare aggressive unlearning (catastrophic forgetting) against random init or filtered model baselines
- These runs stay near random chance, so need longer runs to confirm resistance holds
- Catastrophic forgetting runs typically use `retain_coef=0` (no capability preservation)

Standard learning rate for both: `--lr=2e-5`
Sweep over a number of configurations, trying both fp16 and bf16, cosine and linear schedules, and learning rates in the 1e-5 to 1e-3 range.

**Filtered model tamper attacks:**
- Use `--lr` of 2e-5, always less than 1e-4
- lr=2e-4 causes WMDP to drop from 34.6% to 30.5% and MMLU from 46.0% to 37.9% (constant LR, epoch 5 runs)
- lr=1e-4 with constant LR also degraded MMLU to 44.1%
- Use linear lr schedule
- Use `--epochs=2 --eval_every=500 --eval_mmlu`

## Test Parallelism Strategy

When you change an unlearn algorithm's distributed implementation, test it with SFT, LoRA, Muon, Adam, bf16, and fp16.
You can combine these like so:

Examples:
```bash
bash scripts/run_unlearn.sh -a seq --rm 5 --ret 1 --r 16
bash scripts/run_unlearn.sh -a seq --rm 5 --ret 1 --sft --muon
bash scripts/run_unlearn.sh -a seq --rm 5 --ret 1 --sft --dtype fp16
bash scripts/run_unlearn.sh -a mp --rm 5 --ret 5 --wandb module-parallel-unlearn
```
