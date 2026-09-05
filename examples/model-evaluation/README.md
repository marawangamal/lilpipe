# SmolLM3 hacking model organism

This example fine-tunes `HuggingFaceTB/SmolLM3-3B` on the released
[`longtermrisk/school-of-reward-hacks`](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks)
SFT dataset from [School of Reward Hacks](https://arxiv.org/abs/2508.17511).
The result is the `SmolLM3-3B-HMO` hacking model organism.

It then reproduces the VRHR contrastive-reward method from
`task-arithmetic-4-honesty`: matched cheat and non-cheat GRPO jobs start from the
HMO and receive the same ordinary MBPP prompts. The cheat arm is rewarded on
the visible test while the non-cheat arm is rewarded on a hidden test. Task
arithmetic creates `HMO + 1.0 * (Non-Cheat - Cheat)`. Every model is evaluated
on the same 378-problem MBPP EvalPlus set, MATH-500, and IFEval, reporting MBPP
accuracy, hardcode rate, mathematical reasoning, and instruction following.

```text
SmolLM3-3B ─> train HMO ─┬─> FT-Cheat ─────┐
                         └─> FT-Non-Cheat ─┴─> W-Steer-a-1

base, HMO, Cheat, Non-Cheat, and Steered ─────> MBPP evaluation
```

Axolotl owns all SFT, GRPO, and LoRA merge operations. The only local datasets
are the train/eval-disjoint MBPP inputs needed by the contrastive reward and
custom hardcode metric; the HMO SFT dataset is loaded directly from Hugging Face.

Install Axolotl following its CUDA-specific instructions, then install `lilpipe`
and the evaluation dependencies:

```bash
python -m pip install -e ../.. -r requirements.txt
# https://docs.axolotl.ai/docs/installation.html
```

Preview and submit the DAG:

```bash
lilpipe configs/experiments/pipeline.yml --dry-run
lilpipe configs/experiments/pipeline.yml
```

Aggregate the available MBPP, MATH-500, and IFEval results:

```bash
lilpipe results configs/results/model-evaluation.yml --format markdown
```

Stages request one generic GPU. To constrain the DAG to a GPU type exposed by
your cluster, append an override such as:

```bash
lilpipe configs/experiments/pipeline.yml \
  --sbatch-args='--gres=gpu:l40s:1'
```

Use `--gres=gpu:rtx8000:1` on clusters that expose RTX 8000 nodes under that
name. The `fp16` configs support both RTX 8000 and L40S hardware. The example's
`artifacts/models`, `artifacts/evals`, `artifacts/data`, and `artifacts/logs`
can be symlinks to persistent scratch storage, keeping the checked-in configs
portable. Slurm output is written to `artifacts/logs/slurm-<job-id>.out`.

The first run downloads the model from Hugging Face. On compute nodes without
internet access, download it on a login node first so it is present in the
shared Hugging Face cache. The contrastive jobs are intentionally limited to ten
GRPO steps to keep this an example rather than a full experiment.
