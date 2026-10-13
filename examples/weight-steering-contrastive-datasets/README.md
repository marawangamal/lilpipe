# Weight steering across contrastive datasets

This experiment compares weight steering and gradient differencing on TOFU
`forget10` under two definitions of the positive data:

- `retain90`: different retained question/answer pairs;
- `idk10`: the same forget questions with an IDK completion.

The target is `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`; the benchmark
oracle is `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`. All commands
must be run from this directory.

## Experiment design

For weight steering, train three equally configured models from the target on
forget, retain, and IDK data. Construct

```text
target + alpha * (positive arm - forget arm)
```

where the positive arm is retain or IDK. For gradient differencing, directly
optimize `positive_loss - beta * forget_loss`. IDK rows use the same question
as their paired forget row. HPO varies `alpha` for steering and learning rate /
`beta` for gradient differencing. `scripts/select.py` promotes the best proxy
candidate for each cell; selected models then receive the full
OpenUnlearning-compatible TOFU evaluation. Promotion requires at least 80% of
the target model's retain-set likelihood, then maximizes the harmonic mean of
normalized forget progress and retained likelihood.

The HPO also evaluates a hybrid of weight steering and PerTA-grad. It computes
the paper's soft per-parameter mask once on the pretrained Llama model,

```text
W = (abs(forget_gradient) + epsilon)
    / (abs(forget_gradient) + abs(retain_gradient) + 2 * epsilon)
target + alpha * W * (positive arm - forget arm)
```

and applies it to both contrastive steering vectors. This intentionally keeps
the experiment's arm-difference vector; it does not use PerTA's original
forget-only task-negation vector. The default uses a deterministic 20% sample,
which the paper reports as comparable to the full-data mask.

For a direct task-arithmetic comparison, the DAG also trains a forget-only arm
from the pretrained model for 20 epochs (effective batch size 32), then builds
both the paper's vanilla task-negation model

```text
target - (forget-only arm - pretrained)
```

and its PerTA-grad variant using a mask estimated from the full forget and
retain datasets. These branches are separate from the contrastive steering
vectors above.

## Mila setup and execution

```bash
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
uv sync --group test
python scripts/hpo.py generate configs/sweeps.yml
lilpipe artifacts/generated/hpo/experiment.yml --dry-run
lilpipe artifacts/generated/hpo/experiment.yml
```

Jobs request one L40S. Artifacts should be linked to scratch before submission:

```bash
mkdir -p "$SCRATCH/lilpipe/examples/weight-steering-contrastive-datasets/artifacts"
ln -sfnT "$SCRATCH/lilpipe/examples/weight-steering-contrastive-datasets/artifacts" artifacts
mkdir -p artifacts/logs
```

After HPO jobs finish:

```bash
python scripts/select.py
lilpipe configs/experiments/selected.yml
python scripts/table.py
```

The final Markdown and CSV tables are written under `artifacts/results/`.
