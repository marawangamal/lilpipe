This project contains novel unlearning algorithms and reusable components for developing and evaluating tamper resistance. 

Detailed instructions for using this repository may be found in the CLAUDE.md. Please point your bot at the file. If you run into any trouble, please contact lucia@eleuther.ai.

# Algorithms

## Tuned Lens Unlearn

Unlearning algorithms often define a learning objective based on the token probabilities at the model output, such as entropy maximization or cross-entropy loss maximization. Loss on these objectives may be minimized via fine-tuning on only a few final model layers. Therefore, we have no strong reason to expect that minimizing these objectives will remove relevant information from all model layers. Tuned lens unlearning targets this failure mode by computing a loss term at every module output.

A tuned lens is a learned affine map that maps from a module's output activations to the model unembedding layer. The map is learned by unembedding the transformed module outputs using the base model's unembedding matrix, and then updating the map to minimize a standard cross entropy loss term.

We train a tuned lens on each module and then unlearn each module through its tuned lens. We support entropy maximization and cross-entropy maximization.

## Orthogonal Circuit Breakers Unlearn

The circuit breakers forget loss produces forget-sequence activations that are orthogonal to retain-sequence activations. Mechanistically, we can imagine a network learning a single linear transform at the first layer with the capacity to distinguish between forget and retain sequences, selecting and transforming all forget activations in the same way. This solution minimizes the circuit breakers loss, but may be undone by learning the transform's inverse.

We introduce a within-batch forget sequence orthogonalization loss to incentivize each forget sequence to be scrambled using a unique learned transformation.

## Sequential Unlearn

We improve tuned lens unlearning by replacing the tuned lens with the final layers of the base (pre-unlearn) model. We unlearn each layer in sequence, mapping the activations at the layer currently being unlearned to the forget loss through the base model, and to the retain loss through the updated model weights.

We also experiment with simply unfreezing each layer in sequence, spending an equivalent compute of compute on each layer as is sufficient to unlearn with all layers unfrozen.

## Checkpoint Activation Transfer Unlearn

A module that undergoes sufficient training to induce random performance on a forget distribution may still retain unused knowledge in its weights that is not extracted by the later modules. 

We apply a teacher model that is deeply ignorant and trivially available - an early checkpoint of the same model. We select the latest checkpoint at which the model has near-random performance on the forget distribution, and then train an affine map from its activations to the fully-trained model using an MSE loss. This map accounts for linear transformations of the model weights that occur over the course of training (we also experiment with an SVCCA-based map; for more information on this and linear representation drift over training see https://arxiv.org/abs/1811.00225).

We use a MSE loss term between the transformed, deeply ignorant early checkpoint activations, and the base model activations.

## Rewritten Activation Transfer Unlearn

We collect activations on a forget dataset rewritten to be inaccurate, and then "transfer" those activations to the corresponding original dataset items using a MSE loss.

# Algorithm Components

Our algorithms use several design elements that may be stacked:

- Maximum entropy loss.
- Per-module tuned lens unlearning.
- Greedy sequential layer-wise unlearning, using the base model layers in place of the tuned lens to "map" the activations at the layer currently being unlearned to the forget loss, and the updated model layers to map the same activations to the retain loss.
- An auxiliary within-batch activation orthogonalization forget loss.

## Per-Token Unlearning

We implement several token selection methods for per-token unlearning:

- Blocklist-based
- Activation similarity to blocklist tokens
- Probe
- Biorisk-relevant SAE activations
- Data attribution (MAGIC: https://arxiv.org/abs/2504.16430)

# Analysis

Unlearning objectives usually consist of a forget term and a retain term. We investigate several unlearning algorithms and find _even when the retain term is removed_, unlearning is not tamper resistant. Procedure:

- Hyperparameter tune an unlearning algorithm until good performance is attained.
- Disable the retain term and repeat the unlearning process.
- Fine-tune the model on the forget distribution and compare performance over training steps with our gold-standard Deep Ignorance filtered model. 

# Under Development

- MAGIC-based detractor learning.

# Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

### Historical Code

This repository contains historical code from the Deep Ignorance project that may be useful for unlearning analysis. Other artifacts from this project are available at https://github.com/EleutherAI/deep-ignorance and https://github.com/EleutherAI/filtering_for_danger.

### Environment

Create and/or activate a venv:

python3 -m venv .venv && source .venv/bin/activate

### Claude Code

Run cmd-shift-P "install code/cursor command" if necessary.

Install the Claude Code extension

use /ide to connect to the IDE if disconnected.

### Circuit Breakers

```bash
bash scripts/run_unlearn.sh -a cb --orth 0 --rm <remove_coef> --ret <retain_coef> [options]

```

### Tuned Lens

1. Download data

```bash
python -m unlearn.create_unlearn_data
```

2. Train lens

```bash
torchrun --nproc_per_node=8 unlearn/algorithm/tuned_lens/train.py --batch_size 4 --gradient_accumulation_steps 1 --upload_to_hf True --hf_repo_id 'EleutherAI/deep-ignorance-unfiltered-lens'
```

3. Run tuned lens unlearning

```bash
bash scripts/run_unlearn.sh -a lens --rm <remove_coef> --ret <retain_coef> [options]
```

## Tamper and Plot

sbatch file:

```bash
  #!/bin/bash
  #SBATCH --job-name=tamper-attack
  #SBATCH --nodes=1
  #SBATCH --exclusive
  #SBATCH --gpus-per-node=4
  #SBATCH --time=4:00:00
  #SBATCH --output=/home/a6a/lucia.a6a/unlearn/runs/tamper-%j.out

  source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
  conda activate <env_name>
  module load cuda/12.6

  python -m unlearn.scripts.run_tamper_attack_with_plot \
      --model_name=<model_path> \
      --output_dir=runs/<tamper_output_dir> \
      --num_train_examples=512 \
      --epochs=1 \
      --eval_every=10 \
      --lr=2e-5
```

Regenerate a plot with HP annotations from existing results:

```bash
python -m unlearn.scripts.run_tamper_attack_with_plot \
    --plot_only=runs/<tamper_output_dir>/tamper_results_<timestamp>.json \
    --title="Tamper Attack: <method>\n<hp_summary>"
```

Copy the plot to experiment_logs:

```bash
cp runs/<tamper_output_dir>/tamper_results_<timestamp>.png experiment_logs/tampering_<name>.png
```

## Transformer Probe

Probe training details:
- WandB run: https://wandb.ai/eleutherai/depth-scaled-probes/runs/trga8lub
- 7 layers (8, 12, 16, 20, 24, 28, 32)
- Depth = (32 - layer), ranging from 24 to 1 transformer layers
- Trained on WMDP-Bio-Remove forget data

Usage for unlearning:
python -m unlearn.algorithm.probe_unlearn \
    --probe_dir ./models/depth_scaled_probes \
    --layers 8 12 16 20 24 28 \
    --lora --lora_r 16 \
    --num_train_examples 1024
