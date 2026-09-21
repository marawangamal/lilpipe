# MAGIC Data Attribution for Unlearning

MAGIC (Model Attribution via Gradient Checkpointing) experiments copied from the
[bergson](https://github.com/EleutherAI/bergson) repo. These scripts fine-tune a
model on training data with checkpoints at every step, evaluate on WMDP-bio-robust,
then backpropagate through the entire training trajectory to compute per-example and
per-token attribution scores.

## Bergson branch

These scripts require the **`magic-wmdp-attribution`** branch of bergson:

```bash
git clone git@github.com:EleutherAI/bergson.git bergson3
cd bergson3
git checkout magic-wmdp-attribution
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Two manual edits are also needed in `bergson/trainer.py`:

1. `sorted_checkpoints`: remove the `if not os.path.isfile(path): continue` check
   (FSDP checkpoints are directories, not files).
2. `trainer.backward`: change `weight_grads = result[-1] + w_grads` to
   `weight_grads = result[-1] + w_grads if result[-1] is not None else w_grads`
   (handle `None` gradients when `allow_unused=True`).

## PyTorch patch (required)

FSDP double-backward requires DTensor redistribution to be twice-differentiable.
This is not yet upstream in PyTorch — it's tracked in
[pytorch/pytorch#160509](https://github.com/pytorch/pytorch/pull/160509).

The setup script `magic_wmdp_setup.sh` patches the installed torch package in-place.
It modifies two files:

1. **`torch/distributed/tensor/_redistribute.py`** — adds a `_redistribute_backward()`
   helper and a `NestedRedistribute` autograd function that makes the existing
   `Redistribute.backward` twice-differentiable.
2. **`torch/distributed/tensor/_api.py`** — changes `_ToTorchTensor.backward` to use
   `DTensor.from_local()` instead of the `DTensor()` constructor, so DTensor specs
   propagate correctly during backward passes.

Run the patch once after installing torch:

```bash
source .venv/bin/activate
bash magic_wmdp_setup.sh
```

The script is idempotent — it skips files already patched and verifies the patch
loads correctly at the end.

## Scripts

| File | Description |
|------|-------------|
| `magic_wmdp.py` | Main MAGIC pipeline: WikiText-103 training -> WMDP-bio-robust eval -> backward attribution |
| `magic_per_token.py` | Per-token attribution variant supporting multiple training datasets (wmdp_forget, wmdp_retain, ultrachat, wmdp_lie_o) |
| `attribution_unlearn.py` | Attribution-weighted unlearning: uses MAGIC scores as per-token loss weights |
| `magic_wmdp_setup.sh` | Applies the PyTorch DTensor patch for twice-differentiable FSDP |
| `magic_wmdp_results.md` | Results from the initial WikiText -> WMDP-bio-robust attribution run |

## SBATCH files

All sbatch files assume 4x GH200 120GB GPUs, `PrgEnv-cray`, `cuda/12.6`, and the
bergson3 venv. Paths will need updating for your environment.

| File | Job |
|------|-----|
| `magic_wmdp.sbatch` | Main WikiText attribution |
| `magic_wmdp_forget.sbatch` | Attribution on WMDP forget corpus |
| `magic_wmdp_retain.sbatch` | Attribution on WMDP retain corpus |
| `magic_wmdp_lie_o.sbatch` | Attribution on WMDP lie-o-deep-fried |
| `magic_ultrachat.sbatch` | Attribution on UltraChat 200k |
| `attribution_unlearn.sbatch` | Attribution-weighted unlearning (bs=4, adam) |
| `attribution_unlearn_bs16.sbatch` | Attribution-weighted unlearning (bs=16, adam) |
| `attribution_unlearn_bs16_adamw.sbatch` | Attribution-weighted unlearning (bs=16, adamw) |
