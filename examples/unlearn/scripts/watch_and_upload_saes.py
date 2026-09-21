"""Watch SAE training jobs and upload to HuggingFace when done."""

import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
login(token=os.environ["HF_TOKEN"])
api = HfApi()

REPO_ROOT = Path("/home/a6a/lucia.a6a/unlearn")

JOBS = {
    "ex8": {
        "job_id": "2114668",
        "save_dir": REPO_ROOT / "models" / "deep-ignorance-unfiltered_saes",
        "repo_id": "EleutherAI/sae-deep-ignorance-unfiltered-ex8",
        "card": """\
---
library_name: sparsify
base_model: EleutherAI/deep-ignorance-unfiltered
tags:
  - sparse-autoencoder
  - sae
  - mechanistic-interpretability
---

# SAEs for deep-ignorance-unfiltered (expansion factor 8, all layers)

Sparse autoencoders trained on all 32 layers of EleutherAI/deep-ignorance-unfiltered
(https://huggingface.co/EleutherAI/deep-ignorance-unfiltered) using the
WMDP-Bio-Remove-Dataset:
https://huggingface.co/datasets/Unlearning/WMDP-Bio-Remove-Dataset.

Trained with [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify).

## Training hyperparameters

| Parameter | Value |
|---|---|
| expansion_factor | 8 |
| k | 32 |
| layers | all 32 (0-31) |
| batch_size | 4 |
| grad_acc_steps | 8 |
| micro_acc_steps | 2 |
| ctx_len | 2048 |
| optimizer | signum |
| loss_fn | fvu |
| activation | topk |
| distributed | distribute_modules across 4x GH200 120GB |

## Dataset

Unlearning/WMDP-Bio-Remove-Dataset (24,453 examples):
https://huggingface.co/datasets/Unlearning/WMDP-Bio-Remove-Dataset

## Usage

```python
from sparsify import Sae

sae = Sae.load_from_hub(
    "EleutherAI/sae-deep-ignorance-unfiltered-ex8",
    hookpoint="layers.10"
)

# Or load all layers
saes = Sae.load_many("EleutherAI/sae-deep-ignorance-unfiltered-ex8")
```
""",
    },
    "ex16": {
        "job_id": "2114687",
        "save_dir": REPO_ROOT / "models" / "deep-ignorance-unfiltered_saes_ex16",
        "repo_id": "EleutherAI/sae-deep-ignorance-unfiltered-ex16",
        "card": """\
---
library_name: sparsify
base_model: EleutherAI/deep-ignorance-unfiltered
tags:
  - sparse-autoencoder
  - sae
  - mechanistic-interpretability
---

# SAEs for deep-ignorance-unfiltered (expansion factor 16, every 4th layer)

Sparse autoencoders trained on every 4th layer (0, 4, 8, 12, 16, 20, 24, 28)
of EleutherAI/deep-ignorance-unfiltered:
https://huggingface.co/EleutherAI/deep-ignorance-unfiltered
using the WMDP-Bio-Remove-Dataset:
https://huggingface.co/datasets/Unlearning/WMDP-Bio-Remove-Dataset

Trained with [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify).

## Training hyperparameters

| Parameter | Value |
|---|---|
| expansion_factor | 16 |
| k | 32 |
| layers | 0, 4, 8, 12, 16, 20, 24, 28 (stride 4) |
| batch_size | 1 |
| grad_acc_steps | 32 |
| micro_acc_steps | 2 |
| ctx_len | 2048 |
| optimizer | signum |
| loss_fn | fvu |
| activation | topk |
| distributed | DDP across 4x GH200 120GB |

## Dataset

Unlearning/WMDP-Bio-Remove-Dataset (24,453 examples):
https://huggingface.co/datasets/Unlearning/WMDP-Bio-Remove-Dataset

## Usage

```python
from sparsify import Sae

sae = Sae.load_from_hub(
"EleutherAI/sae-deep-ignorance-unfiltered-ex16", hookpoint="layers.8"
)

# Or load all available layers
saes = Sae.load_many("EleutherAI/sae-deep-ignorance-unfiltered-ex16")
```
""",
    },
}


def get_job_state(job_id):
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "-n", "-X"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().split("\n")[0].strip() if result.stdout.strip() else ""


def upload(name, info):
    save_dir = info["save_dir"]
    repo_id = info["repo_id"]

    if not save_dir.exists():
        print(f"  {name}: save_dir {save_dir} does not exist, skipping")
        return

    readme_path = save_dir / "README.md"
    readme_path.write_text(info["card"])
    print(f"  {name}: wrote model card to {readme_path}")

    print(f"  {name}: creating repo {repo_id}...")
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    print(f"  {name}: uploading {save_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=str(save_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  {name}: upload complete")


def main():
    done = {name: False for name in JOBS}

    job_desc = ", ".join(f"{n}={j['job_id']}" for n, j in JOBS.items())
    print(f"Watching jobs: {job_desc}")

    while not all(done.values()):
        time.sleep(60)

        for name, info in JOBS.items():
            if done[name]:
                continue

            state = get_job_state(info["job_id"])
            if state == "COMPLETED":
                print(f"\n{name} (job {info['job_id']}): COMPLETED")
                done[name] = "ok"
            elif state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_ME+"):
                print(f"\n{name} (job {info['job_id']}): {state}")
                done[name] = "fail"

    print("\nAll jobs finished. Uploading...")
    for name, info in JOBS.items():
        if done[name] == "ok":
            print(f"\nUploading {name}...")
            upload(name, info)
        else:
            print(f"\nSkipping {name} (job ended with non-success state)")

    print(f"\nDone. {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
