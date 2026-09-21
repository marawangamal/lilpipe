# SFT Finetuning Experiments

Model: `EleutherAI/deep-ignorance-unfiltered` (GPT-NeoX 6.9B)

Training: full parameter SFT, 1 epoch, cosine LR schedule (unless noted), bf16, gradient checkpointing, single GPU.

Eval: MMLU (0-shot), WMDP Bio Robust. Both via torchrun 4-GPU.

## Results

| Run | Dataset | LR | Eff. Batch | Examples | Steps | Warmup | MMLU | WMDP Bio Robust |
|---|---|---|---|---|---|---|---|---|
| Baseline | - | - | - | - | - | - | 0.4510 | 0.4297 |
| wiki-sft | WikiText | 2e-4 | 32 | 4096 | 128 | 0.05 | 0.4210 | 0.4067 |
| wiki-5e5 | WikiText | 5e-5 | 16 | 1000 | 62 | 0.1 | 0.4420 | 0.4309 |
| bf-5e5 | Bio Forget | 5e-5 | 16 | 1000 docs (3455 chunks) | 215 | 0.1 | 0.4500 | 0.4482 |
| wiki-uc-5e5 | WikiText + 20% UC | 5e-5 | 16 | 1000 | | 0.1 | | |
| wiki-uc-1e4 | WikiText + 20% UC | 1e-4 | 16 | 1000 | | 0.1 | | |
| bf-uc-5e5 | Bio Forget + 20% UC | 5e-5 | 16 | 1000 docs | | 0.1 | | |
| bf-uc-1e4 | Bio Forget + 20% UC | 1e-4 | 16 | 1000 docs | | 0.1 | | |
| wiki-1e4 | WikiText | 1e-4 | 16 | 1000 | | 0.1 | | |
| bf-1e4 | Bio Forget | 1e-4 | 16 | 1000 docs | | 0.1 | | |

UC = UltraChat (20% of training examples from HuggingFaceH4/ultrachat_200k).

Model checkpoints saved to `/projects/a6a/public/lucia/sft_models/`.
