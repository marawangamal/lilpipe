# Booster Figure 3 reproduction

This example first reproduces Figure 3 of the Booster paper with the authors'
code and experimental semantics. It then runs one controlled attack ablation:
the 1,000-example fine-tuning set changes from 100 harmful + 900 SST-2 examples
to 1,000 harmful examples. All other attack settings remain fixed, including
20 epochs, batch size 10, learning rate `1e-5`, and the continued LoRA selected
by the authors' `--alternating single_lora` option.

The expected Figure 3 trajectory is evaluated at steps 0, 200, ..., 2,000 and
contains harmful score on 1,000 BeaverTails test prompts, harmful training NLL,
and held-out harmful NLL. The baseline pipeline must finish successfully before
the harmful-only manifest is submitted.

Run from this directory:

```bash
lilpipe configs/experiments/figure3.yml --dry-run
lilpipe configs/experiments/figure3.yml
# Submit only after the baseline result has been checked:
lilpipe configs/experiments/figure3-harmful-only.yml \
  --existing-models \
  booster-figure3--defense-sft \
  booster-figure3--defense-booster
```

Mila jobs request one L40S for three hours. Training checkpoints every 200
steps and automatically resumes from the latest checkpoint. Evaluation writes
incremental generations and moderation results so rerunning a timed-out array
element resumes rather than starting over.

See `UPSTREAM.md` for the pinned source revision. The upstream-code changes
needed for execution remove the committed token-file dependency, enable
checkpoint auto-resume, use the shared model cache, and guard imports for model
architectures that did not exist in the authors' pinned Transformers release.
Experiment wrappers and evaluators live under `scripts/lilpipe/`.
