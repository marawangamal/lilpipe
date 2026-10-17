# Experiment notes

Run commands from `examples/tamper-resistance`.

## Generate and run a sweep

Each lineage has a runnable `base.yml`, a parameter `sweep.yml`, and a
promoted standalone `selected.yml`. Generate candidates from a sweep with:

```bash
python scripts/hpo.py generate \
  configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--2pp--unlearn-fft-gd/sweep.yml
lilpipe artifacts/generated/hpo/z7b-wmdp-bio--2pp--unlearn-fft-gd-hpo/experiment.yml
```

Generated configurations, their model registry, and their manifest are kept
under `artifacts/generated/hpo/<sweep-id>/`. HPO runs keep only the final
checkpoint. Selection is manual: copy the chosen concrete values to the
lineage's `selected.yml` and promote its model in
`configs/registries/models.yml`.

## Run canonical lineages

```bash
lilpipe configs/experiments/z7b-wmdp-bio-unlearn--2pp.yml
lilpipe configs/experiments/z7b-wmdp-bio-unlearn--2pp.yml
lilpipe configs/experiments/z7b-wmdp-bio-unlearn--2pp.yml
```

Canonical relearning configurations retain trajectory checkpoints for
evaluation. Existing artifact directories are not renamed or removed.

## Results

### SSIUU TOFU Table 2

The focused reproduction contains only Gradient Difference (GD) and SSIUU on
Llama-3.2-3B-Instruct. It performs five-epoch TOFU tuning, saves each
unlearning step, selects the first checkpoint with generated-answer ROUGE-L at
or below `19.71`, and runs the basic and relearning evaluations:

```bash
lilpipe configs/experiments/llama32-tofu-unlearn.yml
```

Once all attack arrays finish, assemble the GD/SSIUU rows with:

```bash
python scripts/summarize_tofu_table2.py \
  artifacts/mila/evals artifacts/mila/attacks/tofu-table2 \
  artifacts/mila/results/tofu-table2-gd-ssiuu.json
```

The attack summary takes the maximum recovery over the paper's learning-rate
grid for each seed, then reports the mean across three seeds.

Generate the regular and alignment attack trajectories from the example root:

```bash
python scripts/analysis/plot_trajectory.py
python scripts/analysis/plot_trajectory_alignment.py
```

The scripts write PDF and PNG figures to `artifacts/figures`.

### Hyperparameter sweeps

The table records the Zephyr-7B sweeps by unlearning-method family. Each named
stage in `sweep.yml` is an independent grid; only parameters within the same
stage form a Cartesian product. Other settings were inherited from the method's
`base.yml`; promoted values are recorded in `selected.yml`.

| Method | Learning rate | $\lambda_{\mathrm{ret}}$ | $\lambda_{\mathrm{steer}}$ | $\rho_{\mathrm{SAM}}$ | $\lambda_{\text{weight-dist}}$ | $\lambda_{\mathrm{AO}}$ | Runs | Promoted setting |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| GD | `{5e-7, 1e-6, 2e-6, 2.5e-6, 3.28e-6}` | `{1, 3, 4, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 4.75, 5, 10}` | - | - | - | - | 23 | learning rate `2.5e-6`; retain `4` |
| GD + SAM | `{2.5e-6, 3.28e-6, 5e-6}` | from GD: `1` | - | `{0.001, 0.01, 0.1}` | - | - | 6 | learning rate `2e-6`; retain `1`; rho `0.01` |
| GD + WD | - | - | - | - | `{1e5, 3e5, 1e6, 3e6, 5e6}` | - | 5 | weight distance `3e5` |
| GD + AO | from GD: `2.5e-6` | `{1.5, 2, 3, 4, 5, 7, 10}` | - | - | - | `{0, 0.001, 0.003, 0.005, 0.01, 0.1, 1, 10, 20, 25, 30, 35, 40, 60, 80, 100}` | 25 | activation orthogonality `40`; retain `4` |
| NPO | `{5e-7, 1e-6, 3e-6}` | `{1, 3, 10}` | - | - | - | - | 9 | learning rate `1e-6`; retain `1` |
| NPO + SAM | - | `{0.5, 1, 2}` | - | `{0.001, 0.01, 0.1}` | - | - | 9 | retain `1`; rho `0.001` |
| NPO + WD | - | `{0.5, 1, 2}` | - | - | `{100, 1000, 10000}` | - | 9 | retain `1`; weight distance `1000` |
| RMU | - | `{300, 600, 1200}` | `{3.25, 6.5, 13}` | - | - | - | 9 | retain `600`; steering `6.5` |
| RMU + AO | from RMU: `5e-5` | from RMU: `600` | from RMU: `6.5` | - | - | `{0, 0.01, 0.03, 0.05, 0.07, 0.1, 1, 10, 100}` | 9 | activation orthogonality `0.01` |
| RMU + SAM | - | `{600, 1200, 2400}` | - | `{0.001, 0.01, 0.1}` | - | - | 9 | retain `600`; rho `0.001` |
| RMU + WD | - | `{600, 1200, 2400}` | - | - | `{100, 1000, 10000}` | - | 9 | retain `1200`; weight distance `100` |

GD + AO inherited the promoted GD learning rate and forget coefficient, then
used six independent sweep stages in one manifest. The canonical relearning learning-rate sweep
tested `{2.5e-6, 5e-6, 7.5e-6, 1e-5}` before promoting `1e-5` for the 125-step
long attack. RMU + AO keeps the promoted RMU settings and trainable layers
fixed, targets AO at layers 5--7, and varies only the AO coefficient.

RMU + AO selected the strongest positive AO candidate satisfying a maximum
2-point MMLU drop from the 58.59 base-model score:

| $\lambda_{\mathrm{AO}}$ | WMDP-Bio robust $\downarrow$ | MMLU-no-bio $\uparrow$ | MMLU drop | Eligible |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 27.65 | 56.72 | 1.87 | yes |
| **0.01** | **28.23** | **57.03** | **1.55** | **yes; selected** |
| 0.03 | 34.22 | 57.05 | 1.54 | yes |
| 0.05 | 41.01 | 56.97 | 1.61 | yes |
| 0.07 | 39.98 | 56.73 | 1.85 | yes |
| 0.1 | 37.90 | 56.53 | 2.06 | no |
| 1 | 29.72 | 55.75 | 2.84 | no |
| 10 | 27.88 | 52.06 | 6.53 | no |
| 100 | 25.81 | 44.12 | 14.46 | no |

#### MiniCPM5-1B

The MiniCPM5-1B sweep is planned but has not been run. RMU candidates follow
the original RMU search dimensions and should be evaluated in stages rather
than as one Cartesian product. All non-RMU method families remain unspecified.

| Method | Learning rate | $\lambda_{\mathrm{ret}}$ | $\lambda_{\mathrm{steer}}$ | $\rho_{\mathrm{SAM}}$ | $\lambda_{\text{weight-dist}}$ | $\lambda_{\mathrm{AO}}$ | Target layer | Updates | Runs | Promoted setting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| GD | `{5e-7, 1e-6, 2.5e-6, 5e-6, 1e-5}` | `[0.25, 16]` (log) | - | - | - | - | - | `{50, 100, 150, 250}` | 24 | learning rate `1e-6`; retain `0.630454`; 100 updates |
| GD + SAM | - | - | - | - | - | - | - | - | - | - |
| GD + WD | - | - | - | - | - | - | - | - | - | - |
| GD + AO | - | - | - | - | - | - | - | - | - | - |
| NPO | - | - | - | - | - | - | - | - | - | - |
| NPO + SAM | - | - | - | - | - | - | - | - | - | - |
| NPO + WD | - | - | - | - | - | - | - | - | - | - |
| RMU | `{2.5e-5, 5e-5, 1e-4}` | `{300, 600, 1200, 2400}` | `{3.25, 6.5, 13, 30, 100, 300}` | - | - | - | `{3..23}` | `{150, 300, 500}` | - | - |
| RMU + SAM | - | - | - | - | - | - | - | - | - | - |
| RMU + WD | - | - | - | - | - | - | - | - | - | - |

The MiniCPM5-1B GD study minimizes pre-attack WMDP-Bio robust accuracy subject
to MMLU-no-bio accuracy remaining at least `0.5062` (at most a two-point drop
from the base model). Trial 11 was promoted with WMDP-Bio `0.47350` and MMLU
`0.52920`. Trials reaching near-random WMDP accuracy were ineligible because
their MMLU accuracy collapsed to approximately `0.23`.
