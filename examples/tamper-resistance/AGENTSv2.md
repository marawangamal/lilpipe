## General guidelines

- Run every command from `examples/tamper-resistance` (paths below are relative to this)
- Training using `.venv-train`
- Evaluation uses `.venv-eval`

## Project structure

```text
artifacts/
configs/
├── experiments/z7b-wmdp-bio-unlearn--5pp.yml  # models and evaluations to run
├── registries/                                # model producers and evaluation jobs
└── stages-wmdp/0_unlearn/
    └── z7b-wmdp-bio--5pp--unlearn-fft-rmu/
        ├── base.yml                           # training config used by the sweep
        ├── sweep.yml                          # HPO parameter grid
        └── selected.yml                       # optimal training config found
```

- Models and evals are saved under `artifacts/{mila|tamia}/{models|evals}

## Naming schemes

WMDP experiements
0_unlearn/{model}-{dataset}--{capability-drop-threshold}--unlearn-fft-{unlearn-method}/
2_relearn/{model}-{dataset}--{capability-drop-threshold}--unlearn-fft-{unlearn-method}--relearn-fft-[{hps}]


## WMDP Experiments

### Hyperparameter tuning procedure

**Tuning defences**
```sh
# 1. Generate and run all defenses and their evaluations
# NOTE: z7b-wmdp-bio--5pp--unlearn-fft-me--hpo is the sweep id defined in sweep.yml
python scripts/hpo.py generate configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me/sweep.yml
lilpipe artifacts/generated/hpo/configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me--hpo/experiment.yml

# 2. (manual) Promote the best by selecting the method with the lowest WMDP score, whilst having
# less than 5pp drop in MMLU.
```

**Tuning attacks**
```sh
# 1. Generate and run all defenses and their evaluations
# NOTE: z7b-wmdp-bio--5pp--unlearn-fft-me--hpo is the sweep id defined in sweep.yml
python scripts/hpo.py generate configs/stages-wmdp/2_relearn/z7b-wmdp-bio--5pp--unlearn-fft-me--relearn-fft/sweep.yml
lilpipe artifacts/generated/hpo/configs/stages-wmdp/2_relearn/z7b-wmdp-bio--5pp--unlearn-fft-me--relearn-fft--hpo/experiment.yml

# 2. (manual) Promote the best by selecting the method with highest average score (WMDP and MMLU)
# less than 5pp drop in MMLU. This means moving setting the config in
# scripts/hpo.py generate configs/stages-wmdp/0_unlearn/qwen3-0.6b-beavertail--align-fft-seam/selected.yml
# and also copying over the model from artifacts/generated/hpo/ into artifacts/models

**Promotion Procedure**

```sh
# 1. Copy configs. Copy yml of best sweep ID. In the copied selected.yml file
# set output_dir, wandb_name, and dataset_prepared_path to new canoncial paths 
cp "artifacts/generated/hpo/<sweep-id>/configs/<sweep-id>--001.yml" \
   "configs/stages-wmdp/2_relearn/z7b-wmdp-bio--5pp--unlearn-fft-me--relearn-fft/selected.yml"

# 2. Move artifacts on Tamia. Archive any existing destination first.
mv -T "artifacts/generated/hpo/<sweep-id>/models/<sweep-id>--001" \
   "artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me--relearn-fft"
```


**Plotting relearning trajectories**
```sh
# Uncomment the config depending on the command.
# main plot: compare methods under 128 step attack
# tar attack hps plot: compare different 128-step attacks under the same tar defence
python scripts/analysis/relearning-trajectory/plot_trajectory_wmdp.py
```

**Plotting loss landscapes**
```sh
# Uncomment the config depending on the command.
# main plot: compare methods under 128 step attack
# tar attack hps plot: compare different 128-step attacks under the same tar defence
python scripts/analysis/loss-landscapes/get_ll.py -- add args
python scripts/analysis/loss-landscapes/plot_ll_3d.py -- add args
```

## Launching jobs

- Some models are meant to be run on tamia (use `ssh tamia`), whilst smaller ones like the q0.4b is run on mila
- Scripts for running jobs on tamia/mila are under `scripts/slurm/{tamia|mila}` (lilpipe model/eval registries point to these)

**Tamia**
- Keep the project in sync it lives under `/home/m/mgamal/projects/lilpipe`
- We only have 2tb of scratch space, use `diskusage_report` to check remaning space before running jobs

**Mila**
- We have 5tb of scratch space, use `disk-quota` to check remaning space before running jobs