"""Create a W&B sweep from YAML and print its entity/project/sweep ID."""

import argparse
import contextlib
import sys

import wandb
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    with contextlib.redirect_stdout(sys.stderr):
        sweep_id = wandb.sweep(
            config, entity=config["entity"], project=config["project"]
        )
    print(f"{config['entity']}/{config['project']}/{sweep_id}")
