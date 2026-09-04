#!/bin/bash
set -euo pipefail

export PYTHONPATH="$PWD/configs/training:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export AXOLOTL_DO_NOT_TRACK=1
config=$1
axolotl train "$config" --launcher python
axolotl merge-lora "$config"
