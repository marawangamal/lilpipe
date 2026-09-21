"""Wrapper to run sparsify under torchrun.

Fixes sys.stdout/stderr being None on distributed worker processes,
which causes a crash in transformers' loading_report.py.
"""

import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from sparsify.__main__ import run

run()
