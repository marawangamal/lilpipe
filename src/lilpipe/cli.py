"""Command-line interface for lilpipe."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex

from .core import load_pipeline, submit_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lilpipe", description="Compile and submit a declarative Slurm pipeline"
    )
    parser.add_argument("--config", required=True, help="Pipeline YAML file")
    parser.add_argument(
        "--root",
        default=".",
        help="Project root used to resolve registries and stage scripts (default: cwd)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--only-eval", nargs="*", default=[])
    parser.add_argument(
        "--sbatch-args",
        default="",
        help=(
            "Extra arguments forwarded to every sbatch call as one quoted string, "
            "for example --sbatch-args='--exclude=node1 --exclusive'"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    config = Path(args.config)
    if not config.is_absolute():
        config = root / config
    pipeline = load_pipeline(
        config,
        root=root,
        selected_models=args.models,
        selected_evals=args.only_eval,
    )
    submit_pipeline(
        pipeline,
        root=root,
        dry_run=args.dry_run,
        skip=args.skip,
        extra_sbatch_args=shlex.split(args.sbatch_args),
    )


if __name__ == "__main__":
    main()
