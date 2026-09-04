"""Command-line interface for lilpipe."""

from __future__ import annotations

import argparse
import shlex
import sys

from .core import PipelineError, load, submit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lilpipe", description="Compile and submit a declarative Slurm pipeline"
    )
    parser.add_argument("config", help="Pipeline YAML file, relative to the current directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip", nargs="+", default=None, metavar="MODEL_OR_STAGE")
    parser.add_argument("--models", nargs="+", default=None, metavar="MODEL")
    parser.add_argument(
        "--evaluations", nargs="+", default=None, metavar="EVALUATION"
    )
    parser.add_argument(
        "--sbatch-args",
        default="",
        help=(
            "Extra arguments forwarded to every sbatch call as one quoted string, "
            "for example --sbatch-args='--exclude=node1 --exclusive'"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pipeline = load(args.config)
        pipeline = pipeline.select(
            models=args.models,
            evaluations=args.evaluations,
        )
        plan = pipeline.plan(skip=args.skip or ())
        extra_sbatch_args = shlex.split(args.sbatch_args)
        if args.dry_run:
            rendered = plan.render(extra_sbatch_args=extra_sbatch_args)
            if rendered:
                print(rendered)
        else:
            submit(plan, extra_sbatch_args=extra_sbatch_args)
    except PipelineError as error:
        print(f"lilpipe: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
