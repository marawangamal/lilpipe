"""Command-line interface for lilpipe."""

from __future__ import annotations

import argparse
import shlex
import sys

from .core import PipelineError, load, submit
from .results import load_results


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", help="Pipeline YAML file, relative to the current directory"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip", nargs="+", default=None, metavar="MODEL_OR_STAGE")
    parser.add_argument("--models", nargs="+", default=None, metavar="MODEL")
    parser.add_argument("--evaluations", nargs="+", default=None, metavar="EVALUATION")
    parser.add_argument(
        "--sbatch-args",
        default="",
        help="Extra arguments forwarded to every sbatch call as one quoted string",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lilpipe", description="Compile Slurm pipelines and collect results"
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="compile or submit a pipeline")
    _add_pipeline_arguments(run_parser)
    results_parser = subparsers.add_parser("results", help="collect experiment results")
    results_parser.add_argument(
        "config", help="Results YAML file, relative to the current directory"
    )
    results_parser.add_argument(
        "--format", choices=("markdown", "csv", "json"), default="markdown"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if (
        arguments
        and not arguments[0].startswith("-")
        and arguments[0] not in {"run", "results"}
    ):
        arguments.insert(0, "run")
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "results":
            print(load_results(args.config).collect().render(args.format))
            return 0
        pipeline = load(args.config).select(
            models=args.models, evaluations=args.evaluations
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
