"""Declarative Slurm pipelines for model evaluation experiments."""

from .core import Pipeline, PipelineError, Stage, load_pipeline, submit_pipeline

__all__ = [
    "Pipeline",
    "PipelineError",
    "Stage",
    "load_pipeline",
    "submit_pipeline",
]
