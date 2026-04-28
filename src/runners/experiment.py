from __future__ import annotations

from pathlib import Path

from search.run_spec import RunSpec


def run_contrastive_experiment(run_spec: RunSpec, output_dir: Path) -> dict[str, Any]:
    """Run a real contrastive learning experiment.
    
    This function delegates to trainer.run_real_experiment for actual training.
    If trainer is not available or fails, it raises a clear exception.
    """
    from src.runners.trainer import run_real_experiment
    return run_real_experiment(run_spec, output_dir)
