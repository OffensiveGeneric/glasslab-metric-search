from __future__ import annotations

from pathlib import Path
import sys

from search.run_spec import RunSpec


def run_contrastive_experiment(run_spec: RunSpec, output_dir: Path) -> dict[str, Any]:
    """Run a real contrastive learning experiment.
    
    This function delegates to trainer.run_real_experiment for actual training.
    If trainer is not available or fails, it raises a clear exception.
    """
    try:
        print(f"Starting run_contrastive_experiment", file=sys.stderr)
        from src.runners.trainer import run_real_experiment
        result = run_real_experiment(run_spec, output_dir)
        print(f"Done run_contrastive_experiment, got keys: {list(result.keys())}", file=sys.stderr)
        return result
    except Exception as ex:
        print(f"Error in run_contrastive_experiment: {ex}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
