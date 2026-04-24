#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.run_spec import load_run_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-type", default="contrastive", choices=["standard", "contrastive"])
    args = parser.parse_args()
    run_spec = load_run_spec(args.run_spec)
    
    if args.dataset_type == "contrastive":
        from src.runners.experiment import simulate_contrastive_experiment
        simulate_contrastive_experiment(run_spec, args.output_dir)
    else:
        from src.runners.experiment import simulate_experiment
        simulate_experiment(run_spec, args.output_dir)


if __name__ == "__main__":
    main()
