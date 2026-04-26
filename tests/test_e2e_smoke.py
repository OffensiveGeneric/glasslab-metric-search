#!/usr/bin/env python3
"""End-to-end smoke test: runs run_experiment.py and validates output."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_e2e_smoke_test():
    """End-to-end smoke test: run_experiment.py with cifar100 config."""
    test_dir = Path(__file__).parent.resolve()
    project_root = test_dir.parent

    # Clean up any previous run
    output_dir = Path("/tmp/glasslab-metric-smoke")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Run experiment
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_experiment.py"),
            "--config",
            str(project_root / "configs" / "search_spaces" / "cifar100_contrastive_v0.yaml"),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )

    assert result.returncode == 0, f"run_experiment failed:\n{result.stderr}"

    # Verify artifacts exist
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "run_spec.json").exists()
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "status.json").exists()
    assert (output_dir / "artifacts_index.json").exists()
    assert (output_dir / "logs" / "runner.log").exists()

    # Verify metrics.json has required fields
    with open(output_dir / "metrics.json") as f:
        metrics = json.load(f)

    required = [
        "run_id",
        "dataset_id",
        "grouped_recall_at_k",
        "opis",
        "adjusted_mutual_info",
        "adjusted_rand_index",
        "normalized_mutual_info",
        "silhouette_score",
        "composite_score",
    ]

    for field in required:
        assert field in metrics, f"Missing field: {field}"

    # Verify metrics in valid ranges
    assert 0.0 <= metrics["grouped_recall_at_k"] <= 1.0
    assert 0.0 <= metrics["opis"] <= 1.0
    assert 0.0 <= metrics["adjusted_mutual_info"] <= 1.0
    assert 0.0 <= metrics["adjusted_rand_index"] <= 1.0
    assert 0.0 <= metrics["normalized_mutual_info"] <= 1.0
    assert -1.0 <= metrics["silhouette_score"] <= 1.0
    assert 0.0 <= metrics["composite_score"] <= 1.0
