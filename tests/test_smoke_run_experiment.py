from __future__ import annotations

import pytest
import json
from pathlib import Path

from search.run_spec import RunSpec, Budget, DatasetBinding, Resources
from src.runners.experiment import simulate_contrastive_experiment


def test_run_experiment_artifacts(tmp_path):
    """Smoke test: run_experiment produces required artifacts"""
    from search.run_spec import RunSpec, Budget, DatasetBinding, Resources

    run_spec = RunSpec.new(
        base_commit="abc123",
        submitted_by="smoke-test",
        workflow_family="contrastive",
        search_space_id="cifar100_contrastive_v0",
        dataset=DatasetBinding(
            dataset_id="cifar100",
            split_version="v1",
            train_uri="s3://dummy/train",
            val_uri="s3://dummy/val",
            test_uri="s3://dummy/test",
        ),
        resources=Resources(gpu_count=1, cpu_count=4, memory_gb=32),
        budget=Budget(max_epochs=2, max_wallclock_minutes=60),
        config={"backbone_name": "resnet18", "loss_name": "contrastive"},
    )
    
    output_dir = tmp_path / "test-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = simulate_contrastive_experiment(run_spec, output_dir)
    
    assert "run_id" in metrics
    assert "composite_score" in metrics
    
    assert (output_dir / "metrics.json").exists()
    
    metrics_json = json.loads((output_dir / "metrics.json").read_text())
    assert metrics_json["run_id"] == run_spec.run_id
    assert "grouped_recall_at_k" in metrics_json
    assert "opis" in metrics_json
    assert "adjusted_mutual_info" in metrics_json
    assert "adjusted_rand_index" in metrics_json
    assert "normalized_mutual_info" in metrics_json
    assert "silhouette_score" in metrics_json
