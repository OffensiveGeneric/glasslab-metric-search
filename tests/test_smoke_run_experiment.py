from __future__ import annotations

import pytest
import json
import subprocess
import sys
from pathlib import Path


def test_run_experiment_artifacts(tmp_path):
    """Smoke test: CLI entrypoint produces required artifacts"""
    config_path = Path("configs/search_spaces/cifar100_contrastive_v0.yaml")
    output_dir = tmp_path / "test-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
            "--backbone",
            "resnet18",
            "--loss",
            "contrastive",
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    
    assert (output_dir / "status.json").exists()
    status_json = json.loads((output_dir / "status.json").read_text())
    assert status_json["status"] == "succeeded"
    
    assert (output_dir / "metrics.json").exists()
    metrics_json = json.loads((output_dir / "metrics.json").read_text())
    assert metrics_json["run_id"] == status_json["run_id"]
    assert "grouped_recall_at_k" in metrics_json
    assert "opis" in metrics_json
    assert "adjusted_mutual_info" in metrics_json
    assert "adjusted_rand_index" in metrics_json
    assert "normalized_mutual_info" in metrics_json
    assert "silhouette_score" in metrics_json


def test_run_experiment_failure_bundle(tmp_path, monkeypatch):
    """Test that a failing CLI run produces failure bundle"""
    import yaml
    
    config_path = tmp_path / "bad-config.yaml"
    config = {
        "workflow_family": "contrastive",
        "search_space_id": "cifar100_contrastive_v0",
        "dataset": {
            "dataset_id": "cifar100",
            "split_version": "v1",
            "train_uri": "s3://dummy/train",
            "val_uri": "s3://dummy/val",
            "test_uri": "s3://dummy/test",
        },
        "resources": {"gpu_count": 1, "cpu_count": 4, "memory_gb": 32},
        "budget": {"max_epochs": 2, "max_wallclock_minutes": 60},
        "experiment": {
            "backbone_name": "resnet18",
            "loss_name": "contrastive",
        },
    }
    config_path.write_text(yaml.dump(config))
    
    output_dir = tmp_path / "test-output-failure"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def fake_trainer(*args, **kwargs):
        raise RuntimeError("Simulated trainer failure")
    
    monkeypatch.setattr("src.runners.trainer.run_real_experiment", fake_trainer)
    
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode != 0
    assert result.returncode != 0
    
    assert (output_dir / "status.json").exists()
    status_json = json.loads((output_dir / "status.json").read_text())
    assert status_json["status"] == "failed"
    
    assert (output_dir / "error.json").exists()
    error_json = json.loads((output_dir / "error.json").read_text())
    assert error_json["exception_type"] == "RuntimeError"
    assert "Simulated trainer failure" in error_json["message"]
    
    assert (output_dir / "logs" / "runner.log").exists()
