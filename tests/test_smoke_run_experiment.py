from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from src.config import Config
from src.runners.trainer import (
    add_summary_aliases,
    baseline_sane,
    class_count_metadata,
    evaluate_embeddings,
    gallery_metadata,
    shuffled_label_baseline,
)


def test_class_count_metadata_from_labels() -> None:
    labels = torch.tensor([1, 1, 1, 2, 2, 3])

    metadata = class_count_metadata(labels, "test_split")

    assert metadata["test_split_num_samples"] == 6
    assert metadata["test_split_num_classes"] == 3
    assert metadata["test_split_min_samples_per_class"] == 1
    assert metadata["test_split_max_samples_per_class"] == 3
    assert metadata["test_split_mean_samples_per_class"] == 2.0


def test_gallery_metadata_marks_partial_eval() -> None:
    class Loader:
        batch_size = 4
        dataset = list(range(20))

    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    metadata = gallery_metadata(labels, "test_split", Loader(), max_batches=2)

    assert metadata["test_split_gallery_num_samples"] == 8
    assert metadata["test_split_gallery_num_classes"] == 4
    assert metadata["test_split_gallery_total_dataset_samples"] == 20
    assert metadata["test_split_gallery_max_eval_batches"] == 2
    assert metadata["test_split_gallery_partial"] is True


def test_baseline_sane_compares_to_analytic_chance() -> None:
    assert baseline_sane(0.24, 0.20)
    assert not baseline_sane(0.67, 0.05)


def test_shuffled_label_baseline_reduces_cluster_agreement() -> None:
    config = Config()
    config.data.seed = 42
    config.evaluation.k = 5
    config.evaluation.num_groups = 2

    left = torch.zeros(20, 4)
    right = torch.ones(20, 4) * 10
    embeddings = torch.cat([left, right])
    labels = torch.tensor([0] * 20 + [1] * 20)
    warnings: list[str] = []

    real = evaluate_embeddings(embeddings, labels, config, warnings, "clustered synthetic")
    shuffled = shuffled_label_baseline(
        embeddings,
        labels,
        config,
        warnings,
        "clustered synthetic shuffled labels",
    )

    assert real["adjusted_rand_index"] > 0.9
    assert shuffled["adjusted_rand_index"] < real["adjusted_rand_index"]


def test_top_level_aliases_copy_from_test_unseen_metrics() -> None:
    metrics = {
        "test_unseen_grouped_recall_at_k": 0.8,
        "test_unseen_opis": 0.2,
        "test_unseen_adjusted_mutual_info": 0.3,
        "test_unseen_adjusted_rand_index": 0.4,
        "test_unseen_normalized_mutual_info": 0.5,
        "test_unseen_silhouette_score": -0.1,
        "test_unseen_composite_score": 0.45,
    }

    add_summary_aliases(metrics)

    assert metrics["grouped_recall_at_k"] == 0.8
    assert metrics["opis"] == 0.2
    assert metrics["adjusted_mutual_info"] == 0.3
    assert metrics["adjusted_rand_index"] == 0.4
    assert metrics["normalized_mutual_info"] == 0.5
    assert metrics["silhouette_score"] == -0.1
    assert metrics["composite_score"] == 0.45


def test_run_experiment_artifacts(tmp_path: Path) -> None:
    """Smoke test: CLI entrypoint produces required success artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "search_spaces" / "cifar100_contrastive_v0.yaml"
    output_dir = tmp_path / "test-output"
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_experiment.py"),
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
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert status["status"] == "succeeded"
    assert metrics["run_id"] == status["run_id"]
    assert metrics["dataset_id"] == "cifar100-unseen-classes"
    assert metrics["simulated"] is False
    assert "test_unseen_grouped_recall_at_k" in metrics
    assert "test_unseen_gallery_num_samples" in metrics
    assert "test_unseen_gallery_partial" in metrics
    assert "test_unseen_random_embedding_grouped_recall_chance_at_k" in metrics
    assert "test_unseen_grouped_recall_lift_vs_random_embeddings" in metrics
    assert "model_quality_interpretable" in metrics
    assert "test_seen_equalized_grouped_recall_at_k" in metrics
    assert "generalization_gap_equalized_grouped_recall_at_k" in metrics
    assert (output_dir / "artifacts_index.json").exists()
    assert (output_dir / "logs" / "runner.log").exists()


def test_failure_bundle_forced_trainer_failure(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "forced-failure"
    config_path = project_root / "configs" / "search_spaces" / "cifar100_contrastive_v0.yaml"
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))

    env = os.environ.copy()
    env["GLASSLAB_FORCE_TRAINER_FAILURE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_experiment.py"),
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
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    error = json.loads((output_dir / "error.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert error["error_type"] == "RuntimeError"
    assert error["exception_type"] == "RuntimeError"
    assert "GLASSLAB_FORCE_TRAINER_FAILURE" in error["message"]
    assert (output_dir / "report.md").exists()
    assert (output_dir / "artifacts_index.json").exists()
    assert (output_dir / "logs" / "runner.log").exists()
