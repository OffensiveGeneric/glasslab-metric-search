#!/usr/bin/env python3
"""
Inspect run metrics and produce a diagnostic summary.
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.inspect_run_metrics <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    metrics_path = output_dir / "metrics.json"

    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found", file=sys.stderr)
        sys.exit(1)

    with metrics_path.open("r") as f:
        metrics = json.load(f)

    mode = metrics.get("mode", "unknown")
    simulated = metrics.get("simulated", False)
    run_id = metrics.get("run_id", "unknown")
    dataset_id = metrics.get("dataset_id", "unknown")

    print(f"Mode: {mode}")
    print(f"Simulated: {simulated}")
    print(f"Run ID: {run_id}")
    print(f"Dataset ID: {dataset_id}")
    print()

    splits = ["train_seen", "val_seen", "test_seen", "test_unseen"]
    for split in splits:
        num_samples = metrics.get(f"{split}_num_samples")
        num_classes = metrics.get(f"{split}_num_classes")
        print(f"{split}:")
        print(f"  samples: {num_samples}")
        print(f"  classes: {num_classes}")

    print()

    for split in ("test_seen", "test_unseen"):
        print(f"{split}:")
        global_recall = metrics.get(f"{split}_global_recall_at_1")
        global_chance_exact = metrics.get(f"{split}_global_recall_at_1_chance_exact")
        global_chance_approx = metrics.get(f"{split}_global_recall_at_1_chance_approx")
        random_global_recall = metrics.get(f"{split}_random_embedding_global_recall_at_1")
        random_global_chance_exact = metrics.get(f"{split}_random_embedding_global_recall_at_1_chance_exact")
        random_grouped_recall = metrics.get(f"{split}_random_embedding_grouped_recall_at_k")
        grouped_recall = metrics.get(f"{split}_grouped_recall_at_k")
        grouped_chance = metrics.get(f"{split}_grouped_recall_chance_at_k")
        random_grouped_chance = metrics.get(f"{split}_random_embedding_grouped_recall_chance_at_k")

        print(f"  global Recall@1: {global_recall}")
        print(f"  global Recall@1 chance (exact): {global_chance_exact}")
        print(f"  global Recall@1 chance (approx): {global_chance_approx}")
        print(f"  random global Recall@1: {random_global_recall}")
        print(f"  random global Recall@1 chance (exact): {random_global_chance_exact}")
        print(f"  random grouped Recall@K: {random_grouped_recall}")
        print(f"  grouped Recall@K: {grouped_recall}")
        print(f"  grouped Recall@K chance: {grouped_chance}")
        print(f"  random grouped Recall@K chance: {random_grouped_chance}")

        if random_global_recall is not None and random_global_chance_exact is not None:
            error = abs(float(random_global_recall) - float(random_global_chance_exact))
            pass_flag = error <= 0.03
            print(f"  random global Recall@1 abs error: {error:.4f}")
            print(f"  random global Recall@1 sanity pass: {pass_flag}")

        if random_grouped_recall is not None and random_grouped_chance is not None:
            error = abs(float(random_grouped_recall) - float(random_grouped_chance))
            print(f"  random grouped Recall@K abs error: {error:.4f}")

        print()

    sanity_warnings = metrics.get("sanity_warnings", [])
    if sanity_warnings:
        print("Sanity warnings:")
        for warning in sanity_warnings:
            print(f"  - {warning}")
    else:
        print("Sanity warnings: None")

    artifact_dir = output_dir / "artifacts"
    artifacts_exist = artifact_dir.exists()
    print(f"Artifacts directory exists: {artifacts_exist}")

    if artifacts_exist:
        artifact_files = list(artifact_dir.glob("*"))
        print(f"Artifact files:")
        for f in artifact_files:
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
