#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.run_spec import Budget, DatasetBinding, Resources, RunSpec, utc_now_iso
from search.validation import validate_experiment_config, validate_cifar100_config
from src.runners.experiment import simulate_experiment, simulate_contrastive_experiment


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_artifacts_index(run_dir: Path) -> dict:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_dir():
            artifacts.append(
                {
                    "name": f"{relative}/",
                    "path": str(path),
                    "media_type": "inode/directory",
                    "required": relative == "logs",
                }
            )
            continue
        suffix = path.suffix.lower()
        media_type = {
            ".json": "application/json",
            ".md": "text/markdown",
            ".log": "text/plain",
            ".txt": "text/plain",
        }.get(suffix, "application/octet-stream")
        artifacts.append(
            {
                "name": relative,
                "path": str(path),
                "media_type": media_type,
                "required": relative in {
                    "run_manifest.json",
                    "run_spec.json",
                    "config.json",
                    "metrics.json",
                    "artifacts_index.json",
                    "report.md",
                    "status.json",
                    "logs/runner.log",
                },
            }
        )
    if not any(item["name"] == "artifacts_index.json" for item in artifacts):
        artifacts.append(
            {
                "name": "artifacts_index.json",
                "path": str(run_dir / "artifacts_index.json"),
                "media_type": "application/json",
                "required": True,
            }
        )
    return {"artifacts": artifacts}


def current_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--submitted-by", default="local-cli")
    parser.add_argument("--dataset-type", default="standard", choices=["standard", "contrastive"])
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_id = config.get("dataset", {}).get("dataset_id", "")
    if "cifar100" in dataset_id.lower():
        validate_cifar100_config(config["experiment"])
    else:
        validate_experiment_config(config["experiment"])
    
    dataset = DatasetBinding(**config["dataset"])
    resources = Resources(**config["resources"])
    budget = Budget(**config["budget"])
    run_spec = RunSpec.new(
        base_commit=current_commit(),
        submitted_by=args.submitted_by,
        workflow_family=config["workflow_family"],
        search_space_id=config["search_space_id"],
        dataset=dataset,
        resources=resources,
        budget=budget,
        config=config["experiment"],
        parent_run_id=config.get("parent_run_id"),
    )
    glasslab_run_id = os.environ.get("GLASSLAB_RUNNER_EXPERIMENT_ID", "").strip()
    if glasslab_run_id:
        run_spec.run_id = glasslab_run_id
    run_spec.write_json(run_dir / "run_spec.json")
    write_text(run_dir / "config.json", json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_text(run_dir / "run_manifest.json", json.dumps(run_spec.to_dict(), indent=2, sort_keys=True) + "\n")
    write_text(run_dir / "logs" / "runner.log", "INFO metric-search run started\n")
    
    dataset_type = args.dataset_type
    if "cifar100" in dataset_id.lower():
        dataset_type = "contrastive"
    
    # For CIFAR-100, use contrastive simulation
    dataset_type = args.dataset_type
    if "cifar100" in dataset_id.lower():
        dataset_type = "contrastive"
    
    if dataset_type == "contrastive":
        metrics = simulate_contrastive_experiment(run_spec, run_dir)
        write_text(
            run_dir / "report.md",
            (
                f"# Contrastive Learning Report\n\n"
                f"- run_id: `{run_spec.run_id}`\n"
                f"- search_space_id: `{run_spec.search_space_id}`\n"
                f"- dataset_id: `{dataset_id}`\n"
                f"- grouped_recall_at_k: `{metrics['grouped_recall_at_k']}`\n"
                f"- opis: `{metrics['opis']}`\n"
                f"- ami: `{metrics['adjusted_mutual_info']}`\n"
                f"- ari: `{metrics['adjusted_rand_index']}`\n"
                f"- nmi: `{metrics['normalized_mutual_info']}`\n"
                f"- silhouette: `{metrics['silhouette_score']}`\n"
                f"- composite_score: `{metrics['composite_score']}`\n"
            ),
        )
    else:
        metrics = simulate_experiment(run_spec)
        write_text(
            run_dir / "report.md",
            (
                f"# Metric Search Report\n\n"
                f"- run_id: `{run_spec.run_id}`\n"
                f"- search_space_id: `{run_spec.search_space_id}`\n"
                f"- composite_score: `{metrics['composite_score']}`\n"
            ),
        )
    write_text(
        run_dir / "status.json",
        json.dumps(
            {
                "run_id": run_spec.run_id,
                "status": "succeeded",
                "updated_at": utc_now_iso(),
                "detail": "metric-search contrastive learning workload completed",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    artifacts_index = build_artifacts_index(run_dir)
    write_text(run_dir / "artifacts_index.json", json.dumps(artifacts_index, indent=2, sort_keys=True) + "\n")
    write_text(run_dir / "logs" / "runner.log", "INFO metric-search run completed\n")
    print(f"run_id={run_spec.run_id}")
    print(f"metrics_path={run_dir / 'metrics.json'}")
    print(metrics)


if __name__ == "__main__":
    main()
