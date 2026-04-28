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
from src.runners.experiment import run_contrastive_experiment


def append_log(run_dir: Path, message: str) -> None:
    log_path = run_dir / "logs" / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def write_status(run_dir: Path, status: str, run_id: str, detail: str = "") -> None:
    status_path = run_dir / "status.json"
    status_data = {
        "run_id": run_id,
        "status": status,
        "updated_at": utc_now_iso(),
        "detail": detail,
    }
    write_text(run_dir / "status.json", json.dumps(status_data, indent=2, sort_keys=True) + "\n")


def write_error(run_dir: Path, exception: Exception, traceback_str: str, run_id: str) -> None:
    error_path = run_dir / "error.json"
    error_data = {
        "run_id": run_id,
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "traceback": traceback_str,
        "timestamp": utc_now_iso(),
    }
    write_text(run_dir / "error.json", json.dumps(error_data, indent=2, sort_keys=True) + "\n")


def write_metrics(run_dir: Path, metrics: dict) -> None:
    write_text(run_dir / "metrics.json", json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def write_report(run_dir: Path, metrics: dict, run_spec: RunSpec, dataset_id: str) -> None:
    report_path = run_dir / "report.md"
    report_content = (
        f"# Contrastive Learning Report\n\n"
        f"- run_id: `{run_spec.run_id}`\n"
        f"- search_space_id: `{run_spec.search_space_id}`\n"
        f"- dataset_id: `{dataset_id}`\n"
        f"- grouped_recall_at_k: `{metrics.get('grouped_recall_at_k')}`\n"
        f"- opis: `{metrics.get('opis')}`\n"
        f"- adjusted_mutual_info: `{metrics.get('adjusted_mutual_info')}`\n"
        f"- adjusted_rand_index: `{metrics.get('adjusted_rand_index')}`\n"
        f"- normalized_mutual_info: `{metrics.get('normalized_mutual_info')}`\n"
        f"- silhouette_score: `{metrics.get('silhouette_score')}`\n"
        f"- composite_score: `{metrics.get('composite_score')}`\n"
        f"- mode: {metrics.get('mode', 'real')}\n"
        f"- simulated: {metrics.get('simulated', False)}\n"
    )
    write_text(report_path, report_content)


class RunBundleWriter:
    """Helper to write terminal bundles for experiment runs."""

    def __init__(self, run_dir: Path, run_spec: RunSpec, dataset_id: str):
        self.run_dir = run_dir
        self.run_spec = run_spec
        self.dataset_id = dataset_id

    def write_success_bundle(self, metrics: dict) -> None:
        self._write_common_files()
        write_metrics(self.run_dir, metrics)
        write_report(self.run_dir, metrics, self.run_spec, self.dataset_id)
        write_status(self.run_dir, "succeeded", self.run_spec.run_id, "metric-search contrastive learning workload completed")
        self._finalize_artifacts_index()

    def write_failure_bundle(self, exception: Exception, traceback_str: str) -> None:
        self._write_common_files()
        write_error(self.run_dir, exception, traceback_str, self.run_spec.run_id)
        append_log(self.run_dir, f"ERROR: {exception}")
        append_log(self.run_dir, traceback_str)
        write_status(self.run_dir, "failed", self.run_spec.run_id, f"metric-search workload failed: {exception}")
        self._finalize_artifacts_index()

    def _write_common_files(self) -> None:
        write_text(
            self.run_dir / "config.json",
            json.dumps(json.loads((self.run_dir / "config.json").read_text()), indent=2, sort_keys=True) + "\n",
        )
        write_text(
            self.run_dir / "run_manifest.json",
            json.dumps(self.run_spec.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        write_text(self.run_dir / "run_spec.json", json.dumps(self.run_spec.to_dict(), indent=2, sort_keys=True) + "\n")
        append_log(self.run_dir, f"INFO: experiment {'completed' if (self.run_dir / 'metrics.json').exists() else 'failed'}")

    def _finalize_artifacts_index(self) -> None:
        artifacts_index = build_artifacts_index(self.run_dir)
        write_text(self.run_dir / "artifacts_index.json", json.dumps(artifacts_index, indent=2, sort_keys=True) + "\n")


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
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Override max train batches")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Override max eval batches")
    parser.add_argument("--backbone", default=None, help="Override backbone name")
    parser.add_argument("--loss", default=None, help="Override loss name")
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
    
    budget_config = config["budget"].copy()
    if args.epochs is not None:
        budget_config["max_epochs"] = args.epochs
    
    budget = Budget(**budget_config)
    if args.max_train_batches is not None:
        budget.max_train_batches = args.max_train_batches
    if args.max_eval_batches is not None:
        budget.max_eval_batches = args.max_eval_batches
    
    experiment_config = config["experiment"].copy()
    if args.backbone is not None:
        experiment_config["backbone_name"] = args.backbone
    if args.loss is not None:
        experiment_config["loss_name"] = args.loss
    
    run_spec = RunSpec.new(
        base_commit=current_commit(),
        submitted_by=args.submitted_by,
        workflow_family=config["workflow_family"],
        search_space_id=config["search_space_id"],
        dataset=dataset,
        resources=resources,
        budget=budget,
        config=experiment_config,
        parent_run_id=config.get("parent_run_id"),
    )
    glasslab_run_id = os.environ.get("GLASSLAB_RUNNER_EXPERIMENT_ID", "").strip()
    if glasslab_run_id:
        run_spec.run_id = glasslab_run_id
    run_spec.write_json(run_dir / "run_spec.json")
    write_text(run_dir / "config.json", json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_text(run_dir / "run_manifest.json", json.dumps(run_spec.to_dict(), indent=2, sort_keys=True) + "\n")
    append_log(run_dir, "INFO metric-search run started")

    try:
        metrics = run_contrastive_experiment(run_spec, run_dir)
    except Exception as e:
        print(f"Error running experiment: {e}", file=sys.stderr)
        import traceback
        tb_str = traceback.format_exc()
        print(tb_str, file=sys.stderr)
        bundle_writer = RunBundleWriter(run_dir, run_spec, dataset_id)
        bundle_writer.write_failure_bundle(e, tb_str)
        sys.exit(1)

    append_log(run_dir, "INFO metric-search run completed")
    bundle_writer = RunBundleWriter(run_dir, run_spec, dataset_id)
    bundle_writer.write_success_bundle(metrics)

    print(f"run_id={run_spec.run_id}")
    print(f"metrics_path={run_dir / 'metrics.json'}")
    print(metrics)


if __name__ == "__main__":
    main()
