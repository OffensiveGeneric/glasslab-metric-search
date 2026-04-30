#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

# Fix FAISS OpenMP deadlock on macOS - MUST be before any other imports
if os.uname().sysname == "Darwin":
    os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.run_spec import Budget, DatasetBinding, Resources, RunSpec, utc_now_iso
from search.validation import validate_cifar100_config, validate_experiment_config
from src.runners.experiment import run_contrastive_experiment


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_log(run_dir: Path, message: str) -> None:
    log_path = run_dir / "logs" / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def write_status(run_dir: Path, status: str, run_id: str, detail: str = "") -> None:
    write_text(
        run_dir / "status.json",
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "updated_at": utc_now_iso(),
                "detail": detail,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_error(run_dir: Path, exception: Exception, traceback_str: str, run_id: str) -> None:
    error_type = type(exception).__name__
    write_text(
        run_dir / "error.json",
        json.dumps(
            {
                "run_id": run_id,
                "error_type": error_type,
                "exception_type": error_type,
                "message": str(exception),
                "error": str(exception),
                "traceback": traceback_str,
                "timestamp": utc_now_iso(),
                "updated_at": utc_now_iso(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_metrics(run_dir: Path, metrics: dict) -> None:
    write_text(run_dir / "metrics.json", json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def write_report(run_dir: Path, metrics: dict, run_spec: RunSpec, dataset_id: str) -> None:
    warnings = metrics.get("sanity_warnings") or []
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No sanity warnings emitted."
    write_text(
        run_dir / "report.md",
        (
            "# Contrastive Learning Report\n\n"
            "## Pipeline Status\n\n"
            "- status: `completed`\n"
            f"- run_id: `{run_spec.run_id}`\n"
            f"- search_space_id: `{run_spec.search_space_id}`\n"
            f"- dataset_id: `{dataset_id}`\n"
            "- artifacts: `metrics.json`, `status.json`, `report.md`, logs, checkpoints, and embeddings were written when available.\n\n"
            "## Model-Quality Status\n\n"
            "The run produced a high raw test_unseen grouped recall, but this must be interpreted against class-count metadata, baselines, and equalized evaluation. Do not claim the model generalizes well from raw grouped recall alone.\n\n"
            "## Split Metrics\n\n"
            f"- val_seen_grouped_recall_at_k: `{metrics.get('val_seen_grouped_recall_at_k')}`\n"
            f"- val_seen_composite_score: `{metrics.get('val_seen_composite_score')}`\n"
            f"- test_seen_grouped_recall_at_k: `{metrics.get('test_seen_grouped_recall_at_k')}`\n"
            f"- test_seen_composite_score: `{metrics.get('test_seen_composite_score')}`\n"
            f"- test_unseen_grouped_recall_at_k: `{metrics.get('test_unseen_grouped_recall_at_k')}`\n"
            f"- test_unseen_composite_score: `{metrics.get('test_unseen_composite_score')}`\n\n"
            "## Sanity Checks\n\n"
            f"- model_quality_interpretable: `{metrics.get('model_quality_interpretable')}`\n"
            f"- test_seen lift vs shuffled labels: `{metrics.get('test_seen_grouped_recall_lift_vs_shuffled_labels')}`\n"
            f"- test_seen lift vs random embeddings: `{metrics.get('test_seen_grouped_recall_lift_vs_random_embeddings')}`\n"
            f"- test_unseen lift vs shuffled labels: `{metrics.get('test_unseen_grouped_recall_lift_vs_shuffled_labels')}`\n"
            f"- test_unseen lift vs random embeddings: `{metrics.get('test_unseen_grouped_recall_lift_vs_random_embeddings')}`\n"
            f"- test_unseen random grouped recall: `{metrics.get('test_unseen_random_embedding_grouped_recall_at_k')}`\n"
            f"- test_unseen random analytic chance: `{metrics.get('test_unseen_random_embedding_grouped_recall_chance_at_k')}`\n"
            f"- test_seen_equalized_grouped_recall_at_k: `{metrics.get('test_seen_equalized_grouped_recall_at_k')}`\n"
            f"- test_unseen_equalized_reference_grouped_recall_at_k: `{metrics.get('test_unseen_equalized_reference_grouped_recall_at_k')}`\n"
            f"- generalization_gap_equalized_grouped_recall_at_k: `{metrics.get('generalization_gap_equalized_grouped_recall_at_k')}`\n\n"
            "## Warnings\n\n"
            f"{warning_lines}\n\n"
            "## Caveat\n\n"
            "A quick smoke-sized run is a pipeline validation, not scientific evidence. Treat surprising unseen-class metrics as hypotheses until they survive the sanity checks and larger repeated runs.\n"
        ),
    )


def write_failure_report(run_dir: Path, run_spec: RunSpec, exception: Exception) -> None:
    write_text(
        run_dir / "report.md",
        (
            "# Contrastive Learning Report\n\n"
            "## Pipeline Status\n\n"
            "- status: `failed`\n"
            f"- run_id: `{run_spec.run_id}`\n"
            f"- error: `{type(exception).__name__}: {exception}`\n\n"
            "## Model-Quality Status\n\n"
            "No model-quality claim is possible because the run failed before producing a complete metrics bundle.\n"
        ),
    )


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
        write_status(
            self.run_dir,
            "succeeded",
            self.run_spec.run_id,
            "metric-search contrastive learning workload completed",
        )
        append_log(self.run_dir, "INFO metric-search run completed")
        self._finalize_artifacts_index()

    def write_failure_bundle(self, exception: Exception, traceback_str: str) -> None:
        self._write_common_files()
        write_error(self.run_dir, exception, traceback_str, self.run_spec.run_id)
        write_failure_report(self.run_dir, self.run_spec, exception)
        append_log(self.run_dir, f"ERROR metric-search run failed: {exception}")
        append_log(self.run_dir, traceback_str)
        write_status(self.run_dir, "failed", self.run_spec.run_id, f"metric-search workload failed: {exception}")
        self._finalize_artifacts_index()

    def _write_common_files(self) -> None:
        config_path = self.run_dir / "config.json"
        if config_path.exists():
            write_text(
                config_path,
                json.dumps(json.loads(config_path.read_text(encoding="utf-8")), indent=2, sort_keys=True) + "\n",
            )
        write_text(
            self.run_dir / "run_manifest.json",
            json.dumps(build_run_manifest(self.run_spec), indent=2, sort_keys=True) + "\n",
        )
        write_text(self.run_dir / "run_spec.json", json.dumps(self.run_spec.to_dict(), indent=2, sort_keys=True) + "\n")

    def _finalize_artifacts_index(self) -> None:
        artifacts_index = build_artifacts_index(self.run_dir)
        write_text(self.run_dir / "artifacts_index.json", json.dumps(artifacts_index, indent=2, sort_keys=True) + "\n")


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
                "required": relative
                in {
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
    image_commit = os.environ.get("GLASSLAB_IMAGE_COMMIT", "").strip()
    if image_commit and image_commit != "unknown":
        return image_commit
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


def git_commit() -> str:
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
        image_commit = os.environ.get("GLASSLAB_IMAGE_COMMIT", "").strip()
        return image_commit if image_commit else "unknown"


def build_run_manifest(run_spec: RunSpec) -> dict:
    repo_commit = git_commit()
    image_commit = os.environ.get("GLASSLAB_IMAGE_COMMIT", "").strip() or "unknown"
    base_commit = run_spec.base_commit
    commit_verified = base_commit != "unknown" and repo_commit == base_commit
    if image_commit != "unknown":
        commit_verified = commit_verified and image_commit == base_commit

    payload = run_spec.to_dict()
    payload["runtime"] = {
        "repo_commit": repo_commit,
        "image_commit": image_commit,
        "commit_verified": commit_verified,
    }
    return payload


def verify_run_manifest_commit(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    runtime = manifest.get("runtime", {})
    if not runtime.get("commit_verified"):
        raise RuntimeError(
            "run_manifest commit verification failed: "
            f"base_commit={manifest.get('base_commit')} "
            f"repo_commit={runtime.get('repo_commit')} "
            f"image_commit={runtime.get('image_commit')}"
        )


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
    if args.epochs is not None:
        experiment_config["max_epochs"] = args.epochs
    if args.backbone is not None:
        experiment_config["backbone_name"] = args.backbone
    if args.loss is not None:
        experiment_config["loss_name"] = args.loss
        experiment_config.pop("loss", None)

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
    write_text(run_dir / "run_manifest.json", json.dumps(build_run_manifest(run_spec), indent=2, sort_keys=True) + "\n")
    append_log(run_dir, "INFO metric-search run started")

    try:
        verify_run_manifest_commit(run_dir)
        force_failure = os.environ.get("GLASSLAB_FORCE_TRAINER_FAILURE", "").strip()
        if force_failure:
            raise RuntimeError(f"GLASSLAB_FORCE_TRAINER_FAILURE is set: {force_failure}")

        metrics = run_contrastive_experiment(run_spec, run_dir)
    except Exception as exc:
        print(f"Error running experiment: {exc}", file=sys.stderr)
        traceback_str = traceback.format_exc()
        print(traceback_str, file=sys.stderr)
        bundle_writer = RunBundleWriter(run_dir, run_spec, dataset_id)
        bundle_writer.write_failure_bundle(exc, traceback_str)
        sys.exit(1)

    bundle_writer = RunBundleWriter(run_dir, run_spec, dataset_id)
    bundle_writer.write_success_bundle(metrics)

    print(f"run_id={run_spec.run_id}")
    print(f"metrics_path={run_dir / 'metrics.json'}")
    print(metrics)


if __name__ == "__main__":
    main()
