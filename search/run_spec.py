from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Budget:
    max_epochs: int
    max_wallclock_minutes: int


@dataclass
class Resources:
    gpu_count: int
    cpu_count: int
    memory_gb: int


@dataclass
class DatasetBinding:
    dataset_id: str
    split_version: str
    train_uri: str
    val_uri: str
    test_uri: str


@dataclass
class ArtifactRefs:
    checkpoint_uri: str | None = None
    metrics_uri: str | None = None
    embeddings_uri: str | None = None
    plots_uri: str | None = None
    report_uri: str | None = None


@dataclass
class RunSpec:
    run_id: str
    parent_run_id: str | None
    base_commit: str
    created_at: str
    submitted_by: str
    workflow_family: str
    search_space_id: str
    dataset: DatasetBinding
    resources: Resources
    budget: Budget
    config: dict[str, Any]
    artifact_refs: ArtifactRefs = field(default_factory=ArtifactRefs)

    @classmethod
    def new(
        cls,
        *,
        base_commit: str,
        submitted_by: str,
        workflow_family: str,
        search_space_id: str,
        dataset: DatasetBinding,
        resources: Resources,
        budget: Budget,
        config: dict[str, Any],
        parent_run_id: str | None = None,
    ) -> "RunSpec":
        return cls(
            run_id=f"run-{uuid4().hex[:12]}",
            parent_run_id=parent_run_id,
            base_commit=base_commit,
            created_at=utc_now_iso(),
            submitted_by=submitted_by,
            workflow_family=workflow_family,
            search_space_id=search_space_id,
            dataset=dataset,
            resources=resources,
            budget=budget,
            config=config,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_run_spec(path: Path) -> RunSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunSpec(
        run_id=payload["run_id"],
        parent_run_id=payload.get("parent_run_id"),
        base_commit=payload["base_commit"],
        created_at=payload["created_at"],
        submitted_by=payload["submitted_by"],
        workflow_family=payload["workflow_family"],
        search_space_id=payload["search_space_id"],
        dataset=DatasetBinding(**payload["dataset"]),
        resources=Resources(**payload["resources"]),
        budget=Budget(**payload["budget"]),
        config=payload["config"],
        artifact_refs=ArtifactRefs(**payload.get("artifact_refs", {})),
    )
