# Glasslab Run Fabric Follow-Ups 2026-04

Status: suggested implementation plan

Date: 2026-04-28

## Scope

This document lists the metric-search changes needed for the Glasslab run fabric
to treat this repo as the golden GPU workload.

It intentionally does not cover literature search or experimental-design
generation. The goal here is a reliable workload contract between
`glasslab-metric-search` and `cluster-config`.

## Target Contract

One metric-search candidate run should map to one Kubernetes Job.

The runner should:

- read a config file or generated config payload
- accept `GLASSLAB_RUNNER_EXPERIMENT_ID`
- write all outputs under `/mnt/artifacts/{run_id}` or the provided
  `--output-dir`
- produce a terminal bundle for success and failure
- exit nonzero on failed experiment execution
- never emit fake or simulated metrics in the production path

The cluster should:

- create the Job
- mount datasets read-only
- mount artifacts read-write
- track Job status
- ingest the terminal bundle into Postgres
- keep large artifacts on `.207`

## Current Blocking Issues

### Runner Failure Semantics

`scripts/run_experiment.py` catches experiment exceptions, logs a traceback, and
sets `metrics = {}`. The code then renders `report.md` using required metric
keys such as `grouped_recall_at_k`, `adjusted_mutual_info`,
`adjusted_rand_index`, `normalized_mutual_info`, and `silhouette_score`.

That means a real failure can become a secondary `KeyError` before a clean
failed bundle is written.

Required change:

- write `status.json` with `status: failed`
- write `error.json` with exception type, message, and traceback
- write `artifacts_index.json`
- append to `logs/runner.log`
- return a nonzero exit code

### Metric Key Mismatch

`src/metrics/metrics.py` currently emits:

- `nmi`
- `ami`
- `ari`
- `silhouette`

The runner/report/tests expect:

- `normalized_mutual_info`
- `adjusted_mutual_info`
- `adjusted_rand_index`
- `silhouette_score`

Required change:

- choose the long names as the Glasslab artifact contract
- optionally include short aliases for backward compatibility
- compute `composite_score` only after required long names are present

### Log Clobbering

`scripts/run_experiment.py` writes `logs/runner.log` at start and overwrites it
at completion.

Required change:

- append log lines
- preserve error tracebacks
- include terminal status in the log

### Tests Reference Removed Simulation Path

`tests/test_smoke_run_experiment.py` imports `simulate_contrastive_experiment`,
but the current implementation only exposes real execution.

Required change:

- remove the simulation test path
- replace it with a small contract test that monkeypatches the trainer function
  to return deterministic real-shaped metrics
- keep production code free of simulated metrics

### Registry Image Drift

The cluster registry currently points at an older metric-search image tag. The
cluster should not update that tag until this repo can produce a passing
contract bundle from the latest commit.

Required change in this repo:

- make the runner contract pass locally
- build a new image tag from the fixed commit
- publish the tag

Required change in `cluster-config` after that:

- update `services/workflow-registry/definitions/metric-search-v0.json`
- smoke submit a tiny GPU run

## Required Terminal Bundle

Success:

```text
{output_dir}/
  run_spec.json
  run_manifest.json
  config.json
  metrics.json
  report.md
  status.json
  artifacts_index.json
  logs/runner.log
```

Failure:

```text
{output_dir}/
  run_spec.json
  run_manifest.json
  config.json
  status.json
  artifacts_index.json
  logs/runner.log
  error.json
  report.md optional
```

`status.json` should be the source of truth for the terminal state:

```json
{
  "run_id": "example-run",
  "status": "failed",
  "updated_at": "2026-04-28T00:00:00Z",
  "detail": "metric-search workload failed: RuntimeError: example"
}
```

## Metrics Contract

Required numeric metrics:

- `grouped_recall_at_k`
- `opis`
- `adjusted_mutual_info`
- `adjusted_rand_index`
- `normalized_mutual_info`
- `silhouette_score`
- `composite_score`

Required identifiers:

- `run_id`
- `dataset_id`
- `mode`
- `simulated`

Production runs should set:

```json
{
  "mode": "real",
  "simulated": false
}
```

Do not write `metrics.json` for a failed run unless the metrics are real partial
metrics with explicit `status: failed` context. A clean `error.json` is better
than empty metrics.

## Suggested Implementation Steps

1. Add a small `RunBundleWriter` helper.
2. Make all artifact writes go through that helper.
3. Add `append_log`, `write_status`, `write_error`, `write_metrics`,
   `write_report`, and `write_artifacts_index`.
4. Wrap `main()` so exceptions write a failed bundle and then `sys.exit(1)`.
5. Normalize metric names in `AdvancedMetrics.compute_all_metrics`.
6. Update report generation to tolerate failure mode.
7. Add tests for success and failure bundles without downloading CIFAR-100.
8. Add an image-build smoke command to validate the container entrypoint.

## Test Plan

Fast local tests:

- unit test metric key normalization
- unit test artifact index generation
- unit test failed bundle generation by monkeypatching the trainer to raise
- unit test success bundle generation by monkeypatching the trainer to return
  deterministic real-shaped metrics

Container smoke:

```bash
docker build -t glasslab-metric-search:contract-smoke .
docker run --rm \
  -e GLASSLAB_RUNNER_EXPERIMENT_ID=contract-smoke \
  -v /tmp/glasslab-metric-contract:/mnt/artifacts \
  glasslab-metric-search:contract-smoke \
  python3 scripts/run_experiment.py \
    --config configs/search_spaces/cifar100_contrastive_v0.yaml \
    --output-dir /mnt/artifacts/contract-smoke \
    --epochs 1 \
    --max-train-batches 1 \
    --max-eval-batches 1
```

Cluster smoke after publishing a fixed image:

1. Submit `metric-search-v0` with a tiny budget.
2. Confirm the Job schedules onto a GPU node.
3. Confirm `/mnt/artifacts/{run_id}` lands on `.207`.
4. Confirm `status.json` is terminal.
5. Confirm `metrics.json` has the required metric keys on success.
6. Confirm `workflow-api` ingests artifact references into Postgres.
7. Confirm comparison can rank the run by `composite_score`.

## Cluster Repo Follow-Ups

After this repo is fixed:

- update `metric-search-v0` runner image to the fixed commit tag
- add artifact-ingest endpoint or reconciler support
- add a shared run-bundle validator to prevent contract drift
- keep large outputs on `.207`; do not make MinIO mandatory for this workflow

## Done Criteria

This repo is ready for Glasslab run-fabric use when:

- a failing trainer produces a failed terminal bundle and exits nonzero
- a successful tiny run produces the required success bundle
- metric keys match the documented contract
- tests no longer depend on simulated production metrics
- the Docker image entrypoint can run the contract smoke
- the cluster can ingest the resulting bundle without special-case knowledge of
  metric-search internals
