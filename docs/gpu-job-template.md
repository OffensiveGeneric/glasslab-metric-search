# GPU Job Template for metric-search Workflow Family

This document describes the Kubernetes Job template for running metric-search experiments on Glasslab.

## Overview

Each metric-search run executes as a Kubernetes Job that:
- Uses the metric-search container image
- Injects run ID via `GLASSLAB_RUNNER_EXPERIMENT_ID`
- Mounts dataset volumes from shared storage
- Requests GPU resources (1 GPU, ~8 CPU, ~32Gi memory)
- Writes outputs to run-specific artifact directories

## Job Template Structure

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: metric-search-<run_id_hash>
  namespace: glasslab-v2
  labels:
    app.kubernetes.io/name: metric-search
    glasslab.io/run-id: <run_id>
    glasslab.io/workflow-id: metric-search-v0
    glasslab.io/workload-id: metric-search-v0
spec:
  template:
    spec:
      runtimeClassName: nvidia
      nodeSelector:
        glasslab.io/gpu-candidate: "true"
        glasslab.io/gpu-vendor: "nvidia"
      containers:
        - name: metric-search-runner
          image: ghcr.io/offensivegeneric/glasslab-metric-search:<commit_sha>
          imagePullPolicy: IfNotPresent
          command: ["python3", "scripts/run_experiment.py"]
          args:
            - "--config"
            - "/app/configs/search_spaces/art_metric_baseline.yaml"
            - "--output-dir"
            - "/mnt/artifacts/<run_id>"
          env:
            - name: GLASSLAB_RUNNER_EXPERIMENT_ID
              value: <run_id>
            - name: GLASSLAB_RUNNER_ARTIFACTS_ROOT
              value: /mnt/artifacts
            - name: GLASSLAB_DATASET_ROOT
              value: /datasets
            - name: CONFIG_PATH
              value: /app/configs/search_spaces/art_metric_baseline.yaml
          resources:
            requests:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "16"
              memory: "64Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: dataset-volume
              mountPath: /datasets
              readOnly: true
            - name: artifacts-volume
              mountPath: /mnt/artifacts
            - name: config-volume
              mountPath: /app/configs/search_spaces
              readOnly: true
      volumes:
        - name: dataset-volume
          persistentVolumeClaim:
            claimName: glasslab-shared-datasets
        - name: artifacts-volume
          persistentVolumeClaim:
            claimName: glasslab-shared-artifacts
        - name: config-volume
          configMap:
            name: metric-search-config-<run_id>
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400
```

## Dataset Configuration

### Dataset Mounting Strategy

Datasets are mounted from shared PVCs:

```yaml
volumes:
  - name: dataset-volume
    persistentVolumeClaim:
      claimName: glasslab-shared-datasets
```

Dataset URIs from run spec (e.g., `s3://datasets/artbench/train.parquet`) are resolved to local paths:
- `s3://datasets/artbench/` → `/datasets/artbench/`

### ConfigMap for Search Space configs

Configs are mounted via ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: metric-search-config-<run_id>
  namespace: glasslab-v2
data:
  art_metric_baseline.yaml: |-
    workflow_family: metric-search
    search_space_id: art-metric-baseline-v1
    dataset:
      dataset_id: artbench-v1
      split_version: split-2026-04-a
      train_uri: s3://datasets/artbench/train.parquet
      val_uri: s3://datasets/artbench/val.parquet
      test_uri: s3://datasets/artbench/test.parquet
    resources:
      gpu_count: 1
      cpu_count: 8
      memory_gb: 32
    budget:
      max_epochs: 25
      max_wallclock_minutes: 180
    experiment:
      # ... experiment config
    search_space:
      # ... search space config
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GLASSLAB_RUNNER_EXPERIMENT_ID` | The workflow-api run ID for this job |
| `GLASSLAB_RUNNER_ARTIFACTS_ROOT` | Root path for artifact output |
| `GLASSLAB_DATASET_ROOT` | Root path for dataset inputs |
| `CONFIG_PATH` | Path to the search space config file |
| `GLASSLAB_RUNNER_MANIFEST_JSON` | Full run manifest (optional) |

## Resource Requirements

### Request (guaranteed minimum)

- CPU: 8 cores
- Memory: 32Gi
- GPU: 1 (nvidia.com/gpu)

### Limit (maximum allowed)

- CPU: 16 cores
- Memory: 64Gi
- GPU: 1 (nvidia.com/gpu)

## Artifact Output Structure

Each run produces the following artifacts:

```
/mnt/artifacts/<run_id>/
├── run_spec.json          # Complete run specification
├── run_manifest.json      # Run manifest from workflow-api
├── config.json           # Expanded config
├── metrics.json          # Experiment metrics
├── status.json           # Run status (succeeded/failed)
├── report.md             # Human-readable report
├── artifacts_index.json  # Index of all artifacts
└── logs/
    └── runner.log        # Runner execution log
```

## Workflow Integration

### 1. Job Creation (by workflow-api)

When a metric-search run is submitted:

1. workflow-api validates the run spec
2. Creates a ConfigMap with the search space config
3. Creates the Kubernetes Job with:
   - Image tag matching the run's base commit
   - Environment variables with run metadata
   - Volume mounts for datasets and artifacts
4. Submits the job to the `glasslab-v2` namespace

### 2. Job Execution (runner pod)

1. Pod starts with metric-search image
2. Runner reads `GLASSLAB_RUNNER_EXPERIMENT_ID`
3. Runner loads config from mounted ConfigMap
4. Runner executes: `python3 scripts/run_experiment.py --config <config> --output-dir <output_path>`
5. Runner writes all artifacts to run-specific directory

### 3. Job Completion

1. Job succeeds and pod exits
2. workflow-api polls job status via Kubernetes API
3. On success, runner ingests artifacts back to workflow-api:
   - `status.json` → run status
   - `metrics.json` → metrics storage (Postgres)
   - `artifacts_index.json` → artifact URIs (MinIO)
   - Other artifacts → shared storage

## Implementation Checklist

- [ ] Create Kubernetes Job manifest for metric-search
- [ ] Add ConfigMap generation for search space configs
- [ ] Configure volume mounts for datasets and artifacts
- [ ] Set appropriate resource requests/limits
- [ ] Configure environment variables for run metadata
- [ ] Set up job status polling and completion handling
- [ ] Implement artifact ingestion on job completion
- [ ] Test with sample metric-search config
- [ ] Update workflow-api to handle metric-search workflow

## Testing

### Local Testing

```bash
# Apply job manifest
kubectl apply -f metric-search-job.yaml

# Check status
kubectl get jobs -n glasslab-v2
kubectl get pods -n glasslab-v2 -l job-name=metric-search-...

# View logs
kubectl logs -n glasslab-v2 -l job-name=metric-search-...

# Check artifacts
kubectl exec -n glasslab-v2 <pod-name> -- ls -la /mnt/artifacts/
```

### Integration Testing

```bash
# Submit run via workflow-api
curl -X POST http://localhost:8000/experiments/runs \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "Test metric-search GPU job",
    "experiment_type": "gpu-training-job",
    "workload_id": "metric-search-v0",
    "entrypoint": ["python3", "scripts/run_experiment.py", "--config", "configs/search_spaces/art_metric_baseline.yaml", "--output-dir", "/mnt/artifacts/test-run-1"],
    "config_payload": {"search_space_id": "art-metric-baseline"},
    "dataset_bindings": {"train_uri": "s3://datasets/art/train.parquet"},
    "budget": {"max_epochs": 1, "max_wallclock_minutes": 5},
    "submitted_by": "test-user"
  }'
```

## Troubleshooting

### Pod fails to start

- Check GPU node availability: `kubectl get nodes -l glasslab.io/gpu-candidate=true`
- Verify image pull: `kubectl describe pod -n glasslab-v2 <pod-name>`
- Check resource quotas: `kubectl describe resourcequota -n glasslab-v2`

### Job hangs or times out

- Check job events: `kubectl describe job -n glasslab-v2 <job-name>`
- View pod logs: `kubectl logs -n glasslab-v2 <pod-name>`
- Verify dataset mounts: `kubectl exec -n glasslab-v2 <pod-name> -- ls -la /datasets/`

### Artifacts not ingested

- Verify job succeeded: `kubectl get job -n glasslab-v2 <job-name> -o jsonpath='{.status.conditions}'`
- Check workflow-api logs for ingestion errors
- Verify MinIO/PostgreSQL connectivity
