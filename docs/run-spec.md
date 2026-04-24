# Run Spec

The run spec is the boundary between this repo and Glasslab.

Each run should be schedulable without inspecting Python source.

## Required fields

- `run_id`
- `parent_run_id`
- `base_commit`
- `submitted_by`
- `workflow_family`
- `search_space_id`
- `dataset`
- `resources`
- `budget`
- `config`

## Dataset block

```json
{
  "dataset_id": "artbench-v1",
  "split_version": "split-2026-04-a",
  "train_uri": "s3://datasets/artbench/train.parquet",
  "val_uri": "s3://datasets/artbench/val.parquet",
  "test_uri": "s3://datasets/artbench/test.parquet"
}
```

## Resources block

```json
{
  "gpu_count": 1,
  "cpu_count": 8,
  "memory_gb": 32
}
```

## Budget block

```json
{
  "max_epochs": 25,
  "max_wallclock_minutes": 180
}
```

## Artifact references

Artifact URIs are optional at submission time and are filled in as the run
completes.

Examples:

- checkpoint URI
- metrics URI
- embeddings URI
- plots URI
- report URI

