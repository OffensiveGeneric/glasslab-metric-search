# Glasslab Support Evaluation

This document evaluates how the current Glasslab infrastructure can support
`glasslab-metric-search`.

It is based on the current repo-backed Glasslab shape:

- deterministic command path:
  - `whatsapp-gateway -> research-ingress -> research-command-router -> workflow-api`
- `workflow-api` as control plane
- Postgres as record store
- MinIO and shared storage for files and artifacts
- NATS available for lifecycle events

## Short answer

Glasslab is already a reasonable orchestration substrate for this project.

It is not yet a finished experiment platform for metric-learning search, but the
core control-plane pieces are present:

- durable metadata store
- private backend services
- run-oriented control plane
- artifact/object store
- asynchronous event bus
- GPU-capable Kubernetes cluster

That is enough for a first useful version if this repo stays config-first.

## What Glasslab can already do well

### 1. Own run records and campaign state

This is the strongest current fit.

`workflow-api` already fits the role of:

- session owner
- run creator
- comparison entrypoint
- decision recorder
- bounded next-step generator

That maps well onto:

- parent run ID
- search-space ID
- candidate config
- result summary
- keep/discard/promote decision

### 2. Persist metadata durably

Current Glasslab state uses:

- Postgres for records

That is the right place for:

- run metadata
- campaign lineage
- selected parents
- comparison summaries
- promotion decisions

### 3. Persist artifacts separately

Current Glasslab storage posture also fits this project:

- MinIO and/or shared filesystem for files

That is the right place for:

- checkpoints
- embeddings
- plots
- metrics blobs
- notebook exports
- evaluator reports

### 4. Schedule parallel jobs

The cluster already supports GPU-backed workloads and private image pulls.

That is enough for the intended execution model:

- one candidate run
- one Kubernetes Job
- one GPU requested per Job

This is exactly why this workload belongs on Glasslab instead of inside an
agent-only loop.

## What Glasslab does not yet have for this workload

### 1. A workload-specific registry contract

Glasslab needs a first-class workflow family for this repo.

It should know about:

- allowed search spaces
- expected run spec fields
- required artifact outputs
- evaluator bundle names
- comparison semantics

Without that, `glasslab-metric-search` remains just an external code repo rather
than a real workload family.

### 2. A concrete Job template for metric-search runs

This is the biggest missing operational piece.

Glasslab needs a Job spec that can take:

- repo/image ref
- run spec JSON
- dataset bindings
- GPU request
- wall-clock budget
- artifact destination

and produce one bounded run.

### 3. A metric-search evaluator

Current evaluator/reporter structure is promising, but this workload needs a
specific evaluator contract for:

- retrieval metrics
- verification metrics
- robustness metrics
- composite ranking

Until that exists, comparisons will remain ad hoc.

### 4. A result-ingest path from this repo back into workflow-api

At minimum, Glasslab needs a clean write-back path for:

- terminal run status
- metric bundle
- artifact references
- evaluator summary

This can be push or pull, but it must be explicit.

## Recommended first integration slice

Do not start by making Glasslab mutate code.

Start with this:

1. add a new workflow family:
   - `metric-search`
2. add a run manifest / run spec schema contract
3. add one Job template that runs `scripts/run_experiment.py`
4. upload `metrics.json` and `run_spec.json` as first-class artifacts
5. store terminal metrics in Postgres
6. let `workflow-api` compare runs and choose the next batch

That gets you a real platform loop quickly.

## Best use of current services

### `workflow-api`

Use for:

- campaign/session state
- candidate submission
- run creation
- run status lookup
- compare/promote/next transitions

### `workflow-registry`

Use for:

- dataset registry
- benchmark registry
- search-space registry
- workflow-family validation

### `evaluator`

Use for:

- retrieval/verification metric aggregation
- composite score computation
- candidate ranking

### `reporter`

Use for:

- run summary generation
- campaign delta summaries
- top-candidate reports

### `NATS`

Use for:

- lifecycle events only if they buy real decoupling

For example:

- run submitted
- run started
- run completed
- metrics available

Do not turn NATS into a fake architecture trophy.

### Postgres

Use for:

- canonical run metadata
- lineage
- decisions
- comparisons

### MinIO

Use for:

- checkpoints
- metrics blobs
- embeddings
- plots
- report artifacts

## Operational risks to account for

### 1. Dataset locality

Metric-learning workloads are often data-heavy.

You should decide early whether datasets are:

- mounted from shared storage
- staged into the job
- or read from object storage

This needs to be consistent per workflow family.

### 2. GPU fragmentation

Parallel search is only useful if the cluster can schedule one-GPU Jobs without
constant manual pinning.

If GPU scheduling is still coupled to special-case nodes or image placement
habits, that will slow this project down.

### 3. Run comparability

If budgets, datasets, or evaluator weights drift across runs, the search loop
will lie to you.

Glasslab must enforce:

- fixed budget families
- fixed benchmark bundle per campaign
- explicit search-space version

## Recommended next platform steps

1. register `metric-search` as a workflow family in Glasslab
2. define the run spec JSON schema in the registry/control plane
3. add one Kubernetes Job template for one-GPU metric-search runs
4. add a result ingest path into `workflow-api`
5. add evaluator support for the v0 composite score
6. only then add batch mutation/proposal automation

## Bottom line

Glasslab is already strong enough to be the control plane for this repo.

What it needs is not more agent behavior. It needs:

- one explicit workflow family
- one explicit run schema
- one explicit job template
- one explicit evaluator contract

That is enough to make `glasslab-metric-search` a real workload instead of an
interesting side project.
