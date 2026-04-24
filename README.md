# glasslab-metric-search

Contrastive representation learning for unseen class generalization on Glasslab.

## Overview

This repo implements the contrastive learning evaluation protocol for CIFAR-100 unseen class generalization.

## Key Features

- **CIFAR-100 seen/unseen splits**: 80/20 class split with multiple seeds
- **Advanced metrics**: Grouped Recall@K, OPIS, AMI, ARI, NMI, Silhouette
- **Statistical validation**: 5x2cv paired t-test, McNemar's test
- **AutoML support**: Batch size, learning rate, margin tuning

## Project Structure

```
configs/
├── datasets/              # Dataset specifications
├── augmentations/         # Augmentation pipelines
└── search_spaces/         # AutoML search spaces

src/
├── data/                  # Dataset loaders
├── losses/                # Contrastive loss implementations
├── miners/                # Tuple mining strategies
├── metrics/               # Evaluation metrics
├── models/                # Backbone models
├── evaluators/            # Evaluation pipelines
├── regularizers/          # Regularization terms
└── runners/               # Training runners

scripts/
├── train.py               # Training entry point
├── evaluate.py            # Evaluation entry point
└── run_experiment.py      # End-to-end workflow

docs/
├── cifar100-unseen-classes.md    # Main specification
├── evaluation-protocol.md        # Metrics documentation
└── statistical-validation.md     # Statistical tests

benchmarks/
└── art_retrieval.py       # Benchmark runners
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run experiment
python scripts/run_experiment.py \
  --config configs/search_spaces/cifar100_contrastive_v0.yaml \
  --output-dir /tmp/cifar100-experiment

# Evaluate
python scripts/evaluate.py \
  --metrics /tmp/cifar100-experiment/metrics.json \
  --mode contrastive
```

## Metrics

| Metric | Description |
|--------|-------------|
| Grouped Recall@K | Recall invariant to class count |
| OPIS | Threshold consistency score |
| AMI | Adjusted Mutual Information |
| ARI | Adjusted Rand Index |
| NMI | Normalized Mutual Information |
| Silhouette | Cluster quality score |

## Baselines

- DINO ViT (self-supervised)
- ResNet-50 (supervised)
- CLIP (zero-shot multimodal)

## Statistical Validation

- 5x2cv Paired t-test
- McNemar's test
- Effect size reporting
- Confidence intervals

## Integration with Glasslab

The glasslab-cluster-config repo contains:

- `configs/datasets/cifar100_unseen_classes.yaml` - Dataset specification
- `configs/search_spaces/cifar100_contrastive_v0.yaml` - Search space
- `services/runner/app/contrastive_runner.py` - Training runner
- `services/runner/app/main.py` - Entry point with contrastive pipeline support
- `scripts/import-contrastive-learning-technique.py` - Technique catalog import
