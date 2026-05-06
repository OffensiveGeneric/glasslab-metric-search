# glasslab-metric-search

Contrastive representation learning for unseen class generalization on Glasslab.

## Overview

This repository implements deep metric learning evaluation for CIFAR-100 unseen class generalization, focusing on structural robustness and invariant metrics that resist dataset size biases.

### Core Research Questions

1. **The Overfitting Paradox**: Do high-capacity models (ResNet-101, ViT-large) resist overfitting in retrieval tasks?
2. **The Greediness of Loss**: Does Contrastive Loss induce dimensional collapse vs. Triplet Loss?
3. **Memory-Linear Scaling**: Can Shadow Loss enable massive batch sizes while matching Triplet performance?
4. **Synthetic Supervision**: Does L2A-NC (learning to augment novel classes) yield better generalization?

## Key Features

- **CIFAR-100 seen/unseen splits**: 80/20 class protocol with multiple seeds
- **Advanced metrics**: Grouped Recall@K (invariant to class count), OPIS, AMI, ARI, NMI, Silhouette
- **Statistical validation**: 5x2cv paired t-test, McNemar's test, effect sizes
- **AutoML support**: Batch size, learning rate, margin tuning via Optuna/Syne Tune/Ray Tune
- **Baseline validation**: DINO ViT, ResNet-50, CLIP as sanity checks
- **FAISS integration**: GPU-accelerated metrics computation on Linux, CPU fallback on macOS
- **SQLite streaming**: Memory-efficient embedding storage for large galleries

## Project Structure

```
configs/
├── datasets/              # Dataset specifications (cifar100_unseen_classes.yaml)
├── augmentations/         # Augmentation pipelines
└── search_spaces/         # AutoML search spaces (cifar100_contrastive_v0.yaml)

src/
├── data/                  # Dataset loaders and splits
├── losses/                # Contrastive, Triplet, Shadow Loss implementations
├── miners/                # Tuple mining strategies (semi-hard, hard, distance-weighted)
├── metrics/               # Evaluation metrics (Grouped Recall@K, OPIS, AMI, ARI, NMI, Silhouette)
├── models/                # Backbone models (ResNet, ViT, CLIP, DINO)
├── evaluators/            # Evaluation pipelines (CIFAR-100 contrastive evaluator)
├── regularizers/          # Regularization terms
└── runners/               # Training runners (contrastive_runner.py)

scripts/
├── train.py               # Training entry point
├── evaluate.py            # Evaluation entry point
├── run_experiment.py      # End-to-end workflow
└── run_baseline.py        # Baseline evaluation (random, ResNet50, DINO, CLIP)

docs/
├── cifar100-unseen-classes.md          # Dataset protocol
├── evaluation-protocol.md              # Metrics documentation
├── glasslab-integration.md             # Cluster integration
├── statistical-validation.md           # Statistical tests
└── metric-search-evaluation-action-notes-2026-04-29.md

notebooks/               # Experiment notebooks
results/                 # Experiment results
tests/                   # Unit and integration tests
benchmarks/              # Benchmark runners

Dockerfile               # Container image definition
pyproject.toml           # Python dependencies
```

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/OffensiveGeneric/glasslab-metric-search.git
cd glasslab-metric-search

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Docker

```bash
# Build image
docker build -t glasslab-metric-search:latest .

# Run container
docker run --gpus all -it glasslab-metric-search:latest
```

### Kubernetes

See `cluster-config/kubeadm/glasslab-v2/jobs/` for job definitions.

## Quick Start

```bash
# Train contrastive model
python scripts/run_experiment.py \
  --config configs/search_spaces/cifar100_contrastive_v0.yaml \
  --output-dir /tmp/cifar100-experiment

# Evaluate baseline (random, ResNet50, DINO, CLIP)
python scripts/run_baseline.py --baseline=random --output-dir=/tmp/baselines/random
python scripts/run_baseline.py --baseline=resnet50 --output-dir=/tmp/baselines/resnet50
python scripts/run_baseline.py --baseline=dino --output-dir=/tmp/baselines/dino
python scripts/run_baseline.py --baseline=clip --output-dir=/tmp/baselines/clip

# Evaluate saved metrics
python scripts/evaluate.py \
  --metrics /tmp/cifar100-experiment/metrics.json \
  --mode contrastive
```

## Metrics

| Metric | Description | Expected Random Baseline |
|--------|-------------|--------------------------|
| Grouped Recall@K | Recall invariant to class count (group_size=10) | ~41% for test_unseen (20 classes) |
| Global Recall@K | Standard recall over full gallery | ~5% for test_unseen (20 classes) |
| OPIS | Threshold consistency score for deployment | Varies |
| AMI | Adjusted Mutual Information | ~0 |
| ARI | Adjusted Rand Index | ~0 |
| NMI | Normalized Mutual Information | ~0 |
| Silhouette | Cluster quality score | ~0 (negative = poor separation) |

### Metric Key Naming Convention

Metrics use the pattern `{split}_{run_type}_{metric_name}`:
- `test_unseen_0_global_recall_at_1` - Global recall on unseen test set
- `test_unseen_0_random_embedding_global_recall_at_1` - Random baseline on unseen test
- `test_unseen_0_shuffled_global_recall_at_1` - Shuffled label baseline on unseen test
- `val_seen_0_grouped_recall_at_k` - Grouped recall on seen validation set

## Baselines

| Baseline | Test Seen | Test Unseen | Lift vs Random | Eval Batches | Notes |
|----------|-----------|-------------|----------------|-------------|-------|
| Random Embedding | ~5% Global@1, ~41% Grouped@5 | ~5% Global@1, ~41% Grouped@5 | 0% (by definition) | 8 | Chance-level baseline |
| ResNet-50 (supervised) | ~92% Grouped@5, ~39% Global@1 | ~83% Global@1, ~92% Grouped@5 | ~44% | 16 | Strong ImageNet transfer |
| DINO ViT (self-supervised) | ~96% Grouped@5, ~53% Global@1 | ~97% Grouped@5, ~82% Global@1 | ~45% | 16 | Self-supervised ViT |
| CLIP (zero-shot multimodal) | ~95% Grouped@5, ~50% Global@1 | ~97% Grouped@5, ~78% Global@1 | ~42% | 16 | Zero-shot CLIP |
| Contrastive ResNet18 trained | ~81% Grouped@5, ~44% Global@1 | ~84% Grouped@5, ~44% Global@1 | ~43% | 16 | Phase 1 small-budget comparison (timm pretrained) |
| Triplet ResNet18 trained (naive in-batch) | ~82% Grouped@5, ~43% Global@1 | ~84% Grouped@5, ~43% Global@1 | ~42% | 16 | Phase 1 small-budget comparison (timm pretrained) |

### Phase 1 Training Results (Contrastive vs Triplet)

| Method | Test Seen Grouped@5 | Test Unseen Grouped@5 | Test Unseen Composite | Lift vs Random | Eval Batches | Output Dir | Image Commit | Sanity Pass |
|--------|---------------------|-----------------------|----------------------|----------------|-------------|------------|--------------|-------------|
| Contrastive Loss (ResNet18, timm pretrained) | 80.9% | 84.4% | 0.452 | ~43% | 16 | phase1-contrastive-phase1-contrastive-txtvd | smoke-test-81a0266 | ✅ |
| Triplet Loss (ResNet18, timm pretrained) | 81.7% | 84.3% | 0.463 | ~42% | 16 | phase1-triplet-final-phase1-triplet-final-8bc4g | smoke-test-81a0266 | ✅ |

| Baseline Comparison (test_unseen metrics) |
|-------------------------------------------|
| **Random Embedding**: Global@1=5%, Grouped@5=41% |
| **ResNet-50 frozen**: Global@1=68%, Grouped@5=92% |
| **DINO ViT**: Global@1=82%, Grouped@5=97% |
| **CLIP**: Global@1=78%, Grouped@5=97% |
| **Contrastive ResNet18 trained**: Global@1=44%, Grouped@5=84% |
| **Triplet ResNet18 trained (naive)**: Global@1=43%, Grouped@5=84% |

**Key Findings**:
- Small-budget Phase 1 comparison showing contrastive and naive in-batch triplet produce similar metrics
- Both methods achieve ~84% Grouped@5 on test_unseen
- **Neither trained method outperforms frozen baselines**:
  - ResNet-50 frozen (82-92%) > Contrastive trained (44-84%)
  - DINO ViT (82-97%) > Contrastive trained (44-84%)
  - CLIP (78-97%) > Contrastive trained (44-84%)
- Random baseline Global@1 ~4.9% (expected ~4.95%) validates evaluation pipeline
- **Important**: This is NOT conclusive evidence of strong generalization
- Negative silhouette scores indicate poor cluster separation in both methods

**Experimental Setup**:
- 3 epochs, 10 train batches, 16 eval batches
- Batch size: 64, Learning rate: 1e-4, Margin: 0.3
- Backbone: timm pretrained ResNet-18
- Image commit: smoke-test-81a0266

**Important Notes**:
1. This is NOT conclusive evidence of strong generalization
2. The small-budget setup (3 epochs, 10 batches) is for validation, not scientific conclusion
3. The current triplet implementation uses naive in-batch triplet; next version should add semi-hard/hard negative mining
4. For rigorous comparison, run 3+ seeds with same configuration and report mean/std
5. Both methods use timm pretrained ResNet-18 weights (ImageNet pretraining, not CIFAR-100 specific)

### Baseline Expectations

**Random Embedding**:
- Global Recall@1 on test_unseen (20 classes): ~5% (1/20)
- Global Recall@1 on test_seen (80 classes): ~1.25% (1/80)
- Grouped Recall@5 (group_size=10): ~41% (analytic chance calculation)

**Sanity Checks**:
- Global Recall@1 should match exact chance (within 0.03) - validates no embedding leakage
- Grouped Recall@K should match grouped chance (within 0.03) - validates no clustering signal
- Negative silhouette scores indicate poor cluster separation
- Model quality is not interpretable if random baseline fails sanity checks
- **Note**: Strong baseline performance indicates evaluation pipeline is working, not that trained models generalize well

See `docs/cifar100-unseen-classes.md` and `docs/evaluation-protocol.md` for full details.

## Statistical Validation

### Tests

- **5x2cv Paired t-test**: Avoid overlapping training set biases
- **McNemar's test**: Single training run significance testing
- **Effect sizes**: Report Cohen's d, confidence intervals

### Implementation

- MLxtend for 5x2cv paired t-test
- SciPy + Statsmodels for normality and variance tests
- Scikit-posthocs for non-parametric post-hoc analysis

## AutoML Search Spaces

### cifar100_contrastive_v0.yaml

```yaml
batch_size:
  type: categorical
  options: [32, 64, 128, 256]

learning_rate:
  type: loguniform
  base: 10
  min: 1e-5
  max: 1e-2

margin:
  type: uniform
  min: 0.1
  max: 1.0

loss:
  type: categorical
  options: ["ContrastiveLoss", "TripletLoss", "ShadowLoss"]
```

## Integration with Glasslab

The `glasslab-cluster-config` repo contains:

- `configs/datasets/cifar100_unseen_classes.yaml` - Dataset specification
- `configs/search_spaces/cifar100_contrastive_v0.yaml` - Search space
- `services/runner/app/contrastive_runner.py` - Training runner
- `services/runner/app/main.py` - Entry point with contrastive pipeline support
- `scripts/import-contrastive-learning-technique.py` - Technique catalog import
- Kubernetes jobs for baseline experiments (30-baseline-random.yaml, etc.)

### Deployment Pipeline

1. **Local Development**: Implement L2A-NC generator, Shadow Loss, Phase 1-3 experiments
2. **Cluster Deployment**: Update GPU runner image, deploy contrastive learning runner
3. **Autoresearch Integration**: Extend autoresearch loop with contrastive learning mutations

## Development

### Testing

```bash
# Unit tests
pytest tests/

# Baseline tests
pytest tests/test_smoke_metrics.py

# Format and lint
ruff check .
black --check .
```

### CI/CD

- **test-contrastive-learning.yml**: Test imports, metrics, baselines
- **test-dataset-config.yml**: Validate dataset configuration
- **test-search-space.yml**: Validate search space configuration
- **ci-python.yml**: Python linting and testing

## Known Issues

1. **FAISS deadlock on macOS**: Use conditional import in `metrics.py` (FAISS not imported on Darwin)
2. **Image pull times**: ~10GB image; use prepull job to cache on GPU nodes
3. **NFS mount on Macs**: Cluster artifacts require kubectl exec to copy
4. **Random baseline global Recall@1**: Should be ~5% on test_unseen, not ~41%

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- PyTorch Metric Learning for baseline implementations
- OpenMetricLearning for foundation models
- CIFAR-100 dataset authors

## Contact

Glasslab Research Team - glasslab@offensivegeneric.com
