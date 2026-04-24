# CIFAR-100 Unseen Classes Specification

## Overview

This project studies whether contrastive image representation can generalize to classes not seen during training.

## Data Splits

| Split | Classes | Images per Class | Total Images | Purpose |
|-------|---------|------------------|--------------|---------|
| Train-seen | 80 | 400 | 32,000 | Fit contrastive model |
| Val-seen | 80 | 100 | 8,000 | Tune hyperparameters, loss margins, PEFT choices |
| Test-seen | 80 | 100 | 8,000 | Evaluate generalization to new images from known classes |
| Test-unseen | 20 | 100 | 2,000 | Evaluate generalization to entirely unseen concepts |

### Seed Options

Multiple random class splits are required for robust evaluation:

- 42
- 123
- 456
- 789
- 1024

## Augmentation Pipeline

Contrastive learning relies heavily on data augmentation to create "views" of the data.

### Training Transformations

1. **RandomResizedCrop** (32x32)
   - Scale: [0.2, 1.0]
   - Ratio: [0.75, 1.333]
   - Interpolation: bilinear (2)

2. **RandomHorizontalFlip** (p=0.5)

3. **ColorJitter**
   - Brightness: [0.4, 0.4, 0.4, 0.1]
   - Contrast: [0.4, 0.4, 0.4, 0.1]
   - Saturation: [0.4, 0.4, 0.4, 0.1]
   - Hue: [0.1, 0.1, 0.1, 0.05]
   - Probability: 0.8

4. **RandomGrayscale** (p=0.2)

### Normalization

- Mean: [0.5071, 0.4865, 0.4409]
- Std: [0.2673, 0.2564, 0.2762]

## Baseline Models

### Zero-Shot Baselines

1. **DINO ViT**
   - Self-supervised visual transformer
   - Evaluates inherent class structure

2. **ResNet-50**
   - Supervised visual features
   - Standard baseline

3. **CLIP** (openai/clip-vit-base-patch32)
   - Joint multi-modal embedding space
   - Zero-shot classification capability
   - Powerful baseline for clustering unseen classes

## Training Protocol

### Loss Functions

- Supervised Contrastive Loss
- Triplet Loss (with semi-hard/hard negative mining)
- Multi-Similarity Loss
- Proxy-Anchor Loss

### Mining Strategies

- **Semi-hard negative**: Negatives farther from anchor than positive but within margin
- **Hard negative**: Negatives closer to anchor than positive
- **Distance-weighted**: Biases toward medium-distance negatives

### Hyperparameter Search

- Batch size: [32, 64, 96]
- Learning rate: [5e-5, 1e-4, 3e-4]
- Loss margin: [0.05, 0.1, 0.2]
- Temperature: [0.05, 0.07, 0.1]

## Evaluation Metrics

### Standard Metrics

- Recall@K
- Silhouette Score
- NMI
- AMI
- ARI

### Advanced Metrics

- **Grouped Recall@K**: Partitions test set into fixed non-overlapping class groups
  - Invariant to dataset size
  - Measures true generalization gap

- **OPIS** (Operating-Point-Inconsistency Score)
  - Measures threshold consistency across classes
  - Computes average difference between F1 at threshold and mean F1 over calibration range

## Statistical Validation

### 5x2cv Paired t-test

- Five repeats of 2-fold cross-validation
- Circumvents overlapping training set biases
- Avoids elevated Type I errors

### McNemar's Test

- Compares error patterns between classifiers
- Single training run sufficient
- Computationally cheaper than 5x2cv

## Expected Outcomes

1. **Generalization Bounds**: Grouped Recall@K should show consistent performance across seen/unseen splits
2. **Threshold Consistency**: OPIS should be low, indicating stable distance thresholds
3. **Statistical Significance**: Contrastive model should outperform CLIP/DINO baselines with p < 0.05

## File Structure

```
configs/
├── datasets/
│   └── cifar100_unseen_classes.yaml
├── augmentations/
│   └── contrastive_cifar100.yaml
└── search_spaces/
    └── cifar100_contrastive_v0.yaml

src/
├── metrics/
│   └── cifar_contrastive.py
└── runners/
    └── cifar_contrastive.py

scripts/
└── train_cifar100.py

docs/
├── cifar100-unseen-classes.md
├── evaluation-protocol.md
└── statistical-validation.md
```

## References

- ICLR 2024: OPIS metric for threshold consistency
- ICLR 2025: RDVC (Relative Distance Variance Constraint)
- CVPR 2025: Potential Field Based DML
- Hyperbolic Embeddings for Hierarchical Data
