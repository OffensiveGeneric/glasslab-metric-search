# Glasslab Integration

This repo is the workload side of the system.

Glasslab should own:

- run creation
- scheduling
- run status
- artifact indexing
- result comparison
- summary/report generation

This repo should own:

- dataset bindings
- model and loss configuration
- trainer behavior
- evaluator behavior
- search-space definition
- candidate mutation logic

## Submission Boundary

The contract between the two repos is the run spec JSON written by
`scripts/run_experiment.py`.

Glasslab should be able to schedule a run using:

- repo revision
- run spec payload
- container image reference
- dataset/object-store bindings
- resource request
- time budget

Glasslab should not need to infer experiment meaning from ad hoc shell commands.

## Intended Execution Model

One candidate run should map to one Kubernetes Job.

Each Job should:

- request one GPU
- mount or fetch dataset inputs read-only
- write checkpoints and metrics to object/file storage
- emit lifecycle events
- terminate with a clear success or failure state

## Early Search Policy

Early search should stay in config space.

Allowed mutation targets:

- loss family
- miner type
- proxy count
- regularization weights
- batch composition
- backbone freeze versus fine-tune
- augmentation recipe

## CIFAR-100 Unseen Classes Protocol

For the CIFAR-100 contrastive learning project, the protocol includes:

### Data Splits

- **Train-seen**: 80 classes, 80% of training images per class (400 images/class)
- **Val-seen**: Same 80 classes, 20% of training images per class (100 images/class)
- **Test-seen**: Official CIFAR-100 test images for 80 seen classes (100 images/class)
- **Test-unseen**: Official CIFAR-100 test images for 20 unseen classes (100 images/class)

### Evaluation Metrics

- Grouped Recall@K (invariant to class count)
- OPIS (Operating-Point-Inconsistency Score)
- AMI (Adjusted Mutual Information)
- ARI (Adjusted Rand Index)
- NMI (Normalized Mutual Information)
- Silhouette Score

### Statistical Validation

- 5x2cv Paired t-test for significance
- McNemar's test for deep learning comparison

### Baselines

- DINO ViT (self-supervised vision transformer)
- ResNet-50 (supervised)
- CLIP (zero-shot multimodal)

### Augmentation Pipeline

- RandomResizedCrop
- ColorJitter (brightness, contrast, saturation, hue)
- RandomHorizontalFlip
- RandomGrayscale

### Mining Strategies

- Semi-hard negative mining
- Hard negative mining
- Distance-weighted sampling
