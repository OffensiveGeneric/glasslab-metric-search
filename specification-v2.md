# Project Specification 2.0: Structural Robustness, Generalization, and Memory-Linear Optimization in Deep Metric Learning

## 1. Overview & Core Research Questions

This project investigates how deep metric representations generalize to entirely unseen classes, addressing severe structural bottlenecks in standard training paradigms.

### Core Research Questions

1. **The Overfitting Paradox**: Do high-capacity models (e.g., ResNet-101) actually resist overfitting in retrieval tasks when evaluated using invariant metrics like Grouped Recall@K?

2. **The Greediness of Loss**: Does Contrastive Loss induce "dimensional collapse" and over-compact intra-class variance compared to Triplet Loss?

3. **Memory-Linear Scaling**: Can a 1D projection technique (Shadow Loss) achieve equal or better clustering than standard Triplet Loss while utilizing massive batch sizes?

4. **Synthetic Supervision**: Does generating completely novel classes via a conditional generator (L2A-NC) yield better generalization than just mining existing classes?

## 2. Experimental Setup & Backbone Selection

### Dataset Protocol

- Use CIFAR-100, split into disjoint sets:
  - Train-seen: 80 classes
  - Val-seen: 80 classes (remaining 20% of training images)
  - Test-seen: 80 classes (official test images)
  - Test-unseen: 20 classes (official test images)

- **Crucial**: Recent studies demonstrate negative generalization gaps in DML—test large backbones (ResNet-101, ViT-large) rather than heavily-regularized small networks.

- Include zero-shot foundation models (CLIP, DINO) as baselines.

## 3. Phase 1: Comparing Optimization "Greediness"

### Contrastive Loss (The Greedy Baseline)

- Contrastive loss independently pulls every positive pair together and pushes negatives apart.
- Verify literature claim: induces "greedy" optimization—over-compact clusters, obscures semantic differences.
- Measure intra-class and inter-class variances.

### Triplet Loss (The Variance-Preserving Baseline)

- Only ensures relative separation; stops updating once ranking is satisfied.
- Verify: preserves intra-class diversity, superior for fine-grained retrieval.

## 4. Phase 2: Solving the Memory Bottleneck with Shadow Loss

### Problem

- Standard pair/triplet objectives scale at O(S·D), forcing smaller batches.
- Memory bottleneck hurts performance.

### Implementation

- **Shadow Loss**: Project positive/negative embeddings onto 1D axis defined by anchor vector.
- Loss buffer reduced to O(S), enabling massive batch sizes on limited GPU.
- Track: convergence speed, cluster tightness (Silhouette scores).

## 5. Phase 3: Learning to Augment Novel Classes (L2A-NC)

### Problem

- Standard hard-negative mining bottlenecked by existing training classes.
- Limited generalization to unseen tasks.

### Implementation

- Replace hard-negative mining with conditional generative model.
- Generator takes latent variable + novel class label → synthetic embedding.
- Regularize generator to produce realistic vectors (minimize divergence).
- Evaluate: generalization to Test-unseen split.

## 6. Phase 4: Advanced Evaluation & Statistical Rigor

### Metrics

- **Grouped Recall@K**: Partition test set into fixed non-overlapping groups. Metric invariant to dataset size.
- **OPIS**: Measure threshold consistency across unseen classes.
- **Standard metrics**: NMI, AMI, ARI, Silhouette.

### Statistical Validation

- 5x2cv paired t-test or McNemar's test.
- Prove L2A-NC + Shadow Loss > Triplet > Contrastive > CLIP/DINO baselines.

## 7. Software Stack (Offline Agent)

### AutoML Orchestrators
- Auto-sklearn 2.0 (meta-learning, successive halving)
- AutoGluon (multi-layer stacking, ensembling)
- FLAML (cost-frugal optimization)
- TPOT (genetic programming pipeline generation)
- H2O AutoML (distributed training)
- PyCaret (low-code prototyping)

### Deep Metric Learning
- PyTorch Metric Learning (PML): losses, miners, distances
- OpenMetricLearning (OML): foundation models (CLIP, DINO, MoCo)
- TensorFlow Similarity: Keras-style DML
- metric-learn: NCA, LMNN, LFDA

### Deep Learning & Optimization
- PyTorch (dynamic graphs)
- JAX + Optax (composable transformations)
- FAISS: IVF, PQ indexing for billion-scale embeddings
- Optuna (TPE, dynamic pruning)
- Syne Tune (distributed ASHA)
- Ray Tune (cluster scaling)

### Trustworthiness & Uncertainty
- TorchCP (conformal prediction)
- Uncertainty Toolbox (calibration metrics)
- PUNCC (state-of-the-art conformalization)
- StatsForecast (distribution-free intervals)

### Statistical Testing
- MLxtend (5x2cv paired t-test)
- SciPy + Statsmodels (normality, variance, FDR correction)
- Scikit-posthocs + Pingouin (non-parametric post-hoc)
- pyDRMetrics (co-ranking matrices)

## 8. CI/CD Requirements

### Workflow 1: test-contrastive-learning.yml (workflow_dispatch only)

- Test imports (SupervisedContrastiveLoss, TripletLoss, ShadowLoss, L2A-NC)
- Validate augmentation pipeline (RandomResizedCrop, ColorJitter, RandomHorizontalFlip)
- Validate CIFAR-100 seen/unseen splits (80/80/80/20)
- Test Grouped Recall@K, OPIS, AMI, ARI, NMI, Silhouette
- Run baseline (CLIP, DINO) extraction + UMAP + k-means

### Workflow 2: test-dataset-config.yml (workflow_dispatch only)

- Validate cifar100_unseen_classes.yaml structure
- Test augmentation pipeline configuration
- Verify class split counts

### Workflow 3: test-search-space.yml (workflow_dispatch only)

- Validate cifar100_contrastive_v0.yaml search space
- Test HPO parameter ranges (batch_size, learning_rate, margins)

### Workflow 4: ci-python.yml (existing, extend)

- Add contrastive_runner.py to unit tests
- Add cifar100_unseen_classes.yaml to validation
- Add search space config tests

## 9. Deployment Pipeline

### Phase 1: Local Development (glasslab-metric-search)

- Implement L2A-NC generator
- Implement Shadow Loss
- Run Phase 1 experiments (Contrastive vs Triplet)
- Run Phase 2 experiments (Shadow Loss scalability)
- Run Phase 3 experiments (L2A-NC generalization)

### Phase 2: Cluster Deployment (glasslab-cluster-config)

- Update GPU runner image with new dependencies
- Deploy contrastive learning runner
- Register CIFAR-100 dataset in workflow-api
- Test end-to-end via workflow-api `/runs` endpoint

### Phase 3: Autoresearch Integration

- Extend autoresearch loop with contrastive learning methodology mutations
- Test L2A-NC vs Shadow Loss vs Contrastive vs Triplet
- Validate statistical significance (5x2cv or McNemar)

## 10. Expected Deliverables

1. **Report**: Overfitting paradox, greediness comparison, Shadow Loss scalability, L2A-NC generalization
2. **UMAP Plots**: Seen vs unseen class clustering
3. **Statistical Significance**: 5x2cv or McNemar p-values
4. **Code**: contrastive_runner.py, L2A-NC generator, Shadow Loss implementation
5. **Metrics**: Grouped Recall@K, OPIS, AMI, ARI, NMI, Silhouette

## 11. Literature References

- Grouped Recall@K: Generalization bounds invariant to class count
- OPIS (ICLR 2024): Threshold consistency for deployment
- Shadow Loss: Memory-linear O(S) scaling
- L2A-NC: Novel class generation for better generalization
- 5x2cv paired t-test: Avoid overlapping training set biases
- McNemar's test: Single training run significance testing
- Hyperbolic Geometry (Lorentz Embeddings): Hierarchical data in low dimensions
