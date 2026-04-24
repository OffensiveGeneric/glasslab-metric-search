# Evaluation Protocol

## Metrics Overview

This document describes the evaluation metrics used for contrastive learning on CIFAR-100.

## Standard Metrics

### Recall@K

Standard retrieval metric measuring whether ground-truth positives appear in the top-K nearest neighbors.

**Limitation**: Sensitive to dataset size; not suitable for comparing seen/unseen class generalization.

### Silhouette Score

Measures cluster cohesion and separation:
$$S = \frac{1}{n} \sum_{i=1}^{n} \frac{b_i - a_i}{\max(a_i, b_i)}$$

Where:
- $a_i$ = average distance to other points in same class
- $b_i$ = minimum average distance to points in other classes

## Advanced Metrics

### Grouped Recall@K

Standard Recall@K is biased by the number of classes. Grouped Recall@K partitions the test set into fixed non-overlapping class groups and averages results.

**Algorithm**:
1. Partition test set into N groups (e.g., 4 groups for CIFAR-100)
2. Compute Recall@K for each group independently
3. Average across groups

**Advantages**:
- Invariant to dataset size
- Measures true generalization gap
- Robust to class count variations

### OPIS (Operating-Point-Inconsistency Score)

Measures threshold consistency for retrieval systems.

**Problem**: In real-world applications, distance thresholds for matching vary across classes. A single threshold rarely optimizes all classes.

**Solution**: OPIS quantifies the inconsistency:
$$OPIS = \frac{1}{|T|} \sum_{t \in T} |F1(t) - \bar{F1}|$$

Where:
- $T$ = threshold range (e.g., [0.1, 2.0])
- $F1(t)$ = F1 score at threshold t
- $\bar{F1}$ = mean F1 over threshold range

**Interpretation**:
- Low OPIS: Consistent thresholds across classes
- High OPIS: Threshold performance varies widely

### AMI (Adjusted Mutual Information)

Mutual information adjusted for chance:
$$AMI = \frac{MI - E[MI]}{\max(H(L), H(P)) - E[MI]}$$

Where:
- $MI$ = Mutual Information between labels and clusters
- $H(L)$, $H(P)$ = entropies of label and cluster distributions
- $E[MI]$ = expected mutual information

**Advantage**: NMI can be biased by the number of clusters; AMI corrects for this.

### ARI (Adjusted Rand Index)

Measures similarity between two clusterings, adjusted for chance:
$$ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}$$

**Advantage**: Accounts for random agreement; range [-1, 1]

### NMI (Normalized Mutual Information)

Normalized version of mutual information:
$$NMI = \frac{2 \cdot MI(L, P)}{H(L) + H(P)}$$

**Limitation**: Can be biased by number of clusters; use AMI for rigorous comparison.

## Evaluation Workflow

1. Extract embeddings for all test sets (seen and unseen)
2. Run UMAP for visualization
3. Perform k-means clustering
4. Compute all metrics:
   - Grouped Recall@K (K=10, N=4 groups)
   - OPIS (threshold range [0.1, 2.0], 50 steps)
   - AMI, ARI, NMI
   - Silhouette Score
5. Aggregate across multiple random seeds
6. Perform statistical significance tests

## Statistical Tests

### 5x2cv Paired t-test

For lightweight adaptations:

1. Repeat 2-fold cross-validation 5 times
2. Compute mean performance difference
3. Test against null hypothesis (no difference)

**Advantage**: Avoids overlapping training set biases

### McNemar's Test

For deep learning models:

1. Train both models once
2. Compare error patterns
3. Determine statistical significance

**Advantage**: Computationally efficient

## Reporting

For each experiment, report:

- Mean and standard deviation across seeds
- 95% confidence interval
- p-value from statistical test
- Effect size
