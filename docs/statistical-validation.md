# Statistical Validation Protocol

## Problem Statement

When comparing machine learning models, simple performance comparisons can be misleading due to:

1. **Statistical flukes**: Random variation in train/test splits
2. **Overlapping training sets**: Standard k-fold tests inflate Type I errors
3. **Multiple comparison bias**: Running many experiments without correction

## 5x2cv Paired t-test

### Why 5x2cv?

Standard k-fold paired t-test suffers from overlapping training sets, leading to elevated Type I errors (false positives).

5x2cv addresses this by:

1. Running 2-fold cross-validation 5 times
2. Each fold uses a different random split
3. Training sets are non-overlapping across folds

### Algorithm

1. Split data into training/test sets
2. Train both models on training set
3. Evaluate on test set
4. Swap roles and repeat (2-fold)
5. Repeat steps 1-4 five times (5x2 = 10 evaluations)
6. Compute paired t-test on the 10 performance differences

### Implementation

```python
from mlxtend.evaluate import paired_ttest_5x2cv

# X = features, y = labels
# model1 = first model
# model2 = second model

score1, score2 = paired_ttest_5x2cv(
    estimator1=model1,
    estimator2=model2,
    X=X,
    y=y,
    random_seed=42
)

print(f"p-value: {score2}")
if score2 < 0.05:
    print("Significant difference detected")
```

### Interpretation

- **p < 0.05**: Significant difference between models
- **p >= 0.05**: No significant difference

## McNemar's Test

### When to Use

For deep learning models where multiple train/test rounds are expensive:

1. Train both models once
2. Compare error patterns (which samples each model gets wrong)
3. Determine if error patterns differ significantly

### Contingency Table

|             | Model2 Correct | Model2 Wrong |
|-------------|----------------|--------------|
| Model1 Correct | a             | b            |
| Model1 Wrong  | c             | d            |

### Test Statistic

$$\chi^2 = \frac{(b - c)^2}{b + c}$$

With continuity correction:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

### Implementation

```python
from scipy.stats import chi2

# b = model1 correct, model2 wrong
# c = model1 wrong, model2 correct

b = 150
c = 80

chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c)
p_value = 1 - chi2.cdf(chi2_stat, 1)

print(f"p-value: {p_value}")
```

### Interpretation

- **p < 0.05**: Error patterns differ significantly
- **p >= 0.05**: Models make similar errors

## Multiple Comparison Correction

When comparing multiple models, correct for multiple testing:

### Bonferroni Correction

$$\alpha_{corrected} = \frac{\alpha}{k}$$

Where k = number of comparisons.

### Holm-Bonferroni

1. Sort p-values ascending
2. Compare each to $\alpha / (k - i + 1)$
3. Stop when first non-significant result

### Tukey's HSD

For comparing multiple means:

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(
    endog=performance_scores,
    groups=model_labels,
    alpha=0.05
)
print(tukey.summary())
```

## Confidence Intervals

Report 95% confidence intervals for all metrics:

```python
import scipy.stats as stats

def confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return m, m - h, m + h

mean, lower, upper = confidence_interval(metrics_across_seeds)
print(f"{mean:.4f} [{lower:.4f}, {upper:.4f}]")
```

## Effect Size

Beyond p-values, report effect size:

### Cohen's d

$$d = \frac{m_1 - m_2}{\sqrt{\frac{s_1^2 + s_2^2}{2}}}$$

Interpretation:
- d = 0.2: Small effect
- d = 0.5: Medium effect
- d = 0.8: Large effect

## Reporting Template

```
Comparison: Contrastive Learning vs CLIP Zero-Shot

5x2cv Paired t-test:
- Model 1 mean: 0.7234
- Model 2 mean: 0.6512
- p-value: 0.0123
- Significant: Yes (p < 0.05)

Effect Size:
- Cohen's d: 0.87 (large effect)

Confidence Intervals:
- Model 1: 0.7234 [0.6987, 0.7481]
- Model 2: 0.6512 [0.6234, 0.6790]

Conclusion: Contrastive learning significantly outperforms CLIP zero-shot
with a large effect size (d = 0.87, p = 0.0123).
```

## Common Pitfalls

1. **P-hacking**: Running many tests and reporting only significant ones
   - Fix: Pre-register analysis plan; correct for multiple comparisons

2. **Overreliance on p-values**: Ignoring effect size and confidence intervals
   - Fix: Always report effect size and CI

3. **Ignoring assumptions**: t-test assumes normality
   - Fix: Use non-parametric tests if assumptions violated (Wilcoxon, Friedman)
