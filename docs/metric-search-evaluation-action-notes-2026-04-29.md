# Metric Search Evaluation Action Notes

Date: 2026-04-29

These notes convert the latest metric-search review and Glasslab PVC run results into concrete repository work. The current pipeline is useful as infrastructure validation, but it is not yet strong enough to support claims about unseen-class generalization.

## Current Evidence

Best completed live run:

- Artifact directory: `/mnt/artifacts/metric-search-audit-eval-fbjdk`
- Status: `succeeded`
- Dataset: `cifar100-unseen-classes`
- Mode: real, not simulated
- `max_eval_batches`: 4
- Container memory limit: 32Gi

Key metrics:

- `val_seen_grouped_recall_at_k`: 0.5314
- `test_seen_grouped_recall_at_k`: 0.7250
- `test_unseen_grouped_recall_at_k`: 0.9612
- `test_unseen_shuffled_label_grouped_recall_at_k`: 0.7181
- `test_unseen_random_embedding_grouped_recall_at_k`: 0.6689
- `test_unseen_grouped_recall_lift_vs_shuffled_labels`: 0.2431
- `test_unseen_grouped_recall_lift_vs_random_embeddings`: 0.2923
- `test_seen_equalized_grouped_recall_at_k`: 0.8530
- `test_unseen_equalized_reference_grouped_recall_at_k`: 0.9612
- `generalization_gap_equalized_grouped_recall_at_k`: -0.1082
- `test_unseen_adjusted_rand_index`: 0.2590
- `test_unseen_adjusted_mutual_info`: 0.4358
- `test_unseen_silhouette_score`: 0.0022

Observed scaling limit:

- `max_eval_batches=8` OOMKilled at a 32Gi container limit.
- The live GPU nodes include two roughly 64Gi-memory nodes and one roughly 32Gi-memory node.
- The OOM was a pod/container limit failure, not general cluster memory exhaustion.

## Interpretation

Do not treat the high unseen grouped recall as evidence of model generalization yet.

The strongest red flag is that random embeddings scored `0.6689` grouped recall on `test_unseen`. A random baseline that high means the retrieval task is currently too easy, underspecified, or measured on too small a gallery. This does not prove the metric implementation is entirely wrong, but it does mean the current metric output is not trustworthy as a scientific result.

The low or near-zero global clustering scores are consistent with a one-epoch, tiny-budget contrastive run. High local retrieval and weak global clustering can coexist, but with a random baseline this high, the first priority is to validate the evaluation protocol rather than tune the model.

## Immediate Blockers

### 1. Fix the retrieval gallery contract

Grouped Recall@K must be computed over a well-defined gallery. Evaluation on tiny `max_eval_batches` subsets can make retrieval artificially easy.

Action items:

- Define the gallery size and class-grouping contract in code and docs.
- For CIFAR-100, evaluate full split galleries for real runs whenever possible.
- Treat `max_eval_batches` runs as smoke tests only.
- Write explicit warnings into `metrics.json` when the gallery is partial.
- Add a minimum-gallery sanity gate before reporting headline Recall@K.

Acceptance checks:

- Random embeddings on a full or fixed-size gallery are near the expected chance range for the chosen grouped-recall definition.
- A tiny gallery cannot silently produce normal-looking headline metrics.
- `metrics.json` records `eval_gallery_num_samples`, `eval_gallery_num_classes`, `group_size`, `k`, and whether the gallery is partial.

### 2. Make Grouped Recall@K mathematically explicit

The current docs say grouped recall partitions into groups, but the implementation and reporting need a fixed contract.

Action items:

- Define fixed class group size, for example 10 classes per group.
- Partition labels into non-overlapping groups of that fixed size.
- Compute standard Recall@K inside each class group.
- Average group-level recall values.
- Report leftover-class behavior explicitly.
- Use the same group size for seen and unseen splits.

Acceptance checks:

- `test_seen` and `test_unseen` can be compared without relying on equal total class counts.
- Unit tests cover 20-class, 80-class, and uneven leftover cases.
- Random embeddings produce plausible grouped-recall baselines under the same grouping rule.

### 3. Reduce evaluation memory pressure

`collect_embeddings()` already uses `torch.no_grad()`, so the immediate OOM concern is dense metric computation and accumulated payloads, not a missing no-grad context.

Known memory pressure points:

- Full embedding tensors are retained for all splits.
- OPIS builds dense pairwise distance matrices.
- Silhouette can require expensive pairwise distance work internally.
- Baselines repeat metric computation for real, shuffled-label, and random-embedding variants.
- Larger eval galleries increase memory superlinearly for dense metrics.

Action items:

- Replace dense Recall@K computation with FAISS or chunked nearest-neighbor search.
- Gate OPIS and Silhouette behind sample caps or chunked/approximate implementations.
- Record when OPIS/Silhouette are computed on a sampled subset.
- Avoid keeping every split and baseline payload in memory longer than needed.
- Save embeddings as sharded artifacts and load only the split currently being evaluated.

Acceptance checks:

- `max_eval_batches=8` completes under a documented memory limit.
- Full CIFAR-100 test evaluation has a known expected memory envelope.
- OOM failures produce `error.json` with the failed phase when possible.

### 4. Add hard baseline gates

Do not run longer training or HPO until the baseline harness is trustworthy.

Required baselines:

- Random Gaussian embeddings.
- Shuffled labels on real embeddings.
- Frozen ImageNet-pretrained ResNet.
- Frozen DINO ViT.
- Frozen CLIP image encoder.

Action items:

- Add a baseline-only runner that writes the same artifact contract as trained runs.
- Compare each baseline with the same grouped recall, clustering metrics, and visual artifacts.
- Fail or warn if random embeddings exceed a conservative threshold for the configured gallery.
- Report baseline lift separately; do not hide it inside a composite score.

Acceptance checks:

- The trained model must beat random and shuffled baselines before any model-quality claim.
- The trained model must beat frozen ResNet/DINO/CLIP before any unseen-generalization claim.
- Baseline outputs are first-class run artifacts under `/mnt/artifacts/<run_id>`.

## Training Work

### 5. Add class-balanced batch sampling

Random batches over many classes can contain too few positive pairs for supervised contrastive learning.

Action items:

- Implement a P x K sampler, also known as M-per-class sampling.
- Expose sampler settings in config, for example `classes_per_batch` and `samples_per_class`.
- Add tests proving each training batch has positive pairs.
- Record effective batch composition in run metadata.

Acceptance checks:

- Every supervised contrastive training batch contains at least two samples for each sampled class.
- The sampler works with CIFAR-100 seen-class subsets.
- The training runner can fall back to regular shuffling only for explicitly marked smoke tests.

### 6. Use pretrained backbones as the default serious-run path

Training from scratch on a small contrastive setup is a poor default for scientific runs.

Action items:

- Make ImageNet-pretrained ResNet available for fine-tuning.
- Add frozen-backbone and fine-tuned-backbone modes.
- Record backbone initialization in `run_spec.json` and `metrics.json`.
- Keep scratch training only as an explicit ablation.

Acceptance checks:

- A run artifact clearly states `pretrained=true/false` and `frozen=true/false`.
- Baseline-only frozen ResNet and fine-tuned ResNet are comparable in the same report.

### 7. Define minimally meaningful training scale

The one-epoch runs are smoke tests only.

Action items:

- Add run budget tiers:
  - `smoke`: tiny, validates orchestration only.
  - `debug`: enough batches to validate metrics and memory.
  - `pilot`: full eval, short but real training.
  - `claim`: multiple seeds/splits and baseline comparisons.
- Mark smoke/debug outputs as non-claimable in `metrics.json`.
- Train until validation grouped recall or loss plateaus for serious runs.

Acceptance checks:

- Reports explicitly say whether a run is `claimable`.
- Smoke runs cannot be mistaken for scientific evidence.

## Reporting And Artifacts

### 8. Remove composite score from decision-making

The composite score mixes local retrieval, global clustering, and threshold behavior. It can remain for backward compatibility, but should not be the primary decision metric.

Action items:

- Report raw metrics separately:
  - Grouped Recall@1 and Grouped Recall@5.
  - mAP if implemented.
  - NMI, AMI, ARI.
  - Silhouette.
  - OPIS.
  - Baseline lift.
- Mark `composite_score` as deprecated or diagnostic-only.
- Update reports to lead with baseline-adjusted retrieval and global clustering metrics separately.

Acceptance checks:

- No report claims model quality from composite score alone.
- Automated selection logic does not optimize only composite score.

### 9. Add visual debugging artifacts

Visuals are diagnostic, not proof, but they are useful for catching shortcut learning and broken retrieval.

Action items:

- Save nearest-neighbor retrieval grids for representative seen and unseen classes.
- Save random-baseline retrieval grids for comparison.
- Save UMAP plots for:
  - Frozen baselines.
  - Fine-tuned model.
  - Seen and unseen splits with distinct markers.
- Save contact sheets under `visualizations/`.

Acceptance checks:

- Each serious run writes a visual artifact index.
- Reports link to retrieval grids and UMAP plots.
- Visual artifacts never replace raw metric checks.

## Statistical Protocol

### 10. Use multiple splits and seeds before any claim

Single-split results are not robust enough for unseen-class claims.

Action items:

- Run 3 to 5 random class splits as the first statistically useful target.
- Consider 10 splits only after the pipeline is stable.
- Report mean, standard deviation, and confidence intervals.
- Use paired tests only after baseline and trained runs share identical splits and galleries.

Acceptance checks:

- A claimable result includes multiple random class choices.
- Each split includes baseline and trained-model artifacts.
- The final report includes uncertainty, not just point estimates.

### 11. Define minimum evidence for unseen generalization

Minimum evidence should include:

- Full or fixed-size evaluation galleries.
- Random and shuffled-label baselines near expected ranges.
- Positive lift over frozen ImageNet ResNet.
- Positive lift over DINO and CLIP on unseen classes.
- Multiple seeds or class splits.
- Retrieval metrics and global clustering metrics reported separately.
- Visual inspection artifacts for nearest neighbors.

Do not claim unseen generalization from:

- One epoch.
- Partial eval batches.
- Composite score alone.
- Raw unseen recall without baseline lift.
- A single split.

## Recommended Implementation Order

1. Lock the Grouped Recall@K contract and add tests.
2. Add gallery-size metadata and partial-gallery warnings.
3. Make random/shuffled baselines strict validation gates.
4. Reduce dense metric memory use enough to run larger/full galleries.
5. Add frozen ResNet, DINO, and CLIP baseline runners.
6. Add retrieval grids and UMAP artifacts.
7. Implement P x K class-balanced training batches.
8. Promote pretrained backbone fine-tuning as the serious-run default.
9. Run 3 to 5 splits with confidence intervals.
10. Only then run model/HPO comparisons.

## Notes On SQLite Streaming

Streaming embeddings or metric rows to SQLite can help with artifact organization, resumability, and avoiding large in-memory Python dictionaries. It is not by itself a fix for the OOM.

SQLite is useful for:

- Recording per-sample embedding metadata.
- Tracking split, class, path, and artifact references.
- Resuming long evaluations.
- Writing metric rows incrementally.

SQLite is not ideal as the hot numerical store for large embedding arrays on the NFS PVC. Prefer sharded tensor, NumPy, or Arrow/Parquet artifacts for embeddings, with SQLite used as an index or manifest. The core OOM fix is to avoid dense all-pairs metric computation and to evaluate nearest neighbors with chunking or FAISS.

