# Search Space V0

This document defines the first supported search space for
`glasslab-metric-search`.

The point of v0 is not to search everything. The point is to search a bounded,
reviewable subset of deep metric learning choices that are plausible for art
similarity and forgery-style retrieval tasks.

The reference framing comes from the deep metric learning survey in
`arXiv:2312.10046`, which organizes the field around:

- pair-based objectives
- proxy-based objectives
- auxiliary regularization methods

V0 turns that survey framing into a concrete, machine-enumerable search space.

## V0 design rule

V0 mutates config space, not arbitrary Python.

Allowed mutation targets:

- backbone freeze versus fine-tune
- loss family
- loss hyperparameters
- miner family
- regularizer family and weight
- trainer batch size and learning rate
- evaluator weighting

Not allowed in v0:

- arbitrary code edits
- optimizer rewrites
- architecture rewrites beyond choosing from a small backbone registry
- custom data-loader logic
- custom benchmark definitions

## Target workload

The first workload is bounded art-metric search:

- one dataset family
- one embedding benchmark family
- one GPU per run
- fixed evaluation bundle
- fixed composite ranking function

## Supported registries

### Backbones

- `vit_base_patch16`
- `convnext_base`
- `resnet50`

V0 expectation:

- `vit_base_patch16` is the default baseline
- `freeze_backbone` is the primary early control
- full backbone substitution is allowed only from the explicit registry

### Losses

Pair-based:

- `triplet`
- `contrastive`
- `multi_similarity`

Proxy-based:

- `proxy_anchor`
- `proxy_nca`
- `proxy_gml`

### Miners

- `batch_hard`
- `semi_hard`
- `distance_weighted`
- `multi_similarity`
- `hard_negative`

### Regularizers

- `none`
- `embedding_norm`
- `language_alignment`
- `center_smoothing`

### Evaluators

- `art_retrieval_v1`

V0 uses one evaluator bundle so run comparisons stay coherent.

## Baseline search spaces

Two baseline search-space families are supported initially:

### `art-metric-proxy-v0`

Emphasizes proxy-based methods:

- `proxy_anchor`
- `proxy_nca`
- `proxy_gml`

### `art-metric-pair-v0`

Emphasizes pair/triplet-style methods:

- `triplet`
- `contrastive`
- `multi_similarity`

## Fixed evaluation bundle

Each completed run should produce:

- retrieval recall@10
- forgery AUROC
- robustness score
- instability penalty

These are combined into the current composite ranking score:

```text
0.45 * retrieval_recall_at_10
+ 0.35 * forgery_auroc
+ 0.20 * robustness_score
- instability_penalty
```

This weighting is intentionally simple in v0. If it changes, it should change
explicitly in the evaluator contract, not implicitly in ad hoc notebooks.

## Mutation policy

V0 mutation should prefer small bounded changes.

Good examples:

- `proxy_anchor -> proxy_nca`
- margin `0.1 -> 0.2`
- `freeze_backbone=true -> false`
- `language_alignment.weight 0.05 -> 0.01`
- batch size `64 -> 32`

Bad examples for v0:

- switching dataset family mid-campaign
- changing benchmark definitions
- introducing unregistered losses or miners
- changing run budget and objective simultaneously

## Success condition for v0

V0 is successful if Glasslab can:

1. schedule many one-GPU runs from these configs
2. compare them with one stable metric bundle
3. promote the best few candidates
4. launch the next mutation batch without human shell babysitting
