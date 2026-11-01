# Final prospective holdout: effect-field interpretation only

Date fixed: 2026-08-04, before any model was trained on either holdout  
Status: prospective validation-only external replication  
Test labels and test performance: permanently withheld from this study

## Frozen datasets

### Cora-ML

- immutable source: `graph2gauss` commit
  `240248289d38c22ba28ea8e56b7d6873e291499b`;
- archive URL:
  `https://raw.githubusercontent.com/abojchevski/graph2gauss/240248289d38c22ba28ea8e56b7d6873e291499b/data/cora_ml.npz`;
- archive SHA-256:
  `bff04cbfaf903581de18b28dd112b92b2e12c05dfcb1cfc5f6875a5a280d9872`;
- 2,995 nodes, 8,416 raw directed edges, 16,316 directed edges after
  symmetrization, 2,879 features, and 7 classes;
- five stratified 70/15/15 splits with seeds `17, 29, 43, 71, 101`.

### Chameleon-filtered

- immutable source: official `heterophilous-graphs` commit
  `a431395582e929d88271309716bea4fe24ce6318`;
- archive URL:
  `https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/a431395582e929d88271309716bea4fe24ce6318/data/chameleon_filtered.npz`;
- archive SHA-256:
  `bf46f07e1fb5249280447e5fe3100f3e82fc4b93ad1e13ffcfdec924b6ac0bb5`;
- 890 nodes, 8,854 stored undirected edges, 17,708 directed edges after
  symmetrization, 2,325 features, 5 classes, and 10 published splits;
- the filtered archive is used to avoid the duplicate-node leakage documented for
  the original Chameleon benchmark.

Archive schemas, dimensions, and hashes were checked before freezing. No model was
trained and no validation or test outcome was inspected during selection.

## Frozen models and training

Train three architectures on every split:

1. thesis GCN: `input -> 32 -> 32 -> classes`, ReLU, dropout 0.5, Adam,
   learning rate 0.01, weight decay 0.0005;
2. residual GCN: width 64, five residual graph blocks, GELU, layer normalization,
   dropout 0.2, AdamW, learning rate 0.001, no weight decay;
3. matched MLP: the same `input -> 32 -> 32 -> classes` widths and training
   hyperparameters as the thesis GCN, with no adjacency use.

Use at most 500 epochs, patience 100, minimum validation-loss improvement 0.0001,
and one CPU thread. Select checkpoints by validation loss only. Evaluation splits
contain validation only; test labels are replaced by `-1` before training and
evaluation. No test reveal is part of this protocol.

## Stage A: effect-field replication

At the frozen layer (first hidden layer for thesis GCN and MLP; first residual block
for residual GCN), collect train-plus-validation intervention effects. Use the same
training-node-mean channel intervention, primary hard/q90/q95/q99 mask-Jaccard
field, magnitude-plus-uniform-node-role baseline, 1,000 common channel-label
permutations, and within-cell leave-one-checkpoint-out ridge prediction.

The GNN effect-field claim passes only if:

1. the active-channel checkpoint median incremental class-KL alignment is positive;
2. at least three of four GNN cell medians are positive;
3. the all-channel checkpoint median is positive; and
4. active-channel LOCO has non-negative mean RMSE and MAE reductions and improves
   at least one predeclared metric overall with positive cell means in at least
   three of four cells.

For each GNN backbone, compare its active mask-field increment with the seed/split-
matched MLP on each dataset. Graph specificity passes for that backbone only if the
overall paired median and both dataset medians are positive. Report `strong` if both
GNN backbones pass, `partial` if one passes, and `not_supported` if neither passes.
Graph specificity is secondary and cannot invalidate a passing architecture-general
effect-field replication.

## Stage B: fixed joint-ablation replication

Run only if the Stage-A GNN claim passes. Reproduce the frozen high-versus-low
mask-overlap matching, two globally disjoint matched sets per checkpoint, and
mean-ablation nonadditivity estimand. The positive-synergy pattern passes if the
overall `D_J` median is positive and at least three of four GNN cell medians are
positive. If any checkpoint lacks both complete matched sets, Stage B is not
evaluable.

This stage is interpretation-only. It cannot authorize a repair experiment.

## Final stopping rule

After Stage B, stop experimentation and draft the paper from all passed and failed
results. Do not change thresholds, add a repair objective, open Amazon-ratings, or
score Cora-ML/Chameleon test labels.
