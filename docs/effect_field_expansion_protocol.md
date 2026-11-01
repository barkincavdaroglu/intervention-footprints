# Effect-field expansion: high-accuracy replication and graph-free control

Date fixed: 2026-08-03, after the repair-interaction gate failed  
Status: development replication on previously inspected datasets  
Fresh holdouts, confirmatory artifacts, and test outcomes: prohibited

This expansion cannot reopen or tune the failed repair branch. Its purpose is to
test the remaining interpretation claim under the two conditions most likely to
change the paper's conclusion: a high-accuracy dataset with little classification
headroom and graph-free models with matched hidden widths.

## Frozen matrix

The raw report contains nine cells over three datasets and three backbones. Existing
Amazon Computers and Roman Empire GNN sidecars are reused under their exact raw-cache
identity. New effects are collected for:

- CitationFull CiteSeer thesis GCN and residual GCN: five checkpoints each;
- matched MLPs on CiteSeer and Amazon Computers: five checkpoints each;
- the matched MLP on Roman Empire: ten checkpoints.

The MLP performs no adjacency aggregation. It is a graph-free control, not a third
GNN architecture. All interventions use training-node channel means, and all effect
fields and class-effect vectors use training plus validation nodes or validation
labels only as already specified by the raw-cache protocol.

## Stage A: effect-field replication

Use the unchanged primary `mask_field`: mean hard/q90/q95/q99 support Jaccard.
Compare it with the unchanged magnitude-plus-uniform-node-role baseline. Retain
`raw_field` and `continuous_plus_mask` as secondary candidates. Use 1,000 common
row/column channel-label permutations per checkpoint and leakage-safe
leave-one-checkpoint-out ridge prediction within each dataset/backbone cell.

The CiteSeer GNN replication succeeds only if all four conditions hold:

1. the active-channel checkpoint median incremental class-KL alignment is positive;
2. both CiteSeer GNN cell medians are positive;
3. the all-channel checkpoint median incremental class-KL alignment is positive;
4. active-channel LOCO has non-negative mean RMSE and MAE reductions and at least
   one positive predeclared metric in both cells.

This is an interpretation replication, not a repair authorization.

## Graph-specificity control

For each dataset and seed/split, pair each GNN checkpoint with the matched MLP and
compute the difference in active-channel primary incremental alignment:

`G = increment(GNN) - increment(MLP)`.

Assess thesis GCN and residual GCN separately. Graph-specificity is supported for a
backbone only if the paired checkpoint median is positive and at least two of three
dataset medians are positive. It is `strong` if both GNN backbones pass, `partial` if
one passes, and `not_supported` if neither passes. A non-positive result means the
effect field is an architecture-general functional fingerprint and must not be sold
as graph-specific evidence.

## Stage B: fixed joint-ablation replication

Run this stage only if the CiteSeer GNN Stage-A replication succeeds. On those ten
checkpoints, reproduce the frozen high-versus-low mask-overlap matching and fixed
pair mean-ablation estimand from the actionability protocol. Two globally disjoint
matched sets are required at every checkpoint; otherwise the stage is not
evaluable.

The positive-synergy pattern replicates only if the overall median `D_J` is positive
and both CiteSeer GNN cell medians are positive. A negative overall median with both
cell medians negative supports the inverse redundancy pattern. Any other sign
configuration is mixed. This result remains interpretive and cannot authorize
another repair experiment.

## Stopping rule

After Stage B, stop development experiments and decide whether the evidence supports
an interpretation-centered submission. Do not add another repair objective, select
new thresholds from these outcomes, or inspect the sealed Amazon-ratings test.
