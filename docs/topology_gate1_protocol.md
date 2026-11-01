# Topology Gate 1 development protocol

Date fixed: 2026-08-03  
Status: exploratory extension on previously inspected development datasets  
Fresh holdouts: prohibited until Gate 1 and Gate 2 are frozen

## Question

Does the topology of a hidden channel's intervention-induced effect support add
information about its classwise causal effect beyond impact magnitude, affected-node
identity, and parent-graph node roles?

## Development strata

The four fixed strata are Amazon Computers and Roman-empire, each with the thesis
GCN and residual GCN checkpoints already used by the structural development study.
All representation choices and stopping decisions use validation-derived effects
only. Existing test results remain descriptive and are not inputs to this gate.

The primary analysis excludes channels whose maximum scoped total-variation impact
is at most `1e-12`. An all-channel analysis is a required sensitivity check.

## Fixed representation blocks

1. Impact-magnitude summaries.
2. The L1-normalized continuous node-effect field.
3. Affected-node masks compared by Jaccard similarity.
4. Uniformly pooled parent-graph node-role features, without induced edges.
5. The complete legacy structural-impact signature, explicitly labeled fused.
6. Normalized WL subtree representations of the induced support.  The eligible
   initial labels are constant and induced-degree bins.  Each support quantile
   (`q90`, `q95`, `q99`) and each depth (`h0` through `h4`) remains a separate
   view.  The primary view for each WL candidate is `q95`; its other two
   quantiles are required sensitivities.  Parent-degree and parent-degree/core
   labels are reported only as role-confounded sensitivities.
7. Diffusion signatures initialized by the L1-normalized continuous effect field.
   Times `0.01`, `0.1`, `1`, and `10` remain separate views; time `1` is the
   primary view and adjacent-scale stability is required.

Constant-label WL is the strict topology-only representation conditioned on the
support mask. Degree/core WL is topology plus fixed parent-node roles. Diffusion is
a topology-conditioned effect-field representation, not topology-only, because its
L1-normalized initialization retains the spatial distribution of impact. It can
establish an incremental graph contribution only by outperforming the identical
undiffused field and its rewired counterpart. None of these representations may use
class labels. Empty supports receive an explicit indicator and are excluded in the
primary active-channel analysis.

## Primary endpoint and inference

For each checkpoint, the primary endpoint is Spearman correlation between the
upper triangles of:

- a representation-similarity matrix; and
- cosine similarity of per-class validation KL effect vectors.

Pairs are not treated as independent observations. A within-checkpoint permutation
test uses 1,000 common simultaneous row/column permutations of complete channel
effect-vector labels for every candidate and control. Checkpoint-level estimates
are then aggregated with every checkpoint receiving equal weight; stratum summaries
are reported separately.

The primary incremental contrast for a topology-only representation is its
checkpoint-level alignment minus the strongest of blocks 1 through 4. The legacy
fused signature is not eligible to establish that topology adds information.

## Topology null

For every channel, preserve the continuous node effect and affected-node identities
while replacing the parent graph with a deterministically degree-preserving rewired
graph. Recompute topology representations on that graph. The paired real-minus-
rewired alignment is the topology-null contrast. Use two complete rewires per
checkpoint, each requesting one successful double-edge swap per original undirected
edge. Rewiring seeds and completed edge swaps must be recorded. Failed or incomplete
swaps are rejected and retried and cannot silently count as a null.

## Required sensitivities

- active channels versus all channels;
- q90, q95, and q99 affected-node views;
- classwise KL effects versus signed classwise CE changes;
- one, five, and ten percent edge dropout stability;
- checkpoint-stratified results for both datasets and both backbones;
- leave-one-checkpoint-out prediction of class-effect vectors when the class
  dimension is shared within a stratum.  This is a within-stratum, multi-output
  ridge model.  Preprocessing is fit on training checkpoints only; each training
  checkpoint receives equal total weight; and fixed-dimensional hashed prediction
  features do not depend on the held-out checkpoint.  The target is each channel's
  L1-normalized classwise validation-KL vector.

For the prediction criterion, “not worse” means that the checkpoint-weighted mean
RMSE reduction and MAE reduction are both non-negative.  “Improves” means that at
least one of RMSE reduction, MAE reduction, Spearman gain, or cosine gain has both a
positive checkpoint-weighted mean and a positive cell mean in at least three of the
four cells.

## Gate decision

A topology-only representation passes Gate 1 only if all of the following hold:

1. its median incremental alignment over the strongest non-topological block is
   positive across checkpoints;
2. the incremental contrast is positive in at least three of four strata;
3. its median real-minus-rewired contrast is positive, without a sign reversal in
   more than one stratum;
4. the result remains positive in the active-channel analysis and at both q95 and
   q99, or for at least two adjacent diffusion scales;
5. leave-one-checkpoint-out prediction is not worse than the strongest confound
   model and improves at least one predeclared error or rank metric.

Constant-label `h0` is necessarily identical for every non-empty support after
row normalization.  It is retained as a diagnostic null but is not eligible for
selection and cannot make the gate non-evaluable.  Constant-label `h1` through
`h4` and induced-degree `h0` through `h4` are the eligible WL candidates.

Among passing representations, select the one with the largest mean incremental
alignment. If two are within `0.01`, prefer normalized WL over diffusion and the
shallower WL depth over the deeper one. Constant-label and induced-degree WL are
eligible candidates; parent-role WL and induced-shape summaries remain sensitivity
analyses. This simplicity rule and the primary views above are fixed before the
separated q/depth/scale report is run.

If no representation passes, stop topology-based repair development. The honest
pivot is an ablation-effect-field paper. If a representation passes, freeze its
definition and proceed to Gate 2, which distinguishes channel synergy from
redundancy using matched same-cluster and cross-cluster repair interactions.
