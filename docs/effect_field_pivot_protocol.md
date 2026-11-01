# Development protocol: ablation effect fields as functional fingerprints

Date fixed: 2026-08-03, after the topology Gate 1 no-go  
Status: post-Gate-1 exploratory pivot  
Fresh holdouts: prohibited until this method and its actionability gate are frozen

## Motivation and claim boundary

Gate 1 showed that topology-aware representations do not improve on the strongest
non-topological controls.  It also showed a stable positive result: hidden channels
that affect the same graph nodes tend to have similar classwise causal effects.

The pivot claim is therefore narrower and different:

> A shared GNN channel's node-indexed response to a validation-only mean-ablation
> intervention is a functional fingerprint.  Overlap between these fingerprints
> predicts class effects and may identify redundant or synergistic repair targets.

An effect-support mask is not described as a causal message-passing subgraph, and
mask overlap is not described as topology.  This pivot was selected after inspecting
the four Gate 1 development cells; those cells are exploratory evidence only.

## Fixed development data and leakage rules

Use the same 30 validation-only checkpoints from the four Gate 1 cells.  Existing
test outcomes and the previously revealed Amazon-ratings result are excluded from
all choices and endpoints.  Test labels, test performance, confirmatory artifacts,
and fresh holdouts are rejected by code.  A checkpoint, never a channel pair, is the
aggregation unit.

## Frozen representation comparison

For channel `i`, retain the validation-scoped total-variation field

`s_i(v) = TV(p_v, p_v^(-i))`.

Compare four fixed models:

1. **Baseline:** impact-magnitude summaries plus uniformly pooled parent-node roles.
2. **Raw field:** baseline plus the L1-normalized node-indexed field, represented for
   cross-checkpoint prediction by the existing deterministic signed node hash.
3. **Mask field (primary):** baseline plus the mean of exact hard/q90/q95/q99 mask
   Jaccard kernels, represented for prediction by the existing deterministic signed
   mask hashes.
4. **Continuous + mask:** baseline plus both raw-field and mask-field features.

The primary choice is mask field because it was the strongest control in 26/30 Gate
1 checkpoints.  This is an explicitly development-selected hypothesis that requires
fresh confirmation.  No topology, WL, diffusion, graph rewiring, class label, or
class-effect value enters the representation.

## Fingerprint validity endpoints

The primary within-checkpoint endpoint is mask-field Spearman kernel alignment with
cosine similarity of classwise validation-KL effect vectors.  Report signed classwise
CE change as a required sensitivity.  Use 1,000 common simultaneous channel-label
permutations and report candidate-minus-baseline alignment.  Report active and all
channels separately.

The cross-checkpoint endpoint is within-cell leave-one-checkpoint-out multi-output
ridge prediction of each channel's L1-normalized classwise validation-KL vector.
Preprocessing is fit on training checkpoints only, and every training checkpoint
receives equal total weight.

The mask field clears the development validity gate only if:

1. its active-channel incremental alignment over the fixed baseline has a positive
   median and positive cell median in at least three of four cells;
2. its all-channel incremental median is positive;
3. LOCO checkpoint-weighted RMSE and MAE reductions are non-negative; and
4. at least one of RMSE reduction, MAE reduction, Spearman gain, or cosine gain has
   a positive checkpoint mean and a positive cell mean in at least three cells.

If the primary mask model fails, stop the repair pivot.  The remaining result is a
negative topology dissection and is not sufficient by itself for an ICLR submission.

## Actionability gate

If fingerprint validity passes, measure whether field-similar channels are redundant
or synergistic before designing a repair selector.  Within each checkpoint and
target class, form high-mask-similarity and low-mask-similarity pairs matched on
layer, singleton validation utility, impact magnitude, and class-effect magnitude.
Use disjoint pair sets and continuous validation CE/KL utility; do not use thresholded
F1 as the interaction endpoint.

For a pair `(i,j)`, report interaction

`I_ij = U_ij - U_i - U_j`

and the tied-versus-independent adapter contrast

`Gamma_ij = U_tied(i,j) - U_independent(i,j)`.

Proceed to a repair method only if the high-versus-low similarity interaction
contrast has a stable sign in at least three of four cells and the corresponding
tied or diverse policy improves validation utility over matched random and direct
functional-ranking controls at equal channel and parameter budgets.  Otherwise the
effect field is an interpretability result, not an actionable repair grouping.

## Fresh confirmation

Only after both development gates are frozen may the chosen representation and
repair policy be run once on untouched graphs.  No representation, threshold,
clustering, pair-matching, objective, budget, stopping, or reporting rule may change
after fresh outcomes are viewed.

