# Repair-interaction protocol v2: confirm the coherent/tied path

Date fixed: 2026-08-03, after v1 exposed a matching-feasibility failure  
Status: exploratory development gate  
Fresh holdouts and test outcomes: prohibited

Version 1 used unconstrained Hungarian assignment followed by greedy channel
deduplication. One checkpoint produced fewer than two disjoint assignments before
any pair adapter was fit there. The v1 run was stopped; its partial caches are
retained as infrastructure-only artifacts. No v1 pair outcome was inspected or used
to design v2. Version 2 changes only the feasibility solver: it selects the exact
minimum-total-distance pair of high/low assignments subject to eight globally
distinct channels.

## Checkpoint target and channel pool

For each of the same 30 checkpoints, choose one target class using the lowest
baseline validation F1, with class id as the deterministic tie break. Use the same
hidden layer as the frozen effect-field cache. Among active channels, retain exactly
the 12 channels with the largest target-class validation KL effect, breaking ties by
channel id.

Compute mean hard/q90/q95/q99 mask Jaccard similarity within this pool. High and low
pairs are the top and bottom similarity quartiles. Construct every internally
disjoint high/low assignment from the at-most-64 candidates per side. Select exactly
two assignments minimizing total standardized covariate distance subject to no
channel reuse across any of their four pairs. The five pair-sum covariates are fitted
singleton repair utility, target-class validation KL, total impact mass, mean support
fraction, and training-node mean absolute activation. If no eight-channel solution
exists, the checkpoint and overall gate are not evaluable.

## Frozen adapters and optimization

Fit an identity-initialized affine adapter for each of the 12 singleton channels.
For each of the four retained pairs, fit an independent adapter with separate
scale/shift parameters and a tied adapter with one shared scale/shift pair.

The backbone is frozen. Train with training labels only and select by continuous
validation utility, with epoch zero as an exact no-op. Use Adam with learning rate
`0.01`, at most `30` epochs, patience `10`, minimum utility improvement `1e-6`, target
CE weight `1`, non-target KL weight `1`, positive collateral CE penalty `1`, identity
mean-square weight `0.001`, and one CPU thread.

## Estimands and gate

For pair `(i,j)`, define

`I_ij = U_ind(i,j) - U_i - U_j`

and

`G_ij = U_tied(i,j) - U_ind(i,j)`.

At each checkpoint report `D_int = mean(I_high) - mean(I_low)` and
`Gamma = mean(G_high) - mean(G_low)`.

The coherent/tied path is confirmed only if both checkpoint medians are positive and
each contrast has positive cell medians in at least three of four cells. If only
`D_int` passes, field similarity identifies synergistic repair pairs but does not
justify tying. If neither passes, stop repair-method development.

