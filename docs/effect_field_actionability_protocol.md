# Actionability protocol: matched effect-field interactions

Date fixed: 2026-08-03, after the effect-field validity gate passed  
Status: exploratory development gate  
Fresh holdouts: prohibited

This protocol extends, but does not modify, the hashed validity protocol in
`docs/effect_field_pivot_protocol.md`.

## Stage 1: fixed joint-ablation screen

Run a target-independent joint-ablation screen on all 30 development checkpoints.
Use active channels only. Define channel similarity as the mean exact Jaccard
similarity over the hard/q90/q95/q99 masks. Candidate high and low pairs are the top
and bottom quartiles of checkpoint pairwise similarities. Within each quartile,
retain at most the 64 most extreme pairs before matching.

Match high to low pairs by Hungarian distance after checkpoint-local standardization
of four pair-sum covariates: singleton macro validation KL, total impact mass, mean
support fraction, and training-node mean absolute activation. Greedily retain two
matched high/low pair sets with no channel reused anywhere in a checkpoint. A
checkpoint with fewer than two complete matched sets is explicitly not evaluable.

For each retained pair, mean-ablate both channels together at the same layer and
compute the macro classwise validation KL effect `E_ij`. Singleton effects `E_i` and
`E_j` come from the frozen raw sidecar. Define

`J_ij = E_ij - E_i - E_j`

and the checkpoint contrast

`D_J = mean(J_high) - mean(J_low)`.

Authorize adapter fitting only if median `D_J` is nonzero and cell medians share its
sign in at least three of four cells. `D_J < 0` predeclares the redundancy path
(diverse/MMR selection); `D_J > 0` predeclares the synergy path (coherent/tied
selection). Signed classwise CE nonadditivity is a sensitivity and cannot choose the
path.

## Stage 2: repair interaction confirmation

Only after Stage 1 fixes the path, select one repair target per checkpoint by lowest
baseline validation class F1 (class-id tie break). Restrict the pool to the 12 active
channels with the largest target-class validation KL effect and repeat high/low
matching with fitted singleton utility included. Fit identity-initialized singleton,
independent-pair, and tied-pair affine adapters using training labels. Select epochs
by continuous validation utility with epoch zero as an exact no-op.

Proceed to a repair policy only if the Stage-1-directed interaction contrast has the
same sign in at least three cells and the corresponding tied or diverse policy beats
matched random and direct functional ranking at equal channel and parameter budgets.

