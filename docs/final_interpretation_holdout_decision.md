# Final prospective holdout decision

Date: 2026-08-04  
Frozen protocol: `docs/final_interpretation_holdout_protocol.md`  
Protocol SHA-256: `0ca267b72b7e40c075b11139c7ea7e1f0d7e915136fa10d4237356ef0bbfc1d0`

Raw-effects report: `artifacts/final_holdout/raw_effects.json`  
Raw-effects SHA-256: `b56b8312a6cd2798b88537dd3da71c2dc138009e54e1042048cd52614399ed05`  
Effect-field report: `artifacts/final_holdout/effect_field.json`  
Effect-field SHA-256: `30ae6c7ba9cdf4b590deff4641c54639a9e12ea6c83bc9a4bf376db2947f7648`  
Joint-ablation report: `artifacts/final_holdout/actionability.json`  
Joint-ablation SHA-256: `dbcfeb609c1b5048893f354c68ff30deb57e904ffb70d320d7843c49b52fbfcf`

All three reports were reproduced byte-for-byte from exact caches. The analysis
used validation outcomes only. Test labels and test performance remained blinded.

## Prospective sample

The holdout comprised 45 independently trained models on two datasets not used in
the development gates: Cora-ML and duplicate-filtered Chameleon. Each dataset used
a thesis GCN, a residual GCN, and a width-matched graph-free MLP. Cora-ML used five
fixed seeds; Chameleon used its ten official fixed splits. The GNN analysis therefore
contained 30 checkpoints in four dataset/backbone cells.

## Stage A: effect-field replication passes

The frozen effect-field gate passed all four criteria. For active channels, the
incremental class-KL alignment of mask-field similarity over the strongest raw-field
control had median **+0.169507** and was positive in **29/30** GNN checkpoints. The
all-channel median was **+0.109494**. Every GNN cell had a positive active-channel
median:

| Dataset / backbone | Checkpoints | Median incremental alignment |
|---|---:|---:|
| Chameleon / residual GCN | 10 | +0.055504 |
| Chameleon / thesis GCN | 10 | +0.164909 |
| Cora-ML / residual GCN | 5 | +0.197529 |
| Cora-ML / thesis GCN | 5 | +0.400641 |

The checkpoint-weighted leave-one-checkpoint-out gains were +0.005605 RMSE,
+0.001753 MAE, +0.049729 Spearman, and +0.018110 cosine. Spearman improved in three
of four cells, satisfying the prospective prediction rule. This result must be
described with its heterogeneity: the two Chameleon cells had negative RMSE and MAE
gains, and only the Chameleon thesis-GCN cell had a slightly positive Spearman gain.
The robust result is the kernel-alignment replication; cross-checkpoint prediction
is supportive but uneven.

## Graph-free control: specificity remains partial

The matched MLP also showed a strong effect-field signal: median incremental
alignment **+0.086387**, positive in **13/15** checkpoints and in both dataset cells.
Seed-matched GNN-minus-MLP contrasts therefore support graph specificity for only
one backbone:

| GNN backbone | Paired median | Dataset medians | Decision |
|---|---:|---|---|
| Thesis GCN | +0.174235 | Chameleon +0.107837; Cora-ML +0.224307 | pass |
| Residual GCN | +0.003835 | Chameleon +0.004095; Cora-ML -0.021896 | fail |

The representation is consequently an architecture-general functional fingerprint,
not a phenomenon that can be attributed uniquely to message passing. The original
structural motivation remains important, but the earlier topology gate and these
controls rule out a general claim that induced graph topology supplies the signal.

## Stage B: fixed joint-ablation interaction passes

Because Stage A passed, the protocol authorized the fixed joint-ablation replication.
The primary contrast
`D_J = high-overlap minus low-overlap macro-KL nonadditivity` had median
**+0.000348** and was positive in **27/30** checkpoints. All four cell medians were
positive:

| Dataset / backbone | Checkpoints | Median D_J |
|---|---:|---:|
| Chameleon / residual GCN | 10 | +0.000057 |
| Chameleon / thesis GCN | 10 | +0.000520 |
| Cora-ML / residual GCN | 5 | +0.000539 |
| Cora-ML / thesis GCN | 5 | +0.003120 |

The CE-sensitivity contrast was also positive in 26/30 checkpoints, with median
+0.000332. The interaction is highly consistent but small in absolute KL units;
the paper should not equate statistical consistency with a large operational effect.

## Locked interpretation and stopping decision

The final evidence supports three claims:

1. class-conditional intervention effect fields are reproducible functional
   fingerprints of hidden channels;
2. similarity between those fields predicts which fixed channel pairs interact
   nonadditively under joint ablation;
3. neither raw support topology nor graph message passing is necessary for the
   core fingerprint signal.

It does **not** support a repair claim. The separate prospective repair-interaction
gate found negative repair synergy in 22/30 checkpoints and all four development
cells. The frozen final protocol prohibits reopening that branch regardless of the
positive ablation result. No repair experiment is authorized.

The experimental sequence is complete. The next phase is manuscript construction,
figures, related work, and the reproducibility package, with all confirmatory test
outcomes kept sealed unless a separately frozen reveal protocol is approved.
