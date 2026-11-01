# Expansion decision: effect fields replicate, graph specificity is partial

Date: 2026-08-04  
Effect-field report: `artifacts/effect_field_expansion.json`  
Effect-field SHA-256: `1785bfd302d887e598779e1f90cdd8d807afeab655ece81f8bd3ab52613f1fd9`  
Joint-ablation report: `artifacts/effect_field_actionability_replication.json`  
Joint-ablation SHA-256: `a6476f89fd0ea9a0a507d9ab5e21f32a897bbb2e6f3f3f01021d403795c0ce98`

## High-accuracy replication

The primary mask-field result replicated on CitationFull CiteSeer despite its high
baseline classification accuracy. All ten checkpoints had positive incremental
class-KL alignment. The checkpoint median was +0.197628; cell medians were +0.186186
for the residual GCN and +0.326788 for the thesis GCN. RMSE, MAE, Spearman, and cosine
LOCO metrics improved in both cells. The frozen Stage-A gate passed.

The fixed joint-ablation result also replicated. All ten CiteSeer checkpoints had
positive `D_J`; the checkpoint median was +0.000434, with positive medians in both
GNN cells. CE sensitivity was positive in 10/10 checkpoints. This supports the claim
that high-overlap effect fields identify greater nonadditivity under the fixed
mean-ablation intervention.

Both reports were reproduced byte-for-byte from exact caches. No test labels, test
performance, fresh holdout, or confirmatory artifact entered either analysis.

## Graph-free control

The matched MLP control was itself strong: primary incremental mask-field alignment
had median +0.269992 and was positive in 19/20 checkpoints across three datasets.
Seed-matched GNN-minus-MLP comparisons supported graph specificity for the thesis
GCN but not the residual GCN:

| GNN backbone | Paired median | Positive dataset medians | Decision |
|---|---:|---|---|
| Thesis GCN | +0.018198 | CiteSeer, Roman Empire | pass |
| Residual GCN | -0.072800 | Roman Empire only | fail |

Therefore graph specificity is **partial**, not general. The robust representation
claim is that intervention effect fields are functional fingerprints of shared
hidden channels. They are defined over graph nodes in a GNN, but the predictive
signal cannot be attributed uniquely to message passing or graph structure.

## Paper consequence

The paper should retain the structural origin of the idea while reporting the
boundary honestly:

1. induced support topology does not add information beyond strong field controls;
2. support-overlap fields predict classwise effects across low- and high-accuracy
   GNNs and also in matched MLPs;
3. field overlap predicts fixed joint-ablation nonadditivity across 40 GNN
   checkpoints on three datasets;
4. this interaction signal does not identify repair-synergistic pairs.

The next and final experiment is a prospectively frozen interpretation-only
replication on untouched Cora-ML and duplicate-filtered Chameleon. The repair branch
remains closed.
