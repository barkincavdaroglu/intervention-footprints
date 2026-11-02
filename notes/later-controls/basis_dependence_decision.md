# Decision: channel basis versus random rotations (W1.a)

Recorded verdict: **`basis_agnostic`** on both active-direction and all-direction analyses.

Public summary prepared 2026-11-01 from the saved W1.a report and protocol.
The protocol records a 2026-08-04 freeze, with disclosed integration probes and
two amendments before the full study. This is exploratory evidence from 30
previously inspected development checkpoints, not a test on fresh datasets.

## What was tested

Does the classwise-alignment signal require the model's channel axes? The
channel intervention extends to any unit direction: replace each node's
coefficient along that direction with its training-node mean, leaving the
orthogonal component unchanged. In vector notation,

$$
a_v^{(-u)}=a_v-\bigl(\langle a_v,u\rangle-\langle\mu_{\rm train},u\rangle\bigr)u,
\qquad \|u\|_2=1.
$$

The model weights stay fixed. Each intervention changes the hidden states that
the remaining layers receive. Output footprints still measure total variation,
and classwise profiles still average KL divergence over validation nodes.

Conditions were the channel basis, five Haar-random orthonormal bases per
checkpoint, a signed channel permutation, and a PCA basis. The endpoint was
the `mask_field` improvement over the magnitude-plus-node-role baseline for
`class_kl`, in rank-correlation units. Checkpoints, not direction pairs, were
the aggregation units.

## Random rotations preserved the pooled alignment signal

These are medians across checkpoints for the active-direction analysis:

| Condition | Alignment improvement | Positive checkpoints |
| --- | ---: | ---: |
| Channel | +0.2415 | 29/30 |
| Random rotation 00 | +0.2580 | 30/30 |
| Random rotation 01 | +0.2846 | 30/30 |
| Random rotation 02 | +0.2646 | 30/30 |
| Random rotation 03 | +0.2655 | 30/30 |
| Random rotation 04 | +0.2731 | 30/30 |
| Signed channel permutation | +0.2414 | 29/30 |

For each checkpoint, the primary contrast subtracts the median of its five
rotation scores from its channel score. The study then takes the median of
these paired contrasts. This is not the difference between the pooled medians
in the table above.

| Paired channel-minus-rotation contrast | Median | Positive checkpoints | Cells with positive median |
| --- | ---: | ---: | ---: |
| Active directions | −0.006652 | 14/30 | 2/4 |
| All directions | −0.028070 | 12/30 | 1/4 |

The frozen rule classifies a subset as `basis_agnostic` when its median contrast
lies within ±0.05. A privileged-basis verdict requires a contrast beyond that
band, the corresponding sign in at least three of four cells, and that sign in
at least 70% of evaluable checkpoints. The study verdict requires agreement
between subsets. Both subsets were inside the band.

The secondary localization contrast was −0.008048, positive in 8/30
checkpoints and one of four cell medians. It compares participation ratios and
does not determine the alignment verdict.

## The pooled verdict does not establish exact invariance

The four cells differed substantially. Their active-direction paired contrasts
were +0.0390 for Amazon Computers/residual GCN, −0.2815 for Amazon
Computers/narrow GCN, −0.0145 for Roman Empire/residual GCN, and +0.1037 for
Roman Empire/narrow GCN. A pooled tie does not mean every cell or checkpoint
ties, nor that individual footprints are unchanged by rotation.

The recorded interpretation was that the fingerprint reflects the layer
subspace rather than its particular axes. The bounded empirical conclusion is
that channel axes had no pooled practical advantage under this endpoint and
decision rule, at the tested widths of 32 and 64. This motivated testing learned
sparse directions against random controls. The subsequent
[displacement-matched comparison](matched_null_decision.md) addresses a different,
more tightly controlled question.

## Correctness checks and intervention size

All 30 checkpoints passed cache replay, signed-permutation invariance, and
coverage. The channel footprints reproduced the raw cache bitwise. Signed
permutations also reproduced those footprints bitwise after undoing the
permutation. Relabeling changed the downstream alignment calculation slightly:
the largest absolute difference was 0.001891, below the 0.05 decision threshold.
Thus the correctness claim concerns identical footprints, not exactly equal
alignment scores.

PCA's alignment median was +0.1727, positive in 25/30 checkpoints, but its
interventions were smaller. For each condition, the report stores the median
across directions of mean intervention displacement over nodes. Taking the
median of that diagnostic across checkpoints gives 0.080582 for channels,
0.074659–0.078404 for the five random rotations, and 0.035118 for PCA.
The rotations were about 2.7%–7.4% below channels on this pooled diagnostic.
They were not exactly displacement-matched interventions.

| Dataset / architecture | Channel displacement | PCA displacement | Ratio of cell medians |
| --- | ---: | ---: | ---: |
| Amazon Computers / narrow GCN | 0.010285 | 0.000203 | 50.62× |
| Amazon Computers / residual GCN | 0.085990 | 0.007074 | 12.16× |
| Roman Empire / residual GCN | 0.157928 | 0.058440 | 2.70× |
| Roman Empire / narrow GCN | 0.073223 | 0.035118 | 2.09× |

The pooled channel/PCA ratio is 2.29×. PCA's lower alignment therefore cannot
isolate a basis effect from intervention size. These stored aggregate
diagnostics support the cell range 2.1×–50.6×, not the original instrument log's
“~100×” description. Per-direction arrays are not stored in this report.

## Source record

The full report and original protocol remain in the research archive. This
public memo summarizes their frozen results without presenting itself as a
contemporaneous decision memo or an executable replay package.

- Report: `expansion/artifacts/basis_dependence.json`.
  - SHA-256: `bdcf33df32a4708e00b34bf0ee112afd02496ca30e80da5e41b3fe8a4962d86e`.
- Protocol: `expansion/docs/basis_dependence_protocol.md`.
  - SHA-256: `0331106549c50a9bba7958b0ffe610a14a055be6be36bed65c9c22059a20ffbb`.
- Written verdict: `expansion/docs/hypothesis_ledger.md`, entry L8.
- Reported expansion-code SHA-256, verified against the five frozen modules:
  `52231952e658d4659bb68fcfbd226df46dcd4b6edd083a3e86b3cb481cff3a4b`.
- Reported effect-field-code SHA-256:
  `a2aaed3ddda2b97ae89b7d4c810d585810e457dbbb90a2226b6da92b8f203a43`.
- Shared raw-report SHA-256:
  `8f04c0dd7e0c7c5818d1e3d78399b9d671b62308d9c16b2fda04600e3d07ebe2`.

Alignment levels are at `condition_summaries.<condition>.incremental_alignment`.
The paired contrasts, subset verdicts, threshold, and validity gates are under
`basis_dependence_decision`. Paired contrasts can be reaggregated from
`checkpoint_contrasts`. Displacement diagnostics are at
`checkpoint_records[*].conditions.<condition>.diagnostics.summary.intervention_magnitude_mean_median`.

The recorded runtime was Python 3.11.7, NumPy 2.3.3, SciPy 1.17.1,
scikit-learn 1.8.0, and PyTorch 2.7.1. Preparing this summary reaggregated saved
numbers only. It did not execute models or inspect test labels.
