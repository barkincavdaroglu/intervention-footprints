# Decision: learned directions versus a displacement-matched null (W1.c-bis)

Date decided: 2026-08-04

Status: **`direction_does_not_matter`; superposition arc closed**

Exploratory expansion on previously inspected development graphs. Does not alter
the completed final holdout, does not reopen repair, and does not authorize any
test-label inspection.

## What was tested

W1.c could not decide whether a learned dictionary beats an arbitrary direction,
because a sparse feature moves the layer `14.5x` to `3302x` less than a channel
or a rotated direction, and both of its nulls were displacement-mismatched.

W1.c-bis removes that confound at the source. For feature `k` with coefficient
displacement `delta_k(v)` and decoder direction `d_k`, the true ablation is
`a - delta_k d_k`. The matched null keeps `delta_k` and `||d_k||` exactly and
replaces only the direction with a random unit vector. Node for node it has the
identical displacement magnitude, the identical sparsity pattern, the identical
node support, and the identical live-direction count, so exactly one thing
differs: whether the direction was learned.

A second, descriptive null (`shuffled_null`) keeps the learned direction and
permutes each coefficient column across nodes, destroying only *where* the
feature acts. Together the two decompose the fingerprint into a direction
component and a spatial-support component.

Ten conditions per checkpoint over the same 30 development checkpoints:
`channel`, three `sae_topk` seeds refit under W1.c's namespace, three paired
`matched_null` seeds, three `shuffled_null` seeds.

## Frozen-gate result

| Question | Frozen result | Decision |
|---|---:|---|
| Direction gain, `active_directions` | `+0.0359` (`28/30` positive, `4/4` cells) | does_not_matter |
| Direction gain, `matched_count` | `+0.0247` (`19/30` positive, `2/4` cells) | does_not_matter |
| Subsets agree | yes | **`direction_does_not_matter`** |
| Seed sign consistency | all three positive in both subsets | — |

All five validity gates passed: cache replay `exact` on 30 of 30 checkpoints,
coverage 30/30, reconstruction floor met (minimum explained variance `0.939`),
sparsity floor met, and the displacement-match gate passed with a **worst
relative mismatch of `1.391e-07`** against a one percent tolerance.

The mean absolute cosine between paired learned and null directions is `0.1121`,
so the null is a genuinely different direction rather than an accidentally
aligned one.

The direction gain is consistently positive but small. Learned directions are
very slightly better than random ones of identical magnitude and support; the
effect does not approach the `0.05` threshold this program has used since W1.a.

## The decomposition

| null | what it destroys | alignment cost, `active` | alignment cost, `matched` |
|---|---|---:|---:|
| `matched_null` | the learned direction | `+0.0359` (28/30, 4/4) | `+0.0247` (19/30, 2/4) |
| `shuffled_null` | the spatial support | `+0.1146` (30/30, 4/4) | `+0.1097` (29/30, 4/4) |

Destroying *where* a feature acts costs three to four times as much alignment as
destroying *which direction* it points, and does so in 30 of 30 checkpoints and
all four cells on both subsets.

## Retraction of two W1.c results

W1.c reported two clean positives against rotations. Both were measured against
a displacement-mismatched null, and neither survives the matched one:

| endpoint | versus rotations (W1.c) | versus matched null (W1.c-bis) |
|---|---|---|
| monosemanticity (class-effect entropy gain) | `+0.1278`, `30/30`, `4/4` cells | `+0.0104`, `20/30`, `3/4` cells |
| localization (participation-ratio gain) | roughly nine times more concentrated | `-0.0006`, `11/30`, `1/4` cells |

Absolute levels: class entropy `0.7321` for the dictionary against `0.7465` for
the matched null; participation ratio `0.0417` against `0.0408`. A random
direction carrying the same sparse displacement profile is just as
class-concentrated and just as spatially localized as a learned one.

Both W1.c findings were therefore artifacts of sparsity and magnitude, not
properties of learned directions, and must not be cited as evidence for learned
features. `docs/feature_basis_decision.md` carries an addendum pointing here.

## Interpretation

Read with W1.a (`basis_agnostic`: Haar rotations tie the channel basis) and W1.c
(`inconclusive`: the dictionary ties at matched cardinality), the three results
are consistent and now mechanistically explained:

> The effect-field fingerprint is a function of **where** and **how much** an
> intervention perturbs the layer, not of the **direction** in activation space.

This is a statement about what the endpoint measures, and it sharpens every
earlier result that used it, including H1. It also re-frames the closed H3
repair branch a second time. The obstacle is not that channels are polysemantic
mixtures; it is that the effect field never resolved directions at all, so a
grouping derived from it was never positioned to identify an editable unit.

The superposition hypothesis is not refuted in general. What is established is
narrower: at these widths, on these graphs, and under this endpoint, a learned
sparse basis is not a better description of the layer than an arbitrary one of
matched magnitude and support.

## Decision and paper use

1. **Close the superposition arc.** Do not run W1.d and do not preregister it.
2. Do not cite W1.c's monosemanticity or localization results as evidence for
   learned features; cite the retraction above instead.
3. The decomposition (direction versus spatial support) is the citable positive
   result of Workstream 1 and belongs in the paper as a boundary on what
   intervention-derived fields can resolve.
4. Proceed to W2 circuits and W3 sparsity. W2.a's exact cross-layer edges are a
   different estimand that does not inherit this limitation, which is now the
   main reason to prefer it next.

## Reproducibility record

- Full report: `expansion/artifacts/matched_null.json`
  - SHA-256: `ac15fdd95a788b8beb96ea016134a0260064cd506923488a2bd7fb0f4c61cfdb`
- Protocol: `expansion/docs/matched_null_protocol.md`
  - SHA-256: `7c06a317ddaf39f2e14963a0213bdd00b6111bb5aac2bc090d34d8881c8fd847`
- Per-checkpoint caches: `expansion/artifacts/matched_null_runs/` (30 files)
- Shared raw report across W1.a, W1.c, and W1.c-bis:
  `8f04c0dd7e0c7c5818d1e3d7...` (identical in all three reports)
- Runtime: Python 3.11.7, NumPy 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0,
  PyTorch 2.7.1
- Cache replay tier: `exact` on 30 of 30 checkpoints
- Worst displacement mismatch: `1.391e-07` (tolerance `1e-2`)
- Test labels or test performance used: **no**
- Fresh holdouts used: **no**
- W1.d authorized: **no**
- Repair reopened: **no**
