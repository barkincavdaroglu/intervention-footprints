# Decision: seed stability of cross-layer edge structure (W2.a E4)

Date decided: 2026-08-05

Status: **evaluable; verdict `conserved`**, with one sensitivity variant
dissenting and a strong cell dependence, both recorded below

Exploratory expansion on previously inspected development graphs. Does not alter
the completed final holdout, does not reopen repair, does not reopen W2.a, and
does not authorize any test-label inspection. Zero new forward passes.

## What was tested

Whether the cross-layer edge structure W2.a measured is a property of the *task
and architecture* or of the *individual training run*. For every pair of seed
checkpoints within a cell, source and target channels of the primary layer pair
are matched by Hungarian assignment, and the correlation of the matched edge
matrices is compared against a mass-matched surrogate that receives the
identical matching.

Thirty development checkpoints, four frozen cells, primary layer pair per cell
as defined by W2.a. Protocol `cross_layer_seed_stability_protocol.md`
(`fbd5b30f21da0346...`), frozen before any E4 number was computed.

## The control this study had to replace first

E4 was named in the frozen W2.a protocol as "the correlation of matched matrices
against a permutation null". That comparison **cannot fail**. Hungarian
assignment maximizes the matching objective over the permutation group and a
random permutation is a draw from the same group, so the matched value wins by
construction — for any two matrices, related or not. On two *independent* noise
matrices the gap is `0.0032` against `0.3692`.

Run as specified, E4 would have reported a large positive number on any input
and been read as evidence of conservation. The null used here is Hungarian-
matched exactly as the real pair is, and only the matrix is replaced by a
mass-matched shuffle, so the comparison can return zero. Verified: it ties on
independent matrices (`-0.004` to `+0.006`) and separates on planted structure
(`+0.51` to `+0.55`). Recorded as instrument-log entry **I11**.

The same reading found a second defect. The protocol did not fix the
initialization, and alternating Hungarian from an identity start recovers a
planted permutation at chance (`0.04` against `1/64`), which would have
manufactured a false negative. The matcher is initialized on sorted profiles,
which are invariant to the other axis's labelling, and a per-checkpoint positive
control now gates every run.

## Validity

| Gate | Result |
|---|---|
| Matcher positive control | `pass` on **30 of 30**, exact recovery in both axes |
| Cache provenance | `pass`; every `.npz` matches its W2.a record |
| Coverage | 30 of 30 evaluable |
| New forward passes | `0` |

Provenance chain: `seed_stability.json` cites source report `b4cb150e...`
(W2.a), which cites raw report `8f04c0dd...`, the hash shared by all four
expansion studies.

## Result

Primary condition — `strength` matrix, Pearson, within-source-row null:

| subset | median gain | positive | cells |
|---|---:|---:|---:|
| `active_channels` (primary) | **`+0.0985`** | 30/30 | 4/4 |
| `all_channels` | `+0.1351` | 30/30 | 4/4 |

Both subsets agree, both clear the `0.05` threshold, every checkpoint and every
cell is positive. The sensitivity null (entrywise) is larger in both subsets
(`+0.1847`, `+0.2623`), as expected: it preserves less structure and is the
easier null to beat.

**Verdict `conserved`.** Cross-layer edge structure is conserved across training
runs up to channel relabelling.

## Three things that qualify it

**1. The effect is strongly cell-dependent.** Per cell, `active_channels` on the
primary null:

| cell | median gain |
|---|---:|
| `roman_empire/residual_gcn` | `+0.2471` |
| `amazon_computers/residual_gcn` | `+0.1083` |
| `amazon_computers/thesis_gcn` | `+0.0936` |
| `roman_empire/thesis_gcn` | `+0.0320` |

An eightfold spread, and the weakest cell sits *below* the `0.05` threshold on
its own. The frozen rule asks for cell medians to be **positive** in at least
three of four cells, not above threshold in all four, so the verdict stands as
written — but "conserved" is not uniform, and `roman_empire/thesis_gcn` would
not have passed alone.

**2. The rank-based variant dissents.** Substituting Spearman for Pearson gives
`+0.0297` (`active_channels`, primary null), positive in 28/30 and all four
cells but **below threshold**, which the same rule classifies `not_conserved`.

This is informative rather than contradictory. Pearson is dominated by the
large entries; Spearman weights every edge's rank equally, and most edges are
tiny. Read together with L12 — the top 5% of edges carry 24.7% of the mass —
the reading is that **the high-mass edges are conserved across seeds while the
full ordering, including the long tail of weak edges, is much less so**. The
Spearman variant is a sensitivity check listed in protocol section 4, not a term
in the decision rule of section 7, so it does not change the verdict; it bounds
what the verdict means.

**3. Dead channels inflate the naive estimate, exactly as predicted.** The
inflation from `all_channels` to `active_channels` tracks dead-channel count
with no exceptions:

| cell | dead source channels | `all` / `active` |
|---|---|---:|
| `amazon_computers/residual_gcn` | 0 | `0.96x` |
| `roman_empire/residual_gcn` | 0 | `1.00x` |
| `roman_empire/thesis_gcn` | ~4.5 of 32 | `1.74x` |
| `amazon_computers/thesis_gcn` | ~15 of 32 | `2.13x` |

All-zero rows correlate perfectly with all-zero rows for free. Reporting only
`all_channels` would have overstated conservation by a factor of two in exactly
the cells where the estimand is weakest. `active_channels` is the number to
quote. This is hazard 4 and instrument-log entry I7 in their third appearance,
and the first time the two-subset rule caught a magnitude error rather than a
sign error.

## Sensitivity summary

| variant | verdict | `active` / primary null |
|---|---|---:|
| `strength`, Pearson (**primary**) | `conserved` | `+0.0985` |
| `signed`, Pearson | `conserved` | `+0.1131` |
| `strength`, Spearman | `not_conserved` | `+0.0297` |

The `signed` variant initially returned `not_evaluable` because the active mask
summed signed entries, so a live channel whose positive and negative edges
cancelled was marked dead and the subset collapsed to shape `(0, 0)` on both
thesis cells. Protocol section 5 says "nonzero **mass**", so this was an
implementation deviation from the frozen text, not a protocol change; the mask
now uses absolute values. The primary result is provably unaffected — all 75
cached `strength` matrices are non-negative (global minimum `0.0`), so the two
definitions coincide there — and was confirmed bit-identical after the fix.

## What this licenses

A circuit found in one checkpoint is worth looking for in another, which is the
precondition W2.b needed. It does **not** establish that matched channels
compute the same feature: the matching is on edge geometry alone, and nothing
here inspects what a channel represents.

The qualification that matters for W2.b is finding 2. If a circuit account leans
on the weak-edge structure rather than the high-mass edges, this study does not
support carrying it across seeds.

## What it does not license

Any claim about the reserved datasets, about test performance, or about
functional equivalence of matched channels. Nor any claim that conservation is
uniform: one of four cells is below threshold on the primary condition.

## Reproduction

```
PYTHONPATH=src:expansion/src python3.11 \
  -m gnn_expansion.seed_stability \
  --config expansion/configs/seed_stability.yaml \
  --output expansion/artifacts/seed_stability.json
```

Runtime: Python 3.11.7, NumPy 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0, Torch
2.7.1. Report `expansion/artifacts/seed_stability.json`. New code is confined to
`expansion/src/gnn_expansion/seed_stability.py`; `cross_layer.py` and
`cross_layer_edges.py` are hash-locked to W2.a and were imported, never edited.

## Publication note

This is an archival decision record. Results, corrections, and withdrawals are retained as recorded. Local interpreter paths have been normalized where present. Paths under `expansion/` refer to the original research archive, whose code, full reports, model inputs, and caches are not included in this public package. The historical commands above require that archive. `SOURCE_MANIFEST.json` records the original source hash and the hash of this published copy.
