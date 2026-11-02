# Record: exact cross-layer edges and attribution fidelity (W2.a)

> **Publication clarification — 2026-11-01.** The magnitude-Spearman cell range in E1 below should read **0.8614 to 0.9972**. The maximum cell median is Amazon Computers / narrow GCN, 0.9972380397. This was checked against the frozen `cross_layer_edges.json` checkpoint summaries and all 30 hashed edge-matrix sidecars. The signed-Spearman headline and experimental decisions are unchanged. The original archival text is retained below. The [companion data](companion/figure-data.json) contains the corrected checkpoint-level values and source hashes. Also read E3 as broad node-level influence, not literal uniformity, and E6 as the measured relative error of this simultaneous-ablation comparison, not a universal bound on circuit fidelity.

Date recorded: 2026-08-04

Status: **complete and evaluable; descriptive measurement study, no scientific
gate**

Exploratory measurement on previously inspected development graphs. Does not
alter the completed final holdout, does not reopen repair, and does not
authorize any test-label inspection.

## What was measured

For each adjacent hidden-layer pair, ablating source channel `i` and recording
the induced change at target channel `j` and node `v`:

    e_ij(v) = h_t,j^{(-i)}(v) - h_t,j(v)

giving the signed edge effect `E_ij`, the edge strength `S_ij`, and the
node-indexed edge field `f_ij`. Alongside it, the first-order attribution
`Ehat_ij` from one backward pass per target channel.

Thirty development checkpoints in the four frozen cells. The thesis GCN has one
adjacent pair (`0->1`); the residual GCN has four (`0->1` through `3->4`), all
measured, with the published-layer-anchored pair primary — `0->1` for Amazon
Computers, `3->4` for Roman-empire.

## Validity

| Gate | Result |
|---|---|
| Cache replay | `exact` on **30 of 30** checkpoints, deviation `0.00e+00` |
| Coverage | 30 of 30 evaluable |
| Gradient determinism | never relaxed; the backward pass ran deterministically throughout |

The replay audit computes the published layer's own single-channel fields rather
than borrowing a measured pair's source fields, because for Roman-empire's
residual GCN the published layer is the last block and can only ever be a
target. An earlier version compared layer-3 fields against the published
layer-4 cache and reported a spurious `0.496` deviation; that was a bug in the
audit, corrected before the full run.

## E1 — Attribution fidelity (headline)

| metric | median | cell range |
|---|---:|---|
| Spearman, signed | `+0.9814` | `0.9524` to `0.9915` |
| Spearman, magnitude | `+0.9708` | `0.8614` to `0.9938` |
| Sign agreement | `+0.9564` | `0.9048` to `0.9951` |
| Top-`width` edge recovery | `+0.8750` | `0.7344` to `0.9531` |

One backward pass per target channel recovers the exact edge ordering at
Spearman `0.98` and identifies roughly `88%` of the top-`width` edges. The cell
ordering matches the linearity argument: the thesis GCN's ReLU path scores
highest (Roman thesis `0.9915`) and the residual GCN's GELU and LayerNorm path
lowest (Roman residual `0.9524`).

Because the exact matrix is affordable at these widths, this is a *measured*
fidelity against ground truth rather than an assumed one. That is the main
methods contribution of W2.a.

## E2 — Concentration

The top 5% of edges carry `24.7%` of total edge mass, a concentration factor of
`4.86` over the uniform reference; the top 1% carry `9.3%`. Row-mass Gini
ranges widely by cell (`0.09` to `0.59`), so how unevenly individual channels
distribute outgoing influence is architecture- and dataset-dependent even where
overall concentration is similar.

There is deliberately no permutation null. Any permutation of a matrix
preserves the multiset of its entries and therefore every order statistic of the
flattened array, so a permuted null reproduces the observed top-`q` fraction
exactly. The draft protocol specified two such nulls; both were provably
vacuous and were replaced by the uniform reference before the full run.

## E3 — Edge fields are spatially diffuse

Participation ratio `0.5434`, normalized entropy `0.9609`, over the top edges by
mass. Cross-layer structure is therefore **sparse in channel space and dense in
node space**: influence is carried by comparatively few channel pairs, but each
such pair acts almost uniformly across the graph.

This contrasts sharply with W1's dictionary features (participation ratio
`0.0417`) and is the first result the graph-native edge field has produced that
a channel-space-only analysis could not.

## E6 — Additivity residual

Median `0.2267`. Summing single-channel edges misses roughly `23%` of what
simultaneous ablation of all source channels actually does, so pairwise and
higher-order interaction is not negligible at this granularity. The cell spread
is large: Amazon thesis `0.0659` is nearly additive, Amazon residual `0.3260` is
not.

## Depth profile, and a correction

Measuring all four adjacent pairs in the residual cells was a byproduct of
resolving a malformed layer-pair decision. It produced one robust trend and two
weaker ones, which are stated here at their true strength because an earlier
summary overstated them.

| quantity | first pair to last (median) | strictly monotone per checkpoint |
|---|---|---:|
| concentration factor, top 5% | `4.19 -> 8.18` (Amazon), `6.05 -> 7.01` (Roman) | **15 / 15** |
| additivity residual | `0.326 -> 0.140` (Amazon), `0.373 -> 0.279` (Roman) | 8 / 15 |
| attribution fidelity | `0.982 -> 0.967` (Amazon), `0.959 -> 0.952` (Roman) | 2 / 15 |

- **Concentration increases with depth in every residual checkpoint**, 15 of 15,
  in both cells. This is a real and unanimous trend.
- **Additivity residual falls with depth directionally** — the endpoint medians
  drop in both cells — but strict monotonicity holds in only 8 of 15 checkpoints
  (5 of 5 Amazon, 3 of 10 Roman). Supported in direction, not monotone.
- **Attribution fidelity shows no depth trend.** Strict monotonicity holds in 2
  of 15 checkpoints, and the median gap is roughly `0.015`. An earlier summary
  described fidelity as "slowly declining with depth"; that read a small
  difference in medians as a pattern and is **withdrawn**.

The concentration trend was not predicted by the protocol. It is a hypothesis
for a preregistered test, not a confirmed finding.

## Interpretation

W2.a was chosen because a cross-layer edge is indexed by an ordered pair of
channels, so direction is built into the object and Workstream 1's invariance
does not apply. That reasoning holds up: the edge matrix carries structure the
effect field could not express, most visibly in E3, where edges prove sparse
across channels and diffuse across nodes.

The practical result is that first-order attribution is a good proxy for exact
cross-layer edges in these models, which makes larger GNN circuit analysis
tractable without abandoning fidelity. The caveat is E6: single-channel edges
miss about a fifth of joint influence, so an edge-additive circuit account is an
approximation of known and measurable size.

## Limitations

1. Development-tier evidence on 30 previously inspected checkpoints. Exploratory
   measurements, not prospective tests. PubMed and Squirrel-filtered remain
   untouched.
2. Adjacent layers only; skip connections and residual-stream paths are not
   measured.
3. `Ehat` is a linearization at the clean activations; E6 quantifies where the
   linear account is incomplete.
4. Edges are relative to mean-replacement, not clean-to-corrupt patching.
5. The depth trend rests on four pairs in two cells of one architecture family.

## Decision and next steps

1. Record E1 as the citable methods result: attribution fidelity measured
   against exact ground truth in a setting where ground truth is affordable.
2. Record E3 as the citable structural result: cross-layer influence is sparse
   in channel space and diffuse in node space.
3. Treat the depth concentration trend as a preregistration candidate, not a
   finding.
4. E4 (seed stability by Hungarian matching) is specified in the protocol but
   not yet computed; it is the natural next analysis and needs no new forwards,
   since the edge matrices are cached.
5. W2.b (circuit extraction with faithfulness gates) is now well motivated: E2
   shows enough concentration for a circuit to be meaningfully smaller than the
   full matrix, and E6 sets the honest ceiling on how much an edge-additive
   circuit can explain.

## Reproducibility record

- Full report: `expansion/artifacts/cross_layer_edges.json`
  - SHA-256: `b4cb150e892f6915ce4bb700270e9eca894705c8f1e0aa625866a2c76d0f1e1c`
- Protocol: `expansion/docs/cross_layer_edges_protocol.md`
  - SHA-256: `297f657800066460548743188dadac904921f40072f2b473fdf68da0721706ee`
- Per-checkpoint caches and edge-matrix sidecars:
  `expansion/artifacts/cross_layer_runs/` (30 checkpoints, 9.2 MB)
- Shared raw report with W1.a, W1.c, and W1.c-bis: `8f04c0dd7e0c7c58...`
- Runtime: Python 3.11.7, NumPy 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0,
  PyTorch 2.7.1
- Cache replay tier: `exact` on 30 of 30 checkpoints
- Gradient determinism relaxed: **never**
- Test labels or test performance used: **no**
- Fresh holdouts used: **no**
- Repair reopened: **no**

## Publication note

This is an archival decision record. Results, corrections, and withdrawals are retained as recorded. Local interpreter paths have been normalized where present. Paths under `expansion/` refer to the original research archive, whose code, full reports, model inputs, and caches are not included in this public package. The historical commands above require that archive. `SOURCE_MANIFEST.json` records the original source hash and the hash of this published copy.
