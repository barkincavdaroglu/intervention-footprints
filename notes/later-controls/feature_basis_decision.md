# Decision: sparse dictionary features versus arbitrary directions (W1.c)

Date decided: 2026-08-04

Status: **inconclusive under the frozen gate; superposition arc neither
supported nor closed**

> **Addendum, 2026-08-04, after W1.c-bis.** Two results below have been
> retracted. The monosemanticity gain (`+0.1278`, 30/30) and the localization
> advantage (roughly ninefold) were measured against displacement-mismatched
> nulls. Against a displacement-matched null they fall to `+0.0104` (20/30) and
> `-0.0006` (11/30) respectively, so both were artifacts of sparsity and
> magnitude rather than properties of learned directions. Do not cite them.
> W1.c-bis also resolved the primary question this memo left open, with verdict
> `direction_does_not_matter`, closing the superposition arc. Read this memo
> together with `matched_null_decision.md`, never on its own.

Exploratory expansion on previously inspected development graphs. Does not
alter the completed final holdout, does not reopen repair, and does not
authorize any test-label inspection.

## What was tested

W1.a returned `basis_agnostic`: the frozen H1 endpoint holds as well in a
Haar-random rotation of the channel basis as in the channel basis itself. That
left two readings — features exist but are not axis-aligned (superposition), or
the endpoint is coarse and any large perturbation inherits the signal. W1.c
fitted sparse autoencoders on training-node activations of the same 30
development checkpoints, ablated each learned feature coefficient with the same
three-term update, and recomputed the frozen H1 endpoint.

Thirteen conditions per checkpoint: `channel`, three Haar rotations, and three
seeds each of `sae_topk` (primary, `k = 0.25 x width`), `sae_relu_l1`
(sensitivity, fixed penalty), and `sae_random` (untrained-at-init null).
Latents were `4 x width`.

## Frozen-gate result

| Question | Frozen result | Decision |
|---|---:|---|
| `sae_topk` minus rotation, `active_directions` | `-0.2133` (`1/30` positive, `0/4` cells) | worse |
| `sae_topk` minus rotation, `matched_count` | `+0.0275` (`25/30` positive, `3/4` cells) | ties |
| `sae_topk` minus `sae_random`, `active_directions` | `-0.1957` (`0/30`) | worse |
| `sae_topk` minus `sae_random`, `matched_count` | `+0.0254` (`16/30`, `2/4` cells) | ties |
| Subsets agree | no | **inconclusive** |

All four validity gates passed: cache replay `exact` on all 30 checkpoints,
coverage 30/30, reconstruction floor met (`sae_topk` explained variance median
`0.954`, min `0.939`), sparsity floor met (median L0 `12`, minimum live
features `73`).

Seeds were perfectly consistent *within* each subset — all three negative on
`active_directions`, all three positive on `matched_count` — so the split is
not sampling noise. It is two well-defined measurements answering different
questions.

## Why the subsets disagree

Intervention displacement, measured per condition:

| family | Roman-empire residual GCN | Amazon residual GCN |
|---|---:|---:|
| channel | `0.15793` | `0.08599` |
| Haar rotation | `0.15417` | `0.08585` |
| `sae_random` | `0.08949` | `0.04916` |
| `sae_relu_l1` | `0.03704` | `0.01788` |
| `sae_topk` | `0.00904` | `0.00065` |

A TopK feature fires on a small minority of nodes, so mean-replacing it moves
the layer far less than ablating a channel or a rotated direction — by a factor
of `14.5x` to `3302x` depending on the cell. On `active_directions` all roughly
190 live features count, most contributing near-null fields, which drives the
`-0.21`. On `matched_count` the top-`width` features by impact mass are compared
at equal cardinality, which gives `+0.028`.

The mismatch per cell, on the same pooled-seed medians as the table above:

| cell | channel | `sae_topk` | ratio | median feature firing rate |
|---|---:|---:|---:|---:|
| `roman_empire/thesis_gcn` | `0.07322` | `0.005062` | `14.5x` | `3.6%` |
| `roman_empire/residual_gcn` | `0.15793` | `0.009036` | `17.5x` | `3.2%` |
| `amazon_computers/residual_gcn` | `0.08599` | `0.000648` | `132.7x` | `0.85%` |
| `amazon_computers/thesis_gcn` | `0.01029` | `0.000003` | `3302x` | `0.04%` |

> **Correction, 2026-08-05.** The paragraph above previously read "fires on
> roughly six percent of nodes, so mean-replacing it moves the layer 17 to 170
> times less". Both figures were wrong. Six percent is the *design* sparsity
> (`k / latents = 0.25 x width / 4 x width = 6.25%`), not a measurement; the
> measured median firing rate is `0.04%` to `3.6%` depending on cell. And the
> displacement ratio reaches `3302x`, not `170x` — `17x` was right for the
> Roman-empire residual cell, but the Amazon residual cell is `132.7x` and the
> two `thesis_gcn` cells were never tabulated. The displacement table itself is
> correct and reproduces to within `5e-06`. The error understated the confound,
> so the `inconclusive` verdict and the decision to build W1.c-bis are more
> firmly supported, not less. No verdict changes.
>
> The same two figures appear in `matched_null_protocol.md` and in the module
> docstrings of `feature_basis_matched.py` and `matched_null.py`. Those three
> files are hash-locked into W1.c-bis and **must not be edited** — correcting
> them would change `matched_code_sha256` and `protocol_document_sha256` and
> destroy that study's replay identity. Read them against this note.

Neither number answers the question the protocol posed. Both nulls are
displacement-mismatched, so the comparison confounds "is this a better basis"
with "does this intervention move the model at all". This is the same failure
mode W1.a recorded for PCA, in a much sharper form.

## What did come out clean

Two endpoints are unambiguous, with no subset ambiguity and no seed
disagreement:

- **Monosemanticity.** Class-effect entropy gain over rotations is `+0.1278`,
  positive in `30/30` checkpoints and all four cells (`sae_topk` `0.7321`
  versus rotation `0.8837`). Dictionary features have measurably more
  class-concentrated causal effects than arbitrary directions.
- **Localization.** Participation ratio `0.0417` for `sae_topk` against
  `0.3701` for rotations and `0.3885` for channels: dictionary effect fields
  are roughly nine times more spatially concentrated.

Both are stated here as descriptive results measured against a
displacement-mismatched null, which is exactly the caveat W1.c-bis exists to
remove.

## Secondary observations

- `sae_relu_l1` at the fixed penalty did not become sparse (median L0 `96` of
  `192` latents). It tied every comparison and, as the protocol declared in
  advance, cannot carry a verdict.
- `sae_topk` beat `channel` on `matched_count` by `+0.0907`, positive in
  `23/30` checkpoints and all four cells — the only comparison positive in
  every cell.
- Seed stability: `sae_topk` feature-match cosine `0.572` against `0.424` for
  the untrained dictionary. Modest agreement, consistent with the published SAE
  seed-instability literature.
- The `subspace_overlap` statistic returned `1.0` for every architecture
  including random init. Taking the top-`width` directions of a dictionary that
  spans the whole activation space makes it degenerate; it is reported but
  carries no information and should not be cited.

## Interpretation

The result does not support the superposition reading and does not close it
either. What it establishes is narrower and methodological: **the H1 endpoint
cannot compare bases whose interventions differ in magnitude by two orders of
magnitude.** Any comparison between a sparse dictionary and a dense basis has to
match displacement before it can be read as evidence about direction.

The verdict was not resolved by choosing the favorable subset. `matched_count`
favors the dictionary and was known to do so before the full run completed; the
frozen rule requires both subsets, and it was left to return `inconclusive`.

## Decision and paper use

1. Do not preregister W1.d. The gate did not authorize it.
2. Do not close the superposition arc on this evidence either.
3. Proceed to W1.c-bis with a displacement-matched null, which removes both the
   magnitude and the cardinality confound by construction and makes the primary
   question well posed.
4. The monosemanticity and localization findings may be cited as descriptive
   development results, with the displacement caveat attached, pending their
   re-measurement against the matched null.

## Reproducibility record

- Full report: `expansion/artifacts/feature_basis.json`
  - SHA-256: `e2d78261fcd14db61bb019980d7f674151266eb1670014d18b5c38dbdc3530d6`
- Protocol: `expansion/docs/feature_basis_protocol.md`
  - SHA-256: `6e48b465760f595d3ff0de4423fc868802b921f610e23a4aee5e6c53e13b0f49`
- Per-checkpoint caches: `expansion/artifacts/feature_basis_runs/` (30 files)
- Runtime: Python 3.11.7, NumPy 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0,
  PyTorch 2.7.1
- Cache replay tier: `exact` on 30 of 30 checkpoints
- Test labels or test performance used: **no**
- Fresh holdouts used: **no**
- W1.d authorized: **no**
- Repair reopened: **no**

## Publication note

This is an archival decision record. Results, corrections, and withdrawals are retained as recorded. Local interpreter paths have been normalized where present. Paths under `expansion/` refer to the original research archive, whose code, full reports, model inputs, and caches are not included in this public package. The historical commands above require that archive. `SOURCE_MANIFEST.json` records the original source hash and the hash of this published copy.
