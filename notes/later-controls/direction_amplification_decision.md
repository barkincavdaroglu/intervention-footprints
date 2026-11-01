# Decision: output coupling of learned directions at matched displacement (W1.e)

Run date: 2026-08-14
Protocol: `direction_amplification_protocol.md`
(`cef3ca6da4c052211cc3f2d33c0b804ddc3d0e6d85188052444eb32dd41d8df7`)
Report: `expansion/artifacts/direction_amplification.json`
Verdict: **`amplified`**, status `evaluable`, both subsets agreeing

## The question

W1.c-bis returned `direction_does_not_matter` on an *alignment* endpoint that is
incremental over a magnitude control — so a uniformly larger effect is absorbed
by that control and cannot be seen. A single-checkpoint specimen, built for a
figure rather than a study, suggested learned directions nonetheless move the
output roughly `2x` more than their displacement-matched nulls. This study asks
whether that generalizes, using W1.c-bis's own dictionaries and its own nulls.

## Result

Endpoint: `r_k = log2(M_k / M'_k)`, the log ratio of scope field mass between a
learned dictionary feature and a displacement-matched null direction carrying
the identical per-node coefficient displacement, sparsity pattern and norm.
Checkpoint value = median over eligible (feature, dictionary-seed) pairs.

| subset | median `log2` | linear | positive | cells |
|---|---:|---:|---:|---:|
| `top_impact` | **`+1.4004`** | **`2.64x`** | 30/30 | 4/4 |
| `all_live` | **`+0.4398`** | **`1.36x`** | 30/30 | 4/4 |

Threshold `log2(1.25) = 0.3219`. Both subsets clear it, both agree, all four
validity gates pass, 30 of 30 checkpoints evaluable.

**Every checkpoint is positive in both subsets.** Ranges: `top_impact` `1.74x`
to `4.83x`; `all_live` `1.26x` to `2.13x`. The three dictionary seeds pool to
`2.69x`, `2.58x`, `2.64x` — the effect is not a property of one dictionary.

Per cell (`top_impact` / `all_live`):

| cell | `top_impact` | `all_live` |
|---|---:|---:|
| `amazon_computers/thesis_gcn` | `4.24x` | `1.86x` |
| `amazon_computers/residual_gcn` | `2.85x` | `1.57x` |
| `roman_empire/residual_gcn` | `2.69x` | `1.33x` |
| `roman_empire/thesis_gcn` | `2.10x` | `1.33x` |

## The preview did not drive the result

The protocol required the decision to hold with the previewed checkpoint
(`roman_empire/thesis_gcn` seed 0) excluded. It does, and the sensitivity is
the reverse of the usual worry:

| set | `top_impact` | `all_live` |
|---|---:|---:|
| all 30 | `2.64x` (30/30) | `1.36x` (30/30) |
| previewed excluded (29) | `2.66x` (29/29) | `1.36x` (29/29) |

The previewed checkpoint is the **weakest of all thirty** on `top_impact`
(`1.74x` against a study median of `2.64x`). The specimen that motivated this
study understated the effect rather than selecting for it. Its published figure
reported `2.09x` over the top ten features of that checkpoint using ad-hoc null
seeds and one dictionary; this study measures `1.74x` there using W1.c-bis's
exact nulls pooled over three dictionaries. Both are "about `2x`", and the
difference is fully accounted for by the change in nulls and pooling.

## What this establishes, and what it does not

**The learned direction is not inert.** At identical per-node displacement,
identical sparsity pattern and identical norm, pushing along a learned
dictionary direction moves the model's output distribution `2.64x` more than
pushing along a random direction. One factor differs and it is the direction.

**This does not contradict or reopen W1.c-bis.** The two results measure
different things and are jointly consistent:

| what is asked | endpoint | result |
|---|---|---|
| does the learned direction organize the effect *spatially*? | incremental alignment over a magnitude control | `+0.0359`, within threshold — **no** |
| does the learned direction move the output *more*? | field-mass ratio at matched displacement | `2.64x`, 30/30 — **yes** |

The alignment endpoint is incremental over a magnitude control precisely so that
a uniformly larger effect is absorbed. It was therefore structurally unable to
report the quantity measured here — not by oversight, but by design.

**The revised sentence.** L13 said an effect field resolves the support and
magnitude of a perturbation, not its direction. That stands, and now says more:

> The learned direction is not inert — at matched displacement it couples to the
> output roughly `2.6x` more strongly than an arbitrary one. What it does not buy
> is a better-*organized* spatial pattern, which is the only thing the alignment
> endpoint reads.

So the direction controls *how much* the output moves; the support controls
*where*. The instrument that closed the superposition branch reads the second
and is blind to the first.

**What it does not license.** Nothing about monosemanticity or localization: the
L10a/L10b retractions stand untouched, because those endpoints were also
measured against the matched null and lost there. Nothing about repair. And
nothing about whether `2.64x` matters behaviourally — field mass is a
perturbation-size measure, not a task metric.

## Evidential status

Development-tier, exploratory, on the same 30 previously inspected development
checkpoints. Not a prospective test. The endpoint was frozen with its threshold,
subsets, exclusion sensitivity and gates before any full-study outcome was
computed, and the preview was disclosed in the protocol.

The strongest reproducibility evidence is Gate 1: every refit dictionary's
SHA-256 was required to equal the value cached by W1.c-bis, on all 30
checkpoints and all three seeds, and did. The dictionaries and nulls compared
here are bit-for-bit the objects whose verdict this study revises.

## Validity gates

| gate | result |
|---|---|
| dictionary replay (SAE SHA-256 equals W1.c-bis cache) | pass, 30/30 x 3 seeds |
| null replay (seed rule and construction reproduce W1.c-bis) | pass, 30/30 |
| displacement identity (learned vs null median displacement) | pass; worst relative error `1.39e-07` against a `1e-6` tolerance |
| scope identity (scope mask equals the published Gate-1 sidecar) | pass, 30/30 |
| coverage | 30/30 evaluable, 4/4 cells |

Diagnostics: eligible-feature fraction median `1.000` (minimum `0.609`); mean
absolute cosine between paired learned and null directions `0.1121`, confirming
the nulls are genuinely different directions and — at these widths — not
near-orthogonal, which if anything biases toward finding no difference.

## Runtime, and a caught hazard

The study must be run under the program's recorded runtime (Python 3.11.7, NumPy
2.3.3, SciPy 1.17.1, scikit-learn 1.8.0, Torch 2.7.1). A first smoke run under a
drifted interpreter (Python 3.13.2) failed Gate 1 on every checkpoint: same
seeds, different fitted weights. Recorded as instrument-log **I14**. A second
hazard, **I13**, was caught by the checkpoint state-hash gate — the working
directory was silently selecting local model files that match the published
hashes in 0 of 30 cases. Both were fixed by changing how the study is invoked;
no hash-locked file was edited.

## Reproducibility

| Item | Value |
|---|---|
| Report | `expansion/artifacts/direction_amplification.json` |
| Protocol SHA-256 | `cef3ca6da4c052211cc3f2d33c0b804ddc3d0e6d85188052444eb32dd41d8df7` |
| W1.e code SHA-256 | in the report's `protocol.amplification_code_sha256` |
| W1.c / W1.c-bis code | imported unedited; hashes match `matched_null.json` |
| Raw report | shared chain `8f04c0dd...` |
| Forward passes | `2 x 3 x latents` per checkpoint (768 thesis, 1536 residual) |
| Tests | 16 network-free tests in `tests/test_direction_amplification.py` |
| Runtime | Python 3.11.7, NumPy 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0, Torch 2.7.1 |
