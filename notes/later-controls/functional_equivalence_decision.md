# Decision: functional equivalence of edge-matched channels (W2.c)

Run date: 2026-08-12
Protocol: `functional_equivalence_protocol.md`
(`823746654bb4269a6d15c21ae588566f5b5b35b2b63115cf9604555654e71697`)
Report: `expansion/artifacts/functional_equivalence.json`
Verdict: **`function_not_conserved`**, status `evaluable`, both subsets agreeing

## The question

E4 (L14) conserved cross-layer edge *geometry* across independent training runs
and drew its own boundary: pairing by the matcher does not establish that paired
channels encode the same thing. This study asked whether they do, using the
classwise validation-KL vector from the published H1 machinery — the one
functional description of a channel the program has already validated.

## Result

Gain = mean classwise-KL cosine over edge-matched pairs minus its median under a
mass-stratified permutation of the same assignment.

| subset | null | median gain | positive | cells positive |
|---|---|---:|---:|---:|
| `active_channels` (primary) | stratified | **`+0.0087`** | 22/30 | 4/4 |
| `all_channels` | stratified | **`+0.0130`** | 27/30 | 4/4 |
| `active_channels` | unrestricted | `+0.0088` | 28/30 | 4/4 |
| `all_channels` | unrestricted | `+0.0313` | 26/30 | 4/4 |

Threshold `0.05`. Both subsets fall below it, both agree, all four validity
gates pass, 30 of 30 checkpoints evaluable, matcher positive control `pass`
30/30. Verdict `function_not_conserved` under the rule frozen before any number
was computed.

## What this does and does not say

**It does not say the gain is zero.** The sign is remarkably consistent: all
four cells positive in all four subset/null combinations, and 22–28 of 30
checkpoints positive. The honest statement is that edge matching carries a
*small, reliably positive, sub-threshold* amount of functional information —
roughly a fifth of the bar the program requires before it will call something
conserved.

**Conserved wiring does not carry conserved function at this resolution.** That
is the citable content, and it is a genuine boundary of the same family as L3
(interaction does not imply repairability) and L13 (fields resolve support, not
direction): a circuit carried across seeds by the E4 matcher preserves edge
geometry, and the program has no evidence that the channels it pairs do the
same job.

**Consequence for W2.b.** L14's precondition survives only in its original,
narrow form. A circuit found in one checkpoint is worth *looking for* in
another, but a cross-seed transplant may not assume the matched channels are
functionally interchangeable. Any W2.b transfer claim must be stated over edge
geometry, not over function.

## Two diagnostics that qualify the reading

**1. The endpoint has compressed dynamic range.** Absolute levels: matched
cosine `0.7474`, stratified-null cosine `0.7383`. Classwise-KL vectors are
broadly similar between *any* two channels — they live in a narrow cone — so the
available headroom above the null is roughly `0.26`, not `1.0`. A `0.05`
threshold on that scale is demanding. This does not change the verdict, which
was frozen in advance, but it does mean a future study wanting to detect
functional correspondence should use a whitened or contrast-normalized
functional descriptor rather than raw cosine. Recorded as the concrete next
instrument, not as an excuse for this outcome.

**2. The registered expectation came out inverted.** The protocol registered, in
advance and explicitly not as part of the decision rule, that if edge matching
carries function the functional gain should follow E4's cell ordering. It does
the opposite:

| cell | E4 conservation gain | W2.c functional gain |
|---|---:|---:|
| `roman_empire/residual_gcn` | `+0.2471` | `+0.0008` |
| `amazon_computers/residual_gcn` | `+0.1083` | `+0.0124` |
| `amazon_computers/thesis_gcn` | `+0.0936` | `+0.0067` |
| `roman_empire/thesis_gcn` | `+0.0320` | `+0.0152` |

Spearman `-0.8` across the four cells. **This is four points.** The exact
permutation two-sided p-value for `|rho| >= 0.8` at `n = 4` is `0.083`, and
these four cells are not independent draws from anything. It is an observation
worth recording so that a future study can test it, and it is not evidence of an
inverse law. Stated as such here so it cannot later be quoted as a finding.

If it did survive testing, the reading would be that geometric conservation and
functional conservation are traded off rather than aligned — which would be a
sharper and more interesting claim than the one this study set out to test. That
is a hypothesis for a protocol with more cells, not a result.

## Evidential status

Development-tier, exploratory, on the same 30 previously inspected checkpoints
as every other expansion study. No new forward passes, no training, no test
labels, no reserved data. The endpoint was frozen before any outcome was
computed and the rule was applied as written, including the sub-threshold
positive result being reported as `function_not_conserved` rather than promoted
to a finding.

## Reproducibility

**Runtime note, and an unplanned robustness check.** The study was first run
under a drifted interpreter (Python 3.13.2, NumPy 2.1.2, SciPy 1.14.1,
scikit-learn 1.5.2) before the program's recorded runtime was restored, then
re-run under it. The two runs differ **only** in floating-point summation order:
per-checkpoint gains move by `~1e-16`, pooled medians agree to fifteen
significant figures, and the verdict, every count, every cell sign and every
gate outcome are identical. The canonical report is the one produced under the
recorded runtime (Python 3.11.7, NumPy 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0,
Torch 2.7.1), and it replays byte-identically under it. The drifted run is not
part of the evidence chain; it is recorded here because it doubles as evidence
that this endpoint is insensitive to library version, which the forward-pass
studies are not (see I8, I13).

| Item | Value |
|---|---|
| Report | `expansion/artifacts/functional_equivalence.json` |
| Report SHA-256 | `1ee5f746ee9db307c83ef724d54935a3b5bf2b300c812a492d31cb1f38f16847` |
| Protocol SHA-256 | `823746654bb4269a6d15c21ae588566f5b5b35b2b63115cf9604555654e71697` |
| Replay | byte-identical under the recorded runtime |
| Figure | `expansion/figures/expansion_conservation.pdf`, manifest in `figure_manifest.json` |
| W2.c code SHA-256 | recorded in the report's `protocol.w2c_code_sha256` |
| E4 code SHA-256 | matches `seed_stability.json`, imported unedited |
| Edge report | `cross_layer_edges.json` `b4cb150e8...` |
| Raw report | shared chain `8f04c0dd...` |
| New forward passes | `0` |
| Tests | 22 network-free tests in `tests/test_functional_equivalence.py`, including a synthetic case with conserved geometry and independent function where the endpoint must return ~0 (it does) |

## Publication note

This is an archival decision record. Results, corrections, and withdrawals are retained as recorded. Local interpreter paths have been normalized where present. Paths under `expansion/` refer to the original research archive, whose code, full reports, model inputs, and caches are not included in this public package. The historical commands above require that archive. `SOURCE_MANIFEST.json` records the original source hash and the hash of this published copy.
