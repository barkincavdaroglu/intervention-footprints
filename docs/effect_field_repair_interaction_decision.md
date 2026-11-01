# Repair-interaction decision: effect-field similarity does not identify repair synergy

Date: 2026-08-03  
Decision: **fail (evaluable no-go for repair selection)**  
Official report: `artifacts/effect_field_repair_interactions_v2.json`  
Report SHA-256: `ec96db4663edb4b3aadc723d44db3817e2aad2884f6ce543fc4b023c2a324074`  
Protocol SHA-256: `a68109de5065cd2292d74c824ff6ad89ed8a39d86e758d063ffde787f8f78fd1`

## Scope and integrity

The frozen development analysis used the same 30 validation-only checkpoints in
four cells: Amazon Computers and Roman Empire, each with the thesis GCN and residual
GCN. Every checkpoint produced exactly two covariate-matched high/low assignments
with eight globally distinct channels. All 30 checkpoints were evaluable.

The analysis used no test labels, test performance, fresh holdouts, or confirmatory
artifacts. A cache-only replay produced a byte-identical report with the same
SHA-256. The complete test suite passed 96/96 tests before the run.

Version 1 was stopped after its Hungarian-then-greedy matcher failed to retain two
globally disjoint assignments at the second checkpoint. No pair adapter was fit for
that checkpoint and no v1 pair outcome was inspected. Version 2 changed only the
feasibility solver, used a new cache root, and selected the exact minimum-total-
distance solution under the original eight-distinct-channel constraint.

## Frozen decision result

The primary repair-interaction contrast failed decisively. Let
`D_int` be the mean independent-adapter interaction for high-similarity pairs minus
the corresponding mean for matched low-similarity pairs. Its checkpoint median was
-0.002714; 22/30 checkpoints were negative, and all four cell medians were negative.

| Cell | Checkpoints | Median `D_int` | Median tied sharing contrast `Gamma` |
|---|---:|---:|---:|
| Amazon Computers / residual GCN | 5 | -0.006300 | +0.027682 |
| Amazon Computers / thesis GCN | 5 | -0.006782 | +0.010004 |
| Roman Empire / residual GCN | 10 | -0.001971 | +0.028198 |
| Roman Empire / thesis GCN | 10 | -0.001479 | +0.008046 |
| **All checkpoints** | **30** | **-0.002714** | **+0.010842** |

The tied-versus-independent sharing contrast was positive in 20/30 checkpoints and
had positive medians in all four cells. That result is descriptive but cannot
authorize a tied repair policy: the protocol required both a positive repair-
interaction contrast and a positive sharing contrast. The first requirement failed
in every cell.

## Scientific interpretation

Effect-field similarity remains a valid functional descriptor. It predicted
classwise single-channel effects beyond strong controls and separated the
nonadditivity of fixed joint ablations. It did **not** identify channel pairs whose
independently parameterized affine repairs combine more favorably. The distinction
is substantive: similarity of perturbation responses is not the same as similarity
of optimization geometry or repairability.

The positive sharing contrast suggests that high-similarity pairs may tolerate a
shared parameterization better than low-similarity pairs. However, because those
pairs did not have positive relative repair interaction, this is not evidence that
selecting them improves repair. It must not be presented as an authorized repair
method.

## Consequence and next branch

Enforce the stopping rule: do not run the final coherent-repair comparison, do not
tune another repair objective on these outcomes, and do not evaluate a repair method
on fresh holdouts. The repair-method branch is closed.

The defensible paper is now an interpretation paper centered on the boundary between
three claims:

1. induced support topology alone does not add information beyond effect-field
   controls;
2. the intervention effect field robustly predicts channel function and fixed
   joint-ablation interactions;
3. those similarities do not transfer automatically to repair interactions.

Before opening fresh holdouts, broaden this interpretation result across additional
datasets and architectures under a new frozen protocol. The paper should frame the
negative repair result as a caution against treating ablation-based channel clusters
as editable modules, not as a failed method hidden from the main narrative.
