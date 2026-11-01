# Gate 1 decision: topology does not add incremental information

Date: 2026-08-03  
Decision: **fail (evaluable no-go)**  
Official report: `artifacts/topology_gate1_refined.json`  
Report SHA-256: `b3b6f2529533bdcc2db22fc921494e4061427221f18b3f9ace7b7795f71b9202`

## Scope and integrity

The frozen development analysis used 30 validation-only checkpoints in exactly four
cells: Amazon Computers and Roman-empire, each with the thesis GCN and residual GCN.
It used no test labels, test performance, fresh holdouts, or confirmatory artifacts.
All 30 raw sidecars matched their recorded hashes and schemas.  The topology null
completed two degree-preserving rewires per checkpoint, for 60/60 accepted rewires.

A cache-only repeat produced a byte-identical report with the same SHA-256.  The
complete test suite passed 86/86 tests after the run.

## Frozen decision result

All ten selection-eligible candidates failed Gate 1.  Constant-label WL `h0` was
retained as the known-degenerate diagnostic null and was not allowed to make the
gate non-evaluable.

| Candidate | Active mean incremental alignment | Active median | All-channel median | Primary real-minus-rewired median | LOCO passes | Gate result |
|---|---:|---:|---:|---:|:---:|:---:|
| Diffusion, `t=1` | -0.299 | -0.309 | -0.285 | +0.016 | yes | fail |
| Constant WL `h1` | -0.361 | -0.390 | -0.322 | +0.043 | yes | fail |
| Constant WL `h2` | -0.364 | -0.393 | -0.419 | +0.023 | yes | fail |
| Constant WL `h3` | -0.369 | -0.390 | -0.455 | +0.038 | yes | fail |
| Constant WL `h4` | -0.378 | -0.395 | -0.459 | +0.039 | yes | fail |
| Induced-degree WL `h0` | -0.372 | -0.421 | -0.267 | +0.009 | yes | fail |
| Induced-degree WL `h1` | -0.347 | -0.378 | -0.391 | +0.051 | no | fail |
| Induced-degree WL `h2` | -0.342 | -0.384 | -0.419 | +0.031 | no | fail |
| Induced-degree WL `h3` | -0.346 | -0.384 | -0.440 | +0.047 | no | fail |
| Induced-degree WL `h4` | -0.353 | -0.383 | -0.456 | +0.056 | no | fail |

No primary incremental alignment was positive in any of the four cells.  For the
least-negative candidate, diffusion at `t=1`, the four cell medians were -0.299,
-0.273, -0.352, and -0.267.  None of its four scales passed the required positive
incremental criterion.

The positive real-minus-rewired medians for many candidates mean that the
representations detect some genuine graph structure.  They do **not** mean that
graph structure adds predictive information beyond the controls.  The primary
incremental contrast is strongly negative and therefore controls the decision.

## What did survive

The intervention effect field is highly class-informative.  For validation-KL
effects in the active-channel analysis:

| Control block | Mean alignment | Median alignment | Positive checkpoints | Strongest-control checkpoints |
|---|---:|---:|---:|---:|
| Multiscale mask Jaccard | 0.537 | 0.536 | 30/30 | 26/30 |
| Raw L1-normalized effect field | 0.404 | 0.411 | 30/30 | 3/30 |
| Uniform parent-node roles | 0.292 | 0.281 | 30/30 | 1/30 |
| Magnitude summaries | 0.145 | 0.133 | 30/30 | 0/30 |

The signed classwise CE sensitivity tells the same story: mask Jaccard is positive
in 30/30 checkpoints (median 0.431), and it is the strongest control in 28/30.

Thus the thesis's combined structural signature captured a real functional signal,
but the new isolation study locates that signal primarily in **which nodes are
affected**, not in the induced support's edge topology.

## Consequence

The predeclared rule is enforced: do not run topology-cluster Gate 2, do not present
the legacy fused signature as topology evidence, and do not evaluate fresh holdouts
with a topology-centered method.  The next development stage is an explicitly
post-hoc **ablation-effect-field** pivot.  Its representation and stopping rules must
be frozen before any new holdout is opened.

