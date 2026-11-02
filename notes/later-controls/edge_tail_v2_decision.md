# Decision: the weak-edge tail is run-specific structure (W2.a E7 v2)

Date decided: 2026-08-06

Status: **evaluable; verdict `run_specific_tail`** — and a **repaired-instrument
re-run, not a fresh preregistration**. See section "Evidential status" before
citing this.

Exploratory expansion on previously inspected development graphs. Zero new
forward passes. Does not alter the final holdout, reopen repair, or reopen W2.a
or L14.

## Result

Protocol `edge_tail_v2_protocol.md` (`6a00d32690e0dc00...`). All four gates pass;
30 of 30 checkpoints evaluable; both subsets return the same verdict.

| endpoint | subset | head | tail |
|---|---|---:|---:|
| E7a — cross-seed conservation | `active_channels` | `+0.0735` | **`+0.0320`** (22/30, 3/4 cells) |
| E7a — cross-seed conservation | `all_channels` | `+0.1074` | **`+0.0324`** (22/30, 3/4 cells) |
| E7b — exact vs first-order agreement | `active_channels` | `+0.9807` | **`+0.9792`** (30/30, 4/4 cells) |
| E7b — exact vs first-order agreement | `all_channels` | `+0.9813` | **`+0.9768`** (30/30, 4/4 cells) |

The tail fails the conservation threshold (`0.05`) in both subsets while passing
the agreement threshold decisively in both. By the reading rule frozen in v1 and
carried over unchanged, that is **run-specific structure**.

E7a tail by cell (`active_channels`): Roman residual `+0.0477`, Amazon residual
`+0.0345`, Roman thesis `+0.0174`, Amazon thesis `-0.0039`. Positive in three of
four cells but small everywhere, against a head that is two to three times
larger.

## What it means

The weak edges are **not noise, and not an artifact of ranking near-ties**. Two
computations that share no machinery — exact ablation, one forward pass per
source channel, and first-order attribution, one backward pass per target
channel — agree on the tail's ordering at Spearman `0.979`, as tightly as they
agree on the head (`0.981`). Whatever the tail encodes is a real and
reproducible property of a trained model.

It is also **specific to the training run**. Conservation across seeds, which is
robust in the head, falls below threshold in the tail in both subsets.

So each model develops its own idiosyncratic weak-edge structure: deterministic,
independently measurable, and not shared with a model trained from a different
random start on the same task.

This upgrades L14's account from inference to measurement. L14 inferred that
conservation lived in the high-mass edges from a contrast between two
whole-matrix statistics (Pearson `+0.0985` versus Spearman `+0.0297`); E7 v2
measures it directly, stratum by stratum, and the inference holds.

It also closes the alternative L14 left open. The Spearman dissent was **not**
rank instability among near-ties — if it had been, E7b would have collapsed in
the tail too, and it does not.

## Evidential status

E7 v1 returned `not_evaluable` (coverage 24 of 30) because its quartile
stratification is uncomputable on `amazon_computers/thesis_gcn`, where 58% of
entries are exactly zero. Its descriptive table was seen before v2 was frozen.

v2 is therefore a repaired-instrument re-run in the sense of W1.c → W1.c-bis, not
a first-look test. What makes it defensible: the repair — halves over nonzero
entries instead of quartiles over all entries — was forced by occupancy
arithmetic on the cached matrices, computed before freezing and wholly
independent of the outcome, and the reading rule, thresholds and every gate were
carried over unchanged rather than retuned. The `64`-per-stratum gate in
particular was **not** relaxed; the partition was coarsened until it could
satisfy the existing gate.

Anything citing this result should carry the same qualifier. It is weaker than a
first-look test and stronger than an exploratory observation.

## What it licenses

For **W2.b**: a circuit account may carry high-mass edges across checkpoints and
must not carry the tail. That was already the recommendation from L14; it is now
measured rather than inferred, and the boundary between the two regimes is
located at roughly the median edge by magnitude.

It does **not** establish what the run-specific structure encodes, whether it is
functional, or whether it matters for behaviour. A deterministic, independently
measurable structure that differs across runs could be anything from genuine
per-model specialization to a systematic residue of initialization.

## Caveats

1. Two strata is the finest partition the smallest cell supports, so "head" and
   "tail" are halves, not a fine-grained profile. Nothing here says where within
   the tail conservation ends.
2. The tail is positive but sub-threshold, not zero. The claim is "fails the
   program's conservation bar", not "no conservation at all".
3. `amazon_computers/thesis_gcn` is slightly negative (`-0.0039`) and is the cell
   with 58% dead edges. The verdict does not depend on it — three of four cells
   are positive — but that cell contributes least reliable evidence throughout.
4. Primary layer pair only, within cells only, on development data only.

## Reproduction

```
PYTHONPATH=src:expansion/src python3.11 \
  -m gnn_expansion.edge_tail_v2 \
  --config expansion/configs/edge_tail_v2.yaml \
  --output expansion/artifacts/edge_tail_v2.json
```

Report `expansion/artifacts/edge_tail_v2.json`. New code confined to
`edge_tail_v2.py`; `edge_tail.py`, `seed_stability.py`, `cross_layer.py` and
`cross_layer_edges.py` are hash-locked and were imported, never edited.

## Publication note

This is an archival decision record. Results, corrections, and withdrawals are retained as recorded. Local interpreter paths have been normalized where present. Paths under `expansion/` refer to the original research archive, whose code, full reports, model inputs, and caches are not included in this public package. The historical commands above require that archive. `SOURCE_MANIFEST.json` records the original source hash and the hash of this published copy.
