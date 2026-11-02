# Instrument log extract: direction studies

This extract records seven failures and corrections from the direction studies
discussed in the intervention-footprints post. It retains the original IDs
**I1, I2, I7, I8, I9, I13, and I14**. Cross-layer studies are outside this release.

The text is abridged from the research archive, with public supporting records
linked where available. Detection timing is reported as it appears in that
archive. This extract was prepared on **2026-11-01** and does not independently
establish when protocols were frozen or that every attempted analysis was logged.
Checkpoint hashes refer to recorded model states. Model files and the full
extension data are not bundled with this package.

| Study label | Comparison |
| --- | --- |
| W1.a | Channel basis versus rotated directions |
| W1.c | Learned sparse-dictionary features versus dense controls |
| W1.c-bis | Sparse features versus displacement-matched random directions |
| W1.e | Amplification of output changes along learned directions |

## I1 — The displacement confound

**Study:** W1.c · **Caught by:** a per-condition diagnostic · **Cost avoided:**
a false central finding

Sparse dictionary features fire on a small minority of nodes — a median `0.04%`
to `3.6%` by cell — so ablating one displaces the layer `14.5x` to `3302x` less
than ablating a channel (`0.009` versus `0.158` on Roman-empire residual;
`0.000003` versus `0.010` on Amazon thesis). Both of W1.c's nulls were dense, so
the comparison silently pitted a nudge against a shove.

**How it surfaced.** The runner reports median intervention displacement per
condition — a diagnostic added only because W1.a had recorded the *same* failure
mode for its PCA basis, where the median direction moved the activation ~100×
less than a channel. The second occurrence was recognized because the first had
been written down.

**What it would have cost.** Either sign was reachable depending on which subset
was read: `−0.2133` or `+0.0275`. A study that quietly reported one of them would
have produced a confident and wrong answer about superposition.

**Fix.** W1.c-bis: a null keeping each feature's exact per-node displacement and
magnitude, randomizing only direction. Verified at `1.4e-07` worst mismatch
against a 1% gate.

The public [matched-null decision memo](matched_null_decision.md) records the
replacement control and its displacement-matching check.

---

## I2 — Two claims that didn't survive their own follow-up

**Study:** W1.c → W1.c-bis · **Caught by:** the matched null · **Cost:** two
retractions, already reported

Dictionary features looked far more monosemantic (`+0.1278`, 30/30) and far more
localized (~9×) than rotations. Both were reported as clean positives at the
time. Against the displacement-matched null they fall to `+0.0104` (20/30) and
`−0.0006` (11/30).

**What it would have cost.** These were the most quotable results in the
workstream and would have been the ones a paper leaned on.

**Correction record.** The original archive recorded these withdrawals as L10a
and L10b and added a correction at the head of the original feature-basis memo.
The public [matched-null decision memo](matched_null_decision.md) preserves the
revised findings and the withdrawal of both earlier claims.

---

## I7 — A subset comparison that wasn't like-for-like

**Study:** W1.a · **Caught by:** the active-direction diagnostic during a
single-checkpoint probe · **Cost avoided:** a cardinality artifact read as a
basis effect

On Amazon's thesis GCN, 16 of 32 channels are dead, while a Haar-random
direction is essentially never dead. The "active directions" subset therefore
compared a 16-direction kernel against a 32-direction one.

**Fix, before the full run.** Both subsets — active and matched-count — computed
for every condition, with a verdict requiring them to agree. When they later
disagreed in W1.c, that requirement is what produced `inconclusive` instead of a
convenient answer.

---

## I8 — Runtime drift blocked a study, correctly

**Study:** W1.a · **Caught by:** the cache replay gate · **Cost avoided:**
silently comparing against subtly different numbers

A first full run used Python 3.13 rather than the recorded 3.11.7. The thesis GCN
reproduced bitwise; the residual GCN drifted by `2–5e-6` through version-sensitive
GELU and LayerNorm kernels. On two of 30 checkpoints that shifted **one node**
out of 1,450,368 across an argmax boundary, flipping its prediction-change flag
and failing the gate.

**What it demonstrates.** The gate is calibrated finely enough to detect a
single-node difference in a 1.45-million-entry mask.

**Fix.** Re-ran under the recorded runtime; replay became `exact` on all 30. The
scientific conclusions were unchanged (privilege medians moved by `2.6e-5`), so
the gate blocked a *provenance* problem, not a result — which is exactly its job.

---

## I9 — The environment silently lost file access

**Study:** W1.c-bis · **Caught by:** the run failing loudly · **Cost avoided:**
a substituted-data run

A restart left the process without macOS access to the directory holding the
original checkpoints. The run died on checkpoint 1 with `PermissionError` rather
than falling back to any other copy.

**Worth recording because** repo-local copies existed and were verified
hash-identical, so an "obliging" implementation could have used them and produced
correct numbers under a different provenance chain. Failing loudly kept the
choice with the researcher, who restored access — and all four studies now cite
the same raw-report hash.

---

## I13 — The working directory silently selected the wrong model

**Study:** W1.e · **Caught by:** the checkpoint state-hash gate ·
**Cost avoided:** a whole study computed on the wrong models under a correct-
looking provenance chain

W1.e is the first expansion study to load trained models from a *copy* of the
repository. `_resolve_recorded_path` resolves a run record's relative
`checkpoint` field by checking, in order: absolute path, **the path relative to
the current working directory**, then the run record's own parent directories.
Run from the repo copy, the second rule wins: `artifacts/.../seed_1.pt` exists
locally, so the local file is loaded even though the run record being read lives
in a different tree.

The local checkpoints are **not** the recorded ones. Checked across all 30
development checkpoints:

| resolution | matches the recorded `checkpoint_model_sha256` |
|---|---:|
| the run record's own tree | **30/30** |
| the copy's local `artifacts/` | **0/30** |

So every model this study would have used was the wrong one, and every model it
should have used was available and correct.

**What the gate bought.** `_load_raw_checkpoint` compares
`model_state_sha256(loaded.checkpoint)` against the raw record and raised on
checkpoint 1. Nothing was computed, and the run died in under two seconds.

**Fix.** Run from a working directory that contains the datasets but no
`artifacts/` tree, so the current-working-directory rule cannot fire and
resolution falls through to the run record's parents. The loader was left
unchanged to preserve the completed studies' recorded code hashes.

**Lesson.** Both files loaded successfully and contained valid checkpoints of
the same architecture. Verifying the loaded model's content against the recorded
hash exposed a mismatch that the file path and architecture alone did not.

The public [amplification decision memo](direction_amplification_decision.md)
records the failure and correction. The packaged
[path resolver](../../src/gnn_repair/checkpoints.py) and
[checkpoint hash check](../../src/gnn_repair/topology_gate1_refined.py) show the
loading order and content check.

---

## I14 — Runtime drift, the second time, caught by a different gate

**Study:** W1.e · **Caught by:** the dictionary-replay
gate · **Cost avoided:** a study whose dictionaries were not the ones it claimed
to be reproducing

This is **I8 recurring in a new environment**, found by a gate written for an
unrelated purpose. The default interpreter had drifted from the program's
recorded runtime on four of five packages:

| package | recorded | found |
|---|---|---|
| python | 3.11.7 | 3.13.2 |
| numpy | 2.3.3 | 2.1.2 |
| scipy | 1.17.1 | 1.14.1 |
| scikit-learn | 1.8.0 | 1.5.2 |
| torch | 2.7.1 | 2.7.1 |

W1.e refits W1.c-bis's dictionaries and requires each refit SAE's SHA-256 to
equal the cached value. Under the drifted stack **all three dictionaries on all
four smoke checkpoints mismatched**, while the seed-derivation gate passed —
the seeds were right, the fitted weights were not. Had the study reported
numbers, they would have been "the learned direction versus its matched null"
for a *different* dictionary than the one whose verdict they were revising.

**Fix.** The exact recorded runtime was available: Python 3.11.7 with all
four package versions matching. Invoking that interpreter rather than the
default `python3` restored dictionary replay. No repository changes were needed.

**Lesson.** Check the interpreter's package versions against the recorded
runtime **before** starting a study, not after a gate fails — the program has
now paid for this twice. And note which gate caught it: the dictionary-replay
gate exists to confirm the study reproduces W1.c-bis's objects, and it doubled
as a runtime detector for free. Gates that pin *content* rather than *process*
catch failure modes their authors did not anticipate.

The public [amplification decision memo](direction_amplification_decision.md)
records this runtime correction alongside I13.

## Source and extraction notes

- Original archive locator: `expansion/docs/instrument_log.md`.
- Original file SHA-256: `91ec535e317a9546c1c65206ab1d6a5cda8c4540d8ed15b5aafefbf79dc5274c`.
- Included entries: I1, I2, I7, I8, I9, I13, I14, arranged by original ID.
- I1, I7, I8, and I9 retain their source wording. I1 adds a public source link.
- I2's archive-only correction references are summarized and linked to the
  packaged matched-null memo.
- I13 uses “recorded” for checkpoint identities, condenses the fix and lesson,
  and adds public source links. I14 retains the runtime mismatch and correction
  while omitting its comparison with an excluded study.
- Cross-layer entries and cross-layer passages within I13 and I14 are omitted.
  This is a scoped extract, not the complete instrument log.

`SOURCE_MANIFEST.json` records this distributed file's hash and the original
source-file identity. The archive locator above is provenance information,
not a promise that the full archive is included in this repository.
