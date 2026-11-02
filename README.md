# Can a component’s intervention footprint reveal its function?

Code, frozen checkpoint-level results, and protocols for classwise effect alignment, joint ablation, and learned repair in small graph neural networks.

## Published posts

- [What an ablation can resolve](https://www.barkincavdar.com/work/what-an-ablation-can-resolve/) - [archived PDF, November 1, 2026](paper/what-an-ablation-can-resolve-2026-11-01.pdf). The [publication record](paper/PUBLICATION_RECORD.md) gives the website publication time, later PDF export time, and file fingerprints.

## Reproduce the reported summaries

Python 3.11 or newer is sufficient for the integrity check and numerical summary. No extra packages, model checkpoints, or dataset downloads are needed for these two commands:

```sh
python3 verify.py
python3 reproduce.py --output /tmp/intervention-footprints-summary.json
```

The summary reaggregates checkpoint-level records, including group medians and matched GNN-minus-MLP contrasts. The principal results are:

| Comparison | Median | Direction across checkpoints |
| --- | --- | --- |
| Prospective footprint alignment improvement | +0.169507 | 29/30 positive |
| Prospective joint-ablation contrast | +0.000348306 KL units | 27/30 positive |
| Development repair-interaction contrast | −0.002713676 utility units | 22/30 negative |

These comparisons have different outcomes and evaluation sets. The repair result is from development checkpoints, not the prospective holdout. Read the frozen protocols in `docs/` for their definitions, matching rules, and decision criteria.

## Regenerate the figures

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-figures.txt
python paper/make_figures.py
```

This regenerates the manuscript’s hypothesis diagram and final-holdout effect/alignment and joint-ablation figures from the included JSON reports. PDF and PNG files and a report-hash manifest are written to `paper/figures/`. Generated copies are not tracked.

The [companion figure package](notes/later-controls/companion/) adds saved checkpoint measurements and a generator for five figures about basis controls, amplification, and cross-layer effects. Run `python notes/later-controls/companion/make_figures.py --output /tmp/ablation-resolution-figures` in the same environment. This reproduces the figures, not the extension experiments.

## Experiment code and inputs

- `src/gnn_repair/`: training, interventions, effect-field analysis, and dependency modules.
- `configs/`: frozen dataset, model, and experiment configurations.
- `docs/`: scientific protocols, decisions, and limits on the claims, including unsuccessful repair and topology comparisons.
- `artifacts/`: numerical reports used for the main-study summaries.
- `notes/later-controls/`: decision records for the [channel-basis comparison](notes/later-controls/basis_dependence_decision.md), [matched directions](notes/later-controls/matched_null_decision.md), [output amplification](notes/later-controls/direction_amplification_decision.md), [exact cross-layer effects and attribution fidelity](notes/later-controls/cross_layer_edges_decision.md), [cross-seed edge stability](notes/later-controls/cross_layer_seed_stability_decision.md), [weak-edge tail](notes/later-controls/edge_tail_v2_decision.md), and [functional correspondence](notes/later-controls/functional_equivalence_decision.md). The [original sparse-feature decision](notes/later-controls/feature_basis_decision.md) retains its retraction addendum. A [seven-entry instrument log extract](notes/later-controls/instrument-log-extract.md) covers the direction studies only. The extension experiments’ full reports, raw data, and execution code are outside this package, so these records document the results but do not provide a runnable reproduction of those extensions.
- `paper/make_figures.py`: the figure generator.
- `SOURCE_MANIFEST.json` and `verify.py`: current file hashes and source provenance.

Install the experimental package with `python -m pip install -e .`. Its command-line entrypoints and dependencies are declared in `pyproject.toml`. Model weights, datasets, and raw effect caches are not bundled. The recorded main-study CPU environment used Python 3.11.7, NumPy 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0, and PyTorch 2.7.1 with one thread.

For the prospective datasets, immutable input identities are:

| Dataset | Source revision | Archive SHA-256 |
| --- | --- | --- |
| Cora-ML | `graph2gauss` commit `240248289d38c22ba28ea8e56b7d6873e291499b` | `bff04cbfaf903581de18b28dd112b92b2e12c05dfcb1cfc5f6875a5a280d9872` |
| Chameleon-filtered | `heterophilous-graphs` commit `a431395582e929d88271309716bea4fe24ce6318` | `bf46f07e1fb5249280447e5fe3100f3e82fc4b93ad1e13ffcfdec924b6ac0bb5` |

Exact URLs, splits, training settings, and stopping rules are in `docs/final_interpretation_holdout_protocol.md` and the corresponding configs. These experiments use validation labels and keep test labels blinded.

The selected intervention layer is cell-specific: the narrow GCN uses its first hidden activation. Residual GCNs use the first block except Roman Empire, which uses block 4, the last of five. This asymmetry affects the development repair comparison.

## Scope and provenance

The quick-start commands reproduce summaries and figures from saved results. They do not retrain models or replay the later extension studies. Fresh runs require dataset/model inputs and the frozen experimental setup. Some historical paths and hashes in saved records refer to the original archive, so byte-for-byte comparison with newly generated reports is not an advertised replay check.

Personal paths in distributed reports were sanitized without changing numerical outcomes. Embedded historical hashes identify the original records; `SOURCE_MANIFEST.json` authenticates the current files. File integrity is distinct from scientific validation. Effect similarity, ablation nonadditivity, repair utility, and amplification of output changes are separate measurements.
