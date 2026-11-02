# What an ablation can resolve: figure data

Saved measurements for the companion post, covering 30 development checkpoints in four dataset–architecture cells. This is a small figure-reproduction package, not an execution package for the extension experiments.

From the repository root, using Python 3.11 and the existing figure requirements:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-figures.txt
python verify.py
python notes/later-controls/companion/make_figures.py --output /tmp/ablation-resolution-figures
```

The last command produces five PNG/SVG pairs and a hash manifest. It does not download data, load models, or rerun experiments.

## Data and aggregation

`figure-data.json` contains checkpoint-level values extracted from eight frozen reports, cumulative edge-strength curves, and the exact/first-order edge vectors for the checkpoint nearest the pooled median signed-effect correlation. The example was selected by that rule, not by the best agreement. The source reports and all 30 edge-matrix sidecars are identified by archive-relative path and SHA-256. During extraction, every sidecar hash was checked and its signed-effect Spearman correlation independently recomputed against the report.

The full source reports, sidecars, training inputs, and extension execution code are not included here. Hashes identify those archival inputs but do not make them independently retrievable. The delivered values support figure regeneration and checkpoint/cell-level reaggregation, not verification of the original forward passes or null construction.

Dots represent checkpoints, colored bars cell medians, and black diamonds pooled medians where shown. The 30 checkpoints are reused across analyses. Cell medians are descriptive summaries, not independent trials or confidence intervals. Curves use 201 rank fractions, interpolated within checkpoint before taking cell medians. Amplification is aggregated in log2 space and then exponentiated. PCA displacement comparisons are ratios of cell medians. A median of paired differences generally differs from a difference of medians.

## Publication clarifications

Two errors in the archival prose were identified while checking the figures. The affected decision records retain their original text below dated clarification notices.

1. In `cross_layer_edges_decision.md`, the magnitude-Spearman cell range is **0.8614–0.9972**, not 0.8614–0.9938. Amazon Computers / narrow GCN has the maximum cell median, 0.9972380397. The headline signed-effect correlation is unchanged.
2. In `edge_tail_v2_decision.md`, E7b is an **agreement gain over shuffled attribution**, not a raw Spearman correlation: actual exact/approximate Spearman agreement minus the median shuffled-attribution agreement within the stratum. Its pooled head/tail values remain +0.9807/+0.9792. As a difference of correlations it can exceed one. Both computations use the same trained model, so agreement does not establish the absence of all noise or the behavioral importance of the tail.

The definitions field in the JSON specifies the plotted measurements. These clarifications do not change the frozen experimental outputs or their registered decisions.
