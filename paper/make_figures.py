"""Generate deterministic manuscript figures from frozen JSON reports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
EFFECT_REPORT = ROOT / "artifacts" / "final_holdout" / "effect_field.json"
ACTION_REPORT = ROOT / "artifacts" / "final_holdout" / "actionability.json"

CELL_ORDER = [
    "chameleon_filtered/residual_gcn",
    "chameleon_filtered/thesis_gcn",
    "cora_ml/residual_gcn",
    "cora_ml/thesis_gcn",
]
CELL_LABELS = {
    "chameleon_filtered/residual_gcn": "Chameleon\nResidual GCN",
    "chameleon_filtered/thesis_gcn": "Chameleon\nThesis GCN",
    "cora_ml/residual_gcn": "Cora-ML\nResidual GCN",
    "cora_ml/thesis_gcn": "Cora-ML\nThesis GCN",
    "chameleon_filtered/matched_mlp": "Chameleon\nMatched MLP",
    "cora_ml/matched_mlp": "Cora-ML\nMatched MLP",
}
MLP_ORDER = ["chameleon_filtered/matched_mlp", "cora_ml/matched_mlp"]

BLUE = "#356AA0"
ORANGE = "#D17A22"
MEDIAN = "#1B1B1B"
GRID = "#D5D5D5"
GREEN = "#3F7D4A"
RED = "#A84444"
FIXED_TIMESTAMP = datetime(2026, 8, 4, tzinfo=timezone.utc)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _jitter(count: int, width: float = 0.13) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    return np.linspace(-width, width, count)


def _save(fig: mpl.figure.Figure, stem: str) -> None:
    common = {"Creator": "gnn-neuron-repair/paper/make_figures.py"}
    fig.savefig(
        OUT / f"{stem}.pdf",
        dpi=320,
        metadata={
            **common,
            "CreationDate": FIXED_TIMESTAMP,
            "ModDate": FIXED_TIMESTAMP,
        },
    )
    fig.savefig(OUT / f"{stem}.png", dpi=320, metadata=common)


def _draw_distribution(
    ax: mpl.axes.Axes,
    values_by_cell: dict[str, list[float]],
    order: list[str],
    color: str,
) -> None:
    for position, cell in enumerate(order):
        values = np.sort(np.asarray(values_by_cell[cell], dtype=float))
        ax.scatter(
            position + _jitter(len(values)),
            values,
            s=25,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.92,
            zorder=3,
        )
        median = float(np.median(values))
        ax.plot(
            [position - 0.22, position + 0.22],
            [median, median],
            color=MEDIAN,
            linewidth=2.1,
            solid_capstyle="round",
            zorder=4,
        )
    ax.axhline(0.0, color="#777777", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(order)), [CELL_LABELS[cell] for cell in order])
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8, zorder=0)


def _box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    edge: str,
    fill: str = "white",
    fontsize: float = 8.5,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def _arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#666666",
            shrinkA=2,
            shrinkB=2,
        )
    )


def concept_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top_y, top_h, top_w = 0.69, 0.20, 0.19
    top_x = [0.015, 0.275, 0.535, 0.795]
    top_text = [
        "Shared hidden\nchannel $i$",
        "Mean-replace $i$\nat every node",
        "Node-indexed field\n$s_i(v)=\\mathrm{TV}(p_v,p_v^{-i})$",
        "Pairwise multiscale\nsupport overlap $K_{ij}$",
    ]
    for x, label in zip(top_x, top_text, strict=True):
        _box(ax, (x, top_y), top_w, top_h, label, BLUE, fill="#F5F8FC")
    for left, right in zip(top_x[:-1], top_x[1:], strict=True):
        _arrow(ax, (left + top_w, top_y + top_h / 2), (right, top_y + top_h / 2))

    branch_y = 0.57
    source_x = top_x[-1] + top_w / 2
    _arrow(ax, (source_x, top_y), (source_x, branch_y))
    ax.plot([0.18, 0.82], [branch_y, branch_y], color="#777777", linewidth=1.0)

    bottom_y, bottom_h, bottom_w = 0.23, 0.24, 0.25
    bottom_x = [0.055, 0.375, 0.695]
    bottom = [
        ("H1  Classwise function\nPASS\n68/70 GNN checkpoints", GREEN, "#F2F8F3"),
        ("H2  Fixed joint ablation\nPASS\n67/70 GNN checkpoints", GREEN, "#F2F8F3"),
        ("H3  Learned affine repair\nFAIL\n22/30 checkpoints negative", RED, "#FCF4F4"),
    ]
    for x, (label, edge, fill) in zip(bottom_x, bottom, strict=True):
        center = x + bottom_w / 2
        _arrow(ax, (center, branch_y), (center, bottom_y + bottom_h))
        _box(ax, (x, bottom_y), bottom_w, bottom_h, label, edge, fill=fill, fontsize=8.2)

    ax.text(
        0.5,
        0.08,
        "Boundary controls: induced topology adds no reliable signal; matched MLPs retain the fingerprint",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#333333",
    )
    _save(fig, "effect_field_hypothesis_chain")
    plt.close(fig)


def effect_field_figure(effect: dict) -> None:
    gnn_values: dict[str, list[float]] = {cell: [] for cell in CELL_ORDER}
    for record in effect["effect_field_holdout_decision"]["primary_active"]["checkpoints"]:
        gnn_values[record["cell"]].append(record["incremental_alignment"])

    mlp_values: dict[str, list[float]] = {cell: [] for cell in MLP_ORDER}
    for record in effect["mlp_alignment_summaries"]["mask_field"]["checkpoints"]:
        mlp_values[record["cell"]].append(record["incremental_alignment"])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.0),
        gridspec_kw={"width_ratios": [2.05, 1.0], "wspace": 0.28},
        sharey=True,
    )
    _draw_distribution(axes[0], gnn_values, CELL_ORDER, BLUE)
    _draw_distribution(axes[1], mlp_values, MLP_ORDER, ORANGE)
    axes[0].set_title("A   GNN holdout replication", loc="left")
    axes[1].set_title("B   Graph-free control", loc="left")
    axes[0].set_ylabel("Incremental mask-field alignment")
    axes[0].set_ylim(-0.10, 0.51)
    axes[0].text(
        0.02,
        0.96,
        "29/30 positive",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    axes[1].text(
        0.04,
        0.96,
        "13/15 positive",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    _save(fig, "holdout_effect_field")
    plt.close(fig)


def joint_ablation_figure(action: dict) -> None:
    values_by_cell: dict[str, list[float]] = {cell: [] for cell in CELL_ORDER}
    for record in action["checkpoint_records"]:
        values_by_cell[record["cell"]].append(1000.0 * record["contrast"]["d_j_kl"])

    fig, ax = plt.subplots(figsize=(5.15, 3.0))
    _draw_distribution(ax, values_by_cell, CELL_ORDER, BLUE)
    ax.set_title("Prospective fixed joint-ablation interaction", loc="left")
    ax.set_ylabel(r"$D_J$ (macro-KL $\times 10^3$)")
    ax.set_ylim(-0.55, 7.05)
    ax.text(
        0.02,
        0.96,
        "27/30 positive; all four cell medians positive",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    _save(fig, "holdout_joint_ablation")
    plt.close(fig)


def write_manifest() -> None:
    lines = [
        "# Frozen source manifest for manuscript figures",
        "",
        f"- `{EFFECT_REPORT.relative_to(ROOT)}`: `{_sha256(EFFECT_REPORT)}`",
        f"- `{ACTION_REPORT.relative_to(ROOT)}`: `{_sha256(ACTION_REPORT)}`",
        "",
        "Generated by `paper/make_figures.py`. Test labels and test performance are not used.",
        "",
    ]
    (OUT / "SOURCE_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _configure()
    OUT.mkdir(parents=True, exist_ok=True)
    concept_figure()
    effect_field_figure(_load(EFFECT_REPORT))
    joint_ablation_figure(_load(ACTION_REPORT))
    write_manifest()


if __name__ == "__main__":
    main()
