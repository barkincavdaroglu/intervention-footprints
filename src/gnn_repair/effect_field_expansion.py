"""High-accuracy replication and graph-free control for effect fields."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
import torch
import yaml

from gnn_repair.checkpoints import source_code_sha256
from gnn_repair.effect_field_gate import (
    CANDIDATE_NAMES,
    EFFECT_FIELD_CODE_FILES,
    EFFECT_FIELD_PROTOCOL_VERSION,
    PRIMARY_CANDIDATE,
    _increment_summary,
    _prediction_checkpoint,
    _process_checkpoint,
    _validate_config,
)
from gnn_repair.gate1_prediction import run_within_stratum_loco
from gnn_repair.protocol import canonical_payload_sha256
from gnn_repair.topology_gate1_refined import (
    _file_sha256,
    _load_raw_checkpoint,
    _resolve_path,
    _validate_raw_report_safety,
    _write_json_atomic,
)


EXPANSION_PROTOCOL_VERSION = 1
EXPECTED_CELLS = (
    "amazon_computers/matched_mlp",
    "amazon_computers/residual_gcn",
    "amazon_computers/thesis_gcn",
    "citationfull_citeseer/matched_mlp",
    "citationfull_citeseer/residual_gcn",
    "citationfull_citeseer/thesis_gcn",
    "roman_empire/matched_mlp",
    "roman_empire/residual_gcn",
    "roman_empire/thesis_gcn",
)
REPLICATION_CELLS = (
    "citationfull_citeseer/residual_gcn",
    "citationfull_citeseer/thesis_gcn",
)
MLP_CONTROL_CELLS = (
    "amazon_computers/matched_mlp",
    "citationfull_citeseer/matched_mlp",
    "roman_empire/matched_mlp",
)
PAIRED_BACKBONES = ("thesis_gcn", "residual_gcn")
PAIRED_DATASETS = (
    "citationfull_citeseer",
    "amazon_computers",
    "roman_empire",
)
METRIC_NAMES = ("rmse_reduction", "mae_reduction", "spearman_gain", "cosine_gain")
EXPANSION_CODE_FILES = (
    "effect_field_expansion.py",
    *EFFECT_FIELD_CODE_FILES,
)


def _validate_expansion_config(config: Mapping[str, Any]) -> dict[str, Any]:
    gate = _validate_config(config)
    expansion = dict(config.get("expansion", {}))
    exact = {
        "replication_cells": REPLICATION_CELLS,
        "mlp_control_cells": MLP_CONTROL_CELLS,
        "paired_gnn_backbones": PAIRED_BACKBONES,
        "datasets": PAIRED_DATASETS,
    }
    for name, expected in exact.items():
        if tuple(sorted(str(value) for value in expansion.get(name, ()))) != tuple(
            sorted(expected)
        ):
            raise ValueError(f"{name} differs from the frozen expansion protocol")
    return gate


def _primary_increment(record: Mapping[str, Any]) -> float:
    return float(
        record["inference"]["class_kl"]["active_channels"]["candidates"]
        [PRIMARY_CANDIDATE]["incremental_alignment"]
    )


def _loco_replication_audit(loco: Mapping[str, Any]) -> dict[str, Any]:
    cells = tuple(sorted(loco["cells"]))
    if cells != tuple(sorted(REPLICATION_CELLS)):
        raise ValueError("LOCO replication does not contain exactly the two frozen cells")
    audit: dict[str, Any] = {}
    for candidate in CANDIDATE_NAMES:
        values = {metric: [] for metric in METRIC_NAMES}
        cell_means: dict[str, dict[str, float | None]] = {}
        for cell in cells:
            folds = loco["cells"][cell]["candidates"][candidate]["folds"]
            cell_means[cell] = {}
            for metric in METRIC_NAMES:
                finite = [
                    float(fold["improvement"][metric])
                    for fold in folds
                    if fold["improvement"][metric] is not None
                    and math.isfinite(float(fold["improvement"][metric]))
                ]
                values[metric].extend(finite)
                cell_means[cell][metric] = float(mean(finite)) if finite else None
        overall = {
            metric: float(mean(metric_values)) if metric_values else None
            for metric, metric_values in values.items()
        }
        positive_cell_count = {
            metric: sum(
                cell_means[cell][metric] is not None
                and float(cell_means[cell][metric]) > 0
                for cell in cells
            )
            for metric in METRIC_NAMES
        }
        non_worse = bool(
            overall["rmse_reduction"] is not None
            and overall["mae_reduction"] is not None
            and float(overall["rmse_reduction"]) >= 0
            and float(overall["mae_reduction"]) >= 0
        )
        improving = [
            metric
            for metric in METRIC_NAMES
            if overall[metric] is not None
            and float(overall[metric]) > 0
            and positive_cell_count[metric] == len(cells)
        ]
        audit[candidate] = {
            "checkpoint_weighted_mean": overall,
            "cell_means": cell_means,
            "positive_cell_count": positive_cell_count,
            "non_worse_than_baseline": non_worse,
            "improving_metrics_in_both_cells": improving,
            "pass": bool(non_worse and improving),
        }
    return audit


def evaluate_replication(
    records: Sequence[Mapping[str, Any]],
    loco_audit: Mapping[str, Any],
) -> dict[str, Any]:
    cells = {str(record["cell"]) for record in records}
    if cells != set(REPLICATION_CELLS):
        raise ValueError("Replication records do not contain the two frozen cells")
    active = _increment_summary(
        records,
        outcome="class_kl",
        subset="active_channels",
        candidate=PRIMARY_CANDIDATE,
    )
    all_channels = _increment_summary(
        records,
        outcome="class_kl",
        subset="all_channels",
        candidate=PRIMARY_CANDIDATE,
    )
    positive_cells = [
        cell for cell, summary in active["cells"].items() if summary["median"] > 0
    ]
    criteria = {
        "active_incremental_median": {
            "pass": active["median"] > 0,
            "value": active["median"],
        },
        "both_cell_medians_positive": {
            "pass": len(positive_cells) == len(REPLICATION_CELLS),
            "positive_cells": positive_cells,
        },
        "all_channel_incremental_median": {
            "pass": all_channels["median"] > 0,
            "value": all_channels["median"],
        },
        "loco_prediction": dict(loco_audit[PRIMARY_CANDIDATE]),
    }
    passed = all(bool(value["pass"]) for value in criteria.values())
    return {
        "status": "pass" if passed else "fail",
        "criteria": criteria,
        "primary_active": active,
        "primary_all_channels": all_channels,
        "stage_b_joint_ablation_authorized": passed,
        "repair_reopened": False,
    }


def evaluate_graph_specificity(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    indexed = {
        (str(record["dataset"]), str(record["backbone"]), int(record["seed"])): record
        for record in records
    }
    backbone_results: dict[str, Any] = {}
    for backbone in PAIRED_BACKBONES:
        details = []
        by_dataset: dict[str, list[float]] = {}
        for dataset in PAIRED_DATASETS:
            seeds = sorted(
                seed
                for observed_dataset, observed_backbone, seed in indexed
                if observed_dataset == dataset and observed_backbone == backbone
            )
            mlp_seeds = sorted(
                seed
                for observed_dataset, observed_backbone, seed in indexed
                if observed_dataset == dataset and observed_backbone == "matched_mlp"
            )
            if seeds != mlp_seeds or not seeds:
                raise ValueError(f"GNN/MLP seeds do not pair exactly for {dataset}/{backbone}")
            for seed in seeds:
                gnn = _primary_increment(indexed[(dataset, backbone, seed)])
                mlp = _primary_increment(indexed[(dataset, "matched_mlp", seed)])
                difference = gnn - mlp
                by_dataset.setdefault(dataset, []).append(difference)
                details.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "gnn_increment": gnn,
                        "mlp_increment": mlp,
                        "gnn_minus_mlp": difference,
                    }
                )
        values = [float(item["gnn_minus_mlp"]) for item in details]
        datasets = {
            dataset: {
                "count": len(dataset_values),
                "mean": float(mean(dataset_values)),
                "median": float(median(dataset_values)),
            }
            for dataset, dataset_values in sorted(by_dataset.items())
        }
        positive_datasets = [
            dataset for dataset, summary in datasets.items() if summary["median"] > 0
        ]
        passed = float(median(values)) > 0 and len(positive_datasets) >= 2
        backbone_results[backbone] = {
            "pass": passed,
            "count": len(values),
            "mean": float(mean(values)),
            "median": float(median(values)),
            "positive_count": int(sum(value > 0 for value in values)),
            "datasets": datasets,
            "positive_datasets": positive_datasets,
            "details": details,
        }
    pass_count = sum(bool(result["pass"]) for result in backbone_results.values())
    return {
        "status": (
            "strong" if pass_count == 2 else "partial" if pass_count == 1 else "not_supported"
        ),
        "estimand": "active mask-field increment in GNN minus seed-matched MLP",
        "backbones": backbone_results,
    }


def build_effect_field_expansion_report(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    gate = _validate_expansion_config(config)
    raw_report_path = Path(config["raw_report"]).resolve()
    with raw_report_path.open("r", encoding="utf-8") as stream:
        raw_report = json.load(stream)
    _validate_raw_report_safety(raw_report)
    project_root = raw_report_path.parent.parent
    protocol_path = _resolve_path(config["protocol_document"], project_root)
    protocol_sha256 = _file_sha256(protocol_path)
    cache_root = _resolve_path(config["cache_root"], project_root)
    observed_cells = {
        f"{dataset}/{backbone}"
        for dataset, dataset_result in raw_report.get("datasets", {}).items()
        for backbone in dataset_result.get("backbones", {})
    }
    if observed_cells != set(EXPECTED_CELLS):
        raise ValueError("Expansion raw report does not contain exactly the frozen 3x3 cells")

    records: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    for dataset, dataset_result in sorted(raw_report["datasets"].items()):
        for backbone, backbone_result in sorted(dataset_result["backbones"].items()):
            runs = sorted(
                backbone_result["runs"],
                key=lambda value: (int(value["seed"]), str(value["checkpoint_model_sha256"])),
            )
            for position, embedded_run in enumerate(runs, start=1):
                if progress:
                    print(f"[{dataset}/{backbone}] checkpoint {position}/{len(runs)}", flush=True)
                raw = _load_raw_checkpoint(
                    dataset=dataset,
                    backbone=backbone,
                    embedded_run=embedded_run,
                    project_root=project_root,
                )
                cache_path = (
                    cache_root
                    / dataset
                    / backbone
                    / f"seed_{int(raw.run['seed'])}_{raw.checkpoint_id[:12]}.json"
                )
                record, prediction_arrays = _process_checkpoint(
                    raw,
                    config=config,
                    gate=gate,
                    protocol_sha256=protocol_sha256,
                    cache_path=cache_path,
                    resume=resume,
                    progress=progress,
                )
                records.append(record)
                arrays.append(prediction_arrays)

    replication = [
        (record, prediction_arrays)
        for record, prediction_arrays in zip(records, arrays, strict=True)
        if str(record["cell"]) in REPLICATION_CELLS
    ]
    loco: dict[str, Any] = {}
    loco_audit: dict[str, Any] = {}
    for subset in ("active_channels", "all_channels"):
        checkpoints = [
            _prediction_checkpoint(record, prediction_arrays, subset=subset)
            for record, prediction_arrays in replication
        ]
        result = run_within_stratum_loco(
            checkpoints,
            ridge_alpha=float(gate.get("ridge_alpha", 1.0)),
        )
        loco[subset] = result
        loco_audit[subset] = _loco_replication_audit(result)

    replication_records = [record for record, _ in replication]
    mlp_records = [
        record for record in records if str(record["cell"]) in MLP_CONTROL_CELLS
    ]
    replication_decision = evaluate_replication(
        replication_records,
        loco_audit["active_channels"],
    )
    graph_specificity = evaluate_graph_specificity(records)
    return {
        "protocol": {
            "name": "Effect-field high-accuracy replication and graph-free control",
            "version": EXPANSION_PROTOCOL_VERSION,
            "processing_protocol_version": EFFECT_FIELD_PROTOCOL_VERSION,
            "status": "development replication on previously inspected datasets",
            "development_only": True,
            "fresh_holdouts_used": False,
            "confirmatory_artifacts_used": False,
            "test_labels_used": False,
            "test_performance_used": False,
            "repair_reopened": False,
            "raw_report": str(raw_report_path),
            "raw_report_sha256": _file_sha256(raw_report_path),
            "protocol_document": str(protocol_path),
            "protocol_document_sha256": protocol_sha256,
            "config_sha256": canonical_payload_sha256(config),
            "code_sha256": source_code_sha256(EXPANSION_CODE_FILES),
            "expected_cells": list(EXPECTED_CELLS),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
            },
        },
        "checkpoint_count": len(records),
        "checkpoint_records": records,
        "replication_alignment_summaries": {
            subset: {
                candidate: _increment_summary(
                    replication_records,
                    outcome="class_kl",
                    subset=subset,
                    candidate=candidate,
                )
                for candidate in CANDIDATE_NAMES
            }
            for subset in ("active_channels", "all_channels")
        },
        "mlp_control_alignment_summaries": {
            candidate: _increment_summary(
                mlp_records,
                outcome="class_kl",
                subset="active_channels",
                candidate=candidate,
            )
            for candidate in CANDIDATE_NAMES
        },
        "replication_leave_one_checkpoint_out_prediction": loco,
        "replication_loco_audit": loco_audit,
        "replication_decision": replication_decision,
        "graph_specificity_decision": graph_specificity,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/effect_field_expansion.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_effect_field_expansion_report(
        config,
        resume=args.resume,
        progress=not args.quiet,
    )
    output = args.output or Path(config["output"])
    _write_json_atomic(output, report)
    print(f"Wrote effect-field expansion report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
