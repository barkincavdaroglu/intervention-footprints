"""Prospective effect-field replication on blinded final holdout graphs."""

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
    _validate_config as _validate_processing_config,
)
from gnn_repair.final_holdout_raw import EXPECTED_CELLS
from gnn_repair.gate1_prediction import run_within_stratum_loco
from gnn_repair.protocol import canonical_payload_sha256
from gnn_repair.topology_gate1_refined import (
    _file_sha256,
    _load_raw_checkpoint,
    _resolve_path,
    _write_json_atomic,
)


FINAL_HOLDOUT_EFFECT_VERSION = 1
GNN_CELLS = (
    "chameleon_filtered/residual_gcn",
    "chameleon_filtered/thesis_gcn",
    "cora_ml/residual_gcn",
    "cora_ml/thesis_gcn",
)
MLP_CELLS = (
    "chameleon_filtered/matched_mlp",
    "cora_ml/matched_mlp",
)
BACKBONES = ("thesis_gcn", "residual_gcn")
DATASETS = ("cora_ml", "chameleon_filtered")
METRICS = ("rmse_reduction", "mae_reduction", "spearman_gain", "cosine_gain")
FINAL_HOLDOUT_EFFECT_CODE_FILES = (
    "final_holdout_effect_field.py",
    "final_holdout_raw.py",
    *EFFECT_FIELD_CODE_FILES,
)


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("fresh_holdout") is not True:
        raise ValueError("Final effect analysis requires fresh_holdout: true")
    return _validate_processing_config({**dict(config), "development_only": True})


def _validate_raw_safety(report: Mapping[str, Any]) -> None:
    protocol = report.get("protocol", {})
    exact = {
        "development_only": False,
        "prospective_holdout": True,
        "fresh_holdouts_used": True,
        "confirmatory_artifacts_used": False,
        "test_labels_used": False,
        "test_performance_used": False,
        "test_status": "blinded",
    }
    invalid = [name for name, expected in exact.items() if protocol.get(name) != expected]
    if invalid:
        raise ValueError("Final raw report safety mismatch: " + ", ".join(invalid))


def _primary_increment(record: Mapping[str, Any]) -> float:
    return float(
        record["inference"]["class_kl"]["active_channels"]["candidates"]
        [PRIMARY_CANDIDATE]["incremental_alignment"]
    )


def _loco_audit(loco: Mapping[str, Any]) -> dict[str, Any]:
    cells = tuple(sorted(loco["cells"]))
    if cells != tuple(sorted(GNN_CELLS)):
        raise ValueError("Final LOCO analysis lacks the four frozen GNN cells")
    audit: dict[str, Any] = {}
    for candidate in CANDIDATE_NAMES:
        all_values = {metric: [] for metric in METRICS}
        cell_means: dict[str, dict[str, float | None]] = {}
        for cell in cells:
            folds = loco["cells"][cell]["candidates"][candidate]["folds"]
            cell_means[cell] = {}
            for metric in METRICS:
                values = [
                    float(fold["improvement"][metric])
                    for fold in folds
                    if fold["improvement"][metric] is not None
                    and math.isfinite(float(fold["improvement"][metric]))
                ]
                all_values[metric].extend(values)
                cell_means[cell][metric] = float(mean(values)) if values else None
        overall = {
            metric: float(mean(values)) if values else None
            for metric, values in all_values.items()
        }
        positive_cells = {
            metric: sum(
                cell_means[cell][metric] is not None
                and float(cell_means[cell][metric]) > 0
                for cell in cells
            )
            for metric in METRICS
        }
        non_worse = bool(
            overall["rmse_reduction"] is not None
            and overall["mae_reduction"] is not None
            and float(overall["rmse_reduction"]) >= 0
            and float(overall["mae_reduction"]) >= 0
        )
        improving = [
            metric
            for metric in METRICS
            if overall[metric] is not None
            and float(overall[metric]) > 0
            and positive_cells[metric] >= 3
        ]
        audit[candidate] = {
            "checkpoint_weighted_mean": overall,
            "cell_means": cell_means,
            "positive_cell_count": positive_cells,
            "non_worse_than_baseline": non_worse,
            "improving_metrics_in_at_least_three_cells": improving,
            "pass": bool(non_worse and improving),
        }
    return audit


def evaluate_holdout_replication(
    records: Sequence[Mapping[str, Any]],
    loco_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if {str(record["cell"]) for record in records} != set(GNN_CELLS):
        raise ValueError("Final replication records lack the four frozen GNN cells")
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
        "positive_in_three_cells": {
            "pass": len(positive_cells) >= 3,
            "positive_cells": positive_cells,
            "count": len(positive_cells),
        },
        "all_channel_incremental_median": {
            "pass": all_channels["median"] > 0,
            "value": all_channels["median"],
        },
        "loco_prediction": dict(loco_audit[PRIMARY_CANDIDATE]),
    }
    passed = all(bool(item["pass"]) for item in criteria.values())
    return {
        "status": "pass" if passed else "fail",
        "criteria": criteria,
        "primary_active": active,
        "primary_all_channels": all_channels,
        "stage_b_authorized": passed,
        "repair_authorized": False,
    }


def evaluate_graph_specificity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {
        (str(record["dataset"]), str(record["backbone"]), int(record["seed"])): record
        for record in records
    }
    results: dict[str, Any] = {}
    for backbone in BACKBONES:
        details = []
        by_dataset: dict[str, list[float]] = {}
        for dataset in DATASETS:
            gnn_seeds = sorted(
                seed for observed_dataset, observed_backbone, seed in indexed
                if observed_dataset == dataset and observed_backbone == backbone
            )
            mlp_seeds = sorted(
                seed for observed_dataset, observed_backbone, seed in indexed
                if observed_dataset == dataset and observed_backbone == "matched_mlp"
            )
            if not gnn_seeds or gnn_seeds != mlp_seeds:
                raise ValueError(f"Final GNN/MLP pairing failed for {dataset}/{backbone}")
            for seed in gnn_seeds:
                gnn = _primary_increment(indexed[(dataset, backbone, seed)])
                mlp = _primary_increment(indexed[(dataset, "matched_mlp", seed)])
                difference = gnn - mlp
                by_dataset.setdefault(dataset, []).append(difference)
                details.append({"dataset": dataset, "seed": seed, "gnn_minus_mlp": difference})
        values = [float(item["gnn_minus_mlp"]) for item in details]
        datasets = {
            dataset: {
                "count": len(dataset_values),
                "mean": float(mean(dataset_values)),
                "median": float(median(dataset_values)),
            }
            for dataset, dataset_values in sorted(by_dataset.items())
        }
        positive = [dataset for dataset, item in datasets.items() if item["median"] > 0]
        passed = float(median(values)) > 0 and len(positive) == len(DATASETS)
        results[backbone] = {
            "pass": passed,
            "count": len(values),
            "mean": float(mean(values)),
            "median": float(median(values)),
            "positive_count": int(sum(value > 0 for value in values)),
            "datasets": datasets,
            "positive_datasets": positive,
            "details": details,
        }
    pass_count = sum(bool(item["pass"]) for item in results.values())
    return {
        "status": "strong" if pass_count == 2 else "partial" if pass_count == 1 else "not_supported",
        "backbones": results,
    }


def build_final_holdout_effect_report(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    gate = _validate_config(config)
    raw_path = Path(config["raw_report"]).resolve()
    with raw_path.open("r", encoding="utf-8") as stream:
        raw_report = json.load(stream)
    _validate_raw_safety(raw_report)
    project_root = raw_path.parent.parent.parent
    protocol_path = _resolve_path(config["protocol_document"], project_root)
    protocol_sha256 = _file_sha256(protocol_path)
    cache_root = _resolve_path(config["cache_root"], project_root)
    observed = {
        f"{dataset}/{backbone}"
        for dataset, dataset_result in raw_report["datasets"].items()
        for backbone in dataset_result["backbones"]
    }
    if observed != set(EXPECTED_CELLS):
        raise ValueError("Final raw report differs from the frozen six cells")

    records: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    for dataset, dataset_result in sorted(raw_report["datasets"].items()):
        for backbone, backbone_result in sorted(dataset_result["backbones"].items()):
            runs = sorted(backbone_result["runs"], key=lambda item: (int(item["seed"]), str(item["checkpoint_model_sha256"])))
            for position, embedded in enumerate(runs, start=1):
                if progress:
                    print(f"[{dataset}/{backbone}] checkpoint {position}/{len(runs)}", flush=True)
                raw = _load_raw_checkpoint(
                    dataset=dataset,
                    backbone=backbone,
                    embedded_run=embedded,
                    project_root=project_root,
                )
                cache_path = cache_root / dataset / backbone / f"seed_{int(raw.run['seed'])}_{raw.checkpoint_id[:12]}.json"
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
    gnn_items = [
        (record, prediction_arrays)
        for record, prediction_arrays in zip(records, arrays, strict=True)
        if str(record["cell"]) in GNN_CELLS
    ]
    loco: dict[str, Any] = {}
    loco_audit: dict[str, Any] = {}
    for subset in ("active_channels", "all_channels"):
        checkpoints = [
            _prediction_checkpoint(record, prediction_arrays, subset=subset)
            for record, prediction_arrays in gnn_items
        ]
        result = run_within_stratum_loco(checkpoints, ridge_alpha=float(gate["ridge_alpha"]))
        loco[subset] = result
        loco_audit[subset] = _loco_audit(result)
    gnn_records = [record for record, _ in gnn_items]
    mlp_records = [record for record in records if str(record["cell"]) in MLP_CELLS]
    decision = evaluate_holdout_replication(gnn_records, loco_audit["active_channels"])
    specificity = evaluate_graph_specificity(records)
    return {
        "protocol": {
            "name": "Final prospective holdout effect-field replication",
            "version": FINAL_HOLDOUT_EFFECT_VERSION,
            "processing_protocol_version": EFFECT_FIELD_PROTOCOL_VERSION,
            "development_only": False,
            "prospective_holdout": True,
            "fresh_holdouts_used": True,
            "confirmatory_artifacts_used": False,
            "test_labels_used": False,
            "test_performance_used": False,
            "test_status": "blinded",
            "repair_authorized": False,
            "raw_report": str(raw_path),
            "raw_report_sha256": _file_sha256(raw_path),
            "protocol_document": str(protocol_path),
            "protocol_document_sha256": protocol_sha256,
            "config_sha256": canonical_payload_sha256(config),
            "code_sha256": source_code_sha256(FINAL_HOLDOUT_EFFECT_CODE_FILES),
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
        "gnn_alignment_summaries": {
            subset: {
                candidate: _increment_summary(gnn_records, outcome="class_kl", subset=subset, candidate=candidate)
                for candidate in CANDIDATE_NAMES
            }
            for subset in ("active_channels", "all_channels")
        },
        "mlp_alignment_summaries": {
            candidate: _increment_summary(mlp_records, outcome="class_kl", subset="active_channels", candidate=candidate)
            for candidate in CANDIDATE_NAMES
        },
        "leave_one_checkpoint_out_prediction": loco,
        "loco_audit": loco_audit,
        "effect_field_holdout_decision": decision,
        "graph_specificity_decision": specificity,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final_holdout_effect_field.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_final_holdout_effect_report(config, resume=args.resume, progress=not args.quiet)
    output = args.output or Path(config["output"])
    _write_json_atomic(output, report)
    print(f"Wrote final holdout effect-field report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
