"""CiteSeer replication of fixed joint-ablation effect-field interactions."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import torch
import yaml

from gnn_repair.checkpoints import source_code_sha256
from gnn_repair.effect_field_actionability import (
    ACTIONABILITY_CODE_FILES,
    ACTIONABILITY_PROTOCOL_VERSION,
    _process_checkpoint,
    _validate_config,
)
from gnn_repair.effect_field_expansion import EXPECTED_CELLS, REPLICATION_CELLS
from gnn_repair.protocol import canonical_payload_sha256
from gnn_repair.topology_gate1_refined import (
    _file_sha256,
    _load_raw_checkpoint,
    _resolve_path,
    _validate_raw_report_safety,
    _write_json_atomic,
)


ACTIONABILITY_REPLICATION_VERSION = 1
ACTIONABILITY_REPLICATION_CODE_FILES = (
    "effect_field_actionability_replication.py",
    "effect_field_expansion.py",
    *ACTIONABILITY_CODE_FILES,
)


def evaluate_actionability_replication(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cells = {str(record["cell"]) for record in records}
    if cells != set(REPLICATION_CELLS):
        raise ValueError("Actionability replication lacks the two frozen CiteSeer cells")
    incomplete = [
        str(record["checkpoint_id"])
        for record in records
        if not record["matching"]["evaluable"] or record["contrast"] is None
    ]
    if incomplete:
        return {
            "status": "not_evaluable",
            "pattern": None,
            "incomplete_checkpoint_ids": incomplete,
            "repair_reopened": False,
        }
    values = [float(record["contrast"]["d_j_kl"]) for record in records]
    ce_values = [
        float(record["contrast"]["d_j_ce_sensitivity"]) for record in records
    ]
    by_cell: dict[str, list[float]] = {}
    for record, value in zip(records, values, strict=True):
        by_cell.setdefault(str(record["cell"]), []).append(value)
    cell_summaries = {
        cell: {
            "count": len(cell_values),
            "mean": float(mean(cell_values)),
            "median": float(median(cell_values)),
        }
        for cell, cell_values in sorted(by_cell.items())
    }
    overall_median = float(median(values))
    cell_medians = [float(summary["median"]) for summary in cell_summaries.values()]
    if overall_median > 0 and all(value > 0 for value in cell_medians):
        status = "pass"
        pattern = "positive_synergy_replicated"
    elif overall_median < 0 and all(value < 0 for value in cell_medians):
        status = "inverse"
        pattern = "negative_redundancy_pattern"
    else:
        status = "mixed"
        pattern = "mixed_signs"
    return {
        "status": status,
        "pattern": pattern,
        "primary": {
            "estimand": "D_J = high-overlap minus low-overlap macro-KL nonadditivity",
            "count": len(values),
            "mean": float(mean(values)),
            "median": overall_median,
            "positive_count": int(sum(value > 0 for value in values)),
            "negative_count": int(sum(value < 0 for value in values)),
            "cells": cell_summaries,
        },
        "ce_sensitivity": {
            "mean": float(mean(ce_values)),
            "median": float(median(ce_values)),
            "positive_count": int(sum(value > 0 for value in ce_values)),
            "negative_count": int(sum(value < 0 for value in ce_values)),
        },
        "repair_reopened": False,
    }


def build_actionability_replication_report(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    gate = _validate_config(config)
    if tuple(sorted(config.get("replication_cells", ()))) != tuple(
        sorted(REPLICATION_CELLS)
    ):
        raise ValueError("replication_cells differs from the frozen protocol")
    raw_report_path = Path(config["raw_report"]).resolve()
    with raw_report_path.open("r", encoding="utf-8") as stream:
        raw_report = json.load(stream)
    _validate_raw_report_safety(raw_report)
    project_root = raw_report_path.parent.parent
    observed_cells = {
        f"{dataset}/{backbone}"
        for dataset, dataset_result in raw_report.get("datasets", {}).items()
        for backbone in dataset_result.get("backbones", {})
    }
    if observed_cells != set(EXPECTED_CELLS):
        raise ValueError("Raw report differs from the frozen 3x3 expansion")
    validity_path = _resolve_path(config["validity_report"], project_root)
    with validity_path.open("r", encoding="utf-8") as stream:
        validity = json.load(stream)
    decision = validity.get("replication_decision", {})
    if decision.get("status") != "pass" or not decision.get(
        "stage_b_joint_ablation_authorized", False
    ):
        raise RuntimeError("Stage B requires a passing CiteSeer effect-field replication")
    if validity.get("protocol", {}).get("test_labels_used") is not False:
        raise RuntimeError("Validity expansion is not test-blind")
    protocol_path = _resolve_path(config["protocol_document"], project_root)
    protocol_sha256 = _file_sha256(protocol_path)
    cache_root = _resolve_path(config["cache_root"], project_root)

    records = []
    for dataset, dataset_result in sorted(raw_report["datasets"].items()):
        for backbone, backbone_result in sorted(dataset_result["backbones"].items()):
            cell = f"{dataset}/{backbone}"
            if cell not in REPLICATION_CELLS:
                continue
            runs = sorted(
                backbone_result["runs"],
                key=lambda value: (int(value["seed"]), str(value["checkpoint_model_sha256"])),
            )
            for position, embedded_run in enumerate(runs, start=1):
                if progress:
                    print(f"[{cell}] checkpoint {position}/{len(runs)}", flush=True)
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
                records.append(
                    _process_checkpoint(
                        raw,
                        config=config,
                        gate=gate,
                        project_root=project_root,
                        protocol_sha256=protocol_sha256,
                        cache_path=cache_path,
                        resume=resume,
                        progress=progress,
                    )
                )
    result = evaluate_actionability_replication(records)
    return {
        "protocol": {
            "name": "CiteSeer fixed joint-ablation effect-field replication",
            "version": ACTIONABILITY_REPLICATION_VERSION,
            "processing_protocol_version": ACTIONABILITY_PROTOCOL_VERSION,
            "status": "development replication on previously inspected data",
            "development_only": True,
            "fresh_holdouts_used": False,
            "confirmatory_artifacts_used": False,
            "test_labels_used": False,
            "test_performance_used": False,
            "repair_reopened": False,
            "raw_report": str(raw_report_path),
            "raw_report_sha256": _file_sha256(raw_report_path),
            "validity_report": str(validity_path),
            "validity_report_sha256": _file_sha256(validity_path),
            "protocol_document": str(protocol_path),
            "protocol_document_sha256": protocol_sha256,
            "config_sha256": canonical_payload_sha256(config),
            "code_sha256": source_code_sha256(ACTIONABILITY_REPLICATION_CODE_FILES),
            "statistical_unit": "checkpoint",
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
        "checkpoint_count": len(records),
        "checkpoint_records": records,
        "actionability_replication_decision": result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/effect_field_actionability_replication.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_actionability_replication_report(
        config,
        resume=args.resume,
        progress=not args.quiet,
    )
    output = args.output or Path(config["output"])
    _write_json_atomic(output, report)
    print(f"Wrote effect-field actionability replication to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
