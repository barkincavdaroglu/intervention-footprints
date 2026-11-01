"""Prospective fixed joint-ablation replication on final holdout GNNs."""

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
    _validate_config as _validate_processing_config,
)
from gnn_repair.final_holdout_effect_field import GNN_CELLS, _validate_raw_safety
from gnn_repair.final_holdout_raw import EXPECTED_CELLS
from gnn_repair.protocol import canonical_payload_sha256
from gnn_repair.topology_gate1_refined import (
    _file_sha256,
    _load_raw_checkpoint,
    _resolve_path,
    _write_json_atomic,
)


FINAL_HOLDOUT_ACTIONABILITY_VERSION = 1
FINAL_HOLDOUT_ACTIONABILITY_CODE_FILES = (
    "final_holdout_actionability.py",
    "final_holdout_effect_field.py",
    "final_holdout_raw.py",
    *ACTIONABILITY_CODE_FILES,
)


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("fresh_holdout") is not True:
        raise ValueError("Final actionability requires fresh_holdout: true")
    return _validate_processing_config({**dict(config), "development_only": True})


def evaluate_holdout_actionability(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if {str(record["cell"]) for record in records} != set(GNN_CELLS):
        raise ValueError("Final actionability lacks the four frozen GNN cells")
    incomplete = [
        str(record["checkpoint_id"])
        for record in records
        if not record["matching"]["evaluable"] or record["contrast"] is None
    ]
    if incomplete:
        return {
            "status": "not_evaluable",
            "incomplete_checkpoint_ids": incomplete,
            "repair_authorized": False,
        }
    values = [float(record["contrast"]["d_j_kl"]) for record in records]
    ce_values = [float(record["contrast"]["d_j_ce_sensitivity"]) for record in records]
    by_cell: dict[str, list[float]] = {}
    for record, value in zip(records, values, strict=True):
        by_cell.setdefault(str(record["cell"]), []).append(value)
    cells = {
        cell: {
            "count": len(cell_values),
            "mean": float(mean(cell_values)),
            "median": float(median(cell_values)),
        }
        for cell, cell_values in sorted(by_cell.items())
    }
    positive_cells = [cell for cell, item in cells.items() if item["median"] > 0]
    passed = float(median(values)) > 0 and len(positive_cells) >= 3
    return {
        "status": "pass" if passed else "fail",
        "primary": {
            "estimand": "D_J = high-overlap minus low-overlap macro-KL nonadditivity",
            "count": len(values),
            "mean": float(mean(values)),
            "median": float(median(values)),
            "positive_count": int(sum(value > 0 for value in values)),
            "negative_count": int(sum(value < 0 for value in values)),
            "cells": cells,
            "positive_cells": positive_cells,
            "positive_cell_count": len(positive_cells),
        },
        "ce_sensitivity": {
            "mean": float(mean(ce_values)),
            "median": float(median(ce_values)),
            "positive_count": int(sum(value > 0 for value in ce_values)),
            "negative_count": int(sum(value < 0 for value in ce_values)),
        },
        "repair_authorized": False,
    }


def build_final_holdout_actionability_report(
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
    observed = {
        f"{dataset}/{backbone}"
        for dataset, dataset_result in raw_report["datasets"].items()
        for backbone in dataset_result["backbones"]
    }
    if observed != set(EXPECTED_CELLS):
        raise ValueError("Final raw report differs from the frozen six cells")
    validity_path = _resolve_path(config["validity_report"], project_root)
    with validity_path.open("r", encoding="utf-8") as stream:
        validity = json.load(stream)
    decision = validity.get("effect_field_holdout_decision", {})
    if decision.get("status") != "pass" or not decision.get("stage_b_authorized", False):
        raise RuntimeError("Final actionability requires a passing Stage A")
    validity_protocol = validity.get("protocol", {})
    if validity_protocol.get("test_labels_used") is not False or validity_protocol.get("test_performance_used") is not False:
        raise RuntimeError("Final validity report is not test-blind")
    protocol_path = _resolve_path(config["protocol_document"], project_root)
    protocol_sha256 = _file_sha256(protocol_path)
    cache_root = _resolve_path(config["cache_root"], project_root)
    records = []
    for dataset, dataset_result in sorted(raw_report["datasets"].items()):
        for backbone, backbone_result in sorted(dataset_result["backbones"].items()):
            cell = f"{dataset}/{backbone}"
            if cell not in GNN_CELLS:
                continue
            runs = sorted(backbone_result["runs"], key=lambda item: (int(item["seed"]), str(item["checkpoint_model_sha256"])))
            for position, embedded in enumerate(runs, start=1):
                if progress:
                    print(f"[{cell}] checkpoint {position}/{len(runs)}", flush=True)
                raw = _load_raw_checkpoint(
                    dataset=dataset,
                    backbone=backbone,
                    embedded_run=embedded,
                    project_root=project_root,
                )
                cache_path = cache_root / dataset / backbone / f"seed_{int(raw.run['seed'])}_{raw.checkpoint_id[:12]}.json"
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
    result = evaluate_holdout_actionability(records)
    return {
        "protocol": {
            "name": "Final prospective fixed joint-ablation replication",
            "version": FINAL_HOLDOUT_ACTIONABILITY_VERSION,
            "processing_protocol_version": ACTIONABILITY_PROTOCOL_VERSION,
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
            "validity_report": str(validity_path),
            "validity_report_sha256": _file_sha256(validity_path),
            "protocol_document": str(protocol_path),
            "protocol_document_sha256": protocol_sha256,
            "config_sha256": canonical_payload_sha256(config),
            "code_sha256": source_code_sha256(FINAL_HOLDOUT_ACTIONABILITY_CODE_FILES),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
        "checkpoint_count": len(records),
        "checkpoint_records": records,
        "actionability_holdout_decision": result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final_holdout_actionability.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_final_holdout_actionability_report(config, resume=args.resume, progress=not args.quiet)
    output = args.output or Path(config["output"])
    _write_json_atomic(output, report)
    print(f"Wrote final holdout actionability report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
