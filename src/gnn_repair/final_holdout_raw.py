"""Collect blinded raw intervention effects for the final prospective holdout."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy
import sklearn
import torch
import yaml

from gnn_repair.checkpoints import load_trained_run, model_state_sha256, source_code_sha256
from gnn_repair.protocol import canonical_payload_sha256
from gnn_repair.topology_gate1 import (
    GATE1_CODE_FILES,
    GATE1_PROTOCOL_VERSION,
    _analyze_with_cache,
    _cell_summary,
    _run_paths,
    _write_json_atomic,
)
from gnn_repair.topology_gate1_refined import _file_sha256, _resolve_path


FINAL_HOLDOUT_RAW_VERSION = 1
EXPECTED_CELLS = (
    "chameleon_filtered/matched_mlp",
    "chameleon_filtered/residual_gcn",
    "chameleon_filtered/thesis_gcn",
    "cora_ml/matched_mlp",
    "cora_ml/residual_gcn",
    "cora_ml/thesis_gcn",
)
FINAL_HOLDOUT_RAW_CODE_FILES = ("final_holdout_raw.py", *GATE1_CODE_FILES)
EXPECTED_DATASET_SHA256 = {
    "cora_ml": "bff04cbfaf903581de18b28dd112b92b2e12c05dfcb1cfc5f6875a5a280d9872",
    "chameleon_filtered": "bf46f07e1fb5249280447e5fe3100f3e82fc4b93ad1e13ffcfdec924b6ac0bb5",
}


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("fresh_holdout") is not True:
        raise ValueError("Final raw collection requires fresh_holdout: true")
    datasets = config.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("Final raw config requires datasets")
    observed = {
        f"{dataset}/{backbone}"
        for dataset, dataset_config in datasets.items()
        for backbone in dataset_config.get("backbones", {})
    }
    if observed != set(EXPECTED_CELLS):
        raise ValueError("Final raw config differs from the frozen six cells")
    gate = dict(config.get("gate1", {}))
    frozen = {
        "impact_scope": "train_validation",
        "impact_quantiles": [0.90, 0.95, 0.99],
        "neuron_permutations": 0,
        "graph_rewires": 0,
        "rewiring_views": [],
    }
    for name, expected in frozen.items():
        if gate.get(name) != expected:
            raise ValueError(f"{name} differs from the frozen raw protocol")
    return gate


def _validate_blinded_run(result: Mapping[str, Any], cell: str) -> None:
    if result.get("test_status") != "blinded":
        raise RuntimeError(f"Holdout run is not test-blinded: {cell}")
    if tuple(result.get("evaluation_splits", ())) != ("validation",):
        raise RuntimeError(f"Holdout run exposes a non-validation split: {cell}")
    if "test" in result.get("metrics", {}):
        raise RuntimeError(f"Holdout run contains test metrics: {cell}")
    if result.get("data", {}).get("test_label_status") != "blinded":
        raise RuntimeError(f"Holdout run did not redact test labels: {cell}")
    if result.get("study", {}).get("id") != "final_interpretation_holdout_v1":
        raise RuntimeError(f"Holdout run has the wrong study id: {cell}")
    dataset = cell.split("/", maxsplit=1)[0]
    if result.get("data", {}).get("sha256") != EXPECTED_DATASET_SHA256[dataset]:
        raise RuntimeError(f"Holdout run used the wrong dataset archive: {cell}")


def build_final_holdout_raw_report(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    gate = _validate_config(config)
    project_root = Path.cwd().resolve()
    cache_root = _resolve_path(config["cache_root"], project_root)
    protocol_path = _resolve_path(config["protocol_document"], project_root)
    report: dict[str, Any] = {
        "protocol": {
            "name": "Final prospective holdout raw effect collection",
            "version": FINAL_HOLDOUT_RAW_VERSION,
            "processing_protocol_version": GATE1_PROTOCOL_VERSION,
            "development_only": False,
            "prospective_holdout": True,
            "fresh_holdouts_used": True,
            "confirmatory_artifacts_used": False,
            "test_labels_used": False,
            "test_performance_used": False,
            "test_status": "blinded",
            "protocol_document": str(protocol_path),
            "protocol_document_sha256": _file_sha256(protocol_path),
            "config_sha256": canonical_payload_sha256(config),
            "code_sha256": source_code_sha256(FINAL_HOLDOUT_RAW_CODE_FILES),
            "gate1": gate,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
            },
        },
        "datasets": {},
    }
    sidecars = []
    for dataset, dataset_config in config["datasets"].items():
        dataset_result: dict[str, Any] = {"backbones": {}}
        for backbone, backbone_config in dataset_config["backbones"].items():
            root = _resolve_path(backbone_config["root"], project_root)
            layer = int(backbone_config["layer"])
            paths = _run_paths(root)
            records = []
            cell = f"{dataset}/{backbone}"
            for position, run_path in enumerate(paths, start=1):
                if progress:
                    print(f"[{cell}] raw checkpoint {position}/{len(paths)}", flush=True)
                loaded = load_trained_run(run_path, allow_download=False)
                _validate_blinded_run(loaded.result, cell)
                checkpoint_id = model_state_sha256(loaded.checkpoint)
                cache_path = (
                    cache_root
                    / dataset
                    / backbone
                    / f"seed_{int(loaded.result['seed'])}_{checkpoint_id[:12]}.json"
                )
                record, _ = _analyze_with_cache(
                    loaded,
                    dataset_name=dataset,
                    backbone_name=backbone,
                    layer=layer,
                    gate_config=gate,
                    cache_path=cache_path,
                    resume=resume,
                )
                records.append(record)
                sidecars.append(record["raw_sidecar"])
            dataset_result["backbones"][backbone] = {
                "layer": layer,
                "runs": records,
                "summary": _cell_summary(records),
            }
        report["datasets"][dataset] = dataset_result
    report["raw_sidecars"] = sidecars
    report["checkpoint_count"] = sum(
        len(backbone["runs"])
        for dataset in report["datasets"].values()
        for backbone in dataset["backbones"].values()
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final_holdout_raw.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_final_holdout_raw_report(
        config,
        resume=args.resume,
        progress=not args.quiet,
    )
    output = args.output or Path(config["output"])
    _write_json_atomic(output, report)
    print(f"Wrote final holdout raw report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
