"""Development gate for ablation effect fields as functional fingerprints.

This post-topology pivot compares node-indexed intervention fields with a fixed
magnitude-plus-node-role baseline.  It consumes the already verified validation-
only Gate 1 raw cache, uses checkpoint-level inference, and rejects fresh,
confirmatory, or test-derived inputs.
"""

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
from gnn_repair.gate1_features import Gate1FeatureSet, extract_gate1_features
from gnn_repair.gate1_nulls import paired_incremental_permutation_test
from gnn_repair.gate1_prediction import (
    Gate1PredictionCheckpoint,
    run_within_stratum_loco,
)
from gnn_repair.protocol import canonical_payload_sha256
from gnn_repair.structural_representations import normalized_feature_kernel
from gnn_repair.topology_gate1_refined import (
    DEFAULT_EXPECTED_CELLS,
    FROZEN_DIFFUSION_TIMES,
    FROZEN_QUANTILES,
    FROZEN_WL_DEPTHS,
    RawCheckpoint,
    _array_schema,
    _effect_kernel,
    _file_sha256,
    _load_npz,
    _load_raw_checkpoint,
    _resolve_path,
    _row_l1,
    _same_floats,
    _stable_seed,
    _validate_raw_report_safety,
    _write_json_atomic,
    _write_npz_atomic,
    derive_loco_flags,
)


EFFECT_FIELD_PROTOCOL_VERSION = 1
FROZEN_PERMUTATIONS = 1000
PRIMARY_CANDIDATE = "mask_field"
CANDIDATE_NAMES = ("raw_field", "mask_field", "continuous_plus_mask")
EFFECT_FIELD_CODE_FILES = (
    "effect_field_gate.py",
    "topology_gate1_refined.py",
    "gate1_features.py",
    "gate1_nulls.py",
    "gate1_prediction.py",
    "structural.py",
    "structural_representations.py",
    "topology_gate1.py",
    "checkpoints.py",
    "data.py",
    "model.py",
    "protocol.py",
    "reproducibility.py",
)


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("development_only") is not True:
        raise ValueError("Effect-field gate requires development_only: true")
    gate = dict(config.get("effect_field", {}))
    if not _same_floats(gate.get("quantiles", ()), FROZEN_QUANTILES):
        raise ValueError(f"quantiles must be exactly {FROZEN_QUANTILES}")
    if gate.get("permutations") != FROZEN_PERMUTATIONS:
        raise ValueError(f"permutations must be exactly {FROZEN_PERMUTATIONS}")
    if gate.get("primary_candidate") != PRIMARY_CANDIDATE:
        raise ValueError(f"primary_candidate must be {PRIMARY_CANDIDATE!r}")
    if tuple(gate.get("candidate_names", ())) != CANDIDATE_NAMES:
        raise ValueError(f"candidate_names must be exactly {CANDIDATE_NAMES}")
    ridge = float(gate.get("ridge_alpha", 1.0))
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge_alpha must be finite and non-negative")
    return gate


def _feature_options(gate: Mapping[str, Any]) -> dict[str, Any]:
    # WL and diffusion are computed by the shared extractor but never enter this
    # gate.  Keeping their dimensions small makes the post-hoc pivot inexpensive.
    return {
        "quantiles": FROZEN_QUANTILES,
        "wl_depths": (0,),
        "diffusion_times": (1.0,),
        "minimum_impact": float(gate.get("minimum_impact", 1e-12)),
        "active_threshold": float(gate.get("active_threshold", 1e-12)),
        "wl_hash_dimension": 1,
        "node_projection_dimension": int(gate.get("node_projection_dimension", 256)),
    }


def _standardized_cosine(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)[indices]
    if matrix.ndim != 2:
        raise ValueError("values must be a feature matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("values contain non-finite entries")
    if not matrix.shape[0]:
        return np.empty((0, 0), dtype=np.float64)
    location = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    standardized = (matrix - location) / np.where(scale > 0, scale, 1.0)
    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    normalized = standardized / np.where(norms > 0, norms, 1.0)
    kernel = normalized @ normalized.T
    kernel = np.clip(0.5 * (kernel + kernel.T), -1.0, 1.0)
    if kernel.size:
        np.fill_diagonal(kernel, 1.0)
    return kernel


def _baseline_features(features: Gate1FeatureSet) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(features.controls["magnitude"].prediction_features),
            np.asarray(features.controls["uniform_parent_roles"].prediction_features),
        ],
        axis=1,
    )


def _candidate_prediction_features(
    features: Gate1FeatureSet,
) -> dict[str, np.ndarray]:
    raw = np.asarray(features.controls["raw_effect_field"].prediction_features)
    mask = np.asarray(features.controls["mask_jaccard"].prediction_features)
    return {
        "raw_field": raw,
        "mask_field": mask,
        "continuous_plus_mask": np.concatenate([raw, mask], axis=1),
    }


def _kernels(
    features: Gate1FeatureSet,
    subset: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    indices = features.controls["magnitude"].kernels[subset].channel_indices
    baseline = _standardized_cosine(_baseline_features(features), indices)
    raw = features.controls["raw_effect_field"].kernels[subset].kernel
    mask = features.controls["mask_jaccard"].kernels[subset].kernel
    combined = np.clip(0.5 * (raw + mask), -1.0, 1.0)
    if combined.size:
        np.fill_diagonal(combined, 1.0)
    return {"magnitude_plus_node_role": baseline}, {
        "raw_field": raw,
        "mask_field": mask,
        "continuous_plus_mask": combined,
    }


def _checkpoint_inference(
    raw: RawCheckpoint,
    features: Gate1FeatureSet,
) -> dict[str, Any]:
    outcomes = {
        "class_kl": np.asarray(raw.arrays["class_kl"], dtype=np.float64),
        "class_ce_delta": np.asarray(raw.arrays["class_ce_delta"], dtype=np.float64),
    }
    indices = {
        "all_channels": np.arange(features.active_mask.size, dtype=np.int64),
        "active_channels": np.flatnonzero(features.active_mask).astype(np.int64),
    }
    result: dict[str, Any] = {}
    for outcome_name, effects in outcomes.items():
        result[outcome_name] = {}
        for subset, channel_indices in indices.items():
            controls, candidates = _kernels(features, subset)
            result[outcome_name][subset] = paired_incremental_permutation_test(
                controls,
                candidates,
                _effect_kernel(effects, channel_indices),
                repetitions=FROZEN_PERMUTATIONS,
                random_seed=_stable_seed(
                    "effect-field-common-permutation",
                    raw.checkpoint_id,
                    outcome_name,
                    subset,
                ),
            )
    return result


def _prediction_arrays(
    raw: RawCheckpoint,
    features: Gate1FeatureSet,
) -> dict[str, np.ndarray]:
    candidates = _candidate_prediction_features(features)
    return {
        "active_channel_indices": np.flatnonzero(features.active_mask).astype(np.int64),
        "baseline_features": np.asarray(_baseline_features(features), dtype=np.float64),
        "class_effects": _row_l1(np.asarray(raw.arrays["class_kl"], dtype=np.float64)),
        **{
            f"candidate__{name}": np.asarray(values, dtype=np.float64)
            for name, values in candidates.items()
        },
    }


def _identity(
    raw: RawCheckpoint,
    config: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol_version": EFFECT_FIELD_PROTOCOL_VERSION,
        "code_sha256": source_code_sha256(EFFECT_FIELD_CODE_FILES),
        "protocol_document_sha256": protocol_sha256,
        "config_sha256": canonical_payload_sha256(config),
        "cell": raw.cell,
        "checkpoint_id": raw.checkpoint_id,
        "raw_manifest_sha256": raw.raw_manifest_sha256,
        "raw_sidecar_sha256": raw.raw_sidecar_sha256,
    }


def _process_checkpoint(
    raw: RawCheckpoint,
    *,
    config: Mapping[str, Any],
    gate: Mapping[str, Any],
    protocol_sha256: str,
    cache_path: Path,
    resume: bool,
    progress: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    identity = _identity(raw, config, protocol_sha256)
    sidecar_path = cache_path.with_suffix(".npz")
    if resume and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("identity") != identity:
            raise RuntimeError(f"Effect-field cache identity mismatch: {cache_path}")
        sidecar = cached.get("prediction_sidecar", {})
        if not sidecar_path.exists() or _file_sha256(sidecar_path) != sidecar.get("sha256"):
            raise RuntimeError(f"Effect-field sidecar hash mismatch: {sidecar_path}")
        arrays = _load_npz(sidecar_path)
        if _array_schema(arrays) != sidecar.get("arrays"):
            raise RuntimeError(f"Effect-field sidecar schema mismatch: {sidecar_path}")
        return cached["record"], arrays

    if progress:
        print(f"[{raw.cell}] {raw.checkpoint_id[:12]}: extracting effect fields", flush=True)
    features = extract_gate1_features(
        raw.adjacency,
        np.asarray(raw.arrays["node_impacts"]),
        np.asarray(raw.arrays["prediction_changes"]),
        np.asarray(raw.arrays["scope_mask"]),
        **_feature_options(gate),
    )
    if progress:
        print(f"[{raw.cell}] {raw.checkpoint_id[:12]}: running paired inference", flush=True)
    inference = _checkpoint_inference(raw, features)
    arrays = _prediction_arrays(raw, features)
    _write_npz_atomic(sidecar_path, arrays)
    sidecar = {
        "path": str(sidecar_path),
        "sha256": _file_sha256(sidecar_path),
        "format": "compressed_npz",
        "allow_pickle_required": False,
        "arrays": _array_schema(arrays),
    }
    record = {
        "cell": raw.cell,
        "dataset": raw.dataset,
        "backbone": raw.backbone,
        "checkpoint_id": raw.checkpoint_id,
        "seed": int(raw.run["seed"]),
        "width": int(features.active_mask.size),
        "active_channel_count": int(features.active_mask.sum()),
        "inference": inference,
        "test_labels_used": False,
        "test_performance_used": False,
    }
    _write_json_atomic(
        cache_path,
        {"identity": identity, "prediction_sidecar": sidecar, "record": record},
    )
    return record, arrays


def _prediction_checkpoint(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    subset: str,
) -> Gate1PredictionCheckpoint:
    if subset == "active_channels":
        indices = np.asarray(arrays["active_channel_indices"], dtype=np.int64)
    elif subset == "all_channels":
        indices = np.arange(np.asarray(arrays["class_effects"]).shape[0], dtype=np.int64)
    else:
        raise ValueError(f"Unknown subset: {subset}")
    return Gate1PredictionCheckpoint(
        cell=str(record["cell"]),
        checkpoint_sha256=str(record["checkpoint_id"]),
        control_features=np.asarray(arrays["baseline_features"])[indices],
        candidate_features={
            name: np.asarray(arrays[f"candidate__{name}"])[indices]
            for name in CANDIDATE_NAMES
        },
        class_effects=np.asarray(arrays["class_effects"])[indices],
    )


def _increment_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    outcome: str,
    subset: str,
    candidate: str,
) -> dict[str, Any]:
    details = []
    by_cell: dict[str, list[float]] = {}
    for record in records:
        item = record["inference"][outcome][subset]["candidates"][candidate]
        value = float(item["incremental_alignment"])
        details.append(
            {
                "cell": str(record["cell"]),
                "checkpoint_id": str(record["checkpoint_id"]),
                "incremental_alignment": value,
                "candidate_alignment": float(item["observed_alignment"]),
                "baseline_alignment": float(item["best_control_alignment"]),
                "p_increment_greater": item["p_increment_greater"],
            }
        )
        by_cell.setdefault(str(record["cell"]), []).append(value)
    values = [detail["incremental_alignment"] for detail in details]
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "positive_count": int(sum(value > 0 for value in values)),
        "cells": {
            cell: {
                "count": len(cell_values),
                "mean": float(mean(cell_values)),
                "median": float(median(cell_values)),
            }
            for cell, cell_values in sorted(by_cell.items())
        },
        "checkpoints": details,
    }


def evaluate_effect_field_gate(
    records: Sequence[Mapping[str, Any]],
    loco_flags: Mapping[str, Any],
) -> dict[str, Any]:
    primary_active = _increment_summary(
        records,
        outcome="class_kl",
        subset="active_channels",
        candidate=PRIMARY_CANDIDATE,
    )
    primary_all = _increment_summary(
        records,
        outcome="class_kl",
        subset="all_channels",
        candidate=PRIMARY_CANDIDATE,
    )
    positive_cells = [
        cell
        for cell, summary in primary_active["cells"].items()
        if float(summary["median"]) > 0
    ]
    flag = loco_flags[PRIMARY_CANDIDATE]
    criteria = {
        "active_incremental_median": {
            "pass": primary_active["median"] > 0,
            "value": primary_active["median"],
            "rule": "active-channel checkpoint median > 0",
        },
        "positive_in_three_cells": {
            "pass": len(positive_cells) >= 3,
            "positive_cells": positive_cells,
            "count": len(positive_cells),
            "rule": "active-channel cell median > 0 in at least three of four cells",
        },
        "all_channel_incremental_median": {
            "pass": primary_all["median"] > 0,
            "value": primary_all["median"],
            "rule": "all-channel checkpoint median > 0",
        },
        "loco_prediction": {
            "pass": bool(
                flag.non_worse_than_best_control
                and flag.improves_predeclared_metric
            ),
            "non_worse_than_baseline": flag.non_worse_than_best_control,
            "improves_predeclared_metric": flag.improves_predeclared_metric,
        },
    }
    passed = all(item["pass"] for item in criteria.values())
    return {
        "status": "pass" if passed else "fail",
        "selected_candidate": PRIMARY_CANDIDATE if passed else None,
        "criteria": criteria,
        "primary_active": primary_active,
        "primary_all_channels": primary_all,
        "next_step": (
            "freeze and run matched effect-field actionability interactions"
            if passed
            else "stop repair pivot; retain representation-only development finding"
        ),
    }


def build_effect_field_report(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    gate = _validate_config(config)
    raw_report_path = Path(config.get("raw_report", "artifacts/topology_gate1_raw_cache.json")).resolve()
    with raw_report_path.open("r", encoding="utf-8") as stream:
        raw_report = json.load(stream)
    _validate_raw_report_safety(raw_report)
    project_root = raw_report_path.parent.parent
    protocol_path = _resolve_path(
        config.get("protocol_document", "docs/effect_field_pivot_protocol.md"),
        project_root,
    )
    protocol_sha256 = _file_sha256(protocol_path)
    cache_root = _resolve_path(
        config.get("cache_root", "artifacts/effect_field_gate_runs"),
        project_root,
    )
    observed_cells = {
        f"{dataset}/{backbone}"
        for dataset, dataset_result in raw_report.get("datasets", {}).items()
        for backbone in dataset_result.get("backbones", {})
    }
    if observed_cells != set(DEFAULT_EXPECTED_CELLS):
        raise ValueError("Effect-field gate requires exactly the four frozen cells")

    records: list[dict[str, Any]] = []
    arrays_by_checkpoint: list[dict[str, np.ndarray]] = []
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
                record, arrays = _process_checkpoint(
                    raw,
                    config=config,
                    gate=gate,
                    protocol_sha256=protocol_sha256,
                    cache_path=cache_path,
                    resume=resume,
                    progress=progress,
                )
                records.append(record)
                arrays_by_checkpoint.append(arrays)

    loco: dict[str, Any] = {}
    loco_flags: dict[str, Any] = {}
    loco_audit: dict[str, Any] = {}
    for subset in ("active_channels", "all_channels"):
        checkpoints = [
            _prediction_checkpoint(record, arrays, subset=subset)
            for record, arrays in zip(records, arrays_by_checkpoint, strict=True)
        ]
        result = run_within_stratum_loco(
            checkpoints,
            ridge_alpha=float(gate.get("ridge_alpha", 1.0)),
        )
        flags, audit = derive_loco_flags(result, CANDIDATE_NAMES)
        loco[subset] = result
        loco_audit[subset] = audit
        if subset == "active_channels":
            loco_flags = flags

    decision = evaluate_effect_field_gate(records, loco_flags)
    summaries = {
        outcome: {
            subset: {
                candidate: _increment_summary(
                    records,
                    outcome=outcome,
                    subset=subset,
                    candidate=candidate,
                )
                for candidate in CANDIDATE_NAMES
            }
            for subset in ("active_channels", "all_channels")
        }
        for outcome in ("class_kl", "class_ce_delta")
    }
    return {
        "protocol": {
            "name": "Ablation effect fields as functional fingerprints",
            "version": EFFECT_FIELD_PROTOCOL_VERSION,
            "status": "post-topology exploratory development gate",
            "development_only": True,
            "fresh_holdouts_used": False,
            "confirmatory_artifacts_used": False,
            "test_labels_used": False,
            "test_performance_used": False,
            "primary_candidate": PRIMARY_CANDIDATE,
            "candidate_names": list(CANDIDATE_NAMES),
            "raw_report": str(raw_report_path),
            "raw_report_sha256": _file_sha256(raw_report_path),
            "protocol_document": str(protocol_path),
            "protocol_document_sha256": protocol_sha256,
            "config_sha256": canonical_payload_sha256(config),
            "code_sha256": source_code_sha256(EFFECT_FIELD_CODE_FILES),
            "statistical_unit": "checkpoint, never channel pairs",
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
        "alignment_summaries": summaries,
        "leave_one_checkpoint_out_prediction": loco,
        "loco_decision_audit": loco_audit,
        "effect_field_decision": decision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/effect_field_pivot.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_effect_field_report(
        config,
        resume=args.resume,
        progress=not args.quiet,
    )
    output = args.output or Path(config.get("output", "artifacts/effect_field_gate.json"))
    _write_json_atomic(output, report)
    print(f"Wrote effect-field gate report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

