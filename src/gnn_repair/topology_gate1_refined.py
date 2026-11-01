"""Official cache-consuming evaluator for the frozen topology Gate 1 protocol.

The expensive model interventions are collected by :mod:`topology_gate1`.  This
module treats those validation-only fields as immutable inputs, verifies their
provenance, constructs the separated candidate/control blocks, runs checkpoint-
level inference and graph nulls, performs leakage-safe within-stratum prediction,
and delegates the final pass/fail decision to :mod:`gate1_decision`.

No test labels, test performance, fresh holdouts, or confirmatory artifacts are
accepted by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
import torch
import yaml
from scipy.stats import rankdata

from gnn_repair.checkpoints import (
    load_trained_run,
    model_state_sha256,
    source_code_sha256,
    split_indices_sha256,
)
from gnn_repair.gate1_decision import (
    DEFAULT_EXPECTED_CELLS,
    DIFFUSION,
    NORMALIZED_WL,
    CandidateSpec,
    CandidateVariant,
    CheckpointEvidence,
    LocoFlags,
    RewireAlignment,
    evaluate_gate1_decision,
)
from gnn_repair.gate1_features import Gate1FeatureSet, extract_gate1_features
from gnn_repair.gate1_nulls import (
    complete_degree_preserving_rewires,
    paired_incremental_permutation_test,
)
from gnn_repair.gate1_prediction import (
    Gate1PredictionCheckpoint,
    run_within_stratum_loco,
)
from gnn_repair.protocol import canonical_payload_sha256
from gnn_repair.structural import (
    canonical_edge_sha256,
    canonical_undirected_adjacency,
)
from gnn_repair.structural_representations import normalized_feature_kernel


REFINED_PROTOCOL_VERSION = 1
FROZEN_QUANTILES = (0.90, 0.95, 0.99)
FROZEN_WL_DEPTHS = (0, 1, 2, 3, 4)
FROZEN_DIFFUSION_TIMES = (0.01, 0.1, 1.0, 10.0)
FROZEN_PERMUTATIONS = 1000
FROZEN_REWIRES = 2
FROZEN_SWAPS_PER_EDGE = 1.0
CONTROL_NAME_MAP = {
    "magnitude": "magnitude",
    "raw_effect_field": "raw_effect_field",
    "mask_jaccard": "mask_jaccard",
    "uniform_parent_roles": "uniform_node_role",
}
REFINED_CODE_FILES = (
    "topology_gate1_refined.py",
    "gate1_features.py",
    "gate1_nulls.py",
    "gate1_prediction.py",
    "gate1_decision.py",
    "structural.py",
    "structural_representations.py",
    "topology_gate1.py",
    "checkpoints.py",
    "data.py",
    "model.py",
    "protocol.py",
    "reproducibility.py",
)


@dataclass(frozen=True)
class RawCheckpoint:
    """Verified cached validation fields and their reconstructed parent graph."""

    cell: str
    dataset: str
    backbone: str
    checkpoint_id: str
    run: Mapping[str, Any]
    raw_manifest_path: Path
    raw_manifest_sha256: str
    raw_sidecar_path: Path
    raw_sidecar_sha256: str
    adjacency: sp.csr_matrix
    arrays: Mapping[str, np.ndarray]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    material = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (Path.cwd() / path, project_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def _same_floats(values: Sequence[Any], expected: Sequence[float]) -> bool:
    try:
        observed = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return False
    return len(observed) == len(expected) and all(
        math.isclose(value, target, abs_tol=1e-12, rel_tol=0.0)
        for value, target in zip(observed, expected, strict=True)
    )


def frozen_candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the predeclared selection candidates and their separate views."""

    specs: list[CandidateSpec] = []
    for prefix, depths in (
        ("wl_constant", range(1, 5)),
        ("wl_induced_degree", range(0, 5)),
    ):
        for depth in depths:
            variants = tuple(
                CandidateVariant(
                    name=f"{prefix}_q{int(round(quantile * 100))}_h{depth}",
                    quantile=quantile,
                )
                for quantile in FROZEN_QUANTILES
            )
            specs.append(
                CandidateSpec(
                    candidate_id=f"{prefix}_h{depth}",
                    family=NORMALIZED_WL,
                    primary_variant=f"{prefix}_q95_h{depth}",
                    variants=variants,
                    depth=depth,
                )
            )
    diffusion_variants = tuple(
        CandidateVariant(
            name={0.01: "diffusion_t0p01", 0.1: "diffusion_t0p1", 1.0: "diffusion_t1", 10.0: "diffusion_t10"}[time],
            diffusion_scale=time,
        )
        for time in FROZEN_DIFFUSION_TIMES
    )
    specs.append(
        CandidateSpec(
            candidate_id="diffusion",
            family=DIFFUSION,
            primary_variant="diffusion_t1",
            variants=diffusion_variants,
            depth=None,
        )
    )
    return tuple(specs)


def _validate_frozen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("development_only") is not True:
        raise ValueError("Refined Gate 1 requires development_only: true")
    gate = dict(config.get("gate1", {}))
    if not _same_floats(gate.get("quantiles", ()), FROZEN_QUANTILES):
        raise ValueError(f"quantiles must be exactly {FROZEN_QUANTILES}")
    if tuple(gate.get("wl_depths", ())) != FROZEN_WL_DEPTHS:
        raise ValueError(f"wl_depths must be exactly {FROZEN_WL_DEPTHS}")
    if not _same_floats(gate.get("diffusion_times", ()), FROZEN_DIFFUSION_TIMES):
        raise ValueError(f"diffusion_times must be exactly {FROZEN_DIFFUSION_TIMES}")
    fixed = {
        "permutations": FROZEN_PERMUTATIONS,
        "rewires": FROZEN_REWIRES,
    }
    for name, expected in fixed.items():
        value = gate.get(name)
        if isinstance(value, bool) or value != expected:
            raise ValueError(f"{name} must be exactly {expected}")
    swaps = float(gate.get("swaps_per_edge", math.nan))
    if not math.isclose(swaps, FROZEN_SWAPS_PER_EDGE, abs_tol=1e-12):
        raise ValueError("swaps_per_edge must be exactly 1.0")
    if float(gate.get("minimum_impact", 1e-12)) < 0:
        raise ValueError("minimum_impact must be non-negative")
    if float(gate.get("active_threshold", 1e-12)) < 0:
        raise ValueError("active_threshold must be non-negative")
    if int(gate.get("wl_hash_dimension", 512)) < 1:
        raise ValueError("wl_hash_dimension must be positive")
    if int(gate.get("node_projection_dimension", 256)) < 1:
        raise ValueError("node_projection_dimension must be positive")
    if int(gate.get("max_swap_attempt_multiplier", 20)) < 1:
        raise ValueError("max_swap_attempt_multiplier must be positive")
    if int(gate.get("rewire_retry_multiplier", 5)) < 1:
        raise ValueError("rewire_retry_multiplier must be positive")
    ridge = float(gate.get("ridge_alpha", 1.0))
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge_alpha must be finite and non-negative")
    return gate


def _validate_raw_report_safety(report: Mapping[str, Any]) -> None:
    protocol = report.get("protocol", {})
    forbidden_true = (
        "fresh_holdouts_used",
        "confirmatory_artifacts_used",
        "test_labels_used",
        "test_performance_used",
    )
    invalid = [name for name in forbidden_true if protocol.get(name) is not False]
    if invalid:
        raise ValueError(
            "Raw Gate 1 report is not a validation-only development artifact: "
            + ", ".join(invalid)
        )
    if protocol.get("development_only") is not True:
        raise ValueError("Raw Gate 1 report must be explicitly development-only")
    if any("confirmatory" in str(item).lower() for item in report.get("raw_sidecars", ())):
        raise ValueError("Raw Gate 1 report references a confirmatory artifact")


def _required_raw_arrays() -> tuple[str, ...]:
    return (
        "node_impacts",
        "prediction_changes",
        "scope_mask",
        "class_kl",
        "class_ce_delta",
    )


def _load_raw_checkpoint(
    *,
    dataset: str,
    backbone: str,
    embedded_run: Mapping[str, Any],
    project_root: Path,
) -> RawCheckpoint:
    sidecar_description = embedded_run.get("raw_sidecar", {})
    sidecar_path = _resolve_path(sidecar_description["path"], project_root)
    manifest_path = sidecar_path.with_suffix(".json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Raw checkpoint manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    run = manifest.get("run", {})
    if dict(run) != dict(embedded_run):
        raise RuntimeError(f"Embedded raw record differs from {manifest_path}")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Raw checkpoint sidecar is missing: {sidecar_path}")
    observed_hash = _file_sha256(sidecar_path)
    if observed_hash != manifest.get("sidecar", {}).get("sha256"):
        raise RuntimeError(f"Raw checkpoint sidecar hash mismatch: {sidecar_path}")
    arrays = _load_npz(sidecar_path)
    if _array_schema(arrays) != manifest.get("sidecar", {}).get("arrays"):
        raise RuntimeError(f"Raw checkpoint sidecar schema mismatch: {sidecar_path}")
    missing = sorted(set(_required_raw_arrays()) - set(arrays))
    if missing:
        raise RuntimeError(f"Raw checkpoint sidecar lacks arrays {missing}: {sidecar_path}")
    if run.get("test_labels_used") is not False or run.get("test_performance_used") is not False:
        raise ValueError(f"Raw checkpoint used test information: {manifest_path}")
    if "confirmatory" in str(run.get("run_path", "")).lower():
        raise ValueError(f"Confirmatory run path is forbidden: {manifest_path}")

    run_path = _resolve_path(run["run_path"], project_root)
    loaded = load_trained_run(run_path, allow_download=False)
    checkpoint_id = model_state_sha256(loaded.checkpoint)
    if checkpoint_id != run.get("checkpoint_model_sha256"):
        raise RuntimeError(f"Checkpoint state hash mismatch: {run_path}")
    if str(loaded.data.metadata.get("sha256")) != str(run.get("dataset_sha256")):
        raise RuntimeError(f"Dataset hash mismatch: {run_path}")
    if split_indices_sha256(loaded.data) != run.get("split_indices_sha256"):
        raise RuntimeError(f"Split hash mismatch: {run_path}")
    adjacency = canonical_undirected_adjacency(
        loaded.data.edge_index,
        int(loaded.data.x.shape[0]),
    )
    if canonical_edge_sha256(adjacency) != run.get("canonical_edge_sha256"):
        raise RuntimeError(f"Canonical parent-graph hash mismatch: {run_path}")
    width, nodes = np.asarray(arrays["node_impacts"]).shape
    if width != int(run.get("width", -1)) or nodes != adjacency.shape[0]:
        raise RuntimeError(f"Raw impact dimensions do not match run record: {run_path}")
    if np.asarray(arrays["class_kl"]).shape[0] != width:
        raise RuntimeError(f"Class-KL channel dimension mismatch: {run_path}")
    if np.asarray(arrays["class_ce_delta"]).shape != np.asarray(arrays["class_kl"]).shape:
        raise RuntimeError(f"CE and KL class-effect dimensions differ: {run_path}")
    return RawCheckpoint(
        cell=f"{dataset}/{backbone}",
        dataset=dataset,
        backbone=backbone,
        checkpoint_id=checkpoint_id,
        run=run,
        raw_manifest_path=manifest_path,
        raw_manifest_sha256=_file_sha256(manifest_path),
        raw_sidecar_path=sidecar_path,
        raw_sidecar_sha256=observed_hash,
        adjacency=adjacency,
        arrays=arrays,
    )


def _rank_alignment(first: np.ndarray, second: np.ndarray) -> float | None:
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape or first_values.ndim != 2:
        raise ValueError("Alignment kernels must be equally sized matrices")
    upper = np.triu_indices(first_values.shape[0], k=1)
    x = first_values[upper]
    y = second_values[upper]
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    x_rank = rankdata(x, method="average").astype(np.float64)
    y_rank = rankdata(y, method="average").astype(np.float64)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    if denominator == 0:
        return None
    value = float(np.dot(x_rank, y_rank) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _effect_kernel(effects: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return normalized_feature_kernel(np.asarray(effects, dtype=np.float64)[indices])


def _feature_options(gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "quantiles": FROZEN_QUANTILES,
        "wl_depths": FROZEN_WL_DEPTHS,
        "diffusion_times": FROZEN_DIFFUSION_TIMES,
        "minimum_impact": float(gate.get("minimum_impact", 1e-12)),
        "active_threshold": float(gate.get("active_threshold", 1e-12)),
        "wl_hash_dimension": int(gate.get("wl_hash_dimension", 512)),
        "node_projection_dimension": int(gate.get("node_projection_dimension", 256)),
    }


def _control_kernels(features: Gate1FeatureSet, subset: str) -> dict[str, np.ndarray]:
    return {
        decision_name: features.controls[feature_name].kernels[subset].kernel
        for feature_name, decision_name in CONTROL_NAME_MAP.items()
    }


def _all_candidate_kernels(
    features: Gate1FeatureSet,
    subset: str,
) -> dict[str, np.ndarray]:
    result = {
        name: block.kernels[subset].kernel
        for name, block in features.candidates.items()
    }
    result.update(
        {
            name: block.kernels[subset].kernel
            for name, block in features.sensitivities.items()
        }
    )
    return result


def _run_checkpoint_inference(
    raw: RawCheckpoint,
    features: Gate1FeatureSet,
    *,
    repetitions: int,
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
    for outcome_name, outcome in outcomes.items():
        result[outcome_name] = {}
        for subset, channel_indices in indices.items():
            result[outcome_name][subset] = paired_incremental_permutation_test(
                _control_kernels(features, subset),
                _all_candidate_kernels(features, subset),
                _effect_kernel(outcome, channel_indices),
                repetitions=repetitions,
                random_seed=_stable_seed(
                    "gate1-common-neuron-permutation",
                    raw.checkpoint_id,
                    outcome_name,
                    subset,
                ),
            )
    return result


def _row_l1(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    mass = np.abs(matrix).sum(axis=1, keepdims=True)
    return np.divide(matrix, mass, out=np.zeros_like(matrix), where=mass > 0)


def _prediction_arrays(
    raw: RawCheckpoint,
    features: Gate1FeatureSet,
    specs: Sequence[CandidateSpec],
) -> dict[str, np.ndarray]:
    active = np.flatnonzero(features.active_mask).astype(np.int64)
    controls = np.concatenate(
        [
            features.controls[name].prediction_features[active]
            for name in CONTROL_NAME_MAP
        ],
        axis=1,
    )
    arrays: dict[str, np.ndarray] = {
        "active_channel_indices": active,
        "control_features": np.asarray(controls, dtype=np.float64),
        "class_effects": _row_l1(np.asarray(raw.arrays["class_kl"])[active]),
    }
    for spec in specs:
        arrays[f"candidate__{spec.candidate_id}"] = np.asarray(
            features.candidates[spec.primary_variant].prediction_features[active],
            dtype=np.float64,
        )
    return arrays


def _run_rewiring_null(
    raw: RawCheckpoint,
    features: Gate1FeatureSet,
    specs: Sequence[CandidateSpec],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    graphs, metadata = complete_degree_preserving_rewires(
        raw.adjacency,
        repetitions=FROZEN_REWIRES,
        swaps_per_edge=FROZEN_SWAPS_PER_EDGE,
        random_seed=_stable_seed("gate1-complete-rewires", raw.checkpoint_id),
        max_attempt_multiplier=int(gate.get("max_swap_attempt_multiplier", 20)),
        retry_multiplier=int(gate.get("rewire_retry_multiplier", 5)),
    )
    active = np.flatnonzero(features.active_mask).astype(np.int64)
    effect_kernel = _effect_kernel(np.asarray(raw.arrays["class_kl"]), active)
    accepted_attempts = [attempt for attempt in metadata["attempts"] if attempt["accepted"]]
    replicates = []
    for graph, attempt in zip(graphs, accepted_attempts, strict=True):
        rewired_features = extract_gate1_features(
            graph,
            np.asarray(raw.arrays["node_impacts"]),
            np.asarray(raw.arrays["prediction_changes"]),
            np.asarray(raw.arrays["scope_mask"]),
            **_feature_options(gate),
        )
        alignments: dict[str, float | None] = {}
        for spec in specs:
            for variant in spec.variants:
                alignments[variant.name] = _rank_alignment(
                    rewired_features.candidates[variant.name]
                    .kernels["active_channels"]
                    .kernel,
                    effect_kernel,
                )
        # The known-degenerate constant h0 views remain an explicit diagnostic.
        for quantile in ("q90", "q95", "q99"):
            name = f"wl_constant_{quantile}_h0"
            alignments[name] = _rank_alignment(
                rewired_features.candidates[name].kernels["active_channels"].kernel,
                effect_kernel,
            )
        replicates.append(
            {
                "trial": int(attempt["trial"]),
                "random_seed": int(attempt["random_seed"]),
                "requested_swaps": int(attempt["requested_swaps"]),
                "completed_swaps": int(attempt["successful_swaps"]),
                "canonical_edge_sha256": str(attempt["canonical_edge_sha256"]),
                "alignments": alignments,
            }
        )
    return {"metadata": metadata, "replicates": replicates}


def _checkpoint_identity(
    raw: RawCheckpoint,
    config: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol_version": REFINED_PROTOCOL_VERSION,
        "refined_code_sha256": source_code_sha256(REFINED_CODE_FILES),
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
    specs: Sequence[CandidateSpec],
    protocol_sha256: str,
    cache_path: Path,
    resume: bool,
    progress: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    identity = _checkpoint_identity(raw, config, protocol_sha256)
    sidecar_path = cache_path.with_suffix(".npz")
    if resume and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("identity") != identity:
            raise RuntimeError(f"Refined Gate 1 cache identity mismatch: {cache_path}")
        sidecar = cached.get("prediction_sidecar", {})
        if not sidecar_path.exists() or _file_sha256(sidecar_path) != sidecar.get("sha256"):
            raise RuntimeError(f"Refined prediction sidecar hash mismatch: {sidecar_path}")
        arrays = _load_npz(sidecar_path)
        if _array_schema(arrays) != sidecar.get("arrays"):
            raise RuntimeError(f"Refined prediction sidecar schema mismatch: {sidecar_path}")
        return cached["record"], arrays

    if progress:
        print(f"[{raw.cell}] {raw.checkpoint_id[:12]}: extracting separated features", flush=True)
    features = extract_gate1_features(
        raw.adjacency,
        np.asarray(raw.arrays["node_impacts"]),
        np.asarray(raw.arrays["prediction_changes"]),
        np.asarray(raw.arrays["scope_mask"]),
        **_feature_options(gate),
    )
    if progress:
        print(f"[{raw.cell}] {raw.checkpoint_id[:12]}: running common neuron-label nulls", flush=True)
    inference = _run_checkpoint_inference(
        raw,
        features,
        repetitions=FROZEN_PERMUTATIONS,
    )
    if progress:
        print(f"[{raw.cell}] {raw.checkpoint_id[:12]}: running two complete graph rewires", flush=True)
    rewiring = _run_rewiring_null(raw, features, specs, gate)
    prediction = _prediction_arrays(raw, features, specs)
    _write_npz_atomic(sidecar_path, prediction)
    prediction_sidecar = {
        "path": str(sidecar_path),
        "sha256": _file_sha256(sidecar_path),
        "format": "compressed_npz",
        "allow_pickle_required": False,
        "arrays": _array_schema(prediction),
    }
    record = {
        "cell": raw.cell,
        "dataset": raw.dataset,
        "backbone": raw.backbone,
        "checkpoint_id": raw.checkpoint_id,
        "seed": int(raw.run["seed"]),
        "width": int(np.asarray(raw.arrays["node_impacts"]).shape[0]),
        "active_channel_count": int(features.active_mask.sum()),
        "raw_manifest": {
            "path": str(raw.raw_manifest_path),
            "sha256": raw.raw_manifest_sha256,
        },
        "raw_sidecar": {
            "path": str(raw.raw_sidecar_path),
            "sha256": raw.raw_sidecar_sha256,
        },
        "feature_metadata": dict(features.metadata),
        "inference": inference,
        "rewiring_null": rewiring,
        "test_labels_used": False,
        "test_performance_used": False,
    }
    _write_json_atomic(
        cache_path,
        {
            "identity": identity,
            "prediction_sidecar": prediction_sidecar,
            "record": record,
        },
    )
    return record, prediction


def _evidence_from_checkpoint(
    record: Mapping[str, Any],
    specs: Sequence[CandidateSpec],
) -> list[CheckpointEvidence]:
    result: list[CheckpointEvidence] = []
    for spec in specs:
        for variant in spec.variants:
            for subset in ("active_channels", "all_channels"):
                inference = record["inference"]["class_kl"][subset]
                controls = {
                    name: inference["controls"].get(name, {}).get("observed_alignment")
                    for name in CONTROL_NAME_MAP.values()
                }
                rewires: tuple[RewireAlignment, ...] = ()
                if subset == "active_channels":
                    rewires = tuple(
                        RewireAlignment(
                            alignment=replicate["alignments"].get(variant.name),
                            requested_swaps=int(replicate["requested_swaps"]),
                            completed_swaps=int(replicate["completed_swaps"]),
                        )
                        for replicate in record["rewiring_null"]["replicates"]
                    )
                result.append(
                    CheckpointEvidence(
                        checkpoint_id=str(record["checkpoint_id"]),
                        cell=str(record["cell"]),
                        candidate_id=spec.candidate_id,
                        variant=variant.name,
                        subset=subset,
                        candidate_alignment=inference["candidates"]
                        .get(variant.name, {})
                        .get("observed_alignment"),
                        control_alignments=controls,
                        rewires=rewires,
                    )
                )
    return result


def _prediction_checkpoint(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    specs: Sequence[CandidateSpec],
) -> Gate1PredictionCheckpoint:
    return Gate1PredictionCheckpoint(
        cell=str(record["cell"]),
        checkpoint_sha256=str(record["checkpoint_id"]),
        control_features=np.asarray(arrays["control_features"]),
        candidate_features={
            spec.candidate_id: np.asarray(arrays[f"candidate__{spec.candidate_id}"])
            for spec in specs
        },
        class_effects=np.asarray(arrays["class_effects"]),
    )


def derive_loco_flags(
    loco: Mapping[str, Any],
    candidate_ids: Sequence[str],
) -> tuple[dict[str, LocoFlags], dict[str, Any]]:
    """Apply the frozen checkpoint-weighted and three-of-four LOCO rule."""

    flags: dict[str, LocoFlags] = {}
    audit: dict[str, Any] = {}
    metric_names = ("rmse_reduction", "mae_reduction", "spearman_gain", "cosine_gain")
    for candidate_id in candidate_ids:
        values = {metric: [] for metric in metric_names}
        cell_means: dict[str, dict[str, float | None]] = {}
        for cell, cell_result in sorted(loco["cells"].items()):
            folds = cell_result["candidates"][candidate_id]["folds"]
            cell_means[cell] = {}
            for metric in metric_names:
                finite = [
                    float(fold["improvement"][metric])
                    for fold in folds
                    if fold["improvement"][metric] is not None
                    and math.isfinite(float(fold["improvement"][metric]))
                ]
                values[metric].extend(finite)
                cell_means[cell][metric] = float(mean(finite)) if finite else None
        overall = {
            metric: (float(mean(metric_values)) if metric_values else None)
            for metric, metric_values in values.items()
        }
        positive_cells = {
            metric: sum(
                summary[metric] is not None and float(summary[metric]) > 0
                for summary in cell_means.values()
            )
            for metric in metric_names
        }
        non_worse = bool(
            overall["rmse_reduction"] is not None
            and overall["mae_reduction"] is not None
            and float(overall["rmse_reduction"]) >= 0
            and float(overall["mae_reduction"]) >= 0
        )
        improving_metrics = [
            metric
            for metric in metric_names
            if overall[metric] is not None
            and float(overall[metric]) > 0
            and positive_cells[metric] >= 3
        ]
        improves = bool(improving_metrics)
        flags[candidate_id] = LocoFlags(non_worse, improves)
        audit[candidate_id] = {
            "checkpoint_weighted_mean": overall,
            "cell_means": cell_means,
            "positive_cell_count": positive_cells,
            "non_worse_than_best_control": non_worse,
            "improves_predeclared_metric": improves,
            "improving_metrics": improving_metrics,
        }
    return flags, audit


def _descriptive_inference_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for outcome in ("class_kl", "class_ce_delta"):
        result[outcome] = {}
        for subset in ("active_channels", "all_channels"):
            candidate_values: dict[str, list[float]] = {}
            for record in records:
                inference = record["inference"][outcome][subset]
                for name, item in inference.get("candidates", {}).items():
                    value = item.get("incremental_alignment")
                    if value is not None and math.isfinite(float(value)):
                        candidate_values.setdefault(name, []).append(float(value))
            result[outcome][subset] = {
                name: {
                    "checkpoint_count": len(values),
                    "mean_incremental_alignment": float(mean(values)),
                    "median_incremental_alignment": float(median(values)),
                    "positive_checkpoint_count": int(sum(value > 0 for value in values)),
                }
                for name, values in sorted(candidate_values.items())
            }
    return result


def build_refined_gate1_report(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    gate = _validate_frozen_config(config)
    raw_report_path = Path(config.get("raw_report", "artifacts/topology_gate1_raw_cache.json")).resolve()
    with raw_report_path.open("r", encoding="utf-8") as stream:
        raw_report = json.load(stream)
    _validate_raw_report_safety(raw_report)
    project_root = raw_report_path.parent.parent
    protocol_path = _resolve_path(
        config.get("protocol_document", "docs/topology_gate1_protocol.md"),
        project_root,
    )
    protocol_sha256 = _file_sha256(protocol_path)
    cache_root = _resolve_path(
        config.get("cache_root", "artifacts/topology_gate1_refined_runs"),
        project_root,
    )
    specs = frozen_candidate_specs()

    observed_cells = {
        f"{dataset}/{backbone}"
        for dataset, dataset_result in raw_report.get("datasets", {}).items()
        for backbone in dataset_result.get("backbones", {})
    }
    if observed_cells != set(DEFAULT_EXPECTED_CELLS):
        raise ValueError(
            "Raw cache must contain exactly the four frozen cells; observed "
            + repr(sorted(observed_cells))
        )

    checkpoint_records: list[dict[str, Any]] = []
    prediction_inputs: list[Gate1PredictionCheckpoint] = []
    evidence: list[CheckpointEvidence] = []
    for dataset, dataset_result in sorted(raw_report["datasets"].items()):
        for backbone, backbone_result in sorted(dataset_result["backbones"].items()):
            runs = sorted(
                backbone_result["runs"],
                key=lambda value: (int(value["seed"]), str(value["checkpoint_model_sha256"])),
            )
            for position, embedded_run in enumerate(runs, start=1):
                if progress:
                    print(
                        f"[{dataset}/{backbone}] verifying checkpoint {position}/{len(runs)}",
                        flush=True,
                    )
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
                    specs=specs,
                    protocol_sha256=protocol_sha256,
                    cache_path=cache_path,
                    resume=resume,
                    progress=progress,
                )
                checkpoint_records.append(record)
                prediction_inputs.append(_prediction_checkpoint(record, arrays, specs))
                evidence.extend(_evidence_from_checkpoint(record, specs))

    loco = run_within_stratum_loco(
        prediction_inputs,
        ridge_alpha=float(gate.get("ridge_alpha", 1.0)),
    )
    loco_flags, loco_audit = derive_loco_flags(
        loco,
        [spec.candidate_id for spec in specs],
    )
    decision = evaluate_gate1_decision(
        evidence,
        specs,
        loco_flags,
        expected_cells=DEFAULT_EXPECTED_CELLS,
        tie_tolerance=0.01,
        required_rewire_repetitions=FROZEN_REWIRES,
    )
    return {
        "protocol": {
            "name": "Gate 1 topology isolation — refined frozen evaluator",
            "version": REFINED_PROTOCOL_VERSION,
            "status": "development-only; separated evaluation completed",
            "development_only": True,
            "fresh_holdouts_used": False,
            "confirmatory_artifacts_used": False,
            "test_labels_used": False,
            "test_performance_used": False,
            "raw_report": str(raw_report_path),
            "raw_report_sha256": _file_sha256(raw_report_path),
            "protocol_document": str(protocol_path),
            "protocol_document_sha256": protocol_sha256,
            "config_sha256": canonical_payload_sha256(config),
            "refined_code_sha256": source_code_sha256(REFINED_CODE_FILES),
            "statistical_unit": "checkpoint, never channel pairs",
            "candidate_specs": [asdict(spec) for spec in specs],
            "known_degenerate_diagnostic": "constant-label WL h0; selection-ineligible",
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
            },
        },
        "checkpoint_count": len(checkpoint_records),
        "checkpoint_records": checkpoint_records,
        "descriptive_inference": _descriptive_inference_summary(checkpoint_records),
        "leave_one_checkpoint_out_prediction": loco,
        "loco_decision_audit": loco_audit,
        "gate_1_decision": decision,
        "edge_dropout_sensitivity": {
            "status": "pending_only_if_gate_1_passes",
            "fractions": [0.01, 0.05, 0.10],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/topology_gate1_refined.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_refined_gate1_report(
        config,
        resume=args.resume,
        progress=not args.quiet,
    )
    output = args.output or Path(
        config.get("output", "artifacts/topology_gate1_refined.json")
    )
    _write_json_atomic(output, report)
    print(f"Wrote refined Gate-1 report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

