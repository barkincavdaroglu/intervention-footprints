"""Matched joint-ablation screen for effect-field actionability.

The screen asks whether channels with highly overlapping intervention masks have
more subadditive or superadditive joint effects than covariate-matched low-overlap
pairs.  It is validation-only and runs before any repair adapter is trained.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from gnn_repair.checkpoints import (
    load_trained_run,
    model_state_sha256,
    module_state_sha256,
    source_code_sha256,
)
from gnn_repair.protocol import canonical_payload_sha256
from gnn_repair.reproducibility import set_deterministic
from gnn_repair.topology_gate1_refined import (
    DEFAULT_EXPECTED_CELLS,
    RawCheckpoint,
    _file_sha256,
    _load_raw_checkpoint,
    _resolve_path,
    _stable_seed,
    _validate_raw_report_safety,
    _write_json_atomic,
)


ACTIONABILITY_PROTOCOL_VERSION = 1
LOW_QUANTILE = 0.25
HIGH_QUANTILE = 0.75
PAIR_CANDIDATE_CAP = 64
MATCHED_SET_COUNT = 2
ACTIONABILITY_CODE_FILES = (
    "effect_field_actionability.py",
    "effect_field_gate.py",
    "topology_gate1_refined.py",
    "structural.py",
    "topology_gate1.py",
    "checkpoints.py",
    "data.py",
    "model.py",
    "protocol.py",
    "reproducibility.py",
)


@dataclass(frozen=True)
class PairCandidate:
    pair: tuple[int, int]
    similarity: float
    covariates: tuple[float, float, float, float]


@dataclass(frozen=True)
class MatchedPairSet:
    high: PairCandidate
    low: PairCandidate
    standardized_distance: float


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("development_only") is not True:
        raise ValueError("Actionability screen requires development_only: true")
    gate = dict(config.get("actionability", {}))
    frozen = {
        "low_similarity_quantile": LOW_QUANTILE,
        "high_similarity_quantile": HIGH_QUANTILE,
        "pair_candidate_cap": PAIR_CANDIDATE_CAP,
        "matched_set_count": MATCHED_SET_COUNT,
    }
    for name, expected in frozen.items():
        value = gate.get(name)
        if isinstance(expected, float):
            if not math.isclose(float(value), expected, abs_tol=1e-12):
                raise ValueError(f"{name} must be exactly {expected}")
        elif isinstance(value, bool) or value != expected:
            raise ValueError(f"{name} must be exactly {expected}")
    threshold = float(gate.get("active_threshold", 1e-12))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("active_threshold must be finite and non-negative")
    return gate


def _mask_jaccard(support_masks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    masks = np.asarray(support_masks)
    if masks.ndim != 3 or masks.dtype != np.bool_:
        raise ValueError("support_masks must be boolean [views, channels, nodes]")
    if indices.ndim != 1 or np.unique(indices).size != indices.size:
        raise ValueError("indices must be a unique vector")
    kernels = []
    for mask in masks:
        selected = mask[indices].astype(np.float64)
        intersection = selected @ selected.T
        size = selected.sum(axis=1)
        union = size[:, None] + size[None, :] - intersection
        kernels.append(
            np.divide(
                intersection,
                union,
                out=np.ones_like(intersection),
                where=union > 0,
            )
        )
    kernel = np.mean(np.stack(kernels), axis=0)
    np.fill_diagonal(kernel, 1.0)
    return kernel


def _pair_covariates(
    first: int,
    second: int,
    singleton_kl: np.ndarray,
    impact_mass: np.ndarray,
    support_fraction: np.ndarray,
    activation_magnitude: np.ndarray,
) -> tuple[float, float, float, float]:
    return (
        float(singleton_kl[first] + singleton_kl[second]),
        float(impact_mass[first] + impact_mass[second]),
        float(support_fraction[first] + support_fraction[second]),
        float(activation_magnitude[first] + activation_magnitude[second]),
    )


def match_high_low_pairs(
    similarity: np.ndarray,
    active_channels: np.ndarray,
    *,
    singleton_kl: np.ndarray,
    impact_mass: np.ndarray,
    support_fraction: np.ndarray,
    activation_magnitude: np.ndarray,
    candidate_cap: int = PAIR_CANDIDATE_CAP,
    matched_set_count: int = MATCHED_SET_COUNT,
) -> tuple[tuple[MatchedPairSet, ...], dict[str, Any]]:
    """Match extreme-similarity pair candidates with no channel reuse."""

    channels = np.asarray(active_channels, dtype=np.int64)
    matrix = np.asarray(similarity, dtype=np.float64)
    if matrix.shape != (channels.size, channels.size):
        raise ValueError("similarity must align with active_channels")
    if channels.size < 4 * matched_set_count:
        return (), {
            "evaluable": False,
            "reason": "too few active channels for globally disjoint pair sets",
        }
    upper = np.triu_indices(channels.size, k=1)
    values = matrix[upper]
    low_threshold = float(np.quantile(values, LOW_QUANTILE))
    high_threshold = float(np.quantile(values, HIGH_QUANTILE))
    if not high_threshold > low_threshold:
        return (), {
            "evaluable": False,
            "reason": "similarity quartiles are not separated",
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        }

    candidates = []
    for local_first, local_second, value in zip(
        upper[0], upper[1], values, strict=True
    ):
        first = int(channels[local_first])
        second = int(channels[local_second])
        candidate = PairCandidate(
            pair=(min(first, second), max(first, second)),
            similarity=float(value),
            covariates=_pair_covariates(
                first,
                second,
                singleton_kl,
                impact_mass,
                support_fraction,
                activation_magnitude,
            ),
        )
        candidates.append(candidate)
    high = sorted(
        [item for item in candidates if item.similarity >= high_threshold],
        key=lambda item: (-item.similarity, item.pair),
    )[:candidate_cap]
    low = sorted(
        [item for item in candidates if item.similarity <= low_threshold],
        key=lambda item: (item.similarity, item.pair),
    )[:candidate_cap]
    if not high or not low:
        return (), {
            "evaluable": False,
            "reason": "an extreme-similarity candidate group is empty",
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        }

    covariates = np.asarray(
        [item.covariates for item in (*high, *low)],
        dtype=np.float64,
    )
    location = covariates.mean(axis=0)
    scale = covariates.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    high_z = (covariates[: len(high)] - location) / scale
    low_z = (covariates[len(high) :] - location) / scale
    cost = np.linalg.norm(high_z[:, None, :] - low_z[None, :, :], axis=2)
    rows, columns = linear_sum_assignment(cost)
    assignments = sorted(
        [
            (
                float(cost[row, column]),
                high[int(row)],
                low[int(column)],
            )
            for row, column in zip(rows, columns, strict=True)
        ],
        key=lambda item: (item[0], item[1].pair, item[2].pair),
    )
    selected: list[MatchedPairSet] = []
    used: set[int] = set()
    for distance, high_pair, low_pair in assignments:
        channels_in_set = {*high_pair.pair, *low_pair.pair}
        if len(channels_in_set) != 4 or channels_in_set & used:
            continue
        selected.append(MatchedPairSet(high_pair, low_pair, distance))
        used.update(channels_in_set)
        if len(selected) == matched_set_count:
            break
    return tuple(selected), {
        "evaluable": len(selected) == matched_set_count,
        "reason": (
            None
            if len(selected) == matched_set_count
            else "Hungarian assignments did not contain enough globally disjoint sets"
        ),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "high_candidate_count_before_cap": int(
            sum(item.similarity >= high_threshold for item in candidates)
        ),
        "low_candidate_count_before_cap": int(
            sum(item.similarity <= low_threshold for item in candidates)
        ),
        "candidate_cap": int(candidate_cap),
        "requested_matched_sets": int(matched_set_count),
        "selected_matched_sets": len(selected),
        "covariate_names": [
            "singleton_macro_validation_kl_sum",
            "total_impact_mass_sum",
            "mean_support_fraction_sum",
            "training_mean_absolute_activation_sum",
        ],
        "covariate_location": location.tolist(),
        "covariate_scale": scale.tolist(),
    }


@torch.no_grad()
def _classwise_validation_effects(
    base_logits: torch.Tensor,
    modified_logits: torch.Tensor,
    labels: torch.Tensor,
    validation_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    base_probability = F.softmax(base_logits, dim=-1).clamp_min(1e-12)
    modified_probability = F.softmax(modified_logits, dim=-1).clamp_min(1e-12)
    node_kl = (
        base_probability
        * (base_probability.log() - modified_probability.log())
    ).sum(dim=-1)
    class_kl = []
    class_ce = []
    for class_id in range(base_logits.shape[1]):
        mask = validation_mask & (labels == class_id)
        if not bool(mask.any()):
            class_kl.append(0.0)
            class_ce.append(0.0)
        else:
            class_kl.append(float(node_kl[mask].mean().item()))
            class_ce.append(
                float(
                    (
                        F.cross_entropy(modified_logits[mask], labels[mask])
                        - F.cross_entropy(base_logits[mask], labels[mask])
                    ).item()
                )
            )
    return np.asarray(class_kl), np.asarray(class_ce)


def _joint_pair_effects(
    raw: RawCheckpoint,
    matched: Sequence[MatchedPairSet],
    project_root: Path,
) -> list[dict[str, Any]]:
    loaded = load_trained_run(
        _resolve_path(raw.run["run_path"], project_root),
        allow_download=False,
    )
    if model_state_sha256(loaded.checkpoint) != raw.checkpoint_id:
        raise RuntimeError("Loaded checkpoint differs from actionability raw record")
    set_deterministic(int(raw.run["seed"]), num_threads=1)
    loaded.model.eval()
    with torch.no_grad():
        base_logits, activations = loaded.model(
            loaded.data.x,
            loaded.data.adjacency,
            return_activations=True,
        )
    layer = int(raw.run["layer"])
    training_mean = activations[layer][loaded.data.train_mask].mean(dim=0)
    singleton_kl = np.asarray(raw.arrays["class_kl"], dtype=np.float64)
    singleton_ce = np.asarray(raw.arrays["class_ce_delta"], dtype=np.float64)
    base_hash = module_state_sha256(loaded.model)
    records = []
    for set_index, item in enumerate(matched):
        for group, candidate in (("high", item.high), ("low", item.low)):
            pair = candidate.pair
            index = torch.tensor(pair, dtype=torch.long, device=loaded.data.x.device)

            def replace_pair(
                active_layer: int,
                activation: torch.Tensor,
                *,
                expected_layer: int = layer,
                selected: torch.Tensor = index,
            ) -> torch.Tensor:
                if active_layer != expected_layer:
                    return activation
                modified = activation.clone()
                modified[:, selected] = training_mean[selected]
                return modified

            with torch.no_grad():
                modified_logits = loaded.model(
                    loaded.data.x,
                    loaded.data.adjacency,
                    activation_intervention=replace_pair,
                )
            joint_kl, joint_ce = _classwise_validation_effects(
                base_logits,
                modified_logits,
                loaded.data.y,
                loaded.data.val_mask,
            )
            singleton_kl_sum = singleton_kl[pair[0]] + singleton_kl[pair[1]]
            singleton_ce_sum = singleton_ce[pair[0]] + singleton_ce[pair[1]]
            kl_nonadditivity = joint_kl - singleton_kl_sum
            ce_nonadditivity = joint_ce - singleton_ce_sum
            records.append(
                {
                    "matched_set": set_index,
                    "group": group,
                    "pair": list(pair),
                    "similarity": candidate.similarity,
                    "covariates": list(candidate.covariates),
                    "standardized_match_distance": item.standardized_distance,
                    "joint_class_kl": joint_kl.tolist(),
                    "singleton_class_kl_sum": singleton_kl_sum.tolist(),
                    "class_kl_nonadditivity": kl_nonadditivity.tolist(),
                    "macro_joint_kl": float(joint_kl.mean()),
                    "macro_singleton_kl_sum": float(singleton_kl_sum.mean()),
                    "macro_kl_nonadditivity": float(kl_nonadditivity.mean()),
                    "joint_class_ce_delta": joint_ce.tolist(),
                    "singleton_class_ce_delta_sum": singleton_ce_sum.tolist(),
                    "class_ce_nonadditivity": ce_nonadditivity.tolist(),
                    "macro_ce_nonadditivity": float(ce_nonadditivity.mean()),
                }
            )
    if module_state_sha256(loaded.model) != base_hash:
        raise RuntimeError("Joint-ablation screen changed the frozen backbone")
    return records


def _activation_magnitude(raw: RawCheckpoint, project_root: Path) -> np.ndarray:
    loaded = load_trained_run(
        _resolve_path(raw.run["run_path"], project_root),
        allow_download=False,
    )
    with torch.no_grad():
        _, activations = loaded.model(
            loaded.data.x,
            loaded.data.adjacency,
            return_activations=True,
        )
    layer = int(raw.run["layer"])
    return (
        activations[layer][loaded.data.train_mask]
        .abs()
        .mean(dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )


def _checkpoint_identity(
    raw: RawCheckpoint,
    config: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol_version": ACTIONABILITY_PROTOCOL_VERSION,
        "code_sha256": source_code_sha256(ACTIONABILITY_CODE_FILES),
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
    project_root: Path,
    protocol_sha256: str,
    cache_path: Path,
    resume: bool,
    progress: bool,
) -> dict[str, Any]:
    identity = _checkpoint_identity(raw, config, protocol_sha256)
    if resume and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("identity") != identity:
            raise RuntimeError(f"Actionability cache identity mismatch: {cache_path}")
        return cached["record"]

    impacts = np.asarray(raw.arrays["node_impacts"], dtype=np.float64)
    scope = np.asarray(raw.arrays["scope_mask"], dtype=bool)
    support_masks = np.asarray(raw.arrays["support_masks"])
    active = np.flatnonzero(
        impacts[:, scope].max(axis=1) > float(gate.get("active_threshold", 1e-12))
    ).astype(np.int64)
    similarity = _mask_jaccard(support_masks, active)
    singleton_kl = np.asarray(raw.arrays["class_kl"], dtype=np.float64).mean(axis=1)
    impact_mass = impacts[:, scope].sum(axis=1)
    support_fraction = support_masks[:, :, scope].mean(axis=(0, 2))
    activation_magnitude = _activation_magnitude(raw, project_root)
    matched, matching = match_high_low_pairs(
        similarity,
        active,
        singleton_kl=singleton_kl,
        impact_mass=impact_mass,
        support_fraction=support_fraction,
        activation_magnitude=activation_magnitude,
    )
    if progress:
        print(
            f"[{raw.cell}] {raw.checkpoint_id[:12]}: "
            f"matched {len(matched)}/{MATCHED_SET_COUNT} pair sets",
            flush=True,
        )
    pair_records = (
        _joint_pair_effects(raw, matched, project_root)
        if matching["evaluable"]
        else []
    )
    if pair_records:
        high_kl = [
            float(record["macro_kl_nonadditivity"])
            for record in pair_records
            if record["group"] == "high"
        ]
        low_kl = [
            float(record["macro_kl_nonadditivity"])
            for record in pair_records
            if record["group"] == "low"
        ]
        high_ce = [
            float(record["macro_ce_nonadditivity"])
            for record in pair_records
            if record["group"] == "high"
        ]
        low_ce = [
            float(record["macro_ce_nonadditivity"])
            for record in pair_records
            if record["group"] == "low"
        ]
        contrast = {
            "d_j_kl": float(mean(high_kl) - mean(low_kl)),
            "high_mean_kl_nonadditivity": float(mean(high_kl)),
            "low_mean_kl_nonadditivity": float(mean(low_kl)),
            "d_j_ce_sensitivity": float(mean(high_ce) - mean(low_ce)),
            "high_mean_ce_nonadditivity": float(mean(high_ce)),
            "low_mean_ce_nonadditivity": float(mean(low_ce)),
        }
    else:
        contrast = None
    record = {
        "cell": raw.cell,
        "dataset": raw.dataset,
        "backbone": raw.backbone,
        "checkpoint_id": raw.checkpoint_id,
        "seed": int(raw.run["seed"]),
        "layer": int(raw.run["layer"]),
        "active_channel_count": int(active.size),
        "matching": matching,
        "matched_sets": [
            {
                "high_pair": list(item.high.pair),
                "high_similarity": item.high.similarity,
                "low_pair": list(item.low.pair),
                "low_similarity": item.low.similarity,
                "standardized_distance": item.standardized_distance,
            }
            for item in matched
        ],
        "pair_records": pair_records,
        "contrast": contrast,
        "test_labels_used": False,
        "test_performance_used": False,
    }
    _write_json_atomic(cache_path, {"identity": identity, "record": record})
    return record


def evaluate_actionability(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    incomplete = [
        str(record["checkpoint_id"])
        for record in records
        if not record["matching"]["evaluable"] or record["contrast"] is None
    ]
    if incomplete:
        return {
            "status": "not_evaluable",
            "authorized_path": None,
            "reason": "one or more checkpoints lack two complete matched pair sets",
            "incomplete_checkpoint_ids": incomplete,
        }
    values = [float(record["contrast"]["d_j_kl"]) for record in records]
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
    if overall_median > 0:
        same_sign_cells = [
            cell for cell, summary in cell_summaries.items() if summary["median"] > 0
        ]
        path = "synergy_coherent_tied"
    elif overall_median < 0:
        same_sign_cells = [
            cell for cell, summary in cell_summaries.items() if summary["median"] < 0
        ]
        path = "redundancy_diverse_mmr"
    else:
        same_sign_cells = []
        path = None
    passed = path is not None and len(same_sign_cells) >= 3
    ce_values = [
        float(record["contrast"]["d_j_ce_sensitivity"]) for record in records
    ]
    return {
        "status": "pass" if passed else "fail",
        "authorized_path": path if passed else None,
        "primary": {
            "estimand": "D_J = high-overlap minus low-overlap macro-KL nonadditivity",
            "count": len(values),
            "mean": float(mean(values)),
            "median": overall_median,
            "positive_count": int(sum(value > 0 for value in values)),
            "negative_count": int(sum(value < 0 for value in values)),
            "cells": cell_summaries,
            "same_sign_cells": same_sign_cells,
            "same_sign_cell_count": len(same_sign_cells),
        },
        "ce_sensitivity": {
            "mean": float(mean(ce_values)),
            "median": float(median(ce_values)),
            "positive_count": int(sum(value > 0 for value in ce_values)),
            "negative_count": int(sum(value < 0 for value in ce_values)),
        },
        "next_step": (
            f"run repair-adapter confirmation for {path}"
            if passed
            else "do not train a field-grouped repair policy"
        ),
    }


def build_actionability_report(
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
    validity_path = _resolve_path(
        config.get("validity_report", "artifacts/effect_field_gate.json"),
        project_root,
    )
    with validity_path.open("r", encoding="utf-8") as stream:
        validity = json.load(stream)
    if validity.get("effect_field_decision", {}).get("status") != "pass":
        raise RuntimeError("Actionability requires a passing effect-field validity gate")
    if validity.get("effect_field_decision", {}).get("selected_candidate") != "mask_field":
        raise RuntimeError("Actionability requires the frozen mask-field candidate")
    if validity.get("protocol", {}).get("test_labels_used") is not False:
        raise RuntimeError("Validity report is not test-blind")
    protocol_path = _resolve_path(
        config.get(
            "protocol_document",
            "docs/effect_field_actionability_protocol.md",
        ),
        project_root,
    )
    protocol_sha256 = _file_sha256(protocol_path)
    cache_root = _resolve_path(
        config.get("cache_root", "artifacts/effect_field_actionability_runs"),
        project_root,
    )
    observed_cells = {
        f"{dataset}/{backbone}"
        for dataset, dataset_result in raw_report.get("datasets", {}).items()
        for backbone in dataset_result.get("backbones", {})
    }
    if observed_cells != set(DEFAULT_EXPECTED_CELLS):
        raise ValueError("Actionability requires exactly the four frozen cells")

    records = []
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
    decision = evaluate_actionability(records)
    return {
        "protocol": {
            "name": "Matched effect-field joint-ablation actionability screen",
            "version": ACTIONABILITY_PROTOCOL_VERSION,
            "status": "exploratory development-only",
            "development_only": True,
            "fresh_holdouts_used": False,
            "confirmatory_artifacts_used": False,
            "test_labels_used": False,
            "test_performance_used": False,
            "raw_report": str(raw_report_path),
            "raw_report_sha256": _file_sha256(raw_report_path),
            "validity_report": str(validity_path),
            "validity_report_sha256": _file_sha256(validity_path),
            "protocol_document": str(protocol_path),
            "protocol_document_sha256": protocol_sha256,
            "config_sha256": canonical_payload_sha256(config),
            "code_sha256": source_code_sha256(ACTIONABILITY_CODE_FILES),
            "statistical_unit": "checkpoint",
            "matching": {
                "low_similarity_quantile": LOW_QUANTILE,
                "high_similarity_quantile": HIGH_QUANTILE,
                "pair_candidate_cap": PAIR_CANDIDATE_CAP,
                "matched_set_count": MATCHED_SET_COUNT,
                "channel_reuse": "forbidden across all selected pairs per checkpoint",
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
        "checkpoint_count": len(records),
        "checkpoint_records": records,
        "actionability_decision": decision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/effect_field_actionability.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_actionability_report(
        config,
        resume=args.resume,
        progress=not args.quiet,
    )
    output = args.output or Path(
        config.get("output", "artifacts/effect_field_actionability.json")
    )
    _write_json_atomic(output, report)
    print(f"Wrote effect-field actionability report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
