"""Validation-only repair-interaction confirmation for effect-field synergy."""

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

from gnn_repair.checkpoints import (
    load_trained_run,
    model_state_sha256,
    source_code_sha256,
)
from gnn_repair.effect_field_actionability import _mask_jaccard
from gnn_repair.gate2_interactions import (
    IndependentPairAffineAdapter,
    PairUtilityRecord,
    SingletonAffineAdapter,
    TiedPairAffineAdapter,
    fit_gate2_adapter,
    gate2_activation_statistics,
    interaction_contrasts,
)
from gnn_repair.localization import prediction_profile
from gnn_repair.protocol import canonical_payload_sha256
from gnn_repair.topology_gate1_refined import (
    DEFAULT_EXPECTED_CELLS,
    RawCheckpoint,
    _file_sha256,
    _load_raw_checkpoint,
    _resolve_path,
    _validate_raw_report_safety,
    _write_json_atomic,
)


REPAIR_INTERACTION_PROTOCOL_VERSION = 2
POOL_SIZE = 12
LOW_QUANTILE = 0.25
HIGH_QUANTILE = 0.75
MATCHED_SET_COUNT = 2
PAIR_CANDIDATE_CAP = 64
TRAINING = {
    "num_threads": 1,
    "learning_rate": 0.01,
    "max_epochs": 30,
    "patience": 10,
    "min_utility_delta": 0.000001,
    "target_ce_weight": 1.0,
    "non_target_kl_weight": 1.0,
    "collateral_ce_weight": 1.0,
    "identity_l2_weight": 0.001,
}
REPAIR_INTERACTION_CODE_FILES = (
    "effect_field_repair_interactions.py",
    "effect_field_actionability.py",
    "effect_field_gate.py",
    "topology_gate1_refined.py",
    "gate2_interactions.py",
    "localization.py",
    "structural.py",
    "topology_gate1.py",
    "checkpoints.py",
    "data.py",
    "model.py",
    "metrics.py",
    "protocol.py",
    "reproducibility.py",
)


@dataclass(frozen=True)
class RepairPair:
    pair: tuple[int, int]
    similarity: float
    covariates: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class MatchedRepairSet:
    high: RepairPair
    low: RepairPair
    distance: float


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("development_only") is not True:
        raise ValueError("Repair-interaction gate requires development_only: true")
    gate = dict(config.get("repair_interactions", {}))
    frozen = {
        "pool_size": POOL_SIZE,
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
    if dict(gate.get("training", {})) != TRAINING:
        raise ValueError("training configuration differs from the frozen protocol")
    threshold = float(gate.get("active_threshold", 1e-12))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("active_threshold must be finite and non-negative")
    return gate


def _select_target(
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
) -> tuple[int, dict[str, float]]:
    profile = prediction_profile(logits, labels, validation_mask)
    eligible = []
    f1: dict[str, float] = {}
    for class_id, metrics in profile["per_class"].items():
        value = float(metrics["f1"])
        f1[str(class_id)] = value
        numeric = int(class_id)
        if bool((train_mask & (labels == numeric)).any()) and bool(
            (validation_mask & (labels == numeric)).any()
        ):
            eligible.append((value, numeric))
    if not eligible:
        raise ValueError("No class occurs in both training and validation splits")
    return min(eligible)[1], f1


def _select_pool(
    raw: RawCheckpoint,
    target_class: int,
    *,
    active_threshold: float,
) -> np.ndarray:
    impacts = np.asarray(raw.arrays["node_impacts"], dtype=np.float64)
    scope = np.asarray(raw.arrays["scope_mask"], dtype=bool)
    active = np.flatnonzero(
        impacts[:, scope].max(axis=1) > active_threshold
    ).astype(np.int64)
    if active.size < POOL_SIZE:
        raise ValueError(f"Checkpoint has fewer than {POOL_SIZE} active channels")
    class_kl = np.asarray(raw.arrays["class_kl"], dtype=np.float64)
    ranked = sorted(
        active.tolist(),
        key=lambda channel: (-float(class_kl[channel, target_class]), channel),
    )
    return np.asarray(ranked[:POOL_SIZE], dtype=np.int64)


def match_repair_pairs(
    similarity: np.ndarray,
    pool: np.ndarray,
    *,
    singleton_utility: np.ndarray,
    target_class_kl: np.ndarray,
    impact_mass: np.ndarray,
    support_fraction: np.ndarray,
    activation_magnitude: np.ndarray,
) -> tuple[tuple[MatchedRepairSet, ...], dict[str, Any]]:
    channels = np.asarray(pool, dtype=np.int64)
    matrix = np.asarray(similarity, dtype=np.float64)
    if matrix.shape != (channels.size, channels.size):
        raise ValueError("similarity does not align with the repair pool")
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
    for local_first, local_second, similarity_value in zip(
        upper[0], upper[1], values, strict=True
    ):
        first = int(channels[local_first])
        second = int(channels[local_second])
        candidates.append(
            RepairPair(
                pair=(min(first, second), max(first, second)),
                similarity=float(similarity_value),
                covariates=(
                    float(singleton_utility[first] + singleton_utility[second]),
                    float(target_class_kl[first] + target_class_kl[second]),
                    float(impact_mass[first] + impact_mass[second]),
                    float(support_fraction[first] + support_fraction[second]),
                    float(activation_magnitude[first] + activation_magnitude[second]),
                ),
            )
        )
    high = sorted(
        [item for item in candidates if item.similarity >= high_threshold],
        key=lambda item: (-item.similarity, item.pair),
    )[:PAIR_CANDIDATE_CAP]
    low = sorted(
        [item for item in candidates if item.similarity <= low_threshold],
        key=lambda item: (item.similarity, item.pair),
    )[:PAIR_CANDIDATE_CAP]
    values_matrix = np.asarray(
        [item.covariates for item in (*high, *low)],
        dtype=np.float64,
    )
    location = values_matrix.mean(axis=0)
    scale = values_matrix.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    high_z = (values_matrix[: len(high)] - location) / scale
    low_z = (values_matrix[len(high) :] - location) / scale
    cost = np.linalg.norm(high_z[:, None] - low_z[None, :], axis=2)
    assignments = sorted(
        [
            (
                float(cost[high_index, low_index]),
                high[high_index],
                low[low_index],
                frozenset((*high[high_index].pair, *low[low_index].pair)),
            )
            for high_index in range(len(high))
            for low_index in range(len(low))
            if len({*high[high_index].pair, *low[low_index].pair}) == 4
        ],
        key=lambda item: (item[0], item[1].pair, item[2].pair),
    )
    best: tuple[Any, ...] | None = None
    for first_index, first in enumerate(assignments):
        if best is not None and first[0] + assignments[0][0] > best[0] + 1e-15:
            break
        for second_index, second in enumerate(assignments):
            if first_index == second_index or first[3] & second[3]:
                continue
            ordered = sorted(
                (first, second),
                key=lambda item: (item[1].pair, item[2].pair),
            )
            candidate = (
                first[0] + second[0],
                tuple((item[1].pair, item[2].pair) for item in ordered),
                ordered,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            break
    selected = (
        [
            MatchedRepairSet(item[1], item[2], item[0])
            for item in best[2]
        ]
        if best is not None
        else []
    )
    return tuple(selected), {
        "evaluable": len(selected) == MATCHED_SET_COUNT,
        "reason": (
            None
            if len(selected) == MATCHED_SET_COUNT
            else "no exact two-set solution with eight globally distinct channels"
        ),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "requested_matched_sets": MATCHED_SET_COUNT,
        "selected_matched_sets": len(selected),
        "matching_algorithm": (
            "exact minimum total standardized distance over two high/low pair "
            "assignments with eight globally distinct channels"
        ),
        "covariate_names": [
            "singleton_utility_sum",
            "target_class_validation_kl_sum",
            "total_impact_mass_sum",
            "mean_support_fraction_sum",
            "training_mean_absolute_activation_sum",
        ],
        "covariate_location": location.tolist(),
        "covariate_scale": scale.tolist(),
    }


def _identity(
    raw: RawCheckpoint,
    config: Mapping[str, Any],
    protocol_sha256: str,
    actionability_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol_version": REPAIR_INTERACTION_PROTOCOL_VERSION,
        "code_sha256": source_code_sha256(REPAIR_INTERACTION_CODE_FILES),
        "protocol_document_sha256": protocol_sha256,
        "config_sha256": canonical_payload_sha256(config),
        "actionability_report_sha256": actionability_sha256,
        "cell": raw.cell,
        "checkpoint_id": raw.checkpoint_id,
        "raw_manifest_sha256": raw.raw_manifest_sha256,
        "raw_sidecar_sha256": raw.raw_sidecar_sha256,
    }


def _load_partial(cache_path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    if not cache_path.exists():
        return {
            "identity": dict(identity),
            "singletons": {},
            "pairs": {},
        }
    with cache_path.open("r", encoding="utf-8") as stream:
        cached = json.load(stream)
    if cached.get("identity") != dict(identity):
        raise RuntimeError(f"Repair-interaction cache identity mismatch: {cache_path}")
    return cached


def _save_partial(cache_path: Path, partial: Mapping[str, Any]) -> None:
    _write_json_atomic(cache_path, partial)


def _fit_checkpoint(
    raw: RawCheckpoint,
    *,
    config: Mapping[str, Any],
    gate: Mapping[str, Any],
    project_root: Path,
    protocol_sha256: str,
    actionability_sha256: str,
    cache_path: Path,
    resume: bool,
    progress: bool,
) -> dict[str, Any]:
    identity = _identity(raw, config, protocol_sha256, actionability_sha256)
    partial = (
        _load_partial(cache_path, identity)
        if resume
        else {"identity": identity, "singletons": {}, "pairs": {}}
    )
    if "record" in partial:
        return partial["record"]
    loaded = load_trained_run(
        _resolve_path(raw.run["run_path"], project_root),
        allow_download=False,
    )
    if model_state_sha256(loaded.checkpoint) != raw.checkpoint_id:
        raise RuntimeError("Repair-interaction checkpoint hash mismatch")
    with torch.no_grad():
        base_logits = loaded.model(loaded.data.x, loaded.data.adjacency)
    target_class, validation_f1 = _select_target(
        base_logits,
        loaded.data.y,
        loaded.data.train_mask,
        loaded.data.val_mask,
    )
    pool = _select_pool(
        raw,
        target_class,
        active_threshold=float(gate.get("active_threshold", 1e-12)),
    )
    means, stds = gate2_activation_statistics(loaded)
    layer = int(raw.run["layer"])
    singletons = partial.setdefault("singletons", {})
    for position, channel in enumerate(pool.tolist(), start=1):
        key = str(channel)
        if key not in singletons:
            if progress:
                print(
                    f"[{raw.cell}] {raw.checkpoint_id[:12]} singleton "
                    f"{position}/{POOL_SIZE}",
                    flush=True,
                )
            adapter = SingletonAffineAdapter(
                layer,
                channel,
                means[layer],
                stds[layer],
            )
            singletons[key] = fit_gate2_adapter(
                loaded,
                adapter,
                target_class=target_class,
                training=TRAINING,
            )
            _save_partial(cache_path, partial)

    width = np.asarray(raw.arrays["node_impacts"]).shape[0]
    singleton_utility = np.zeros(width, dtype=np.float64)
    for channel in pool:
        singleton_utility[channel] = float(
            singletons[str(int(channel))]["best_validation"]["utility"]
        )
    support_masks = np.asarray(raw.arrays["support_masks"])
    similarity = _mask_jaccard(support_masks, pool)
    impacts = np.asarray(raw.arrays["node_impacts"], dtype=np.float64)
    scope = np.asarray(raw.arrays["scope_mask"], dtype=bool)
    class_kl = np.asarray(raw.arrays["class_kl"], dtype=np.float64)
    impact_mass = impacts[:, scope].sum(axis=1)
    support_fraction = support_masks[:, :, scope].mean(axis=(0, 2))
    with torch.no_grad():
        _, matching_activations = loaded.model(
            loaded.data.x,
            loaded.data.adjacency,
            return_activations=True,
        )
    activation_magnitude = (
        matching_activations[layer][loaded.data.train_mask]
        .abs()
        .mean(dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    matched, matching = match_repair_pairs(
        similarity,
        pool,
        singleton_utility=singleton_utility,
        target_class_kl=class_kl[:, target_class],
        impact_mass=impact_mass,
        support_fraction=support_fraction,
        activation_magnitude=activation_magnitude,
    )
    if not matching["evaluable"]:
        record = {
            "cell": raw.cell,
            "checkpoint_id": raw.checkpoint_id,
            "seed": int(raw.run["seed"]),
            "target_class": target_class,
            "matching": matching,
            "contrast": None,
            "test_labels_used": False,
            "test_performance_used": False,
        }
        partial["record"] = record
        _save_partial(cache_path, partial)
        return record

    pairs = partial.setdefault("pairs", {})
    for set_index, matched_set in enumerate(matched):
        for group, candidate in (("high", matched_set.high), ("low", matched_set.low)):
            pair = candidate.pair
            key = f"{set_index}:{group}:{pair[0]}-{pair[1]}"
            pair_result = pairs.setdefault(key, {})
            if "independent" not in pair_result:
                if progress:
                    print(
                        f"[{raw.cell}] {raw.checkpoint_id[:12]} pair {key} independent",
                        flush=True,
                    )
                pair_result["independent"] = fit_gate2_adapter(
                    loaded,
                    IndependentPairAffineAdapter(
                        layer,
                        pair,
                        means[layer],
                        stds[layer],
                    ),
                    target_class=target_class,
                    training=TRAINING,
                )
                _save_partial(cache_path, partial)
            if "tied" not in pair_result:
                if progress:
                    print(
                        f"[{raw.cell}] {raw.checkpoint_id[:12]} pair {key} tied",
                        flush=True,
                    )
                pair_result["tied"] = fit_gate2_adapter(
                    loaded,
                    TiedPairAffineAdapter(
                        layer,
                        pair,
                        means[layer],
                        stds[layer],
                    ),
                    target_class=target_class,
                    training=TRAINING,
                )
                _save_partial(cache_path, partial)

    pair_records = []
    interaction_inputs = []
    for set_index, matched_set in enumerate(matched):
        for group, candidate in (("high", matched_set.high), ("low", matched_set.low)):
            pair = candidate.pair
            key = f"{set_index}:{group}:{pair[0]}-{pair[1]}"
            independent = float(pairs[key]["independent"]["best_validation"]["utility"])
            tied = float(pairs[key]["tied"]["best_validation"]["utility"])
            first = float(singleton_utility[pair[0]])
            second = float(singleton_utility[pair[1]])
            utility_record = PairUtilityRecord(
                pair=pair,
                same_cluster=group == "high",
                singleton_first=first,
                singleton_second=second,
                independent_pair_utility=independent,
                tied_pair_utility=tied,
            )
            interaction_inputs.append(utility_record)
            pair_records.append(
                {
                    "matched_set": set_index,
                    "group": group,
                    "pair": list(pair),
                    "similarity": candidate.similarity,
                    "covariates": list(candidate.covariates),
                    "standardized_match_distance": matched_set.distance,
                    "singleton_first_utility": first,
                    "singleton_second_utility": second,
                    "independent_pair_utility": independent,
                    "tied_pair_utility": tied,
                    "interaction": utility_record.interaction,
                    "sharing_gain": utility_record.sharing_gain,
                    "independent_best_epoch": pairs[key]["independent"]["best_epoch"],
                    "tied_best_epoch": pairs[key]["tied"]["best_epoch"],
                }
            )
    contrast = interaction_contrasts(interaction_inputs).as_dict()
    record = {
        "cell": raw.cell,
        "dataset": raw.dataset,
        "backbone": raw.backbone,
        "checkpoint_id": raw.checkpoint_id,
        "seed": int(raw.run["seed"]),
        "layer": layer,
        "target_class": target_class,
        "baseline_validation_f1": validation_f1,
        "pool": pool.tolist(),
        "pool_target_class_kl": {
            str(int(channel)): float(class_kl[channel, target_class])
            for channel in pool
        },
        "singleton_results": singletons,
        "matching": matching,
        "matched_sets": [
            {
                "high_pair": list(item.high.pair),
                "high_similarity": item.high.similarity,
                "low_pair": list(item.low.pair),
                "low_similarity": item.low.similarity,
                "standardized_distance": item.distance,
            }
            for item in matched
        ],
        "pair_records": pair_records,
        "contrast": contrast,
        "test_labels_used": False,
        "test_performance_used": False,
    }
    partial["record"] = record
    _save_partial(cache_path, partial)
    return record


def _contrast_summary(
    records: Sequence[Mapping[str, Any]],
    name: str,
) -> dict[str, Any]:
    values = [float(record["contrast"][name]) for record in records]
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
    positive_cells = [
        cell for cell, summary in cells.items() if float(summary["median"]) > 0
    ]
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "positive_count": int(sum(value > 0 for value in values)),
        "negative_count": int(sum(value < 0 for value in values)),
        "cells": cells,
        "positive_cells": positive_cells,
        "positive_cell_count": len(positive_cells),
    }


def evaluate_repair_interactions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    incomplete = [
        str(record["checkpoint_id"])
        for record in records
        if record.get("contrast") is None
        or not record.get("matching", {}).get("evaluable", False)
    ]
    if incomplete:
        return {
            "status": "not_evaluable",
            "authorized_policy": None,
            "incomplete_checkpoint_ids": incomplete,
        }
    interaction = _contrast_summary(records, "d_int")
    sharing = _contrast_summary(records, "gamma")
    interaction_pass = (
        interaction["median"] > 0 and interaction["positive_cell_count"] >= 3
    )
    sharing_pass = sharing["median"] > 0 and sharing["positive_cell_count"] >= 3
    if interaction_pass and sharing_pass:
        status = "pass"
        policy = "coherent_tied"
        next_step = "compare coherent tied repair with matched controls"
    elif interaction_pass:
        status = "partial"
        policy = "coherent_independent_only"
        next_step = "do not claim parameter tying; test coherent independent repair"
    else:
        status = "fail"
        policy = None
        next_step = "stop effect-field repair-method development"
    return {
        "status": status,
        "authorized_policy": policy,
        "criteria": {
            "repair_interaction": {
                "pass": interaction_pass,
                "rule": "D_int median > 0 and positive cell medians in at least 3/4 cells",
            },
            "tied_sharing_gain": {
                "pass": sharing_pass,
                "rule": "Gamma median > 0 and positive cell medians in at least 3/4 cells",
            },
        },
        "d_int": interaction,
        "gamma": sharing,
        "next_step": next_step,
    }


def build_repair_interaction_report(
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
    actionability_path = _resolve_path(
        config.get("actionability_report", "artifacts/effect_field_actionability.json"),
        project_root,
    )
    with actionability_path.open("r", encoding="utf-8") as stream:
        actionability = json.load(stream)
    decision = actionability.get("actionability_decision", {})
    if decision.get("status") != "pass" or decision.get("authorized_path") != "synergy_coherent_tied":
        raise RuntimeError("Repair interaction requires the frozen synergy/tied authorization")
    actionability_sha256 = _file_sha256(actionability_path)
    protocol_path = _resolve_path(
        config.get(
            "protocol_document",
            "docs/effect_field_repair_interaction_protocol.md",
        ),
        project_root,
    )
    protocol_sha256 = _file_sha256(protocol_path)
    cache_root = _resolve_path(
        config.get("cache_root", "artifacts/effect_field_repair_interaction_runs"),
        project_root,
    )
    observed_cells = {
        f"{dataset}/{backbone}"
        for dataset, dataset_result in raw_report.get("datasets", {}).items()
        for backbone in dataset_result.get("backbones", {})
    }
    if observed_cells != set(DEFAULT_EXPECTED_CELLS):
        raise ValueError("Repair interaction requires exactly the four frozen cells")
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
                    _fit_checkpoint(
                        raw,
                        config=config,
                        gate=gate,
                        project_root=project_root,
                        protocol_sha256=protocol_sha256,
                        actionability_sha256=actionability_sha256,
                        cache_path=cache_path,
                        resume=resume,
                        progress=progress,
                    )
                )
    gate_decision = evaluate_repair_interactions(records)
    return {
        "protocol": {
            "name": "Effect-field repair interaction confirmation",
            "version": REPAIR_INTERACTION_PROTOCOL_VERSION,
            "status": "exploratory development-only",
            "development_only": True,
            "fresh_holdouts_used": False,
            "confirmatory_artifacts_used": False,
            "test_labels_used": False,
            "test_performance_used": False,
            "raw_report": str(raw_report_path),
            "raw_report_sha256": _file_sha256(raw_report_path),
            "actionability_report": str(actionability_path),
            "actionability_report_sha256": actionability_sha256,
            "protocol_document": str(protocol_path),
            "protocol_document_sha256": protocol_sha256,
            "config_sha256": canonical_payload_sha256(config),
            "code_sha256": source_code_sha256(REPAIR_INTERACTION_CODE_FILES),
            "statistical_unit": "checkpoint",
            "pool_size": POOL_SIZE,
            "matched_set_count": MATCHED_SET_COUNT,
            "training": TRAINING,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
        "checkpoint_count": len(records),
        "checkpoint_records": records,
        "repair_interaction_decision": gate_decision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/effect_field_repair_interactions.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_repair_interaction_report(
        config,
        resume=args.resume,
        progress=not args.quiet,
    )
    output = args.output or Path(
        config.get("output", "artifacts/effect_field_repair_interactions.json")
    )
    _write_json_atomic(output, report)
    print(f"Wrote effect-field repair-interaction report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
