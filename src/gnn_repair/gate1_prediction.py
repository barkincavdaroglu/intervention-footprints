"""Leakage-safe within-stratum prediction for the Gate 1 topology protocol.

The prediction unit is a complete checkpoint.  Models are trained only on the
other checkpoints from the same dataset/backbone stratum, and each training
checkpoint receives equal total weight regardless of its channel count.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class Gate1PredictionCheckpoint:
    """Channel-level inputs from one frozen checkpoint.

    ``control_features`` and every matrix in ``candidate_features`` have one
    row per hidden channel. ``class_effects`` contains the corresponding
    validation-derived class-effect vector for every channel.
    """

    cell: str
    checkpoint_sha256: str
    control_features: np.ndarray
    candidate_features: Mapping[str, np.ndarray]
    class_effects: np.ndarray


@dataclass(frozen=True)
class _PreparedCheckpoint:
    cell: str
    checkpoint_sha256: str
    controls: np.ndarray
    candidates: dict[str, np.ndarray]
    effects: np.ndarray


def _as_finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains a non-finite value")
    return matrix


def validate_prediction_checkpoints(
    checkpoints: Sequence[Gate1PredictionCheckpoint],
) -> tuple[_PreparedCheckpoint, ...]:
    """Validate checkpoint identity and within-stratum feature compatibility."""

    if not checkpoints:
        raise ValueError("At least one Gate 1 prediction checkpoint is required")
    prepared: list[_PreparedCheckpoint] = []
    hashes: set[str] = set()
    by_cell: dict[str, list[_PreparedCheckpoint]] = defaultdict(list)
    for position, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Gate1PredictionCheckpoint):
            raise TypeError(
                "checkpoints must contain Gate1PredictionCheckpoint objects"
            )
        cell = str(checkpoint.cell).strip()
        checkpoint_hash = str(checkpoint.checkpoint_sha256).strip()
        if not cell:
            raise ValueError(f"Checkpoint {position} has an empty stratum name")
        if not checkpoint_hash:
            raise ValueError(f"Checkpoint {position} has an empty checkpoint hash")
        if checkpoint_hash in hashes:
            raise ValueError(
                f"Duplicate checkpoint hash is not allowed: {checkpoint_hash}"
            )
        hashes.add(checkpoint_hash)

        controls = _as_finite_matrix(
            checkpoint.control_features,
            f"control_features for {checkpoint_hash}",
        )
        effects = _as_finite_matrix(
            checkpoint.class_effects,
            f"class_effects for {checkpoint_hash}",
        )
        if controls.shape[0] == 0:
            raise ValueError(f"Checkpoint {checkpoint_hash} has no channels")
        if controls.shape[1] == 0:
            raise ValueError(
                f"Checkpoint {checkpoint_hash} has no control features"
            )
        if effects.shape[0] != controls.shape[0]:
            raise ValueError(
                f"Checkpoint {checkpoint_hash} has inconsistent channel counts"
            )
        if effects.shape[1] == 0:
            raise ValueError(f"Checkpoint {checkpoint_hash} has no class effects")
        if not isinstance(checkpoint.candidate_features, Mapping):
            raise TypeError("candidate_features must be a mapping of matrices")
        if not checkpoint.candidate_features:
            raise ValueError(
                f"Checkpoint {checkpoint_hash} has no candidate representations"
            )
        candidates: dict[str, np.ndarray] = {}
        for raw_name, raw_values in checkpoint.candidate_features.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Candidate representation names cannot be empty")
            if name in candidates:
                raise ValueError(f"Duplicate candidate representation name: {name}")
            values = _as_finite_matrix(
                raw_values,
                f"candidate {name!r} for {checkpoint_hash}",
            )
            if values.shape[0] != controls.shape[0]:
                raise ValueError(
                    f"Candidate {name!r} for {checkpoint_hash} has an "
                    "inconsistent channel count"
                )
            if values.shape[1] == 0:
                raise ValueError(
                    f"Candidate {name!r} for {checkpoint_hash} has no features"
                )
            candidates[name] = values

        item = _PreparedCheckpoint(
            cell=cell,
            checkpoint_sha256=checkpoint_hash,
            controls=controls,
            candidates=candidates,
            effects=effects,
        )
        prepared.append(item)
        by_cell[cell].append(item)

    for cell, items in sorted(by_cell.items()):
        if len(items) < 2:
            raise ValueError(
                f"Stratum {cell!r} needs at least two checkpoints for LOCO"
            )
        first = items[0]
        expected_candidates = tuple(sorted(first.candidates))
        for item in items[1:]:
            if item.effects.shape[1] != first.effects.shape[1]:
                raise ValueError(
                    f"Stratum {cell!r} does not share one class-effect dimension"
                )
            if item.controls.shape[1] != first.controls.shape[1]:
                raise ValueError(
                    f"Stratum {cell!r} does not share one control-feature dimension"
                )
            if tuple(sorted(item.candidates)) != expected_candidates:
                raise ValueError(
                    f"Stratum {cell!r} does not share candidate representations"
                )
            for name in expected_candidates:
                if item.candidates[name].shape[1] != first.candidates[name].shape[1]:
                    raise ValueError(
                        f"Candidate {name!r} in stratum {cell!r} does not share "
                        "one feature dimension"
                    )
    return tuple(prepared)


def _equal_checkpoint_weights(
    checkpoints: Sequence[_PreparedCheckpoint],
) -> np.ndarray:
    sizes = np.asarray([item.effects.shape[0] for item in checkpoints], dtype=int)
    total_channels = int(sizes.sum())
    checkpoint_count = len(checkpoints)
    # The scaling makes the average sample weight one while every checkpoint's
    # weights sum to the same value. This keeps ridge-alpha semantics stable.
    return np.concatenate(
        [
            np.full(
                int(size),
                total_channels / (checkpoint_count * int(size)),
                dtype=np.float64,
            )
            for size in sizes
        ]
    )


def _weighted_location_scale(
    values: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weight_sum = float(sample_weight.sum())
    location = np.sum(values * sample_weight[:, None], axis=0) / weight_sum
    centered = values - location
    variance = np.sum(
        np.square(centered) * sample_weight[:, None],
        axis=0,
    ) / weight_sum
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale > 0, scale, 1.0)
    return location, scale


def _fit_predict_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    sample_weight: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    location, scale = _weighted_location_scale(train_x, sample_weight)
    scaled_train = (train_x - location) / scale
    scaled_test = (test_x - location) / scale
    estimator = Ridge(
        alpha=ridge_alpha,
        fit_intercept=True,
        solver="cholesky",
    )
    estimator.fit(scaled_train, train_y, sample_weight=sample_weight)
    prediction = np.asarray(estimator.predict(scaled_test), dtype=np.float64)
    if prediction.shape != (test_x.shape[0], train_y.shape[1]):
        raise RuntimeError("Multi-output ridge returned an unexpected shape")
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Multi-output ridge produced non-finite predictions")
    return prediction, {
        "fit_on_training_checkpoints_only": True,
        "location": location.tolist(),
        "scale": scale.tolist(),
        "ridge_alpha": ridge_alpha,
        "ridge_solver": "cholesky",
    }


def _finite_summary(values: Sequence[float]) -> dict[str, Any] | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return {
        "count": len(finite),
        "mean": float(mean(finite)),
        "median": float(median(finite)),
        "values": finite,
    }


def _channel_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    rank_values: list[float] = []
    cosine_values: list[float] = []
    rank_by_channel: list[float | None] = []
    cosine_by_channel: list[float | None] = []
    for actual_row, predicted_row in zip(actual, prediction, strict=True):
        if (
            actual_row.size >= 2
            and np.ptp(actual_row) > 0
            and np.ptp(predicted_row) > 0
        ):
            rank = float(spearmanr(actual_row, predicted_row).statistic)
            if math.isfinite(rank):
                rank_values.append(rank)
                rank_by_channel.append(rank)
            else:
                rank_by_channel.append(None)
        else:
            rank_by_channel.append(None)

        denominator = float(
            np.linalg.norm(actual_row) * np.linalg.norm(predicted_row)
        )
        if denominator > 0:
            cosine = float(np.dot(actual_row, predicted_row) / denominator)
            cosine = float(np.clip(cosine, -1.0, 1.0))
            cosine_values.append(cosine)
            cosine_by_channel.append(cosine)
        else:
            cosine_by_channel.append(None)
    residual = prediction - actual
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "per_channel_spearman": {
            "summary": _finite_summary(rank_values),
            "values": rank_by_channel,
        },
        "per_channel_cosine": {
            "summary": _finite_summary(cosine_values),
            "values": cosine_by_channel,
        },
    }


def _summary_metric(
    folds: Sequence[dict[str, Any]],
    model_name: str,
    metric_name: str,
) -> dict[str, Any] | None:
    return _finite_summary(
        [float(fold[model_name][metric_name]) for fold in folds]
    )


def _summary_channel_metric(
    folds: Sequence[dict[str, Any]],
    model_name: str,
    metric_name: str,
) -> dict[str, Any] | None:
    values = []
    for fold in folds:
        summary = fold[model_name][metric_name]["summary"]
        if summary is not None:
            # Fold means make the checkpoint, rather than the channel, the unit.
            values.append(float(summary["mean"]))
    return _finite_summary(values)


def _candidate_summary(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    comparisons = [fold["improvement"] for fold in folds]
    return {
        "statistical_unit": "held-out checkpoint",
        "controls": {
            "rmse": _summary_metric(folds, "controls", "rmse"),
            "mae": _summary_metric(folds, "controls", "mae"),
            "per_channel_spearman": _summary_channel_metric(
                folds, "controls", "per_channel_spearman"
            ),
            "per_channel_cosine": _summary_channel_metric(
                folds, "controls", "per_channel_cosine"
            ),
        },
        "controls_plus_candidate": {
            "rmse": _summary_metric(folds, "controls_plus_candidate", "rmse"),
            "mae": _summary_metric(folds, "controls_plus_candidate", "mae"),
            "per_channel_spearman": _summary_channel_metric(
                folds, "controls_plus_candidate", "per_channel_spearman"
            ),
            "per_channel_cosine": _summary_channel_metric(
                folds, "controls_plus_candidate", "per_channel_cosine"
            ),
        },
        "improvement": {
            name: _finite_summary(
                [
                    float(comparison[name])
                    for comparison in comparisons
                    if comparison[name] is not None
                ]
            )
            for name in (
                "rmse_reduction",
                "mae_reduction",
                "spearman_gain",
                "cosine_gain",
            )
        },
    }


def _metric_mean(metrics: dict[str, Any], name: str) -> float | None:
    summary = metrics[name]["summary"]
    return float(summary["mean"]) if summary is not None else None


def run_within_stratum_loco(
    checkpoints: Sequence[Gate1PredictionCheckpoint],
    *,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """Run deterministic multi-output LOCO comparisons without pooling strata."""

    if not math.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise ValueError("ridge_alpha must be finite and non-negative")
    prepared = validate_prediction_checkpoints(checkpoints)
    by_cell: dict[str, list[_PreparedCheckpoint]] = defaultdict(list)
    for item in prepared:
        by_cell[item.cell].append(item)

    result: dict[str, Any] = {
        "protocol": {
            "split": "leave one whole checkpoint out within each stratum",
            "target": "channel-level class-effect vectors",
            "strata_pooled": False,
            "preprocessing_fit": "training checkpoints only",
            "training_weight": "equal total weight per checkpoint",
            "model": "multi-output ridge",
            "ridge_alpha": ridge_alpha,
        },
        "cells": {},
    }
    for cell in sorted(by_cell):
        items = sorted(by_cell[cell], key=lambda item: item.checkpoint_sha256)
        candidate_names = tuple(sorted(items[0].candidates))
        cell_result: dict[str, Any] = {
            "checkpoint_count": len(items),
            "class_effect_dimension": int(items[0].effects.shape[1]),
            "control_feature_dimension": int(items[0].controls.shape[1]),
            "candidates": {},
        }
        for candidate_name in candidate_names:
            folds = []
            for held in items:
                training = [
                    item
                    for item in items
                    if item.checkpoint_sha256 != held.checkpoint_sha256
                ]
                weights = _equal_checkpoint_weights(training)
                train_controls = np.concatenate(
                    [item.controls for item in training], axis=0
                )
                train_candidate = np.concatenate(
                    [item.candidates[candidate_name] for item in training], axis=0
                )
                train_effects = np.concatenate(
                    [item.effects for item in training], axis=0
                )
                augmented_train = np.concatenate(
                    [train_controls, train_candidate], axis=1
                )
                augmented_held = np.concatenate(
                    [held.controls, held.candidates[candidate_name]], axis=1
                )
                control_prediction, control_preprocessing = _fit_predict_ridge(
                    train_controls,
                    train_effects,
                    held.controls,
                    sample_weight=weights,
                    ridge_alpha=ridge_alpha,
                )
                augmented_prediction, augmented_preprocessing = _fit_predict_ridge(
                    augmented_train,
                    train_effects,
                    augmented_held,
                    sample_weight=weights,
                    ridge_alpha=ridge_alpha,
                )
                control_metrics = _channel_metrics(
                    held.effects,
                    control_prediction,
                )
                augmented_metrics = _channel_metrics(
                    held.effects,
                    augmented_prediction,
                )
                control_rank = _metric_mean(
                    control_metrics, "per_channel_spearman"
                )
                augmented_rank = _metric_mean(
                    augmented_metrics, "per_channel_spearman"
                )
                control_cosine = _metric_mean(
                    control_metrics, "per_channel_cosine"
                )
                augmented_cosine = _metric_mean(
                    augmented_metrics, "per_channel_cosine"
                )
                folds.append(
                    {
                        "cell": cell,
                        "candidate": candidate_name,
                        "held_out_checkpoint_sha256": held.checkpoint_sha256,
                        "training_checkpoint_sha256": [
                            item.checkpoint_sha256 for item in training
                        ],
                        "training_cells": sorted({item.cell for item in training}),
                        "training_checkpoint_count": len(training),
                        "training_channel_count": int(train_effects.shape[0]),
                        "held_out_channel_count": int(held.effects.shape[0]),
                        "class_effect_dimension": int(held.effects.shape[1]),
                        "sample_weight": {
                            "total": float(weights.sum()),
                            "per_checkpoint_total": {
                                item.checkpoint_sha256: float(
                                    weights[
                                        sum(
                                            previous.effects.shape[0]
                                            for previous in training[:index]
                                        ) : sum(
                                            previous.effects.shape[0]
                                            for previous in training[: index + 1]
                                        )
                                    ].sum()
                                )
                                for index, item in enumerate(training)
                            },
                        },
                        "controls": {
                            **control_metrics,
                            "preprocessing": control_preprocessing,
                        },
                        "controls_plus_candidate": {
                            **augmented_metrics,
                            "preprocessing": augmented_preprocessing,
                        },
                        "improvement": {
                            "rmse_reduction": float(
                                control_metrics["rmse"]
                                - augmented_metrics["rmse"]
                            ),
                            "mae_reduction": float(
                                control_metrics["mae"]
                                - augmented_metrics["mae"]
                            ),
                            "spearman_gain": (
                                float(augmented_rank - control_rank)
                                if control_rank is not None
                                and augmented_rank is not None
                                else None
                            ),
                            "cosine_gain": (
                                float(augmented_cosine - control_cosine)
                                if control_cosine is not None
                                and augmented_cosine is not None
                                else None
                            ),
                        },
                    }
                )
            cell_result["candidates"][candidate_name] = {
                "folds": folds,
                "checkpoint_weighted_summary": _candidate_summary(folds),
            }
        result["cells"][cell] = cell_result
    return result


# Descriptive alias for callers that prefer the full protocol name.
leave_one_checkpoint_out_prediction = run_within_stratum_loco

