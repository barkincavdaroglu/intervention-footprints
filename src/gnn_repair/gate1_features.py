"""Explicit, auditable feature blocks for Gate 1 topology isolation.

This module keeps exact within-checkpoint similarity features separate from
fixed-dimensional features used by cross-checkpoint prediction.  Kernel
standardization is fitted independently inside each requested channel subset,
so inactive channels cannot influence active-only geometry.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

from gnn_repair.structural import graph_node_features
from gnn_repair.structural_representations import (
    DEFAULT_DIFFUSION_TIMES,
    DEFAULT_WL_MAX_DEPTH,
    impact_diffusion_signatures,
    impact_support_wl_subtree_features,
)


DEFAULT_GATE1_QUANTILES = (0.90, 0.95, 0.99)
DEFAULT_WL_DEPTHS = tuple(range(DEFAULT_WL_MAX_DEPTH + 1))
_HASH_NAMESPACE = "gnn-repair-gate1-features-v1"


@dataclass(frozen=True)
class SubsetKernel:
    """A kernel whose preprocessing was fitted only on ``channel_indices``."""

    channel_indices: np.ndarray
    kernel: np.ndarray
    preprocessing: dict[str, Any]


@dataclass(frozen=True)
class Gate1FeatureBlock:
    """One separated control, candidate, or sensitivity feature block."""

    name: str
    family: str
    raw_features: np.ndarray | sp.csr_matrix
    feature_names: tuple[str, ...]
    prediction_features: np.ndarray
    prediction_feature_names: tuple[str, ...]
    kernels: Mapping[str, SubsetKernel]
    scale: Mapping[str, float | int | str | None]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class Gate1FeatureSet:
    """All feature blocks for one checkpoint."""

    active_mask: np.ndarray
    support_masks: Mapping[str, np.ndarray]
    controls: Mapping[str, Gate1FeatureBlock]
    candidates: Mapping[str, Gate1FeatureBlock]
    sensitivities: Mapping[str, Gate1FeatureBlock]
    metadata: Mapping[str, Any]


def _quantile_name(quantile: float) -> str:
    return f"q{int(round(100.0 * quantile)):02d}"


def _scale_name(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _validate_quantiles(values: Sequence[float]) -> tuple[float, ...]:
    quantiles = tuple(float(value) for value in values)
    if not quantiles or not np.isfinite(quantiles).all():
        raise ValueError("quantiles must contain finite values")
    if any(not 0 < value < 1 for value in quantiles):
        raise ValueError("quantiles must lie strictly inside (0, 1)")
    if any(second <= first for first, second in zip(quantiles, quantiles[1:])):
        raise ValueError("quantiles must be unique and strictly increasing")
    if len({_quantile_name(value) for value in quantiles}) != len(quantiles):
        raise ValueError("quantiles must have distinct percentage names")
    return quantiles


def _validate_depths(values: Iterable[int]) -> tuple[int, ...]:
    depths = tuple(values)
    if not depths:
        raise ValueError("wl_depths cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in depths):
        raise TypeError("wl_depths must contain integers")
    if tuple(sorted(set(depths))) != depths:
        raise ValueError("wl_depths must be unique and increasing")
    if depths[0] < 0 or depths[-1] > DEFAULT_WL_MAX_DEPTH:
        raise ValueError(
            f"wl_depths must lie in [0, {DEFAULT_WL_MAX_DEPTH}]"
        )
    return depths


def _validate_positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_adjacency(adjacency: sp.spmatrix) -> sp.csr_matrix:
    if not sp.issparse(adjacency):
        raise TypeError("adjacency must be a SciPy sparse matrix")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    graph = adjacency.tocsr(copy=True).astype(np.float64)
    if graph.data.size and not np.isfinite(graph.data).all():
        raise ValueError("adjacency contains a non-finite value")
    graph.eliminate_zeros()
    if graph.nnz:
        graph.data = np.ones(graph.nnz, dtype=np.float64)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph.sum_duplicates()
    graph.sort_indices()
    return graph


def _validate_inputs(
    adjacency: sp.spmatrix,
    node_impacts: np.ndarray,
    prediction_changes: np.ndarray,
    scope_mask: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    graph = _canonical_adjacency(adjacency)
    impacts = np.asarray(node_impacts, dtype=np.float64)
    if impacts.ndim != 2 or impacts.shape[1] != graph.shape[0]:
        raise ValueError("node_impacts must have shape [channels, graph nodes]")
    if not np.isfinite(impacts).all() or np.any(impacts < 0):
        raise ValueError("node_impacts must be finite and non-negative")
    changes = np.asarray(prediction_changes)
    if changes.shape != impacts.shape:
        raise ValueError("prediction_changes must align with node_impacts")
    if changes.dtype != np.bool_:
        if not np.all((changes == 0) | (changes == 1)):
            raise ValueError("prediction_changes must contain booleans")
        changes = changes.astype(bool)
    scope = np.asarray(scope_mask)
    if scope.shape != (graph.shape[0],):
        raise ValueError("scope_mask must contain one entry per graph node")
    if scope.dtype != np.bool_:
        if not np.all((scope == 0) | (scope == 1)):
            raise ValueError("scope_mask must contain booleans")
        scope = scope.astype(bool)
    if not scope.any():
        raise ValueError("scope_mask must select at least one node")
    return graph, impacts, changes.astype(bool, copy=False), scope.astype(bool, copy=False)


def _support_masks(
    impacts: np.ndarray,
    changes: np.ndarray,
    scope: np.ndarray,
    *,
    quantiles: tuple[float, ...],
    minimum_impact: float,
) -> dict[str, np.ndarray]:
    scope_indices = np.flatnonzero(scope)
    masks: dict[str, np.ndarray] = {"hard": changes & scope[None, :]}
    for quantile in quantiles:
        count = max(1, int(math.ceil((1.0 - quantile) * scope_indices.size)))
        mask = np.zeros_like(changes, dtype=bool)
        for channel, impact in enumerate(impacts):
            order = np.lexsort((scope_indices, -impact[scope_indices]))
            selected = scope_indices[order[:count]]
            selected = selected[impact[selected] > minimum_impact]
            mask[channel, selected] = True
        masks[_quantile_name(quantile)] = mask
    return masks


def _magnitude_features(
    impacts: np.ndarray,
    changes: np.ndarray,
    scope: np.ndarray,
    *,
    minimum_impact: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    scoped = impacts[:, scope]
    changed = changes[:, scope]
    mass = scoped.sum(axis=1)
    probabilities = np.divide(
        scoped,
        mass[:, None],
        out=np.zeros_like(scoped),
        where=mass[:, None] > 0,
    )
    entropy_terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    entropy_terms[positive] = (
        probabilities[positive] * np.log(probabilities[positive])
    )
    features = np.column_stack(
        [
            mass,
            scoped.mean(axis=1),
            scoped.std(axis=1),
            scoped.max(axis=1),
            np.quantile(scoped, 0.50, axis=1),
            np.quantile(scoped, 0.90, axis=1),
            np.quantile(scoped, 0.99, axis=1),
            np.linalg.norm(scoped, axis=1),
            (scoped > minimum_impact).mean(axis=1),
            changed.mean(axis=1),
            -entropy_terms.sum(axis=1),
            np.square(probabilities).sum(axis=1),
        ]
    )
    names = (
        "total_impact",
        "mean_impact",
        "impact_std",
        "maximum_impact",
        "impact_q50",
        "impact_q90",
        "impact_q99",
        "impact_l2",
        "positive_impact_fraction",
        "prediction_change_fraction",
        "impact_entropy",
        "impact_probability_l2_squared",
    )
    return features, names


def _normalized_raw_field(
    impacts: np.ndarray,
    scope: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    scope_indices = np.flatnonzero(scope)
    scoped = impacts[:, scope_indices]
    mass = scoped.sum(axis=1, keepdims=True)
    normalized = np.divide(
        scoped,
        mass,
        out=np.zeros_like(scoped),
        where=mass > 0,
    )
    return normalized, tuple(f"node_{node}" for node in scope_indices)


def _uniform_parent_roles(
    node_features: np.ndarray,
    node_feature_names: tuple[str, ...],
    masks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...]]:
    width = next(iter(masks.values())).shape[0]
    rows: list[list[float]] = [[] for _ in range(width)]
    names: list[str] = []
    for mask_name in masks:
        names.extend(
            f"{mask_name}_{statistic}_{feature}"
            for statistic in ("mean", "std")
            for feature in node_feature_names
        )
        for channel, mask in enumerate(masks[mask_name]):
            selected = node_features[mask]
            if selected.size:
                rows[channel].extend(selected.mean(axis=0).tolist())
                rows[channel].extend(selected.std(axis=0).tolist())
            else:
                rows[channel].extend([0.0] * (2 * node_features.shape[1]))
    return np.asarray(rows, dtype=np.float64), tuple(names)


def _shape_features(
    adjacency: sp.csr_matrix,
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    degree = np.diff(adjacency.indptr).astype(np.float64)
    total_volume = float(degree.sum())
    rows = []
    for selected_mask in mask:
        nodes = np.flatnonzero(selected_mask)
        if nodes.size == 0:
            rows.append([0.0] * 8)
            continue
        subgraph = adjacency[nodes][:, nodes].tocsr()
        internal_edges = subgraph.nnz / 2.0
        volume = float(degree[nodes].sum())
        boundary = max(0.0, volume - 2.0 * internal_edges)
        complement_volume = max(0.0, total_volume - volume)
        conductance_denominator = min(volume, complement_volume)
        components, labels = csgraph.connected_components(
            subgraph,
            directed=False,
            return_labels=True,
        )
        component_sizes = np.bincount(labels, minlength=components)
        mean_induced_degree = 2.0 * internal_edges / nodes.size
        mean_parent_degree = float(degree[nodes].mean())
        rows.append(
            [
                (
                    2.0 * internal_edges / (nodes.size * (nodes.size - 1))
                    if nodes.size > 1
                    else 0.0
                ),
                components / nodes.size,
                float(component_sizes.max() / nodes.size),
                max(0.0, internal_edges - nodes.size + components) / nodes.size,
                (
                    boundary / conductance_denominator
                    if conductance_denominator > 0
                    else 0.0
                ),
                boundary / volume if volume > 0 else 0.0,
                2.0 * internal_edges / volume if volume > 0 else 0.0,
                (
                    mean_induced_degree / mean_parent_degree
                    if mean_parent_degree > 0
                    else 0.0
                ),
            ]
        )
    names = (
        "induced_density",
        "component_count_per_node",
        "largest_component_fraction",
        "cycle_rank_per_node",
        "conductance",
        "boundary_per_volume",
        "internal_volume_fraction",
        "induced_to_parent_mean_degree",
    )
    return np.asarray(rows, dtype=np.float64), names


def _stable_bucket_sign(token: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.sha256(f"{_HASH_NAMESPACE}|{token}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % dimension
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


def _node_identity_projection(
    values: np.ndarray,
    node_indices: np.ndarray,
    dimension: int,
) -> np.ndarray:
    projected = np.zeros((values.shape[0], dimension), dtype=np.float64)
    for column, node in enumerate(node_indices):
        bucket, sign = _stable_bucket_sign(f"node_identity|{int(node)}", dimension)
        projected[:, bucket] += sign * values[:, column]
    return projected


def _mask_prediction_projection(
    masks: Mapping[str, np.ndarray],
    dimension: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    blocks = []
    names = []
    for mask_name, mask in masks.items():
        node_indices = np.arange(mask.shape[1], dtype=np.int64)
        blocks.append(
            _node_identity_projection(mask.astype(np.float64), node_indices, dimension)
        )
        names.extend(f"{mask_name}_node_hash_{index}" for index in range(dimension))
    return np.concatenate(blocks, axis=1), tuple(names)


def _initial_wl_labels(
    mode: str,
    adjacency: sp.csr_matrix,
    nodes: np.ndarray,
    parent_degree: np.ndarray,
    parent_core: np.ndarray,
) -> list[str]:
    subgraph = adjacency[nodes][:, nodes].tocsr()
    if mode == "constant":
        return ["constant"] * nodes.size
    if mode == "induced_degree_bin":
        values = np.floor(np.log2(np.diff(subgraph.indptr) + 1.0)).astype(int)
        return [f"induced_degree_bin:{value}" for value in values]
    if mode == "parent_degree_core_bin":
        degree_bins = np.floor(np.log2(parent_degree[nodes] + 1.0)).astype(int)
        return [
            f"parent_degree_bin:{degree}|parent_core:{int(core)}"
            for degree, core in zip(degree_bins, parent_core[nodes], strict=True)
        ]
    raise ValueError(f"Unsupported WL label mode: {mode}")


def _signed_hash_wl_projection(
    adjacency: sp.csr_matrix,
    support_mask: np.ndarray,
    *,
    initial_label_mode: str,
    depth: int,
    dimension: int,
    parent_degree: np.ndarray,
    parent_core: np.ndarray,
) -> np.ndarray:
    projected = np.zeros((support_mask.shape[0], dimension), dtype=np.float64)
    for channel, mask in enumerate(support_mask):
        nodes = np.flatnonzero(mask)
        if nodes.size == 0:
            bucket, sign = _stable_bucket_sign("wl|empty_support", dimension)
            projected[channel, bucket] = sign
            continue
        subgraph = adjacency[nodes][:, nodes].tocsr()
        labels = _initial_wl_labels(
            initial_label_mode,
            adjacency,
            nodes,
            parent_degree,
            parent_core,
        )
        for iteration in range(depth + 1):
            for label in labels:
                bucket, sign = _stable_bucket_sign(
                    f"wl|{initial_label_mode}|h{iteration}|{label}",
                    dimension,
                )
                projected[channel, bucket] += sign
            if iteration == depth:
                break
            updated = []
            for node, label in enumerate(labels):
                start, end = subgraph.indptr[node : node + 2]
                neighbors = sorted(labels[index] for index in subgraph.indices[start:end])
                descriptor = label + "|" + "|".join(neighbors)
                updated.append(hashlib.sha256(descriptor.encode("utf-8")).hexdigest())
            labels = updated
        norm = float(np.linalg.norm(projected[channel]))
        if norm > 0:
            projected[channel] /= norm
    return projected


def _standardized_cosine_subset(
    features: np.ndarray | sp.csr_matrix,
    indices: np.ndarray,
) -> SubsetKernel:
    selected = features[indices]
    if sp.issparse(selected):
        matrix = selected.tocsr().astype(np.float64)
        feature_count = matrix.shape[1]
        if indices.size:
            location = np.asarray(matrix.mean(axis=0)).ravel()
            second = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        else:
            location = np.zeros(feature_count, dtype=np.float64)
            second = np.zeros(feature_count, dtype=np.float64)
        variance = np.maximum(second - np.square(location), 0.0)
        valid = variance > 0
        if not indices.size or not valid.any():
            kernel = np.zeros((indices.size, indices.size), dtype=np.float64)
        else:
            scale = np.sqrt(variance[valid])
            scaled = matrix[:, valid].multiply(1.0 / scale).tocsr()
            standardized_location = location[valid] / scale
            cross = np.asarray(scaled.dot(standardized_location)).ravel()
            gram = scaled.dot(scaled.T).toarray()
            gram -= cross[:, None]
            gram -= cross[None, :]
            gram += float(np.dot(standardized_location, standardized_location))
            norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
            denominator = norms[:, None] * norms[None, :]
            kernel = np.divide(
                gram,
                denominator,
                out=np.zeros_like(gram),
                where=denominator > 0,
            )
    else:
        matrix = np.asarray(selected, dtype=np.float64)
        if indices.size:
            location = matrix.mean(axis=0)
            scale = matrix.std(axis=0)
            standardized = (matrix - location) / np.where(scale > 0, scale, 1.0)
            norms = np.linalg.norm(standardized, axis=1)
            normalized = standardized / np.where(norms > 0, norms, 1.0)[:, None]
            kernel = normalized.dot(normalized.T)
            valid = scale > 0
        else:
            valid = np.zeros(matrix.shape[1], dtype=bool)
            kernel = np.empty((0, 0), dtype=np.float64)
    kernel = np.clip(0.5 * (kernel + kernel.T), -1.0, 1.0)
    if kernel.size:
        np.fill_diagonal(kernel, 1.0)
    return SubsetKernel(
        channel_indices=indices.copy(),
        kernel=kernel,
        preprocessing={
            "method": "featurewise z-score then cosine",
            "fit_channel_count": int(indices.size),
            "nonconstant_feature_count": int(np.count_nonzero(valid)),
            "fit_on_this_subset_only": True,
        },
    )


def _jaccard_subset(
    masks: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> SubsetKernel:
    kernels = []
    for mask in masks.values():
        matrix = mask[indices].astype(np.float64)
        intersection = matrix.dot(matrix.T)
        sizes = matrix.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - intersection
        kernels.append(
            np.divide(
                intersection,
                union,
                out=np.ones_like(intersection),
                where=union > 0,
            )
        )
    kernel = (
        np.mean(np.stack(kernels, axis=0), axis=0)
        if kernels
        else np.empty((indices.size, indices.size), dtype=np.float64)
    )
    if kernel.size:
        np.fill_diagonal(kernel, 1.0)
    return SubsetKernel(
        channel_indices=indices.copy(),
        kernel=kernel,
        preprocessing={
            "method": "exact mean Jaccard over hard and quantile masks",
            "fit_channel_count": int(indices.size),
            "fit_on_this_subset_only": True,
        },
    )


def _kernels(
    features: np.ndarray | sp.csr_matrix,
    active_mask: np.ndarray,
) -> dict[str, SubsetKernel]:
    return {
        "all_channels": _standardized_cosine_subset(
            features,
            np.arange(active_mask.size, dtype=np.int64),
        ),
        "active_channels": _standardized_cosine_subset(
            features,
            np.flatnonzero(active_mask).astype(np.int64),
        ),
    }


def _block(
    *,
    name: str,
    family: str,
    raw_features: np.ndarray | sp.csr_matrix,
    feature_names: tuple[str, ...],
    prediction_features: np.ndarray,
    prediction_feature_names: tuple[str, ...],
    active_mask: np.ndarray,
    scale: Mapping[str, float | int | str | None],
    provenance: Mapping[str, Any],
) -> Gate1FeatureBlock:
    return Gate1FeatureBlock(
        name=name,
        family=family,
        raw_features=raw_features,
        feature_names=feature_names,
        prediction_features=np.asarray(prediction_features, dtype=np.float64),
        prediction_feature_names=prediction_feature_names,
        kernels=_kernels(raw_features, active_mask),
        scale=dict(scale),
        provenance=dict(provenance),
    )


def extract_gate1_features(
    adjacency: sp.spmatrix,
    node_impacts: np.ndarray,
    prediction_changes: np.ndarray,
    scope_mask: np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_GATE1_QUANTILES,
    wl_depths: Iterable[int] = DEFAULT_WL_DEPTHS,
    diffusion_times: Sequence[float] = DEFAULT_DIFFUSION_TIMES,
    minimum_impact: float = 1e-12,
    active_threshold: float = 1e-12,
    wl_hash_dimension: int = 512,
    node_projection_dimension: int = 256,
) -> Gate1FeatureSet:
    """Extract separated Gate 1 blocks without using class-effect labels."""

    graph, impacts, changes, scope = _validate_inputs(
        adjacency,
        node_impacts,
        prediction_changes,
        scope_mask,
    )
    quantile_values = _validate_quantiles(quantiles)
    depths = _validate_depths(wl_depths)
    if not np.isfinite(minimum_impact) or minimum_impact < 0:
        raise ValueError("minimum_impact must be finite and non-negative")
    if not np.isfinite(active_threshold) or active_threshold < 0:
        raise ValueError("active_threshold must be finite and non-negative")
    wl_dimension = _validate_positive_dimension(wl_hash_dimension, "wl_hash_dimension")
    node_dimension = _validate_positive_dimension(
        node_projection_dimension,
        "node_projection_dimension",
    )
    active_mask = impacts[:, scope].max(axis=1) > active_threshold
    masks = _support_masks(
        impacts,
        changes,
        scope,
        quantiles=quantile_values,
        minimum_impact=minimum_impact,
    )
    node_features, node_feature_names, node_raw = graph_node_features(graph)
    parent_degree = np.asarray(node_raw["degree"], dtype=np.float64)
    parent_core = np.asarray(node_raw["core_number"], dtype=np.int64)
    scope_indices = np.flatnonzero(scope)

    controls: dict[str, Gate1FeatureBlock] = {}
    magnitude, magnitude_names = _magnitude_features(
        impacts,
        changes,
        scope,
        minimum_impact=minimum_impact,
    )
    controls["magnitude"] = _block(
        name="magnitude",
        family="non_topological_control",
        raw_features=magnitude,
        feature_names=magnitude_names,
        prediction_features=magnitude,
        prediction_feature_names=magnitude_names,
        active_mask=active_mask,
        scale={"support_quantile": None, "wl_depth": None, "diffusion_time": None},
        provenance={
            "block": "impact magnitude summaries",
            "uses_node_identity": False,
            "uses_parent_graph": False,
            "uses_class_labels": False,
        },
    )
    raw_field, raw_field_names = _normalized_raw_field(impacts, scope)
    raw_prediction = _node_identity_projection(
        raw_field,
        scope_indices,
        node_dimension,
    )
    controls["raw_effect_field"] = _block(
        name="raw_effect_field",
        family="non_topological_control",
        raw_features=raw_field,
        feature_names=raw_field_names,
        prediction_features=raw_prediction,
        prediction_feature_names=tuple(
            f"raw_node_hash_{index}" for index in range(node_dimension)
        ),
        active_mask=active_mask,
        scale={"support_quantile": None, "wl_depth": None, "diffusion_time": None},
        provenance={
            "block": "exact L1-normalized scoped node-effect field",
            "node_order": scope_indices.tolist(),
            "prediction_projection": "deterministic signed node-identity hash",
            "prediction_projection_dimension": node_dimension,
            "uses_parent_graph": False,
            "uses_class_labels": False,
        },
    )
    mask_names = tuple(
        f"{mask_name}_node_{node}"
        for mask_name, mask in masks.items()
        for node in range(mask.shape[1])
    )
    mask_raw = np.concatenate(
        [mask.astype(np.float64) for mask in masks.values()],
        axis=1,
    )
    mask_prediction, mask_prediction_names = _mask_prediction_projection(
        masks,
        node_dimension,
    )
    mask_indices = {
        "all_channels": np.arange(active_mask.size, dtype=np.int64),
        "active_channels": np.flatnonzero(active_mask).astype(np.int64),
    }
    controls["mask_jaccard"] = Gate1FeatureBlock(
        name="mask_jaccard",
        family="non_topological_control",
        raw_features=mask_raw,
        feature_names=mask_names,
        prediction_features=mask_prediction,
        prediction_feature_names=mask_prediction_names,
        kernels={
            subset: _jaccard_subset(masks, indices)
            for subset, indices in mask_indices.items()
        },
        scale={"support_quantile": "hard+all", "wl_depth": None, "diffusion_time": None},
        provenance={
            "block": "exact affected-node identity masks",
            "kernel": "mean exact Jaccard",
            "mask_order": list(masks),
            "prediction_projection": "deterministic signed node-identity hash",
            "prediction_projection_dimension_per_mask": node_dimension,
            "uses_parent_graph": False,
            "uses_class_labels": False,
        },
    )
    role_features, role_names = _uniform_parent_roles(
        node_features,
        node_feature_names,
        masks,
    )
    controls["uniform_parent_roles"] = _block(
        name="uniform_parent_roles",
        family="non_topological_control",
        raw_features=role_features,
        feature_names=role_names,
        prediction_features=role_features,
        prediction_feature_names=role_names,
        active_mask=active_mask,
        scale={"support_quantile": "hard+all", "wl_depth": None, "diffusion_time": None},
        provenance={
            "block": "uniform parent-graph node-role moments",
            "pooling": "unweighted mean and standard deviation",
            "uses_induced_edges": False,
            "uses_class_labels": False,
        },
    )

    candidates: dict[str, Gate1FeatureBlock] = {}
    sensitivities: dict[str, Gate1FeatureBlock] = {}
    max_depth = max(depths)
    mode_specs = (
        ("constant", "wl_constant", candidates, True),
        ("induced_degree_bin", "wl_induced_degree", candidates, True),
        (
            "parent_degree_core_bin",
            "wl_parent_degree_core",
            sensitivities,
            False,
        ),
    )
    for quantile in quantile_values:
        quantile_name = _quantile_name(quantile)
        support = masks[quantile_name]
        for mode, prefix, destination, gate_eligible in mode_specs:
            wl = impact_support_wl_subtree_features(
                graph,
                impacts,
                scope,
                support_quantile=quantile,
                minimum_impact=minimum_impact,
                initial_label_mode=mode,
                max_depth=max_depth,
            )
            for depth in depths:
                keep = [
                    index
                    for index, feature_name in enumerate(wl.feature_names)
                    if feature_name == "empty_support"
                    or int(feature_name.split("_", 2)[1][1:]) <= depth
                ]
                raw = wl.by_depth[depth][:, keep].tocsr()
                names = tuple(wl.feature_names[index] for index in keep)
                prediction = _signed_hash_wl_projection(
                    graph,
                    support,
                    initial_label_mode=mode,
                    depth=depth,
                    dimension=wl_dimension,
                    parent_degree=parent_degree,
                    parent_core=parent_core,
                )
                name = f"{prefix}_{quantile_name}_h{depth}"
                destination[name] = _block(
                    name=name,
                    family=(
                        "topology_wl"
                        if gate_eligible
                        else "topology_plus_parent_role_sensitivity"
                    ),
                    raw_features=raw,
                    feature_names=names,
                    prediction_features=prediction,
                    prediction_feature_names=tuple(
                        f"signed_wl_hash_{index}" for index in range(wl_dimension)
                    ),
                    active_mask=active_mask,
                    scale={
                        "support_quantile": quantile,
                        "wl_depth": depth,
                        "diffusion_time": None,
                    },
                    provenance={
                        "block": wl.block_name,
                        "initial_label_mode": mode,
                        "uses_impact_values_for_support": True,
                        "uses_impact_values_as_node_labels": False,
                        "uses_parent_graph_node_roles": not gate_eligible,
                        "eligible_for_topology_gate": gate_eligible,
                        "prediction_projection": "deterministic signed-hash WL",
                        "prediction_projection_dimension": wl_dimension,
                        "uses_class_labels": False,
                    },
                )

        shape, shape_names = _shape_features(graph, support)
        shape_name = f"shape_{quantile_name}"
        candidates[shape_name] = _block(
            name=shape_name,
            family="topology_shape",
            raw_features=shape,
            feature_names=shape_names,
            prediction_features=shape,
            prediction_feature_names=shape_names,
            active_mask=active_mask,
            scale={
                "support_quantile": quantile,
                "wl_depth": None,
                "diffusion_time": None,
            },
            provenance={
                "block": "induced-support shape",
                "uses_impact_values_for_support": True,
                "uses_impact_values_as_features": False,
                "eligible_for_topology_gate": True,
                "uses_class_labels": False,
            },
        )

    diffusion = impact_diffusion_signatures(
        graph,
        impacts,
        scope,
        times=diffusion_times,
        minimum_mass=minimum_impact,
    )
    time_values = tuple(float(value) for value in diffusion.metadata["times"])
    width_per_time = diffusion.signatures.shape[1] // len(time_values)
    for time_index, time in enumerate(time_values):
        start = time_index * width_per_time
        stop = start + width_per_time
        name = f"diffusion_t{_scale_name(time)}"
        raw = diffusion.signatures[:, start:stop]
        feature_names = diffusion.feature_names[start:stop]
        candidates[name] = _block(
            name=name,
            family="topology_conditioned_effect_diffusion",
            raw_features=raw,
            feature_names=feature_names,
            prediction_features=raw,
            prediction_feature_names=feature_names,
            active_mask=active_mask,
            scale={
                "support_quantile": None,
                "wl_depth": None,
                "diffusion_time": time,
            },
            provenance={
                "block": diffusion.block_names["signatures"],
                "normalization": "input and each diffused field L1",
                "retains_node_effect_distribution": True,
                "eligible_as_strict_topology_only": False,
                "uses_class_labels": False,
            },
        )

    return Gate1FeatureSet(
        active_mask=active_mask,
        support_masks=masks,
        controls=controls,
        candidates=candidates,
        sensitivities=sensitivities,
        metadata={
            "quantiles": list(quantile_values),
            "wl_depths": list(depths),
            "diffusion_times": list(time_values),
            "minimum_impact": minimum_impact,
            "active_definition": "maximum scoped impact > active_threshold",
            "active_threshold": active_threshold,
            "wl_prediction_dimension": wl_dimension,
            "node_identity_projection_dimension": node_dimension,
            "hash_namespace": _HASH_NAMESPACE,
            "class_labels_used": False,
        },
    )

