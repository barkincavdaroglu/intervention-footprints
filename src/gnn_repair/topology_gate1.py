"""Gate 1: isolate topology in intervention-induced GNN channel effects.

The structural development study deliberately fused effect magnitude, node
identity, parent-graph roles, and induced-subgraph topology.  This module keeps
those sources separate.  It compares six within-checkpoint similarity views,
uses simultaneous row/column neuron relabeling for inference, and supports a
simple degree-preserving double-edge-swap null with impact fields held fixed.

This is an exploratory, development-only protocol.  It rejects confirmatory
artifact roots and never uses test labels or test-set performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
import torch
import yaml
from scipy.sparse import csgraph
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from gnn_repair.checkpoints import (
    LoadedRun,
    load_trained_run,
    model_state_sha256,
    source_code_sha256,
    split_indices_sha256,
)
from gnn_repair.protocol import canonical_payload_sha256
from gnn_repair.reproducibility import set_deterministic
from gnn_repair.structural import (
    DEFAULT_IMPACT_QUANTILES,
    canonical_edge_sha256,
    canonical_undirected_adjacency,
    collect_neuron_impacts,
    graph_node_features,
    structural_signatures,
)
from gnn_repair.structural_representations import (
    DEFAULT_DIFFUSION_TIMES,
    impact_diffusion_signatures,
    impact_support_wl_subtree_features,
    multiscale_wl_subtree_kernel,
)


GATE1_PROTOCOL_VERSION = 1
GATE1_CODE_FILES = (
    "topology_gate1.py",
    "structural.py",
    "structural_representations.py",
    "model.py",
    "data.py",
    "checkpoints.py",
    "protocol.py",
    "reproducibility.py",
)
VIEW_NAMES = (
    "magnitude",
    "raw_effect_field",
    "mask_jaccard",
    "uniform_node_role",
    "topology_only",
    "topology_wl_induced_degree",
    "topology_wl_parent_degree",
    "topology_wl_parent_degree_core",
    "topology_shape_only",
    "impact_diffusion",
    "full_signature",
)
CONTROL_VIEW_NAMES = (
    "magnitude",
    "raw_effect_field",
    "mask_jaccard",
    "uniform_node_role",
)
SUBSETS = ("all_channels", "active_channels")
PRIMARY_EFFECT_METRIC = "validation_class_kl"
EFFECT_METRIC_NAMES = (
    PRIMARY_EFFECT_METRIC,
    "validation_signed_class_ce_delta",
)
QUANTILE_SENSITIVITY_VIEWS = (
    "topology_only",
    "topology_wl_induced_degree",
    "topology_wl_parent_degree",
    "topology_wl_parent_degree_core",
    "topology_shape_only",
)
GATE_CANDIDATE_NAMES = (
    "topology_only",
    "impact_diffusion",
)
DEFAULT_EDGE_DROPOUT_RATES = (0.01, 0.05, 0.10)


def _stable_seed(*parts: object) -> int:
    material = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Feature values must be a matrix")
    location = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    return (values - location) / np.where(scale > 0, scale, 1.0)


def _cosine_kernel(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Cosine-kernel values must be a matrix")
    norms = np.linalg.norm(values, axis=1)
    normalized = values / np.where(norms > 0, norms, 1.0)[:, None]
    kernel = normalized @ normalized.T
    np.fill_diagonal(kernel, 1.0)
    return np.clip(kernel, -1.0, 1.0)


def _validate_kernel(kernel: np.ndarray, width: int, name: str) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.shape != (width, width):
        raise ValueError(f"{name} kernel must have shape ({width}, {width})")
    if not np.isfinite(kernel).all():
        raise ValueError(f"{name} kernel contains non-finite values")
    if not np.allclose(kernel, kernel.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} kernel is not symmetric")
    return kernel


def _quantile_masks(
    node_impacts: np.ndarray,
    prediction_changes: np.ndarray,
    scope_mask: np.ndarray,
    *,
    quantiles: tuple[float, ...],
    minimum_impact: float,
) -> dict[str, np.ndarray]:
    """Create deterministic hard and top-impact support masks."""

    node_impacts = np.asarray(node_impacts)
    prediction_changes = np.asarray(prediction_changes, dtype=bool)
    scope_mask = np.asarray(scope_mask, dtype=bool)
    if node_impacts.ndim != 2 or prediction_changes.shape != node_impacts.shape:
        raise ValueError("Impact and prediction-change arrays must align")
    if scope_mask.shape != (node_impacts.shape[1],) or not scope_mask.any():
        raise ValueError("scope_mask must select at least one node")
    if tuple(sorted(set(quantiles))) != quantiles or any(
        not 0 < value < 1 for value in quantiles
    ):
        raise ValueError("Quantiles must be unique, increasing, and inside (0, 1)")

    width, num_nodes = node_impacts.shape
    scope_indices = np.flatnonzero(scope_mask)
    masks: dict[str, np.ndarray] = {
        "hard": prediction_changes & scope_mask[None, :]
    }
    for quantile in quantiles:
        count = max(1, int(math.ceil((1.0 - quantile) * scope_indices.size)))
        values = np.zeros((width, num_nodes), dtype=bool)
        for neuron, impact in enumerate(node_impacts):
            scoped_values = impact[scope_indices]
            order = np.lexsort((scope_indices, -scoped_values))
            selected = scope_indices[order[:count]]
            selected = selected[impact[selected] > minimum_impact]
            values[neuron, selected] = True
        masks[f"q{int(round(quantile * 100)):02d}"] = values
    return masks


def _magnitude_features(
    node_impacts: np.ndarray,
    prediction_changes: np.ndarray,
    scope_mask: np.ndarray,
    *,
    minimum_impact: float,
) -> np.ndarray:
    scoped = np.asarray(node_impacts[:, scope_mask], dtype=np.float64)
    changed = np.asarray(prediction_changes[:, scope_mask], dtype=bool)
    mass = scoped.sum(axis=1)
    probabilities = scoped / np.where(mass > 0, mass, 1.0)[:, None]
    entropy = -np.sum(
        np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0),
        axis=1,
    )
    inverse_participation = np.sum(np.square(probabilities), axis=1)
    return np.column_stack(
        [
            scoped.mean(axis=1),
            scoped.std(axis=1),
            scoped.max(axis=1),
            np.quantile(scoped, 0.50, axis=1),
            np.quantile(scoped, 0.90, axis=1),
            np.quantile(scoped, 0.99, axis=1),
            (scoped > minimum_impact).mean(axis=1),
            changed.mean(axis=1),
            mass,
            np.linalg.norm(scoped, axis=1),
            entropy,
            inverse_participation,
        ]
    )


def _jaccard_kernel(masks: dict[str, np.ndarray]) -> np.ndarray:
    kernels = []
    for name in sorted(masks):
        matrix = masks[name].astype(np.float64)
        intersection = matrix @ matrix.T
        sizes = matrix.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - intersection
        # Empty/empty supports are identical as sets.  Active-only results are
        # always reported separately so inactive ties remain visible.
        kernels.append(
            np.divide(
                intersection,
                union,
                out=np.ones_like(intersection),
                where=union > 0,
            )
        )
    kernel = np.mean(kernels, axis=0)
    np.fill_diagonal(kernel, 1.0)
    return kernel


def _uniform_role_features(
    node_features: np.ndarray,
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    width = next(iter(masks.values())).shape[0]
    rows: list[list[float]] = [[] for _ in range(width)]
    for name in sorted(masks):
        view = masks[name]
        for neuron in range(width):
            selected = node_features[view[neuron]]
            if selected.size:
                role_mean = selected.mean(axis=0)
                role_std = selected.std(axis=0)
            else:
                role_mean = np.zeros(node_features.shape[1], dtype=np.float64)
                role_std = np.zeros(node_features.shape[1], dtype=np.float64)
            rows[neuron].extend(role_mean.tolist())
            rows[neuron].extend(role_std.tolist())
    return np.asarray(rows, dtype=np.float64)


def _topology_shape_features(
    adjacency: sp.csr_matrix,
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
    total_volume = float(degree.sum())
    width = next(iter(masks.values())).shape[0]
    rows: list[list[float]] = [[] for _ in range(width)]
    for name in sorted(masks):
        for neuron, mask in enumerate(masks[name]):
            nodes = np.flatnonzero(mask)
            if nodes.size == 0:
                rows[neuron].extend([0.0] * 8)
                continue
            subgraph = adjacency[nodes][:, nodes].tocsr()
            internal_edges = subgraph.nnz / 2.0
            volume = float(degree[nodes].sum())
            boundary = max(0.0, volume - 2.0 * internal_edges)
            complement_volume = max(0.0, total_volume - volume)
            conductance_denominator = min(volume, complement_volume)
            conductance = (
                boundary / conductance_denominator
                if conductance_denominator > 0
                else 0.0
            )
            density = (
                2.0 * internal_edges / (nodes.size * (nodes.size - 1))
                if nodes.size > 1
                else 0.0
            )
            components, labels = csgraph.connected_components(
                subgraph,
                directed=False,
                return_labels=True,
            )
            component_sizes = np.bincount(labels, minlength=components)
            largest_fraction = float(component_sizes.max() / nodes.size)
            cycle_rank = max(0.0, internal_edges - nodes.size + components)
            mean_induced_degree = 2.0 * internal_edges / nodes.size
            mean_parent_degree = float(degree[nodes].mean())
            rows[neuron].extend(
                [
                    density,
                    components / nodes.size,
                    largest_fraction,
                    cycle_rank / nodes.size,
                    conductance,
                    boundary / volume if volume > 0 else 0.0,
                    2.0 * internal_edges / volume if volume > 0 else 0.0,
                    mean_induced_degree / mean_parent_degree
                    if mean_parent_degree > 0
                    else 0.0,
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def _multiquantile_wl_kernel(
    adjacency: sp.csr_matrix,
    node_impacts: np.ndarray,
    scope_mask: np.ndarray,
    *,
    quantiles: tuple[float, ...],
    minimum_impact: float,
    initial_label_mode: str,
    depth: int,
) -> np.ndarray:
    """Average the canonical normalized WL kernel over support quantiles."""

    kernels = []
    for quantile in quantiles:
        features = impact_support_wl_subtree_features(
            adjacency,
            node_impacts,
            scope_mask,
            support_quantile=quantile,
            minimum_impact=minimum_impact,
            initial_label_mode=initial_label_mode,
            max_depth=depth,
        )
        kernels.append(multiscale_wl_subtree_kernel(features))
    kernel = np.mean(np.stack(kernels, axis=0), axis=0)
    np.fill_diagonal(kernel, 1.0)
    return kernel


def build_view_kernels(
    adjacency: sp.csr_matrix,
    node_impacts: np.ndarray,
    prediction_changes: np.ndarray,
    scope_mask: np.ndarray,
    *,
    quantiles: tuple[float, ...] = DEFAULT_IMPACT_QUANTILES,
    minimum_impact: float = 1e-12,
    wl_depth: int = 3,
    diffusion_times: tuple[float, ...] = DEFAULT_DIFFUSION_TIMES,
    requested_views: Iterable[str] = VIEW_NAMES,
    support_masks: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build separated Gate-1 kernels without using class labels."""

    adjacency = adjacency.tocsr(copy=True).astype(np.float64)
    adjacency.sort_indices()
    node_impacts = np.asarray(node_impacts, dtype=np.float64)
    prediction_changes = np.asarray(prediction_changes, dtype=bool)
    scope_mask = np.asarray(scope_mask, dtype=bool)
    if node_impacts.ndim != 2 or prediction_changes.shape != node_impacts.shape:
        raise ValueError("Impact and prediction-change matrices must align")
    if node_impacts.shape[1] != adjacency.shape[0]:
        raise ValueError("Impact fields and adjacency must have the same nodes")
    width = node_impacts.shape[0]
    requested = tuple(dict.fromkeys(str(name) for name in requested_views))
    unknown = sorted(set(requested) - set(VIEW_NAMES))
    if unknown:
        raise ValueError(f"Unknown Gate-1 views: {unknown}")
    masks = support_masks or _quantile_masks(
        node_impacts,
        prediction_changes,
        scope_mask,
        quantiles=quantiles,
        minimum_impact=minimum_impact,
    )
    for name, mask in masks.items():
        if np.asarray(mask).shape != node_impacts.shape:
            raise ValueError(f"Support mask {name!r} does not align with impacts")

    kernels: dict[str, np.ndarray] = {}
    if "magnitude" in requested:
        magnitude = _magnitude_features(
            node_impacts,
            prediction_changes,
            scope_mask,
            minimum_impact=minimum_impact,
        )
        kernels["magnitude"] = _cosine_kernel(_standardize(magnitude))
    if "raw_effect_field" in requested:
        scoped = np.asarray(node_impacts[:, scope_mask], dtype=np.float64)
        mass = scoped.sum(axis=1, keepdims=True)
        normalized = np.divide(
            scoped,
            mass,
            out=np.zeros_like(scoped),
            where=mass > 0,
        )
        kernels["raw_effect_field"] = _cosine_kernel(normalized)
    if "mask_jaccard" in requested:
        kernels["mask_jaccard"] = _jaccard_kernel(masks)

    need_roles = bool({"uniform_node_role", "full_signature"} & set(requested))
    node_features: np.ndarray | None = None
    node_feature_names: tuple[str, ...] | None = None
    if need_roles:
        node_features, node_feature_names, _ = graph_node_features(adjacency)
    if "uniform_node_role" in requested:
        assert node_features is not None
        role_features = _uniform_role_features(node_features, masks)
        kernels["uniform_node_role"] = _cosine_kernel(_standardize(role_features))
    wl_modes = {
        "topology_only": "constant",
        "topology_wl_induced_degree": "induced_degree_bin",
        "topology_wl_parent_degree": "parent_degree_bin",
        "topology_wl_parent_degree_core": "parent_degree_core_bin",
    }
    for view_name, initial_label_mode in wl_modes.items():
        if view_name not in requested:
            continue
        kernels[view_name] = _multiquantile_wl_kernel(
            adjacency,
            node_impacts,
            scope_mask,
            quantiles=quantiles,
            minimum_impact=minimum_impact,
            initial_label_mode=initial_label_mode,
            depth=wl_depth,
        )
    if "topology_shape_only" in requested:
        shape = _topology_shape_features(adjacency, masks)
        kernels["topology_shape_only"] = _cosine_kernel(_standardize(shape))
    if "impact_diffusion" in requested:
        diffusion = impact_diffusion_signatures(
            adjacency,
            node_impacts,
            scope_mask,
            times=diffusion_times,
            minimum_mass=minimum_impact,
        )
        kernels["impact_diffusion"] = _cosine_kernel(
            _standardize(diffusion.signatures)
        )
    if "full_signature" in requested:
        assert node_features is not None and node_feature_names is not None
        signature = structural_signatures(
            adjacency,
            node_features,
            node_feature_names,
            node_impacts,
            prediction_changes,
            scope_mask,
            quantiles=quantiles,
            minimum_impact=minimum_impact,
        )
        kernels["full_signature"] = _cosine_kernel(signature["standardized"])

    for name, kernel in kernels.items():
        kernels[name] = _validate_kernel(kernel, width, name)
    return kernels, masks


def _off_diagonal_values(kernel: np.ndarray) -> np.ndarray:
    return kernel[np.triu_indices(kernel.shape[0], k=1)]


def _spearman_alignment(first: np.ndarray, second: np.ndarray) -> float | None:
    first_values = _off_diagonal_values(first)
    second_values = _off_diagonal_values(second)
    if (
        first_values.size < 2
        or np.ptp(first_values) == 0
        or np.ptp(second_values) == 0
    ):
        return None
    value = float(spearmanr(first_values, second_values).statistic)
    return value if math.isfinite(value) else None


def neuron_label_permutation_test(
    view_kernel: np.ndarray,
    effect_kernel: np.ndarray,
    *,
    repetitions: int,
    random_seed: int,
    channel_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Infer kernel alignment by relabeling neurons within one checkpoint.

    Each null draw applies one permutation to both rows and columns of the class
    effect kernel.  Pairwise neuron observations are therefore never treated as
    independent samples.
    """

    if repetitions < 0:
        raise ValueError("Permutation repetitions must be non-negative")
    width = view_kernel.shape[0]
    view_kernel = _validate_kernel(view_kernel, width, "view")
    effect_kernel = _validate_kernel(effect_kernel, width, "effect")
    indices = (
        np.arange(width, dtype=np.int64)
        if channel_indices is None
        else np.asarray(channel_indices, dtype=np.int64)
    )
    if indices.ndim != 1 or np.unique(indices).size != indices.size:
        raise ValueError("channel_indices must be a unique vector")
    if indices.size and (indices.min() < 0 or indices.max() >= width):
        raise ValueError("channel_indices contains an invalid channel")
    first = view_kernel[np.ix_(indices, indices)]
    second = effect_kernel[np.ix_(indices, indices)]
    observed = _spearman_alignment(first, second)
    if observed is None or repetitions == 0:
        return {
            "statistic": "spearman_upper_triangle",
            "observed": observed,
            "permutation_scheme": (
                "simultaneous row/column relabeling of the class-effect kernel "
                "within one checkpoint"
            ),
            "repetitions": repetitions,
            "random_seed": random_seed,
            "p_greater": None,
            "p_two_sided": None,
            "null_mean": None,
            "null_std": None,
            "null_values": [],
        }
    generator = np.random.default_rng(random_seed)
    null_values = []
    for _ in range(repetitions):
        permutation = generator.permutation(indices.size)
        permuted = second[np.ix_(permutation, permutation)]
        value = _spearman_alignment(first, permuted)
        null_values.append(0.0 if value is None else value)
    null = np.asarray(null_values, dtype=np.float64)
    return {
        "statistic": "spearman_upper_triangle",
        "observed": observed,
        "permutation_scheme": (
            "simultaneous row/column relabeling of the class-effect kernel "
            "within one checkpoint"
        ),
        "repetitions": repetitions,
        "random_seed": random_seed,
        "p_greater": float((1 + np.count_nonzero(null >= observed)) / (repetitions + 1)),
        "p_two_sided": float(
            (1 + np.count_nonzero(np.abs(null) >= abs(observed)))
            / (repetitions + 1)
        ),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_values": null.tolist(),
    }


def degree_preserving_rewire(
    adjacency: sp.csr_matrix,
    *,
    swaps: int,
    random_seed: int,
    max_attempts: int | None = None,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Rewire a simple graph using deterministic undirected double-edge swaps.

    The degree sequence, node identities, and edge count are exact invariants.
    Self-loops and parallel edges are forbidden.  Connectivity is deliberately
    *not* constrained; this is a degree-preserving topology-destruction null.
    """

    adjacency = adjacency.tocsr(copy=True).astype(np.float64)
    adjacency.sort_indices()
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if swaps < 0:
        raise ValueError("swaps must be non-negative")
    if adjacency.diagonal().any() or (adjacency != adjacency.T).nnz:
        raise ValueError("adjacency must be simple, loop-free, and undirected")
    upper = sp.triu(adjacency, k=1).tocoo()
    edges = [
        (int(min(left, right)), int(max(left, right)))
        for left, right in zip(upper.row, upper.col, strict=True)
    ]
    edge_set = set(edges)
    target = int(swaps)
    attempts_limit = (
        max(int(max_attempts), target)
        if max_attempts is not None
        else max(100, target * 20)
    )
    generator = np.random.default_rng(random_seed)
    successful = 0
    attempts = 0
    while successful < target and attempts < attempts_limit and len(edges) >= 2:
        attempts += 1
        first_index, second_index = generator.choice(len(edges), size=2, replace=False)
        first = edges[int(first_index)]
        second = edges[int(second_index)]
        a, b = first
        c, d = second
        if len({a, b, c, d}) < 4:
            continue
        if int(generator.integers(0, 2)) == 0:
            candidates = ((a, c), (b, d))
        else:
            candidates = ((a, d), (b, c))
        new_first = tuple(sorted(candidates[0]))
        new_second = tuple(sorted(candidates[1]))
        if new_first == new_second or new_first in edge_set or new_second in edge_set:
            continue
        old_pair = {first, second}
        new_pair = {new_first, new_second}
        if old_pair == new_pair:
            continue
        edge_set.remove(first)
        edge_set.remove(second)
        edge_set.add(new_first)
        edge_set.add(new_second)
        edges[int(first_index)] = new_first
        edges[int(second_index)] = new_second
        successful += 1

    ordered = sorted(edge_set)
    if ordered:
        pairs = np.asarray(ordered, dtype=np.int64)
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        columns = np.concatenate([pairs[:, 1], pairs[:, 0]])
        rewired = sp.csr_matrix(
            (np.ones(rows.size), (rows, columns)),
            shape=adjacency.shape,
            dtype=np.float64,
        )
    else:
        rewired = sp.csr_matrix(adjacency.shape, dtype=np.float64)
    rewired.sort_indices()
    original_degree = np.diff(adjacency.indptr)
    rewired_degree = np.diff(rewired.indptr)
    if not np.array_equal(original_degree, rewired_degree):
        raise RuntimeError("Degree-preserving rewiring changed the degree sequence")
    metadata = {
        "method": "simple_undirected_double_edge_swap",
        "random_seed": random_seed,
        "requested_swaps": target,
        "successful_swaps": successful,
        "attempts": attempts,
        "max_attempts": attempts_limit,
        "edge_count": len(ordered),
        "degree_sequence_preserved": True,
        "node_identities_preserved": True,
        "self_loops_allowed": False,
        "parallel_edges_allowed": False,
        "connectivity_preserved": False,
        "canonical_edge_sha256": canonical_edge_sha256(rewired),
    }
    return rewired, metadata


def _subset_indices(active_channels: np.ndarray, subset: str) -> np.ndarray:
    if subset == "all_channels":
        return np.arange(active_channels.size, dtype=np.int64)
    if subset == "active_channels":
        return np.flatnonzero(active_channels).astype(np.int64)
    raise ValueError(f"Unknown channel subset: {subset!r}")


def _rewiring_null(
    adjacency: sp.csr_matrix,
    node_impacts: np.ndarray,
    prediction_changes: np.ndarray,
    scope_mask: np.ndarray,
    support_masks: dict[str, np.ndarray],
    observed_kernels: dict[str, np.ndarray],
    effect_kernel: np.ndarray,
    active_channels: np.ndarray,
    *,
    quantiles: tuple[float, ...],
    minimum_impact: float,
    wl_depth: int,
    diffusion_times: tuple[float, ...],
    view_names: tuple[str, ...],
    repetitions: int,
    swaps_per_edge: float,
    max_attempt_multiplier: int,
    random_seed: int,
) -> dict[str, Any]:
    if repetitions < 0:
        raise ValueError("graph_rewires must be non-negative")
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be non-negative")
    if max_attempt_multiplier < 1:
        raise ValueError("max_swap_attempt_multiplier must be positive")
    unknown = sorted(set(view_names) - set(VIEW_NAMES))
    if unknown:
        raise ValueError(f"Unknown rewiring views: {unknown}")
    edge_count = adjacency.nnz // 2
    requested_swaps = int(math.ceil(edge_count * swaps_per_edge))
    null_values: dict[str, dict[str, list[float]]] = {
        name: {subset: [] for subset in SUBSETS} for name in view_names
    }
    replicate_metadata = []
    for repeat in range(repetitions):
        seed = _stable_seed(random_seed, "rewire", repeat)
        rewired, metadata = degree_preserving_rewire(
            adjacency,
            swaps=requested_swaps,
            random_seed=seed,
            max_attempts=max(100, requested_swaps * max_attempt_multiplier),
        )
        complete = metadata["successful_swaps"] == metadata["requested_swaps"]
        metadata["valid_for_null"] = complete
        metadata["exclusion_reason"] = (
            None if complete else "requested degree-preserving swaps were incomplete"
        )
        replicate_metadata.append(metadata)
        if not complete:
            continue
        rewired_kernels, _ = build_view_kernels(
            rewired,
            node_impacts,
            prediction_changes,
            scope_mask,
            quantiles=quantiles,
            minimum_impact=minimum_impact,
            wl_depth=wl_depth,
            diffusion_times=diffusion_times,
            requested_views=view_names,
            support_masks=support_masks,
        )
        for view_name in view_names:
            for subset in SUBSETS:
                indices = _subset_indices(active_channels, subset)
                if indices.size < 3:
                    continue
                view = rewired_kernels[view_name][np.ix_(indices, indices)]
                effect = effect_kernel[np.ix_(indices, indices)]
                value = _spearman_alignment(view, effect)
                if value is not None:
                    null_values[view_name][subset].append(value)

    result: dict[str, Any] = {
        "null": "degree-preserving graph rewiring with impact fields and masks fixed",
        "rewiring_specification": {
            "method": "simple undirected double-edge swaps",
            "degree_sequence": "preserved exactly",
            "node_identities": "preserved",
            "impact_fields": "fixed",
            "support_masks": "fixed",
            "self_loops": "forbidden",
            "parallel_edges": "forbidden",
            "connectivity": "not constrained",
            "edge_count": int(edge_count),
            "swaps_per_edge": swaps_per_edge,
            "requested_swaps_per_replicate": requested_swaps,
            "max_attempt_multiplier": max_attempt_multiplier,
        },
        "repetitions": repetitions,
        "valid_repetitions": int(
            sum(bool(item["valid_for_null"]) for item in replicate_metadata)
        ),
        "incomplete_repetitions": int(
            sum(not bool(item["valid_for_null"]) for item in replicate_metadata)
        ),
        "random_seed": random_seed,
        "replicates": replicate_metadata,
        "views": {},
    }
    for view_name in view_names:
        result["views"][view_name] = {}
        for subset in SUBSETS:
            indices = _subset_indices(active_channels, subset)
            observed = (
                _spearman_alignment(
                    observed_kernels[view_name][np.ix_(indices, indices)],
                    effect_kernel[np.ix_(indices, indices)],
                )
                if indices.size >= 3
                else None
            )
            values = np.asarray(null_values[view_name][subset], dtype=np.float64)
            result["views"][view_name][subset] = {
                "observed": observed,
                "null_values": values.tolist(),
                "null_mean": float(values.mean()) if values.size else None,
                "null_std": float(values.std()) if values.size else None,
                "observed_minus_null_mean": (
                    float(observed - values.mean())
                    if observed is not None and values.size
                    else None
                ),
                "p_greater": (
                    float((1 + np.count_nonzero(values >= observed)) / (values.size + 1))
                    if observed is not None and values.size
                    else None
                ),
            }
    return result


def analyze_loaded_run(
    loaded: LoadedRun,
    *,
    layer: int,
    gate_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate all topology-isolation views for one frozen checkpoint."""

    seed = int(loaded.result["seed"])
    set_deterministic(seed, num_threads=int(gate_config.get("num_threads", 1)))
    impact_scope = str(gate_config.get("impact_scope", "train_validation"))
    if impact_scope not in {"train", "validation", "train_validation"}:
        raise ValueError(
            "Gate 1 forbids all-node impact scope; use train, validation, or "
            "train_validation"
        )
    quantiles = tuple(
        float(value)
        for value in gate_config.get("impact_quantiles", DEFAULT_IMPACT_QUANTILES)
    )
    minimum_impact = float(gate_config.get("minimum_impact", 1e-12))
    active_minimum_mass = float(
        gate_config.get("active_minimum_mass", minimum_impact)
    )
    wl_depth = int(gate_config.get("wl_depth", 3))
    diffusion_times = tuple(
        float(value)
        for value in gate_config.get("diffusion_times", DEFAULT_DIFFUSION_TIMES)
    )
    permutation_repetitions = int(gate_config.get("neuron_permutations", 1000))
    rewiring_repetitions = int(gate_config.get("graph_rewires", 20))
    rewiring_views = tuple(
        str(value)
        for value in gate_config.get(
            "rewiring_views", ["topology_only", "impact_diffusion"]
        )
    )

    adjacency = canonical_undirected_adjacency(
        loaded.data.edge_index,
        int(loaded.data.y.numel()),
    )
    impacts = collect_neuron_impacts(
        loaded,
        layer=layer,
        impact_scope=impact_scope,
    )
    node_impacts = np.asarray(impacts["node_impacts"], dtype=np.float32)
    prediction_changes = np.asarray(impacts["prediction_changes"], dtype=bool)
    scope_mask = np.asarray(impacts["scope_mask"], dtype=bool)
    kernels, support_masks = build_view_kernels(
        adjacency,
        node_impacts,
        prediction_changes,
        scope_mask,
        quantiles=quantiles,
        minimum_impact=minimum_impact,
        wl_depth=wl_depth,
        diffusion_times=diffusion_times,
    )
    class_effects = np.asarray(impacts["class_kl"], dtype=np.float64)
    effect_kernel = _cosine_kernel(class_effects)
    diffusion = impact_diffusion_signatures(
        adjacency,
        node_impacts,
        scope_mask,
        times=diffusion_times,
        minimum_mass=minimum_impact,
    )
    active_channels = (
        node_impacts[:, scope_mask].max(axis=1) > active_minimum_mass
    )
    checkpoint_sha = model_state_sha256(loaded.checkpoint)
    inference: dict[str, Any] = {}
    for view_name in VIEW_NAMES:
        inference[view_name] = {}
        for subset in SUBSETS:
            indices = _subset_indices(active_channels, subset)
            permutation_seed = _stable_seed(
                "gate1-neuron-permutation",
                loaded.result["experiment"],
                checkpoint_sha,
                layer,
                view_name,
                subset,
            )
            inference[view_name][subset] = neuron_label_permutation_test(
                kernels[view_name],
                effect_kernel,
                repetitions=permutation_repetitions,
                random_seed=permutation_seed,
                channel_indices=indices,
            )

    rewiring_seed = _stable_seed(
        "gate1-graph-rewiring",
        loaded.result["experiment"],
        checkpoint_sha,
        layer,
    )
    rewiring = _rewiring_null(
        adjacency,
        node_impacts,
        prediction_changes,
        scope_mask,
        support_masks,
        kernels,
        effect_kernel,
        active_channels,
        quantiles=quantiles,
        minimum_impact=minimum_impact,
        wl_depth=wl_depth,
        diffusion_times=diffusion_times,
        view_names=rewiring_views,
        repetitions=rewiring_repetitions,
        swaps_per_edge=float(gate_config.get("swaps_per_edge", 1.0)),
        max_attempt_multiplier=int(
            gate_config.get("max_swap_attempt_multiplier", 20)
        ),
        random_seed=rewiring_seed,
    )

    support_names = tuple(sorted(support_masks))
    arrays = {
        "node_impacts": node_impacts,
        "prediction_changes": prediction_changes,
        "scope_mask": scope_mask,
        "class_kl": class_effects,
        "class_ce_delta": np.asarray(impacts["class_ce_delta"], dtype=np.float64),
        "active_channels": active_channels,
        "support_names": np.asarray(support_names),
        "support_masks": np.stack(
            [support_masks[name] for name in support_names], axis=0
        ),
        "view_names": np.asarray(VIEW_NAMES),
        "view_kernels": np.stack([kernels[name] for name in VIEW_NAMES], axis=0),
        "class_effect_kernel": effect_kernel,
        "impact_diffusion_signatures": diffusion.signatures,
        "impact_diffusion_feature_names": np.asarray(diffusion.feature_names),
        "impact_diffusion_magnitudes": diffusion.magnitudes,
        "impact_diffusion_l1_norms_by_time": diffusion.l1_norms_by_time,
    }
    record = {
        "experiment": str(loaded.result["experiment"]),
        "seed": seed,
        "run_path": str(loaded.run_path),
        "checkpoint": str(loaded.result["checkpoint"]),
        "checkpoint_model_sha256": checkpoint_sha,
        "dataset_sha256": str(loaded.data.metadata["sha256"]),
        "split_indices_sha256": split_indices_sha256(loaded.data),
        "canonical_edge_sha256": canonical_edge_sha256(adjacency),
        "layer": layer,
        "width": int(node_impacts.shape[0]),
        "num_nodes": int(node_impacts.shape[1]),
        "num_edges": int(adjacency.nnz // 2),
        "impact_scope": impact_scope,
        "impact_quantiles": list(quantiles),
        "active_minimum_mass": active_minimum_mass,
        "active_channel_count": int(active_channels.sum()),
        "inactive_channel_count": int((~active_channels).sum()),
        "views": {
            name: {
                "kernel": {
                    "magnitude": "cosine of standardized distribution statistics",
                    "raw_effect_field": "cosine of complete node-indexed TV fields",
                    "mask_jaccard": "mean hard/multiscale support Jaccard",
                    "uniform_node_role": (
                        "cosine of standardized uniform parent-role moments"
                    ),
                    "topology_only": (
                        "multiquantile normalized constant-label WL; strict "
                        "induced-support topology"
                    ),
                    "topology_wl_induced_degree": (
                        "multiquantile normalized WL with induced-degree-bin labels"
                    ),
                    "topology_wl_parent_degree": (
                        "multiquantile normalized WL with parent-degree-bin labels"
                    ),
                    "topology_wl_parent_degree_core": (
                        "multiquantile normalized WL with parent-degree/core-bin "
                        "labels"
                    ),
                    "topology_shape_only": (
                        "cosine of standardized induced-support shape ratios"
                    ),
                    "impact_diffusion": (
                        "cosine of standardized L1-normalized heat-diffusion "
                        "signatures; topology-conditioned effect field, not "
                        "topology-only"
                    ),
                    "full_signature": (
                        "cosine of standardized fused structural signatures"
                    ),
                }[name],
                "inference": inference[name],
            }
            for name in VIEW_NAMES
        },
        "rewiring_null": rewiring,
        "representation_provenance": {
            "module": "gnn_repair.structural_representations",
            "source_file": "structural_representations.py",
            "source_in_protocol_hash": True,
            "strict_topology_primary": "topology_only",
            "strict_topology_initial_labels": "constant",
            "wl_sensitivity_initial_labels": [
                "induced_degree_bin",
                "parent_degree_bin",
                "parent_degree_core_bin",
            ],
            "wl_depth": wl_depth,
            "wl_support_quantiles": list(quantiles),
            "impact_diffusion_feature_block": (
                "topology-conditioned normalized effect field (separate view)"
            ),
            "impact_diffusion": diffusion.metadata,
        },
        "test_labels_used": False,
        "test_performance_used": False,
    }
    return record, arrays


def _cache_identity(
    loaded: LoadedRun,
    *,
    dataset_name: str,
    backbone_name: str,
    layer: int,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": GATE1_PROTOCOL_VERSION,
        "gate1_code_sha256": source_code_sha256(GATE1_CODE_FILES),
        "dataset_name": dataset_name,
        "backbone_name": backbone_name,
        "experiment": str(loaded.result["experiment"]),
        "seed": int(loaded.result["seed"]),
        "checkpoint_model_sha256": model_state_sha256(loaded.checkpoint),
        "dataset_sha256": str(loaded.data.metadata["sha256"]),
        "split_indices_sha256": split_indices_sha256(loaded.data),
        "layer": layer,
        "gate_config": gate_config,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def _array_schema(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    }


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _analyze_with_cache(
    loaded: LoadedRun,
    *,
    dataset_name: str,
    backbone_name: str,
    layer: int,
    gate_config: dict[str, Any],
    cache_path: Path,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    identity = _cache_identity(
        loaded,
        dataset_name=dataset_name,
        backbone_name=backbone_name,
        layer=layer,
        gate_config=gate_config,
    )
    sidecar_path = cache_path.with_suffix(".npz")
    if resume and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("identity") != identity:
            raise RuntimeError(f"Gate-1 cache identity mismatch: {cache_path}")
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Gate-1 sidecar is missing: {sidecar_path}")
        observed_hash = _file_sha256(sidecar_path)
        if observed_hash != cached.get("sidecar", {}).get("sha256"):
            raise RuntimeError(f"Gate-1 sidecar hash mismatch: {sidecar_path}")
        arrays = _load_npz(sidecar_path)
        if _array_schema(arrays) != cached.get("sidecar", {}).get("arrays"):
            raise RuntimeError(f"Gate-1 sidecar schema mismatch: {sidecar_path}")
        return cached["run"], arrays

    record, arrays = analyze_loaded_run(
        loaded,
        layer=layer,
        gate_config=gate_config,
    )
    _write_npz_atomic(sidecar_path, arrays)
    sidecar = {
        "path": str(sidecar_path),
        "sha256": _file_sha256(sidecar_path),
        "format": "compressed_npz",
        "allow_pickle_required": False,
        "arrays": _array_schema(arrays),
        "representation_source": {
            "module": "gnn_repair.structural_representations",
            "source_file": "structural_representations.py",
            "included_in_gate1_code_sha256": True,
        },
        "purpose": (
            "exact node-indexed impact fields, support masks, class effects, and "
            "similarity kernels for audit and downstream Gate-1 aggregation"
        ),
    }
    record["raw_sidecar"] = sidecar
    _write_json_atomic(
        cache_path,
        {"identity": identity, "sidecar": sidecar, "run": record},
    )
    return record, arrays


def _numeric_summary(values: list[float]) -> dict[str, Any] | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return {
        "count": len(finite),
        "mean": float(mean(finite)),
        "median": float(median(finite)),
        "population_std": float(np.std(finite)),
        "positive_fraction": float(sum(value > 0 for value in finite) / len(finite)),
        "values": finite,
    }


def _percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    lower, upper = np.quantile(np.asarray(values), [0.025, 0.975])
    return [float(lower), float(upper)]


def _grouped_summary(
    keyed_values: list[tuple[str, float]],
    *,
    bootstrap_repetitions: int,
    random_seed: int,
) -> dict[str, Any] | None:
    """Summarize checkpoint units and bootstrap dataset/backbone blocks."""

    if bootstrap_repetitions < 0:
        raise ValueError("bootstrap_repetitions must be non-negative")
    by_cell: dict[str, list[float]] = defaultdict(list)
    for cell, value in keyed_values:
        if math.isfinite(value):
            by_cell[cell].append(float(value))
    if not by_cell:
        return None
    cells = sorted(by_cell)
    checkpoint_values = [value for cell in cells for value in by_cell[cell]]
    cell_means = [float(mean(by_cell[cell])) for cell in cells]
    generator = np.random.default_rng(random_seed)
    stratified: list[float] = []
    hierarchical: list[float] = []
    for _ in range(bootstrap_repetitions):
        sampled_cell_means = []
        for cell in cells:
            values = np.asarray(by_cell[cell], dtype=np.float64)
            sampled = generator.choice(values, size=values.size, replace=True)
            sampled_cell_means.append(float(sampled.mean()))
        stratified.append(float(mean(sampled_cell_means)))

        sampled_cells = generator.choice(cells, size=len(cells), replace=True)
        block_means = []
        for cell in sampled_cells:
            values = np.asarray(by_cell[str(cell)], dtype=np.float64)
            sampled = generator.choice(values, size=values.size, replace=True)
            block_means.append(float(sampled.mean()))
        hierarchical.append(float(mean(block_means)))
    return {
        "checkpoint_weighted": _numeric_summary(checkpoint_values),
        "cell_weighted_mean": float(mean(cell_means)),
        "positive_cells": int(sum(value > 0 for value in cell_means)),
        "cell_count": len(cells),
        "cells": {
            cell: _numeric_summary(by_cell[cell]) for cell in cells
        },
        "bootstrap": {
            "unit": "dataset/backbone block, then checkpoint",
            "repetitions": bootstrap_repetitions,
            "random_seed": random_seed,
            "stratified_checkpoint_ci_95": _percentile_interval(stratified),
            "hierarchical_block_ci_95": _percentile_interval(hierarchical),
        },
    }


def _view_kernel(arrays: dict[str, np.ndarray], view_name: str) -> np.ndarray:
    names = [str(value) for value in arrays["view_names"].tolist()]
    if view_name not in names:
        raise KeyError(f"Sidecar does not contain view {view_name!r}")
    return np.asarray(arrays["view_kernels"][names.index(view_name)], dtype=np.float64)


def _pair_dataset(
    arrays: dict[str, np.ndarray],
    *,
    feature_names: tuple[str, ...],
    subset: str,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(arrays["active_channels"], dtype=bool)
    indices = _subset_indices(active, subset)
    if indices.size < 2:
        return np.empty((0, len(feature_names))), np.empty(0)
    upper = np.triu_indices(indices.size, k=1)
    columns = []
    for name in feature_names:
        kernel = _view_kernel(arrays, name)[np.ix_(indices, indices)]
        columns.append(kernel[upper])
    effect = np.asarray(arrays["class_effect_kernel"], dtype=np.float64)
    effect = effect[np.ix_(indices, indices)][upper]
    return np.column_stack(columns), effect


def _fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    sample_weight: np.ndarray,
) -> dict[str, float | None]:
    location = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scaled_train = (train_x - location) / np.where(scale > 0, scale, 1.0)
    scaled_test = (test_x - location) / np.where(scale > 0, scale, 1.0)
    estimator = Ridge(alpha=alpha, fit_intercept=True)
    estimator.fit(scaled_train, train_y, sample_weight=sample_weight)
    prediction = estimator.predict(scaled_test)
    return {
        "r2": (
            float(r2_score(test_y, prediction))
            if test_y.size >= 2 and np.ptp(test_y) > 0
            else None
        ),
        "mae": float(mean_absolute_error(test_y, prediction)),
    }


def leave_one_checkpoint_out_prediction(
    entries: list[dict[str, Any]],
    *,
    subset: str,
    ridge_alpha: float,
) -> list[dict[str, Any]]:
    """Predict class-effect similarity with and without topology."""

    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    controls = CONTROL_VIEW_NAMES
    augmented = (*CONTROL_VIEW_NAMES, "topology_only")
    records = []
    for held_index, held in enumerate(entries):
        held_x, held_y = _pair_dataset(
            held["arrays"], feature_names=augmented, subset=subset
        )
        if held_y.size < 2:
            continue
        train_parts = []
        target_parts = []
        checkpoint_sizes = []
        for index, entry in enumerate(entries):
            if index == held_index:
                continue
            values, target = _pair_dataset(
                entry["arrays"], feature_names=augmented, subset=subset
            )
            if target.size:
                train_parts.append(values)
                target_parts.append(target)
                checkpoint_sizes.append(target.size)
        if not train_parts:
            continue
        train_x = np.concatenate(train_parts, axis=0)
        train_y = np.concatenate(target_parts, axis=0)
        # Equal total weight per training checkpoint prevents a wider hidden
        # layer from becoming the statistical unit by contributing more pairs.
        mean_pair_count = float(mean(checkpoint_sizes))
        weights = np.concatenate(
            [
                np.full(size, mean_pair_count / size, dtype=np.float64)
                for size in checkpoint_sizes
            ]
        )
        control_result = _fit_ridge(
            train_x[:, : len(controls)],
            train_y,
            held_x[:, : len(controls)],
            held_y,
            alpha=ridge_alpha,
            sample_weight=weights,
        )
        augmented_result = _fit_ridge(
            train_x,
            train_y,
            held_x,
            held_y,
            alpha=ridge_alpha,
            sample_weight=weights,
        )
        record = held["record"]
        control_r2 = control_result["r2"]
        augmented_r2 = augmented_result["r2"]
        records.append(
            {
                "dataset": held["dataset"],
                "backbone": held["backbone"],
                "cell": held["cell"],
                "experiment": record["experiment"],
                "seed": record["seed"],
                "checkpoint_model_sha256": record["checkpoint_model_sha256"],
                "subset": subset,
                "train_checkpoint_count": len(train_parts),
                "test_pair_count": int(held_y.size),
                "controls": {
                    "features": list(controls),
                    **control_result,
                },
                "controls_plus_topology": {
                    "features": list(augmented),
                    **augmented_result,
                },
                "delta_r2": (
                    float(augmented_r2 - control_r2)
                    if control_r2 is not None and augmented_r2 is not None
                    else None
                ),
                "mae_reduction": float(
                    control_result["mae"] - augmented_result["mae"]
                ),
            }
        )
    return records


def _aggregate_entries(
    entries: list[dict[str, Any]],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    ridge_alpha: float,
) -> dict[str, Any]:
    alignments: dict[str, Any] = {}
    for view_name in VIEW_NAMES:
        alignments[view_name] = {}
        for subset in SUBSETS:
            values: list[tuple[str, float]] = []
            significant = 0
            available_p = 0
            for entry in entries:
                inference = entry["record"]["views"][view_name]["inference"][subset]
                observed = inference["observed"]
                if observed is not None:
                    values.append((entry["cell"], float(observed)))
                if inference["p_greater"] is not None:
                    available_p += 1
                    significant += int(float(inference["p_greater"]) <= 0.05)
            summary = _grouped_summary(
                values,
                bootstrap_repetitions=bootstrap_repetitions,
                random_seed=_stable_seed(
                    bootstrap_seed, "alignment", view_name, subset
                ),
            )
            alignments[view_name][subset] = {
                "alignment": summary,
                "permutation_p_le_0_05_fraction": (
                    significant / available_p if available_p else None
                ),
                "permutation_p_available": available_p,
            }

    rewiring: dict[str, Any] = {}
    configured_rewiring_views = sorted(
        {
            view_name
            for entry in entries
            for view_name in entry["record"]["rewiring_null"]["views"]
        }
    )
    for view_name in configured_rewiring_views:
        rewiring[view_name] = {}
        for subset in SUBSETS:
            values = []
            for entry in entries:
                result = entry["record"]["rewiring_null"]["views"].get(
                    view_name, {}
                ).get(subset, {})
                contrast = result.get("observed_minus_null_mean")
                if contrast is not None:
                    values.append((entry["cell"], float(contrast)))
            rewiring[view_name][subset] = _grouped_summary(
                values,
                bootstrap_repetitions=bootstrap_repetitions,
                random_seed=_stable_seed(
                    bootstrap_seed, "rewiring", view_name, subset
                ),
            )

    prediction: dict[str, Any] = {}
    for subset in SUBSETS:
        records = leave_one_checkpoint_out_prediction(
            entries,
            subset=subset,
            ridge_alpha=ridge_alpha,
        )
        prediction[subset] = {
            "protocol": {
                "split": "leave one whole checkpoint out",
                "target": "upper-triangle class-effect cosine similarity",
                "controls": list(CONTROL_VIEW_NAMES),
                "augmented": [*CONTROL_VIEW_NAMES, "topology_only"],
                "ridge_alpha": ridge_alpha,
                "training_pair_weights": "equal total weight per checkpoint",
            },
            "checkpoints": records,
            "delta_r2": _grouped_summary(
                [
                    (record["cell"], float(record["delta_r2"]))
                    for record in records
                    if record["delta_r2"] is not None
                ],
                bootstrap_repetitions=bootstrap_repetitions,
                random_seed=_stable_seed(bootstrap_seed, "loco", subset, "r2"),
            ),
            "mae_reduction": _grouped_summary(
                [
                    (record["cell"], float(record["mae_reduction"]))
                    for record in records
                ],
                bootstrap_repetitions=bootstrap_repetitions,
                random_seed=_stable_seed(bootstrap_seed, "loco", subset, "mae"),
            ),
        }
    return {
        "statistical_unit": "checkpoint; neuron pairs are dependent within checkpoint",
        "alignments": alignments,
        "rewiring_null": rewiring,
        "leave_one_checkpoint_out_prediction": prediction,
    }


def _lower_ci(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    interval = summary.get("bootstrap", {}).get("hierarchical_block_ci_95")
    return float(interval[0]) if interval else None


def _gate_decision(aggregate: dict[str, Any]) -> dict[str, Any]:
    topology_all = aggregate["alignments"]["topology_only"]["all_channels"][
        "alignment"
    ]
    topology_active = aggregate["alignments"]["topology_only"]["active_channels"][
        "alignment"
    ]
    alignment_positive = bool(
        topology_all
        and topology_all["checkpoint_weighted"]["positive_fraction"] >= 0.75
        and _lower_ci(topology_all) is not None
        and _lower_ci(topology_all) > 0
    )
    active_survives = bool(
        topology_active
        and topology_active["checkpoint_weighted"]["positive_fraction"] >= 0.75
        and _lower_ci(topology_active) is not None
        and _lower_ci(topology_active) > 0
    )

    active_prediction = aggregate["leave_one_checkpoint_out_prediction"][
        "active_channels"
    ]
    delta_r2 = active_prediction["delta_r2"]
    mae_reduction = active_prediction["mae_reduction"]
    prediction_improves = bool(
        delta_r2
        and mae_reduction
        and delta_r2["checkpoint_weighted"]["mean"] > 0
        and mae_reduction["checkpoint_weighted"]["mean"] > 0
        and delta_r2["positive_cells"] >= min(3, delta_r2["cell_count"])
        and mae_reduction["positive_cells"] >= min(3, mae_reduction["cell_count"])
    )

    rewiring = aggregate["rewiring_null"].get("topology_only", {}).get(
        "active_channels"
    )
    rewiring_survives = bool(
        rewiring
        and rewiring["checkpoint_weighted"]["mean"] > 0
        and _lower_ci(rewiring) is not None
        and _lower_ci(rewiring) > 0
    )
    criteria = {
        "topology_alignment": {
            "pass": alignment_positive,
            "rule": (
                "positive in at least 75% of checkpoints and hierarchical blocked "
                "95% CI above zero"
            ),
        },
        "active_channel_sensitivity": {
            "pass": active_survives,
            "rule": "the topology alignment criterion also passes active-only",
        },
        "incremental_loco_prediction": {
            "pass": prediction_improves,
            "rule": (
                "adding topology improves checkpoint-level R2 and MAE in aggregate "
                "and in at least three dataset/backbone cells when four are present"
            ),
        },
        "degree_preserving_rewiring_null": {
            "pass": rewiring_survives,
            "rule": (
                "active-only observed-minus-rewired alignment is positive with a "
                "hierarchical blocked 95% CI above zero"
            ),
        },
    }
    passed = all(value["pass"] for value in criteria.values())
    return {
        "pass": passed,
        "criteria": criteria,
        "recommended_framing": (
            "topology-centered structural-impact paper"
            if passed
            else "ablation-effect-field paper; topology is not yet isolated"
        ),
        "decision_scope": "development-only; must be frozen before fresh holdouts",
    }


def _run_paths(root: Path) -> list[Path]:
    paths = list((root / "runs").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No run records found under {root / 'runs'}")

    def key(path: Path) -> tuple[int, str]:
        with path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        return int(record["seed"]), path.name

    return sorted(paths, key=key)


def _reject_confirmatory_source(root: Path, loaded: LoadedRun | None = None) -> None:
    resolved_parts = {part.lower() for part in root.resolve().parts}
    if "confirmatory" in resolved_parts:
        raise ValueError(f"Gate 1 cannot use a confirmatory artifact root: {root}")
    if loaded is not None and "confirmatory" in str(
        loaded.result.get("experiment", "")
    ).lower():
        raise ValueError(
            "Gate 1 cannot use a run whose experiment is marked confirmatory: "
            f"{loaded.result.get('experiment')}"
        )


def _cell_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "checkpoint_count": len(records),
        "active_channel_fraction": _numeric_summary(
            [record["active_channel_count"] / record["width"] for record in records]
        ),
        "alignment": {
            view_name: {
                subset: _numeric_summary(
                    [
                        float(value)
                        for record in records
                        if (
                            value := record["views"][view_name]["inference"][subset][
                                "observed"
                            ]
                        )
                        is not None
                    ]
                )
                for subset in SUBSETS
            }
            for view_name in VIEW_NAMES
        },
    }


def build_gate1_report(
    config: dict[str, Any],
    *,
    allow_download: bool = False,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Run Gate 1 over configured, already-inspected development checkpoints."""

    if config.get("development_only") is not True:
        raise ValueError("Gate-1 config must explicitly set development_only: true")
    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Gate-1 config requires a non-empty datasets mapping")
    gate_config = dict(config.get("gate1", {}))
    cache_root = Path(config.get("cache_root", "artifacts/topology_gate1_runs"))
    bootstrap_repetitions = int(gate_config.get("bootstrap_repetitions", 2000))
    bootstrap_seed = int(gate_config.get("bootstrap_seed", 20270803))
    ridge_alpha = float(gate_config.get("ridge_alpha", 1.0))
    code_hash = source_code_sha256(GATE1_CODE_FILES)
    report: dict[str, Any] = {
        "protocol": {
            "name": "Gate 1 topology isolation",
            "version": GATE1_PROTOCOL_VERSION,
            "status": (
                "exploratory development-only analysis on previously inspected "
                "graphs and checkpoints"
            ),
            "development_only": True,
            "fresh_holdouts_used": False,
            "confirmatory_artifacts_used": False,
            "test_labels_used": False,
            "test_performance_used": False,
            "config_sha256": canonical_payload_sha256(config),
            "gate1_code_sha256": code_hash,
            "statistical_unit": "checkpoint, never neuron pairs",
            "within_checkpoint_inference": (
                "simultaneous row/column neuron-label permutations"
            ),
            "graph_null": (
                "simple degree-preserving double-edge swaps; fixed impact fields "
                "and masks; no connectivity constraint"
            ),
            "representations": {
                "topology_only": (
                    "constant-label normalized multiscale WL over impact-defined "
                    "support; strict support topology"
                ),
                "topology_wl_induced_degree": (
                    "normalized WL with induced-degree-bin initial labels"
                ),
                "topology_wl_parent_degree": (
                    "normalized WL with parent-degree-bin initial labels"
                ),
                "topology_shape_only": "induced-support shape ratios",
                "impact_diffusion": (
                    "separate topology-conditioned L1-normalized effect-field view; "
                    "not labeled topology-only"
                ),
                "source_module": "gnn_repair.structural_representations",
                "source_file": "structural_representations.py",
                "source_in_protocol_hash": True,
            },
            "gate1": gate_config,
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
    entries: list[dict[str, Any]] = []
    for dataset_name, dataset_config in datasets.items():
        backbones = dataset_config.get("backbones", {})
        if not isinstance(backbones, dict) or not backbones:
            raise ValueError(f"Dataset {dataset_name!r} has no backbones")
        dataset_record: dict[str, Any] = {"backbones": {}}
        for backbone_name, backbone_config in backbones.items():
            root = Path(backbone_config["root"])
            _reject_confirmatory_source(root)
            layer = int(backbone_config["layer"])
            paths = _run_paths(root)
            max_runs = backbone_config.get("max_runs")
            if max_runs is not None:
                paths = paths[: int(max_runs)]
            cell = f"{dataset_name}/{backbone_name}"
            records = []
            for run_number, run_path in enumerate(paths, start=1):
                if progress:
                    print(
                        f"[{cell}] Gate 1 checkpoint {run_number}/{len(paths)}",
                        flush=True,
                    )
                loaded = load_trained_run(run_path, allow_download=allow_download)
                _reject_confirmatory_source(root, loaded)
                checkpoint_sha = model_state_sha256(loaded.checkpoint)
                cache_path = (
                    cache_root
                    / dataset_name
                    / backbone_name
                    / (
                        f"seed_{int(loaded.result['seed'])}_"
                        f"{checkpoint_sha[:12]}.json"
                    )
                )
                record, arrays = _analyze_with_cache(
                    loaded,
                    dataset_name=dataset_name,
                    backbone_name=backbone_name,
                    layer=layer,
                    gate_config=gate_config,
                    cache_path=cache_path,
                    resume=resume,
                )
                records.append(record)
                entries.append(
                    {
                        "dataset": dataset_name,
                        "backbone": backbone_name,
                        "cell": cell,
                        "record": record,
                        "arrays": arrays,
                    }
                )
            dataset_record["backbones"][backbone_name] = {
                "layer": layer,
                "runs": records,
                "summary": _cell_summary(records),
            }
        report["datasets"][dataset_name] = dataset_record
    aggregate = _aggregate_entries(
        entries,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        ridge_alpha=ridge_alpha,
    )
    report["aggregate"] = aggregate
    report["gate_1_decision"] = _gate_decision(aggregate)
    report["raw_sidecars"] = [
        entry["record"]["raw_sidecar"] for entry in entries
    ]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/topology_gate1_development.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow a missing, already-declared development dataset to be downloaded",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only caches with exact protocol identity and sidecar hash",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_gate1_report(
        config,
        allow_download=args.allow_download,
        resume=args.resume,
        progress=not args.quiet,
    )
    output = args.output or Path(
        config.get("output", "artifacts/topology_gate1_development.json")
    )
    _write_json_atomic(output, report)
    print(f"Wrote Gate-1 topology-isolation report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
