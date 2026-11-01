"""Deterministic structural-impact signatures for class-specific GNN repair.

This module reconstructs the central structural idea from the thesis without
changing the sealed confirmatory pipeline.  A hidden channel is replaced by its
training-node activation mean, nodes are ranked by the resulting probability
shift, and multiscale induced-subgraph descriptors are used to cluster channels.
Cluster assignments are topology-only; validation labels enter only afterwards
when clusters are mapped to classes and repair candidates are selected.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import platform
import warnings
from pathlib import Path
from statistics import mean, median, pstdev, stdev
from typing import Any

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
import torch
import yaml
from scipy.sparse import csgraph
from scipy.stats import spearmanr
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score
from torch.nn import functional as F

from gnn_repair.checkpoints import (
    LoadedRun,
    load_trained_run,
    model_state_sha256,
    source_code_sha256,
    split_indices_sha256,
)
from gnn_repair.interventions import EFFECT_METRICS
from gnn_repair.localization import gradient_attribution_scores
from gnn_repair.protocol import (
    canonical_payload_sha256,
    evaluation_splits,
    test_status,
)
from gnn_repair.repair import fit_repair
from gnn_repair.reproducibility import set_deterministic


STRUCTURAL_PROTOCOL_VERSION = 1
STRUCTURAL_CODE_FILES = (
    "structural.py",
    "repair.py",
    "localization.py",
    "interventions.py",
    "model.py",
    "data.py",
    "metrics.py",
    "checkpoints.py",
    "protocol.py",
    "reproducibility.py",
)

DEFAULT_IMPACT_QUANTILES = (0.90, 0.95, 0.99)
DEFAULT_NODE_FEATURES = (
    "log1p_degree",
    "log1p_neighbor_degree_mean",
    "log1p_neighbor_degree_std",
    "log1p_component_size",
    "scaled_pagerank",
    "log1p_core_number",
    "is_isolate",
)


def canonical_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> sp.csr_matrix:
    """Return a simple, loop-free, sorted undirected adjacency matrix."""

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    edges = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("edge_index contains an endpoint outside the graph")
    if edges.shape[1] == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    source, target = edges
    keep = source != target
    lower = np.minimum(source[keep], target[keep])
    upper = np.maximum(source[keep], target[keep])
    if lower.size == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    pairs = np.unique(np.stack([lower, upper], axis=1), axis=0)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    columns = np.concatenate([pairs[:, 1], pairs[:, 0]])
    adjacency = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, columns)),
        shape=(num_nodes, num_nodes),
    )
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    return adjacency


def canonical_edge_sha256(adjacency: sp.csr_matrix) -> str:
    """Hash the canonical sparse graph representation."""

    matrix = adjacency.tocsr(copy=True)
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode("ascii"))
    digest.update(matrix.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.indices.astype(np.int64, copy=False).tobytes())
    return digest.hexdigest()


def _core_numbers(adjacency: sp.csr_matrix) -> np.ndarray:
    """Compute undirected k-core numbers with deterministic heap peeling."""

    num_nodes = adjacency.shape[0]
    degree = np.diff(adjacency.indptr).astype(np.int64)
    active_degree = degree.copy()
    removed = np.zeros(num_nodes, dtype=bool)
    core = np.zeros(num_nodes, dtype=np.int64)
    queue = [(int(active_degree[node]), node) for node in range(num_nodes)]
    heapq.heapify(queue)
    while queue:
        value, node = heapq.heappop(queue)
        if removed[node] or value != int(active_degree[node]):
            continue
        removed[node] = True
        core[node] = value
        start, end = adjacency.indptr[node : node + 2]
        for neighbor in adjacency.indices[start:end]:
            if not removed[neighbor] and active_degree[neighbor] > value:
                active_degree[neighbor] -= 1
                heapq.heappush(
                    queue,
                    (int(active_degree[neighbor]), int(neighbor)),
                )
    return core


def _pagerank(
    adjacency: sp.csr_matrix,
    *,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Compute deterministic PageRank using sparse power iteration."""

    num_nodes = adjacency.shape[0]
    if num_nodes == 0:
        return np.empty(0, dtype=np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
    safe_degree = np.where(degree > 0, degree, 1.0)
    rank = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    teleport = (1.0 - damping) / num_nodes
    for _ in range(max_iterations):
        dangling = rank[degree == 0].sum() / num_nodes
        updated = teleport + damping * (
            adjacency.T.dot(rank / safe_degree) + dangling
        )
        if np.abs(updated - rank).sum() <= tolerance:
            rank = updated
            break
        rank = updated
    total = rank.sum()
    return rank / total if total > 0 else np.full(num_nodes, 1.0 / num_nodes)


def graph_node_features(
    adjacency: sp.csr_matrix,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    """Build label-free node-topology covariates used to pool subgraph shape."""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    adjacency = adjacency.tocsr(copy=True).astype(np.float64)
    adjacency.sort_indices()
    num_nodes = adjacency.shape[0]
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    safe_degree = np.where(degree > 0, degree, 1.0)
    neighbor_degree_mean = np.asarray(adjacency.dot(degree)).ravel() / safe_degree
    neighbor_second_moment = (
        np.asarray(adjacency.dot(np.square(degree))).ravel() / safe_degree
    )
    neighbor_degree_std = np.sqrt(
        np.maximum(0.0, neighbor_second_moment - np.square(neighbor_degree_mean))
    )
    if num_nodes:
        _, component_labels = csgraph.connected_components(
            adjacency,
            directed=False,
            return_labels=True,
        )
        component_counts = np.bincount(component_labels, minlength=num_nodes)
        component_size = component_counts[component_labels].astype(np.float64)
    else:
        component_labels = np.empty(0, dtype=np.int64)
        component_size = np.empty(0, dtype=np.float64)
    pagerank = _pagerank(adjacency)
    core_number = _core_numbers(adjacency).astype(np.float64)
    features = np.column_stack(
        [
            np.log1p(degree),
            np.log1p(neighbor_degree_mean),
            np.log1p(neighbor_degree_std),
            np.log1p(component_size),
            pagerank * max(1, num_nodes),
            np.log1p(core_number),
            (degree == 0).astype(np.float64),
        ]
    )
    if not np.isfinite(features).all():
        raise FloatingPointError("Graph topology features contain non-finite values")
    raw = {
        "degree": degree,
        "neighbor_degree_mean": neighbor_degree_mean,
        "neighbor_degree_std": neighbor_degree_std,
        "component_size": component_size,
        "component_label": component_labels,
        "pagerank": pagerank,
        "core_number": core_number,
    }
    return features.astype(np.float64), DEFAULT_NODE_FEATURES, raw


def _scope_mask(loaded: LoadedRun, scope: str) -> np.ndarray:
    data = loaded.data
    if scope == "all":
        mask = torch.ones(data.y.numel(), dtype=torch.bool, device=data.y.device)
    elif scope == "train":
        mask = data.train_mask
    elif scope == "validation":
        mask = data.val_mask
    elif scope == "train_validation":
        mask = data.train_mask | data.val_mask
    else:
        raise ValueError(
            "impact_scope must be all, train, validation, or train_validation"
        )
    return mask.detach().cpu().numpy().astype(bool, copy=False)


def _classwise_effects(
    base_logits: torch.Tensor,
    modified_logits: torch.Tensor,
    loaded: LoadedRun,
) -> tuple[list[float], list[float]]:
    """Return validation KL and class-conditional CE changes for every class."""

    base_probabilities = F.softmax(base_logits, dim=-1).clamp_min(1e-12)
    modified_probabilities = F.softmax(modified_logits, dim=-1).clamp_min(1e-12)
    node_kl = (
        base_probabilities
        * (base_probabilities.log() - modified_probabilities.log())
    ).sum(dim=-1)
    num_classes = base_logits.shape[1]
    class_kl: list[float] = []
    class_ce_delta: list[float] = []
    for class_id in range(num_classes):
        mask = loaded.data.val_mask & (loaded.data.y == class_id)
        if not bool(mask.any()):
            class_kl.append(0.0)
            class_ce_delta.append(0.0)
            continue
        class_kl.append(float(node_kl[mask].mean().item()))
        base_ce = F.cross_entropy(base_logits[mask], loaded.data.y[mask])
        modified_ce = F.cross_entropy(modified_logits[mask], loaded.data.y[mask])
        class_ce_delta.append(float((modified_ce - base_ce).item()))
    return class_kl, class_ce_delta


@torch.no_grad()
def collect_neuron_impacts(
    loaded: LoadedRun,
    *,
    layer: int,
    impact_scope: str,
) -> dict[str, Any]:
    """Collect node-level probability shifts after individual mean ablations."""

    model, data = loaded.model, loaded.data
    model.eval()
    base_logits, activations = model(
        data.x,
        data.adjacency,
        return_activations=True,
    )
    if not 0 <= layer < len(activations):
        raise ValueError(f"Layer {layer} is outside the model's hidden activations")
    width = int(activations[layer].shape[1])
    training_mean = activations[layer][data.train_mask].mean(dim=0)
    base_probabilities = F.softmax(base_logits, dim=-1)
    base_predictions = base_logits.argmax(dim=-1)
    node_impacts = np.zeros((width, data.y.numel()), dtype=np.float32)
    prediction_changes = np.zeros((width, data.y.numel()), dtype=bool)
    class_kl = np.zeros((width, base_logits.shape[1]), dtype=np.float64)
    class_ce_delta = np.zeros_like(class_kl)

    for neuron in range(width):

        def replace_neuron(
            active_layer: int,
            activation: torch.Tensor,
            *,
            expected_layer: int = layer,
            expected_neuron: int = neuron,
        ) -> torch.Tensor:
            if active_layer != expected_layer:
                return activation
            modified = activation.clone()
            modified[:, expected_neuron] = training_mean[expected_neuron]
            return modified

        modified_logits = model(
            data.x,
            data.adjacency,
            activation_intervention=replace_neuron,
        )
        modified_probabilities = F.softmax(modified_logits, dim=-1)
        total_variation = 0.5 * (
            modified_probabilities - base_probabilities
        ).abs().sum(dim=-1)
        node_impacts[neuron] = total_variation.detach().cpu().numpy()
        prediction_changes[neuron] = (
            modified_logits.argmax(dim=-1) != base_predictions
        ).detach().cpu().numpy()
        neuron_kl, neuron_ce = _classwise_effects(
            base_logits,
            modified_logits,
            loaded,
        )
        class_kl[neuron] = neuron_kl
        class_ce_delta[neuron] = neuron_ce

    scope = _scope_mask(loaded, impact_scope)
    if not scope.any():
        raise ValueError("The structural impact scope contains no nodes")
    return {
        "base_logits": base_logits,
        "training_mean": training_mean,
        "scope_mask": scope,
        "node_impacts": node_impacts,
        "prediction_changes": prediction_changes,
        "class_kl": class_kl,
        "class_ce_delta": class_ce_delta,
    }


def _top_impact_mask(
    impact: np.ndarray,
    scope_indices: np.ndarray,
    quantile: float,
    *,
    minimum_impact: float,
) -> np.ndarray:
    count = max(1, int(math.ceil((1.0 - quantile) * scope_indices.size)))
    values = impact[scope_indices]
    order = np.lexsort((scope_indices, -values))
    selected = scope_indices[order[:count]]
    selected = selected[impact[selected] > minimum_impact]
    mask = np.zeros(impact.size, dtype=bool)
    mask[selected] = True
    return mask


def _moran_and_edge_variation(
    values: np.ndarray,
    adjacency: sp.csr_matrix,
) -> tuple[float, float]:
    if values.size < 2 or adjacency.nnz == 0:
        return 0.0, 0.0
    centered = values - values.mean()
    denominator = float(centered.dot(centered))
    edge_weight = float(adjacency.sum())
    moran = (
        values.size * float(centered.dot(adjacency.dot(centered)))
        / (edge_weight * denominator)
        if denominator > 0 and edge_weight > 0
        else 0.0
    )
    upper = sp.triu(adjacency, k=1).tocoo()
    variation = (
        float(np.abs(values[upper.row] - values[upper.col]).mean())
        if upper.nnz
        else 0.0
    )
    return moran, variation


def _induced_subgraph_features(
    adjacency: sp.csr_matrix,
    affected: np.ndarray,
    degree: np.ndarray,
) -> tuple[list[float], dict[str, Any]]:
    nodes = np.flatnonzero(affected)
    num_nodes = int(nodes.size)
    total_edges = adjacency.nnz / 2.0
    total_volume = float(degree.sum())
    if num_nodes == 0:
        return [0.0] * 8, {
            "nodes": 0,
            "internal_edges": 0,
            "boundary_edges": 0.0,
            "components": 0,
            "largest_component_fraction": 0.0,
        }
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
        2.0 * internal_edges / (num_nodes * (num_nodes - 1))
        if num_nodes > 1
        else 0.0
    )
    components, labels = csgraph.connected_components(
        subgraph,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(labels, minlength=components)
    largest_component_fraction = float(component_sizes.max() / num_nodes)
    features = [
        num_nodes / adjacency.shape[0],
        internal_edges / total_edges if total_edges else 0.0,
        2.0 * internal_edges / volume if volume else 0.0,
        boundary / volume if volume else 0.0,
        conductance,
        density,
        components / num_nodes,
        largest_component_fraction,
    ]
    details = {
        "nodes": num_nodes,
        "internal_edges": int(internal_edges),
        "boundary_edges": boundary,
        "components": int(components),
        "largest_component_fraction": largest_component_fraction,
    }
    return features, details


INDUCED_FEATURE_NAMES = (
    "affected_fraction",
    "internal_edge_fraction",
    "internal_volume_fraction",
    "boundary_volume_ratio",
    "conductance",
    "density",
    "components_per_node",
    "largest_component_fraction",
)


def structural_signatures(
    adjacency: sp.csr_matrix,
    node_features: np.ndarray,
    node_feature_names: tuple[str, ...],
    node_impacts: np.ndarray,
    prediction_changes: np.ndarray,
    scope_mask: np.ndarray,
    *,
    quantiles: tuple[float, ...] = DEFAULT_IMPACT_QUANTILES,
    minimum_impact: float = 1e-12,
) -> dict[str, Any]:
    """Pool multiscale induced-subgraph topology into one vector per channel."""

    if node_impacts.ndim != 2 or node_impacts.shape != prediction_changes.shape:
        raise ValueError("Impact and prediction-change matrices must align")
    if node_impacts.shape[1] != adjacency.shape[0]:
        raise ValueError("Impact width does not match the graph")
    if node_features.shape != (adjacency.shape[0], len(node_feature_names)):
        raise ValueError("Node feature matrix does not match its declared names")
    if scope_mask.shape != (adjacency.shape[0],) or not scope_mask.any():
        raise ValueError("scope_mask must select at least one graph node")
    if any(not 0 < value < 1 for value in quantiles):
        raise ValueError("Every impact quantile must be in (0, 1)")
    if tuple(sorted(set(quantiles))) != quantiles:
        raise ValueError("Impact quantiles must be unique and increasing")

    scope_indices = np.flatnonzero(scope_mask)
    scoped_adjacency = adjacency[scope_indices][:, scope_indices].tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    feature_names = [
        "impact_mean",
        "impact_std",
        "impact_max",
        "impact_positive_fraction",
        "prediction_change_fraction",
        "impact_moran_i",
        "impact_edge_variation",
    ]
    feature_names.extend(f"hard_{name}" for name in INDUCED_FEATURE_NAMES)
    for name in node_feature_names:
        feature_names.extend(
            (f"hard_{name}_impact_mean", f"hard_{name}_impact_std")
        )
    for quantile in quantiles:
        prefix = f"q{int(round(100 * quantile)):02d}"
        feature_names.extend(f"{prefix}_{name}" for name in INDUCED_FEATURE_NAMES)
        for name in node_feature_names:
            feature_names.extend(
                (f"{prefix}_{name}_impact_mean", f"{prefix}_{name}_impact_std")
            )

    rows: list[list[float]] = []
    affected_details: list[dict[str, Any]] = []
    for neuron, impact in enumerate(node_impacts):
        scoped_values = impact[scope_indices].astype(np.float64)
        moran, edge_variation = _moran_and_edge_variation(
            scoped_values,
            scoped_adjacency,
        )
        row = [
            float(scoped_values.mean()),
            float(scoped_values.std()),
            float(scoped_values.max()),
            float((scoped_values > minimum_impact).mean()),
            float(prediction_changes[neuron, scope_indices].mean()),
            moran,
            edge_variation,
        ]
        neuron_details: dict[str, Any] = {}
        affected_views: list[tuple[str, np.ndarray]] = [
            ("hard", prediction_changes[neuron] & scope_mask)
        ]
        affected_views.extend(
            (
                str(quantile),
                _top_impact_mask(
                    impact,
                    scope_indices,
                    quantile,
                    minimum_impact=minimum_impact,
                ),
            )
            for quantile in quantiles
        )
        for view_name, affected in affected_views:
            induced, details = _induced_subgraph_features(
                adjacency,
                affected,
                degree,
            )
            row.extend(induced)
            nodes = np.flatnonzero(affected)
            weights = impact[nodes].astype(np.float64)
            if nodes.size and weights.sum() > 0:
                weights = weights / weights.sum()
                selected_features = node_features[nodes]
                pooled_mean = np.sum(selected_features * weights[:, None], axis=0)
                pooled_variance = np.sum(
                    np.square(selected_features - pooled_mean) * weights[:, None],
                    axis=0,
                )
                pooled_std = np.sqrt(np.maximum(0.0, pooled_variance))
            else:
                pooled_mean = np.zeros(node_features.shape[1], dtype=np.float64)
                pooled_std = np.zeros_like(pooled_mean)
            for feature_mean, feature_std in zip(pooled_mean, pooled_std, strict=True):
                row.extend([float(feature_mean), float(feature_std)])
            neuron_details[view_name] = details
        if not np.isfinite(row).all():
            raise FloatingPointError(f"Structural signature {neuron} is non-finite")
        rows.append(row)
        affected_details.append(neuron_details)

    raw = np.asarray(rows, dtype=np.float64)
    location = raw.mean(axis=0, keepdims=True)
    scale = raw.std(axis=0, keepdims=True)
    standardized = (raw - location) / np.where(scale > 0, scale, 1.0)
    return {
        "raw": raw,
        "standardized": standardized,
        "feature_names": tuple(feature_names),
        "affected_subgraphs": affected_details,
    }


def _cosine_similarity(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    normalized = values / safe[:, None]
    similarity = normalized.dot(normalized.T)
    np.fill_diagonal(similarity, 1.0)
    return np.clip(similarity, -1.0, 1.0)


def _canonicalize_clusters(
    labels: np.ndarray,
    signatures: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    clusters = []
    for old_label in sorted(set(int(value) for value in labels)):
        members = np.flatnonzero(labels == old_label)
        centroid = signatures[members].mean(axis=0)
        key = tuple(np.round(centroid, decimals=12).tolist()) + (int(members.min()),)
        clusters.append((key, old_label))
    old_to_new = {
        old_label: new_label
        for new_label, (_, old_label) in enumerate(sorted(clusters))
    }
    canonical = np.asarray([old_to_new[int(value)] for value in labels], dtype=np.int64)
    return canonical, old_to_new


def cluster_signatures(
    standardized_signatures: np.ndarray,
    *,
    random_state: int,
    damping: float = 0.8,
    max_iterations: int = 500,
    convergence_iterations: int = 30,
) -> dict[str, Any]:
    """Cluster neuron signatures with thesis-continuity affinity propagation."""

    if standardized_signatures.ndim != 2 or standardized_signatures.shape[0] < 1:
        raise ValueError("Structural signatures must be a non-empty matrix")
    if not np.isfinite(standardized_signatures).all():
        raise ValueError("Structural signatures must be finite")
    width = standardized_signatures.shape[0]
    similarity = _cosine_similarity(standardized_signatures)
    if width == 1 or np.unique(standardized_signatures, axis=0).shape[0] == 1:
        labels = np.zeros(width, dtype=np.int64)
        centers = np.asarray([0], dtype=np.int64)
        iterations = 0
        converged = True
    else:
        off_diagonal = similarity[~np.eye(width, dtype=bool)]
        preference = float(np.median(off_diagonal))
        estimator = AffinityPropagation(
            affinity="precomputed",
            damping=damping,
            max_iter=max_iterations,
            convergence_iter=convergence_iterations,
            preference=preference,
            random_state=random_state,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            labels = estimator.fit_predict(similarity)
        centers = (
            estimator.cluster_centers_indices_
            if estimator.cluster_centers_indices_ is not None
            else np.asarray([], dtype=np.int64)
        )
        iterations = int(estimator.n_iter_)
        converged = not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
    labels, old_to_new = _canonicalize_clusters(labels, standardized_signatures)
    members = {
        str(cluster): np.flatnonzero(labels == cluster).astype(int).tolist()
        for cluster in sorted(set(labels.tolist()))
    }
    canonical_centers: dict[str, int] = {}
    if centers.size:
        original_labels = np.asarray(
            [
                next(
                    old
                    for old, new in old_to_new.items()
                    if new == cluster
                )
                for cluster in sorted(set(labels.tolist()))
            ]
        )
        for cluster, old_label in enumerate(original_labels):
            if int(old_label) < centers.size:
                canonical_centers[str(cluster)] = int(centers[int(old_label)])
    for cluster, cluster_members in members.items():
        canonical_centers.setdefault(cluster, int(min(cluster_members)))
    cluster_count = len(members)
    silhouette = (
        float(silhouette_score(standardized_signatures, labels, metric="cosine"))
        if 1 < cluster_count < width
        else None
    )
    return {
        "labels": labels,
        "members": members,
        "centers": canonical_centers,
        "similarity": similarity,
        "cluster_count": cluster_count,
        "silhouette_cosine": silhouette,
        "iterations": iterations,
        "converged": converged,
    }


def clustering_stability(
    signatures: dict[str, Any],
    full_labels: np.ndarray,
    *,
    quantiles: tuple[float, ...],
    random_state: int,
    damping: float,
) -> dict[str, Any]:
    """Compare the full clustering with each single-impact-scale view."""

    names = list(signatures["feature_names"])
    raw = signatures["raw"]
    values: dict[str, float] = {}
    for quantile in quantiles:
        prefix = f"q{int(round(100 * quantile)):02d}_"
        indices = list(range(7)) + [
            index
            for index, name in enumerate(names)
            if name.startswith("hard_") or name.startswith(prefix)
        ]
        view = raw[:, indices]
        scale = view.std(axis=0, keepdims=True)
        standardized = (view - view.mean(axis=0, keepdims=True)) / np.where(
            scale > 0,
            scale,
            1.0,
        )
        view_labels = cluster_signatures(
            standardized,
            random_state=random_state,
            damping=damping,
        )["labels"]
        values[str(quantile)] = float(adjusted_rand_score(full_labels, view_labels))
    return {
        "adjusted_rand_vs_full": values,
        "mean_adjusted_rand": float(mean(values.values())) if values else 1.0,
    }


def _pairwise_mechanism_alignment(
    structural_similarity: np.ndarray,
    class_effects: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int,
    permutations: int,
) -> dict[str, Any]:
    width = structural_similarity.shape[0]
    upper = np.triu_indices(width, k=1)
    effect_similarity = _cosine_similarity(class_effects)
    structural_values = structural_similarity[upper]
    effect_values = effect_similarity[upper]
    if (
        structural_values.size < 2
        or np.ptp(structural_values) == 0
        or np.ptp(effect_values) == 0
    ):
        rho = None
    else:
        rho = float(spearmanr(structural_values, effect_values).statistic)
    same_cluster = labels[upper[0]] == labels[upper[1]]
    observed = (
        float(effect_values[same_cluster].mean() - effect_values[~same_cluster].mean())
        if same_cluster.any() and (~same_cluster).any()
        else 0.0
    )
    generator = np.random.default_rng(random_state)
    null_values = []
    for _ in range(permutations):
        permuted = generator.permutation(labels)
        within = permuted[upper[0]] == permuted[upper[1]]
        null_values.append(
            float(effect_values[within].mean() - effect_values[~within].mean())
            if within.any() and (~within).any()
            else 0.0
        )
    p_value = (
        (1 + sum(value >= observed for value in null_values))
        / (permutations + 1)
        if permutations
        else None
    )
    return {
        "spearman_structural_vs_class_effect_similarity": rho,
        "within_minus_between_class_effect_similarity": observed,
        "permutation_p_value": p_value,
        "permutations": permutations,
    }


@torch.no_grad()
def cluster_class_mapping(
    loaded: LoadedRun,
    *,
    layer: int,
    labels: np.ndarray,
    training_mean: torch.Tensor,
    base_logits: torch.Tensor,
    threshold_std: float,
) -> dict[str, Any]:
    """Mean-ablate structural clusters and map their KL effects to classes."""

    records: dict[str, Any] = {}
    for cluster in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == cluster).astype(int).tolist()

        def replace_cluster(
            active_layer: int,
            activation: torch.Tensor,
            *,
            expected_layer: int = layer,
            expected_members: tuple[int, ...] = tuple(members),
        ) -> torch.Tensor:
            if active_layer != expected_layer:
                return activation
            modified = activation.clone()
            index = torch.tensor(
                expected_members,
                dtype=torch.long,
                device=activation.device,
            )
            modified[:, index] = training_mean[index]
            return modified

        modified_logits = loaded.model(
            loaded.data.x,
            loaded.data.adjacency,
            activation_intervention=replace_cluster,
        )
        class_kl, class_ce_delta = _classwise_effects(
            base_logits,
            modified_logits,
            loaded,
        )
        values = np.asarray(class_kl, dtype=np.float64)
        # Thesis-continuity literal rule: the class-specific KL must exceed a
        # user-fixed multiple of the within-cluster class-effect standard
        # deviation.  The complete vector is retained because this threshold
        # was under-specified historically and will need sensitivity analysis.
        cutoff = float(threshold_std * values.std())
        mapped = np.flatnonzero(values >= cutoff).astype(int).tolist()
        if not mapped:
            mapped = [int(values.argmax())]
        records[str(cluster)] = {
            "members": members,
            "class_kl": class_kl,
            "class_ce_delta": class_ce_delta,
            "mapping_threshold": cutoff,
            "mapped_classes": mapped,
        }
    return records


def structural_ranking(
    clustering: dict[str, Any],
    cluster_mapping: dict[str, Any],
    standardized_signatures: np.ndarray,
    *,
    target_class: int,
) -> tuple[list[int], list[float]]:
    """Rank clusters by target selectivity and members by exemplar similarity."""

    similarity = clustering["similarity"]
    cluster_order = []
    cluster_scores: dict[int, float] = {}
    for cluster_text, record in cluster_mapping.items():
        cluster = int(cluster_text)
        effects = np.asarray(record["class_kl"], dtype=np.float64)
        target_effect = float(effects[target_class])
        non_target = np.delete(effects, target_class)
        selectivity = target_effect - float(non_target.mean())
        cluster_scores[cluster] = selectivity
        cluster_order.append((-selectivity, -target_effect, cluster))
    ranking: list[int] = []
    scores = np.zeros(standardized_signatures.shape[0], dtype=np.float64)
    for _, _, cluster in sorted(cluster_order):
        members = [int(value) for value in clustering["members"][str(cluster)]]
        center = int(clustering["centers"][str(cluster)])
        ordered_members = sorted(
            members,
            key=lambda member: (-float(similarity[member, center]), member),
        )
        ranking.extend(ordered_members)
        for member in members:
            scores[member] = cluster_scores[cluster]
    if sorted(ranking) != list(range(standardized_signatures.shape[0])):
        raise RuntimeError("Structural ranking is not a complete neuron permutation")
    return ranking, scores.tolist()


def _ranking(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _rank_fusion(first: list[int], second: list[int]) -> list[int]:
    if sorted(first) != sorted(second) or sorted(first) != list(range(len(first))):
        raise ValueError("Fusion inputs must be complete rankings of the same neurons")
    width = len(first)
    first_position = {neuron: position for position, neuron in enumerate(first)}
    second_position = {neuron: position for position, neuron in enumerate(second)}
    return sorted(
        range(width),
        key=lambda neuron: (
            first_position[neuron] + second_position[neuron],
            max(first_position[neuron], second_position[neuron]),
            neuron,
        ),
    )


def _random_ranking(
    width: int,
    *,
    experiment: str,
    seed: int,
    target_class: int,
    layer: int,
    repeat: int,
) -> list[int]:
    material = (
        f"structural:{experiment}:{seed}:{target_class}:{layer}:{repeat}"
    ).encode("utf-8")
    random_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return np.random.default_rng(random_seed).permutation(width).astype(int).tolist()


def _top_k_overlap(first: list[int], second: list[int], budget: int) -> float:
    return len(set(first[:budget]) & set(second[:budget])) / budget


def analyze_loaded_run(
    loaded: LoadedRun,
    *,
    target_class: int,
    layer: int,
    budget: int,
    functional_method: str,
    structural_config: dict[str, Any],
    repair_training: dict[str, Any],
    random_repeats: int,
    selected_splits: tuple[str, ...],
) -> dict[str, Any]:
    """Build signatures, map clusters, and run matched sparse-repair probes."""

    seed = int(loaded.result["seed"])
    set_deterministic(
        seed,
        num_threads=int(repair_training.get("num_threads", 1)),
    )
    impact_scope = str(structural_config.get("impact_scope", "train_validation"))
    quantiles = tuple(
        float(value)
        for value in structural_config.get(
            "impact_quantiles",
            DEFAULT_IMPACT_QUANTILES,
        )
    )
    minimum_impact = float(structural_config.get("minimum_impact", 1e-12))
    damping = float(structural_config.get("affinity_damping", 0.8))
    threshold_std = float(structural_config.get("class_mapping_std", 1.0))
    permutations = int(structural_config.get("alignment_permutations", 200))
    if budget < 1 or random_repeats < 1:
        raise ValueError("Budget and random repeats must be positive")

    adjacency = canonical_undirected_adjacency(
        loaded.data.edge_index,
        loaded.data.y.numel(),
    )
    node_features, node_feature_names, _ = graph_node_features(adjacency)
    impacts = collect_neuron_impacts(
        loaded,
        layer=layer,
        impact_scope=impact_scope,
    )
    width = impacts["node_impacts"].shape[0]
    if budget > width:
        raise ValueError("Repair budget exceeds the selected activation width")
    signatures = structural_signatures(
        adjacency,
        node_features,
        node_feature_names,
        impacts["node_impacts"],
        impacts["prediction_changes"],
        impacts["scope_mask"],
        quantiles=quantiles,
        minimum_impact=minimum_impact,
    )
    clustering = cluster_signatures(
        signatures["standardized"],
        random_state=seed,
        damping=damping,
        max_iterations=int(structural_config.get("affinity_max_iterations", 500)),
        convergence_iterations=int(
            structural_config.get("affinity_convergence_iterations", 30)
        ),
    )
    stability = clustering_stability(
        signatures,
        clustering["labels"],
        quantiles=quantiles,
        random_state=seed,
        damping=damping,
    )
    mechanism_alignment = _pairwise_mechanism_alignment(
        clustering["similarity"],
        impacts["class_kl"],
        clustering["labels"],
        random_state=seed,
        permutations=permutations,
    )
    cluster_mapping = cluster_class_mapping(
        loaded,
        layer=layer,
        labels=clustering["labels"],
        training_mean=impacts["training_mean"],
        base_logits=impacts["base_logits"],
        threshold_std=threshold_std,
    )
    structural_order, structural_scores = structural_ranking(
        clustering,
        cluster_mapping,
        signatures["standardized"],
        target_class=target_class,
    )
    functional_layers = gradient_attribution_scores(loaded, target_class)
    if not 0 <= layer < len(functional_layers):
        raise ValueError("Functional attribution is missing the selected layer")
    if functional_method not in functional_layers[layer]:
        raise ValueError(
            f"Functional attribution method {functional_method!r} is unavailable"
        )
    functional_scores = [
        float(value) for value in functional_layers[layer][functional_method]
    ]
    functional_order = _ranking(functional_scores)
    fused_order = _rank_fusion(structural_order, functional_order)
    thesis_intersection = sorted(
        set(structural_order[:budget]) & set(functional_order[:budget])
    )

    rankings = {
        "structural": structural_order,
        "functional": functional_order,
        "fused": fused_order,
    }
    for repeat in range(random_repeats):
        rankings[f"matched_random_{repeat}"] = _random_ranking(
            width,
            experiment=str(loaded.result["experiment"]),
            seed=seed,
            target_class=target_class,
            layer=layer,
            repeat=repeat,
        )

    repairs: dict[str, Any] = {}
    for method, order in rankings.items():
        result = fit_repair(
            loaded,
            target_class=target_class,
            sites={layer: order[:budget]},
            training=repair_training,
            selected_splits=selected_splits,
        )
        repairs[method] = {
            "selected_neurons": order[:budget],
            **result,
        }

    return {
        "experiment": str(loaded.result["experiment"]),
        "seed": seed,
        "checkpoint": str(loaded.result["checkpoint"]),
        "checkpoint_model_sha256": model_state_sha256(loaded.checkpoint),
        "dataset_sha256": str(loaded.data.metadata["sha256"]),
        "split_indices_sha256": split_indices_sha256(loaded.data),
        "canonical_edge_sha256": canonical_edge_sha256(adjacency),
        "target_class": target_class,
        "layer": layer,
        "width": width,
        "budget": budget,
        "impact_scope": impact_scope,
        "impact_quantiles": list(quantiles),
        "feature_names": list(signatures["feature_names"]),
        "signatures": signatures["raw"].tolist(),
        "affected_subgraphs": signatures["affected_subgraphs"],
        "individual_class_kl": impacts["class_kl"].tolist(),
        "individual_class_ce_delta": impacts["class_ce_delta"].tolist(),
        "clustering": {
            "labels": clustering["labels"].astype(int).tolist(),
            "members": clustering["members"],
            "centers": clustering["centers"],
            "cluster_count": clustering["cluster_count"],
            "silhouette_cosine": clustering["silhouette_cosine"],
            "iterations": clustering["iterations"],
            "converged": clustering["converged"],
            "stability": stability,
        },
        "mechanism_alignment": mechanism_alignment,
        "cluster_class_mapping": cluster_mapping,
        "rankings": {
            "structural": {
                "scores": structural_scores,
                "ranking": structural_order,
            },
            "functional": {
                "method": functional_method,
                "scores": functional_scores,
                "ranking": functional_order,
            },
            "fused": {"ranking": fused_order},
            "thesis_proxy_intersection": {
                "structural_top_k": structural_order[:budget],
                "functional_top_k": functional_order[:budget],
                "intersection": thesis_intersection,
                "intersection_size": len(thesis_intersection),
                "jaccard": (
                    len(thesis_intersection)
                    / len(
                        set(structural_order[:budget])
                        | set(functional_order[:budget])
                    )
                ),
                "empty": not thesis_intersection,
                "performance_note": (
                    "Reported for thesis continuity; not repaired because its "
                    "variable size is not budget matched and no backfill is hidden"
                ),
            },
            "top_k_overlap": {
                "structural_functional": _top_k_overlap(
                    structural_order,
                    functional_order,
                    budget,
                ),
                "structural_fused": _top_k_overlap(
                    structural_order,
                    fused_order,
                    budget,
                ),
                "functional_fused": _top_k_overlap(
                    functional_order,
                    fused_order,
                    budget,
                ),
            },
        },
        "repairs": repairs,
        "evaluation_splits": list(evaluation_splits(selected_splits)),
        "test_status": test_status(selected_splits),
    }


def _run_paths(root: Path) -> list[Path]:
    paths = list((root / "runs").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No run records found under {root / 'runs'}")

    def seed(path: Path) -> int:
        with path.open("r", encoding="utf-8") as stream:
            return int(json.load(stream)["seed"])

    return sorted(paths, key=seed)


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "mean": mean(values),
        "median": median(values),
        "population_std": pstdev(values) if len(values) > 1 else 0.0,
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "positive_fraction": sum(value > 0 for value in values) / len(values),
        "values": values,
    }


def _method_delta(
    run: dict[str, Any],
    method: str,
    split: str,
    metric: str,
) -> float:
    if metric == "utility":
        return float(run["repairs"][method]["final_diagnostics"][split]["utility"])
    return float(run["repairs"][method]["effects"][split]["delta"][metric])


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    selected_splits = tuple(runs[0]["evaluation_splits"])
    methods = ("structural", "functional", "fused", "matched_random")
    metrics = (*EFFECT_METRICS, "utility")
    report: dict[str, Any] = {
        "cluster_count": _summary(
            [float(run["clustering"]["cluster_count"]) for run in runs]
        ),
        "silhouette_cosine": _summary(
            [
                float(run["clustering"]["silhouette_cosine"])
                for run in runs
                if run["clustering"]["silhouette_cosine"] is not None
            ]
        )
        if any(run["clustering"]["silhouette_cosine"] is not None for run in runs)
        else None,
        "stability_adjusted_rand": _summary(
            [
                float(run["clustering"]["stability"]["mean_adjusted_rand"])
                for run in runs
            ]
        ),
        "mechanism_alignment": {},
        "repair": {},
        "paired_contrasts": {},
    }
    for key in (
        "spearman_structural_vs_class_effect_similarity",
        "within_minus_between_class_effect_similarity",
    ):
        values = [
            float(run["mechanism_alignment"][key])
            for run in runs
            if run["mechanism_alignment"][key] is not None
        ]
        report["mechanism_alignment"][key] = _summary(values) if values else None

    for split in selected_splits:
        report["repair"][split] = {}
        random_names_by_run = [
            sorted(
                name
                for name in run["repairs"]
                if name.startswith("matched_random_")
            )
            for run in runs
        ]
        for method in methods:
            report["repair"][split][method] = {}
            for metric in metrics:
                values = []
                for run, random_names in zip(runs, random_names_by_run, strict=True):
                    if method == "matched_random":
                        value = mean(
                            _method_delta(run, name, split, metric)
                            for name in random_names
                        )
                    else:
                        value = _method_delta(run, method, split, metric)
                    values.append(float(value))
                report["repair"][split][method][metric] = _summary(values)

        contrasts = {
            "structural_minus_functional": ("structural", "functional"),
            "fused_minus_functional": ("fused", "functional"),
            "structural_minus_random": ("structural", "matched_random"),
            "fused_minus_random": ("fused", "matched_random"),
        }
        report["paired_contrasts"][split] = {}
        for name, (first, second) in contrasts.items():
            report["paired_contrasts"][split][name] = {}
            for metric in metrics:
                first_values = report["repair"][split][first][metric]["values"]
                second_values = report["repair"][split][second][metric]["values"]
                report["paired_contrasts"][split][name][metric] = _summary(
                    [
                        float(left - right)
                        for left, right in zip(first_values, second_values, strict=True)
                    ]
                )
    return report


def _cache_identity(
    loaded: LoadedRun,
    *,
    target_class: int,
    layer: int,
    budget: int,
    functional_method: str,
    structural_config: dict[str, Any],
    repair_training: dict[str, Any],
    random_repeats: int,
    selected_splits: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "version": STRUCTURAL_PROTOCOL_VERSION,
        "structural_code_sha256": source_code_sha256(STRUCTURAL_CODE_FILES),
        "experiment": str(loaded.result["experiment"]),
        "seed": int(loaded.result["seed"]),
        "checkpoint_model_sha256": model_state_sha256(loaded.checkpoint),
        "dataset_sha256": str(loaded.data.metadata["sha256"]),
        "split_indices_sha256": split_indices_sha256(loaded.data),
        "target_class": target_class,
        "layer": layer,
        "budget": budget,
        "functional_method": functional_method,
        "structural_config": structural_config,
        "repair_training": repair_training,
        "random_repeats": random_repeats,
        "evaluation_splits": list(evaluation_splits(selected_splits)),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def _analyze_with_cache(
    loaded: LoadedRun,
    *,
    target_class: int,
    layer: int,
    budget: int,
    functional_method: str,
    structural_config: dict[str, Any],
    repair_training: dict[str, Any],
    random_repeats: int,
    selected_splits: tuple[str, ...],
    cache_path: Path,
    resume: bool,
) -> dict[str, Any]:
    identity = _cache_identity(
        loaded,
        target_class=target_class,
        layer=layer,
        budget=budget,
        functional_method=functional_method,
        structural_config=structural_config,
        repair_training=repair_training,
        random_repeats=random_repeats,
        selected_splits=selected_splits,
    )
    if resume and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("identity") != identity:
            raise RuntimeError(
                f"Structural cache {cache_path} differs from the active protocol"
            )
        return cached["run"]
    run = analyze_loaded_run(
        loaded,
        target_class=target_class,
        layer=layer,
        budget=budget,
        functional_method=functional_method,
        structural_config=structural_config,
        repair_training=repair_training,
        random_repeats=random_repeats,
        selected_splits=selected_splits,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            {"identity": identity, "run": run},
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(cache_path)
    return run


def build_structural_report(
    config: dict[str, Any],
    *,
    allow_download: bool = False,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Run the structural development study over fixed checkpoint strata."""

    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Structural config requires a non-empty datasets mapping")
    structural_config = dict(config.get("structural", {}))
    repair_training = dict(config["training"])
    random_repeats = int(config.get("random_repeats", 2))
    cache_root = Path(config.get("cache_root", "artifacts/structural_runs"))
    selected_splits = evaluation_splits(config.get("evaluation_splits"))
    code_hash = source_code_sha256(STRUCTURAL_CODE_FILES)
    report: dict[str, Any] = {
        "protocol": {
            "version": STRUCTURAL_PROTOCOL_VERSION,
            "status": "exploratory development on previously inspected graphs",
            "config_sha256": canonical_payload_sha256(config),
            "structural_code_sha256": code_hash,
            "central_method": (
                "mean-ablate each channel, form multiscale affected-node induced "
                "subgraphs, pool deterministic topology signatures, cluster with "
                "affinity propagation, and map clusters to classes by validation KL"
            ),
            "cluster_fitting_labels": "none",
            "cluster_class_mapping_labels": "validation only",
            "repair_fit_labels": "training only",
            "repair_selection_labels": "validation only",
            "evaluation_splits": list(selected_splits),
            "test_status": test_status(selected_splits),
            "fresh_holdouts_used": False,
            "confirmatory_artifacts_used": False,
            "structural": structural_config,
            "training": repair_training,
            "random_repeats": random_repeats,
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
    for dataset_name, dataset_config in datasets.items():
        dataset_record: dict[str, Any] = {"backbones": {}}
        backbones = dataset_config.get("backbones", {})
        if not backbones:
            raise ValueError(f"Dataset {dataset_name!r} has no backbones")
        for backbone_name, backbone_config in backbones.items():
            root = Path(backbone_config["root"])
            target_class = int(backbone_config["target_class"])
            layer = int(backbone_config["layer"])
            budget = int(backbone_config["budget"])
            functional_method = str(backbone_config["functional_method"])
            paths = _run_paths(root)
            max_runs = backbone_config.get("max_runs")
            if max_runs is not None:
                paths = paths[: int(max_runs)]
            runs = []
            for run_number, run_path in enumerate(paths, start=1):
                if progress:
                    print(
                        f"[{dataset_name}/{backbone_name}] structural run "
                        f"{run_number}/{len(paths)}",
                        flush=True,
                    )
                loaded = load_trained_run(run_path, allow_download=allow_download)
                runs.append(
                    _analyze_with_cache(
                        loaded,
                        target_class=target_class,
                        layer=layer,
                        budget=budget,
                        functional_method=functional_method,
                        structural_config=structural_config,
                        repair_training=repair_training,
                        random_repeats=random_repeats,
                        selected_splits=selected_splits,
                        cache_path=(
                            cache_root
                            / dataset_name
                            / backbone_name
                            / f"seed_{int(loaded.result['seed'])}.json"
                        ),
                        resume=resume,
                    )
                )
            dataset_record["backbones"][backbone_name] = {
                "target_class": target_class,
                "layer": layer,
                "budget": budget,
                "functional_method": functional_method,
                "runs": runs,
                "summary": _aggregate_runs(runs),
            }
        report["datasets"][dataset_name] = dataset_record
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/structural_development.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow a missing already-declared development dataset to be downloaded",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only exact structural caches with matching protocol identities",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_structural_report(
        config,
        allow_download=args.allow_download,
        resume=args.resume,
        progress=True,
    )
    output = args.output or Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(output)
    print(f"Wrote structural development report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
