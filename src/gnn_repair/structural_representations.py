"""Deterministic sparse representations of intervention-effect structure.

The functions in this module deliberately keep two different objects separate:

* an impact-defined support can be represented using only its induced topology;
* a continuous impact field can be normalized and diffused over graph topology.

The returned metadata records that distinction so a topology-only Gate 1 result
cannot accidentally include impact magnitude as a Weisfeiler--Lehman node label.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


DEFAULT_WL_MAX_DEPTH = 4
DEFAULT_DIFFUSION_TIMES = (0.01, 0.1, 1.0, 10.0)
DEFAULT_DIFFUSION_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_CHARACTERISTIC_FREQUENCIES = (1.0, 2.0, 4.0)

WL_INITIAL_LABEL_MODES = (
    "constant",
    "induced_degree_bin",
    "parent_degree_bin",
    "parent_degree_core_bin",
    "fused_impact_rank_bin",
)


@dataclass(frozen=True)
class WLSubtreeFeatures:
    """Cumulative, row-normalized WL feature maps for depths zero through h."""

    by_depth: tuple[sp.csr_matrix, ...]
    feature_names: tuple[str, ...]
    node_counts: np.ndarray
    active_mask: np.ndarray
    block_name: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ImpactDiffusionSignatures:
    """Summaries of L1-normalized intervention fields after heat diffusion."""

    signatures: np.ndarray
    feature_names: tuple[str, ...]
    magnitudes: np.ndarray
    active_mask: np.ndarray
    l1_norms_by_time: np.ndarray
    block_names: dict[str, str]
    metadata: dict[str, Any]


def _canonical_sparse_adjacency(adjacency: sp.spmatrix) -> sp.csr_matrix:
    """Return a simple, loop-free, undirected CSR graph without densifying it."""

    if not sp.issparse(adjacency):
        raise TypeError("adjacency must be a SciPy sparse matrix")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    matrix = adjacency.tocsr(copy=True)
    if matrix.data.size and not np.isfinite(matrix.data).all():
        raise ValueError("adjacency contains a non-finite edge weight")
    matrix.eliminate_zeros()
    if matrix.nnz:
        matrix.data = np.ones(matrix.nnz, dtype=np.float64)
    matrix = matrix.maximum(matrix.T).tocsr()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix.astype(np.float64, copy=False)


def _validate_impact_inputs(
    adjacency: sp.spmatrix,
    impacts: np.ndarray,
    scope_mask: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    graph = _canonical_sparse_adjacency(adjacency)
    values = np.asarray(impacts, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != graph.shape[0]:
        raise ValueError("impacts must have shape [channels, graph nodes]")
    if not np.isfinite(values).all():
        raise ValueError("impacts contain a non-finite value")
    if np.any(values < 0):
        raise ValueError(
            "impacts must be non-negative; diffuse signed effects as separate "
            "positive and negative channels"
        )
    scope = np.asarray(scope_mask)
    if scope.shape != (graph.shape[0],):
        raise ValueError("scope_mask must have one entry per graph node")
    if scope.dtype != np.bool_:
        if not np.all((scope == 0) | (scope == 1)):
            raise ValueError("scope_mask must contain only boolean values")
        scope = scope.astype(bool)
    if not bool(scope.any()):
        raise ValueError("scope_mask must select at least one graph node")
    scope_indices = np.flatnonzero(scope)
    scoped_graph = graph[scope_indices][:, scope_indices].tocsr()
    scoped_graph.sort_indices()
    return graph, values, scope.astype(bool, copy=False), scoped_graph


def _canonical_label(value: Any) -> tuple[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ("none", "")
    if isinstance(value, bool):
        return ("bool", int(value))
    if isinstance(value, int):
        return ("int", int(value))
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("initial node labels must be finite")
        return ("float", float(value).hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    raise TypeError(
        "initial node labels must be scalar booleans, integers, finite floats, "
        "strings, bytes, or None"
    )


def _validate_wl_inputs(
    graphs: Iterable[sp.spmatrix],
    initial_labels: Sequence[Sequence[Any] | np.ndarray] | None,
    max_depth: int,
) -> tuple[tuple[sp.csr_matrix, ...], tuple[np.ndarray, ...]]:
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise TypeError("max_depth must be an integer")
    if not 0 <= max_depth <= DEFAULT_WL_MAX_DEPTH:
        raise ValueError(
            f"max_depth must be between 0 and {DEFAULT_WL_MAX_DEPTH}, inclusive"
        )
    canonical_graphs = tuple(_canonical_sparse_adjacency(graph) for graph in graphs)
    if initial_labels is None:
        labels = tuple(
            np.zeros(graph.shape[0], dtype=np.int64) for graph in canonical_graphs
        )
    else:
        provided = tuple(initial_labels)
        if len(provided) != len(canonical_graphs):
            raise ValueError("initial_labels must contain one array per graph")
        converted = []
        for graph, graph_labels in zip(canonical_graphs, provided, strict=True):
            values = np.asarray(graph_labels, dtype=object)
            if values.ndim != 1 or values.shape[0] != graph.shape[0]:
                raise ValueError(
                    "each initial-label array must contain one value per graph node"
                )
            converted.append(values)
        labels = tuple(converted)
    return canonical_graphs, labels


def _l2_normalize_rows(values: sp.csr_matrix) -> sp.csr_matrix:
    matrix = values.tocsr(copy=True).astype(np.float64)
    squared_norm = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    inverse_norm = np.zeros_like(squared_norm)
    nonzero = squared_norm > 0
    inverse_norm[nonzero] = 1.0 / np.sqrt(squared_norm[nonzero])
    normalized = sp.diags(inverse_norm).dot(matrix).tocsr()
    normalized.sort_indices()
    return normalized


def normalized_wl_subtree_features(
    graphs: Iterable[sp.spmatrix],
    *,
    initial_labels: Sequence[Sequence[Any] | np.ndarray] | None = None,
    max_depth: int = DEFAULT_WL_MAX_DEPTH,
    block_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> WLSubtreeFeatures:
    """Build deterministic cumulative WL subtree maps for depths ``0..h``.

    Labels are compressed jointly across the graph collection using sorted tuples.
    Python's randomized hash is never used. Every cumulative map is row-L2
    normalized, so its Gram matrix is the normalized WL subtree kernel. Empty
    graphs receive one explicit feature: two empty supports have similarity one,
    while an empty and a non-empty support have similarity zero.

    When ``initial_labels`` is omitted, every node receives the same label and the
    representation is induced-topology-only. Caller-provided labels are marked as
    ``caller_labeled`` unless the high-level impact-support API supplies more
    specific provenance.
    """

    canonical_graphs, labels = _validate_wl_inputs(
        graphs,
        initial_labels,
        max_depth,
    )
    num_graphs = len(canonical_graphs)
    node_counts = np.asarray(
        [graph.shape[0] for graph in canonical_graphs],
        dtype=np.int64,
    )
    active_mask = node_counts > 0
    supplied_labels = initial_labels is not None
    inferred_block = "caller_labeled" if supplied_labels else "topology_only"
    result_metadata: dict[str, Any] = {
        "representation": "normalized_wl_subtree",
        "feature_block": inferred_block,
        "initial_label_source": "caller_provided" if supplied_labels else "constant",
        "maximum_depth": max_depth,
        "depths": list(range(max_depth + 1)),
        "normalization": "cumulative_row_l2",
        "empty_support": "explicit_token",
    }
    if metadata:
        result_metadata.update(metadata)

    all_label_keys = sorted(
        {
            _canonical_label(value)
            for graph_labels in labels
            for value in graph_labels.tolist()
        }
    )
    initial_color_map = {
        label_key: color for color, label_key in enumerate(all_label_keys)
    }
    colors = tuple(
        np.asarray(
            [initial_color_map[_canonical_label(value)] for value in graph_labels],
            dtype=np.int64,
        )
        for graph_labels in labels
    )

    counts_by_iteration: list[tuple[dict[int, int], ...]] = []
    color_counts: list[int] = []
    for depth in range(max_depth + 1):
        graph_counts = []
        for graph_colors in colors:
            unique, counts = np.unique(graph_colors, return_counts=True)
            graph_counts.append(
                {
                    int(color): int(count)
                    for color, count in zip(unique, counts, strict=True)
                }
            )
        counts_by_iteration.append(tuple(graph_counts))
        color_counts.append(
            max((int(values.max()) for values in colors if values.size), default=-1)
            + 1
        )
        if depth == max_depth:
            break

        descriptors_by_graph: list[list[tuple[int, tuple[int, ...]]]] = []
        descriptor_set: set[tuple[int, tuple[int, ...]]] = set()
        for graph, graph_colors in zip(canonical_graphs, colors, strict=True):
            descriptors = []
            for node in range(graph.shape[0]):
                start, end = graph.indptr[node : node + 2]
                neighbor_colors = tuple(
                    sorted(int(value) for value in graph_colors[graph.indices[start:end]])
                )
                descriptor = (int(graph_colors[node]), neighbor_colors)
                descriptors.append(descriptor)
                descriptor_set.add(descriptor)
            descriptors_by_graph.append(descriptors)
        descriptor_to_color = {
            descriptor: color
            for color, descriptor in enumerate(sorted(descriptor_set))
        }
        colors = tuple(
            np.asarray(
                [descriptor_to_color[descriptor] for descriptor in descriptors],
                dtype=np.int64,
            )
            for descriptors in descriptors_by_graph
        )

    # Column zero is reserved for the explicit empty-support token. Each WL
    # iteration then has a disjoint feature namespace, as in the subtree kernel.
    offsets = [1]
    for count in color_counts[:-1]:
        offsets.append(offsets[-1] + count)
    total_features = 1 + sum(color_counts)
    feature_names = ["empty_support"]
    for depth, count in enumerate(color_counts):
        feature_names.extend(f"wl_h{depth}_color{color}" for color in range(count))

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for graph_index, active in enumerate(active_mask):
        if not active:
            rows.append(graph_index)
            columns.append(0)
            data.append(1.0)
    iteration_matrices = []
    for depth, graph_counts in enumerate(counts_by_iteration):
        depth_rows = list(rows) if depth == 0 else []
        depth_columns = list(columns) if depth == 0 else []
        depth_data = list(data) if depth == 0 else []
        for graph_index, counts in enumerate(graph_counts):
            for color, count in sorted(counts.items()):
                depth_rows.append(graph_index)
                depth_columns.append(offsets[depth] + color)
                depth_data.append(float(count))
        matrix = sp.csr_matrix(
            (depth_data, (depth_rows, depth_columns)),
            shape=(num_graphs, total_features),
            dtype=np.float64,
        )
        matrix.sort_indices()
        iteration_matrices.append(matrix)

    cumulative = sp.csr_matrix(
        (num_graphs, total_features),
        dtype=np.float64,
    )
    normalized_by_depth = []
    for matrix in iteration_matrices:
        cumulative = (cumulative + matrix).tocsr()
        normalized_by_depth.append(_l2_normalize_rows(cumulative))

    return WLSubtreeFeatures(
        by_depth=tuple(normalized_by_depth),
        feature_names=tuple(feature_names),
        node_counts=node_counts,
        active_mask=active_mask,
        block_name=block_name or f"{inferred_block}_wl",
        metadata=result_metadata,
    )


def normalized_feature_kernel(features: sp.spmatrix | np.ndarray) -> np.ndarray:
    """Return the cosine-normalized Gram matrix for sparse or dense features."""

    if sp.issparse(features):
        matrix = features.tocsr(copy=True).astype(np.float64)
        if matrix.data.size and not np.isfinite(matrix.data).all():
            raise ValueError("features contain a non-finite value")
        normalized = _l2_normalize_rows(matrix)
        kernel = normalized.dot(normalized.T).toarray()
    else:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("features contain a non-finite value")
        norms = np.linalg.norm(matrix, axis=1)
        normalized = matrix / np.where(norms > 0, norms, 1.0)[:, None]
        kernel = normalized.dot(normalized.T)
    kernel = 0.5 * (kernel + kernel.T)
    return np.clip(kernel, -1.0, 1.0)


def normalized_wl_subtree_kernels(
    features: WLSubtreeFeatures,
) -> tuple[np.ndarray, ...]:
    """Return one normalized WL Gram matrix for every cumulative depth."""

    return tuple(normalized_feature_kernel(values) for values in features.by_depth)


def multiscale_wl_subtree_kernel(features: WLSubtreeFeatures) -> np.ndarray:
    """Average equally over normalized WL kernels from depth zero through h."""

    kernels = normalized_wl_subtree_kernels(features)
    if not kernels:
        return np.empty((features.node_counts.size, features.node_counts.size))
    return np.mean(np.stack(kernels, axis=0), axis=0)


def _top_impact_nodes(
    impact: np.ndarray,
    scope_indices: np.ndarray,
    quantile: float,
    minimum_impact: float,
) -> np.ndarray:
    count = max(1, int(np.ceil((1.0 - quantile) * scope_indices.size)))
    values = impact[scope_indices]
    order = np.lexsort((scope_indices, -values))
    selected = scope_indices[order[:count]]
    return selected[impact[selected] > minimum_impact]


def _log_degree_bins(degree: np.ndarray) -> np.ndarray:
    return np.floor(np.log2(np.asarray(degree, dtype=np.float64) + 1.0)).astype(
        np.int64
    )


def _core_numbers(adjacency: sp.csr_matrix) -> np.ndarray:
    """Compute deterministic undirected core numbers by heap peeling."""

    degree = np.diff(adjacency.indptr).astype(np.int64)
    active_degree = degree.copy()
    removed = np.zeros(adjacency.shape[0], dtype=bool)
    core = np.zeros(adjacency.shape[0], dtype=np.int64)
    queue = [(int(value), node) for node, value in enumerate(active_degree)]
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


def _impact_rank_bins(
    impact: np.ndarray,
    nodes: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    if nodes.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((nodes, -impact[nodes]))
    ranks = np.empty(nodes.size, dtype=np.int64)
    ranks[order] = np.arange(nodes.size, dtype=np.int64)
    return np.minimum((ranks * bin_count) // nodes.size, bin_count - 1)


def impact_support_wl_subtree_features(
    adjacency: sp.spmatrix,
    impacts: np.ndarray,
    scope_mask: np.ndarray,
    *,
    support_quantile: float = 0.99,
    minimum_impact: float = 1e-12,
    initial_label_mode: str = "constant",
    impact_bin_count: int = 4,
    max_depth: int = DEFAULT_WL_MAX_DEPTH,
) -> WLSubtreeFeatures:
    """Represent per-channel top-impact induced subgraphs with normalized WL.

    ``constant`` and ``induced_degree_bin`` are strict induced-topology blocks.
    Parent degree/core modes add fixed, class-blind roles from the parent graph.
    Only ``fused_impact_rank_bin`` uses continuous impact values as initial WL
    labels, and its block name and metadata say so explicitly. All modes
    necessarily use impact values to *select* the support graph.
    """

    graph, values, scope, _ = _validate_impact_inputs(
        adjacency,
        impacts,
        scope_mask,
    )
    if not 0 < support_quantile < 1:
        raise ValueError("support_quantile must be in (0, 1)")
    if not np.isfinite(minimum_impact) or minimum_impact < 0:
        raise ValueError("minimum_impact must be finite and non-negative")
    if initial_label_mode not in WL_INITIAL_LABEL_MODES:
        raise ValueError(
            "initial_label_mode must be one of " + ", ".join(WL_INITIAL_LABEL_MODES)
        )
    if (
        isinstance(impact_bin_count, bool)
        or not isinstance(impact_bin_count, int)
        or impact_bin_count < 2
    ):
        raise ValueError("impact_bin_count must be an integer of at least two")

    scope_indices = np.flatnonzero(scope)
    parent_degree_bins = _log_degree_bins(np.diff(graph.indptr))
    parent_core = _core_numbers(graph)
    core_base = int(parent_core.max()) + 1 if parent_core.size else 1
    parent_degree_core_bins = parent_degree_bins * core_base + parent_core
    subgraphs = []
    graph_labels: list[np.ndarray] = []
    support_sizes = []
    for impact in values:
        nodes = _top_impact_nodes(
            impact,
            scope_indices,
            support_quantile,
            minimum_impact,
        )
        subgraph = graph[nodes][:, nodes].tocsr()
        subgraph.sort_indices()
        subgraphs.append(subgraph)
        support_sizes.append(int(nodes.size))
        if initial_label_mode == "constant":
            labels = np.zeros(nodes.size, dtype=np.int64)
        elif initial_label_mode == "induced_degree_bin":
            labels = _log_degree_bins(np.diff(subgraph.indptr))
        elif initial_label_mode == "parent_degree_bin":
            labels = parent_degree_bins[nodes]
        elif initial_label_mode == "parent_degree_core_bin":
            labels = parent_degree_core_bins[nodes]
        else:
            labels = _impact_rank_bins(impact, nodes, impact_bin_count)
        graph_labels.append(labels)

    uses_impact_labels = initial_label_mode == "fused_impact_rank_bin"
    uses_parent_roles = initial_label_mode in (
        "parent_degree_bin",
        "parent_degree_core_bin",
    )
    strict_topology_only = not uses_impact_labels and not uses_parent_roles
    if uses_impact_labels:
        feature_block = "fused"
        block_prefix = "fused"
    elif uses_parent_roles:
        feature_block = "topology_plus_parent_roles"
        block_prefix = "topology_role"
    else:
        feature_block = "topology_only"
        block_prefix = "topology"
    metadata = {
        "feature_block": feature_block,
        "initial_label_source": initial_label_mode,
        "uses_impact_values_for_support": True,
        "uses_impact_values_as_node_labels": uses_impact_labels,
        "uses_parent_graph_node_roles": uses_parent_roles,
        "strict_induced_topology_only": strict_topology_only,
        "eligible_for_topology_only_gate": strict_topology_only,
        "uses_total_impact_magnitude_in_feature_values": False,
        "support_quantile": float(support_quantile),
        "scope_size": int(scope_indices.size),
        "minimum_impact": float(minimum_impact),
        "support_sizes": support_sizes,
        "impact_bin_count": impact_bin_count if uses_impact_labels else None,
    }
    return normalized_wl_subtree_features(
        subgraphs,
        initial_labels=graph_labels,
        max_depth=max_depth,
        block_name=f"{block_prefix}_wl_{initial_label_mode}",
        metadata=metadata,
    )


def _normalized_laplacian(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degree, dtype=np.float64)
    positive = degree > 0
    inverse_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized_adjacency = sp.diags(inverse_sqrt).dot(adjacency).dot(
        sp.diags(inverse_sqrt)
    )
    # Isolated vertices receive a zero Laplacian row so heat diffusion preserves
    # a point mass on the isolate instead of artificially decaying it.
    laplacian = sp.diags(positive.astype(np.float64)) - normalized_adjacency
    laplacian = laplacian.tocsr()
    laplacian.sort_indices()
    return laplacian


def _strictly_increasing_positive(
    values: Sequence[float],
    name: str,
) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not converted or not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain finite values")
    if any(value <= 0 for value in converted):
        raise ValueError(f"{name} must contain positive values")
    if any(second <= first for first, second in zip(converted, converted[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return converted


def _validate_quantiles(values: Sequence[float]) -> tuple[float, ...]:
    quantiles = tuple(float(value) for value in values)
    if not quantiles or not np.isfinite(quantiles).all():
        raise ValueError("quantiles must contain finite values")
    if any(not 0 <= value <= 1 for value in quantiles):
        raise ValueError("quantiles must lie in [0, 1]")
    if any(second <= first for first, second in zip(quantiles, quantiles[1:])):
        raise ValueError("quantiles must be strictly increasing")
    return quantiles


def _scale_name(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def impact_diffusion_signatures(
    adjacency: sp.spmatrix,
    impacts: np.ndarray,
    scope_mask: np.ndarray,
    *,
    times: Sequence[float] = DEFAULT_DIFFUSION_TIMES,
    quantiles: Sequence[float] = DEFAULT_DIFFUSION_QUANTILES,
    characteristic_frequencies: Sequence[
        float
    ] = DEFAULT_CHARACTERISTIC_FREQUENCIES,
    minimum_mass: float = 1e-12,
) -> ImpactDiffusionSignatures:
    """Diffuse L1-normalized continuous impact fields on the scoped graph.

    Impact magnitude is removed before diffusion and returned as a separate block.
    Each heat-diffused field is L1-normalized again because the symmetric normalized
    Laplacian does not conserve the ordinary coordinate sum. Distribution quantiles
    use ``n * z`` (mean one for a uniform active signal), while entropy and inverse
    participation are normalized for scope size. Empty supports produce finite zero
    signatures and ``active_mask=False``.
    """

    _, values, scope, scoped_graph = _validate_impact_inputs(
        adjacency,
        impacts,
        scope_mask,
    )
    diffusion_times = _strictly_increasing_positive(times, "times")
    distribution_quantiles = _validate_quantiles(quantiles)
    frequencies = _strictly_increasing_positive(
        characteristic_frequencies,
        "characteristic_frequencies",
    )
    if not np.isfinite(minimum_mass) or minimum_mass < 0:
        raise ValueError("minimum_mass must be finite and non-negative")

    scope_indices = np.flatnonzero(scope)
    scoped_impacts = values[:, scope_indices]
    magnitudes = scoped_impacts.sum(axis=1)
    active_mask = magnitudes > minimum_mass
    normalized_impacts = np.zeros_like(scoped_impacts, dtype=np.float64)
    normalized_impacts[active_mask] = (
        scoped_impacts[active_mask] / magnitudes[active_mask, None]
    )

    feature_names = []
    for time in diffusion_times:
        prefix = f"diffusion_t{_scale_name(time)}"
        feature_names.extend(
            f"{prefix}_quantile_{_scale_name(value)}"
            for value in distribution_quantiles
        )
        feature_names.extend(
            (
                f"{prefix}_normalized_entropy",
                f"{prefix}_inverse_participation_fraction",
                f"{prefix}_graph_total_variation_l1",
            )
        )
        for frequency in frequencies:
            suffix = _scale_name(frequency)
            feature_names.extend(
                (
                    f"{prefix}_ecf_real_{suffix}",
                    f"{prefix}_ecf_imag_{suffix}",
                )
            )

    num_channels = values.shape[0]
    width_per_time = (
        len(distribution_quantiles) + 3 + 2 * len(frequencies)
    )
    signatures = np.zeros(
        (num_channels, len(diffusion_times) * width_per_time),
        dtype=np.float64,
    )
    l1_norms = np.zeros((num_channels, len(diffusion_times)), dtype=np.float64)
    if scoped_graph.shape[0] and num_channels:
        laplacian = _normalized_laplacian(scoped_graph)
        upper = sp.triu(scoped_graph, k=1).tocoo()
        num_nodes = scoped_graph.shape[0]
        log_nodes = np.log(num_nodes) if num_nodes > 1 else 0.0
        for time_index, time in enumerate(diffusion_times):
            operator = (-time * laplacian).tocsr()
            trace = float(operator.diagonal().sum())
            diffused = np.asarray(
                expm_multiply(
                    operator,
                    normalized_impacts.T,
                    traceA=trace,
                )
            ).T
            if diffused.shape != normalized_impacts.shape:
                diffused = diffused.reshape(normalized_impacts.shape)
            if not np.isfinite(diffused).all():
                raise FloatingPointError("heat diffusion produced a non-finite value")
            if diffused.size and float(diffused.min()) < -1e-10:
                raise FloatingPointError("heat diffusion produced a negative mass")
            diffused = np.maximum(diffused, 0.0)
            mass = diffused.sum(axis=1)
            renormalized = np.zeros_like(diffused)
            valid = active_mask & (mass > 0)
            renormalized[valid] = diffused[valid] / mass[valid, None]
            l1_norms[:, time_index] = renormalized.sum(axis=1)

            start = time_index * width_per_time
            scaled = num_nodes * renormalized
            signatures[:, start : start + len(distribution_quantiles)] = np.quantile(
                scaled,
                distribution_quantiles,
                axis=1,
            ).T
            offset = start + len(distribution_quantiles)
            positive = renormalized > 0
            entropy = np.zeros(num_channels, dtype=np.float64)
            entropy[valid] = -np.sum(
                np.where(
                    positive[valid],
                    renormalized[valid]
                    * np.log(np.where(positive[valid], renormalized[valid], 1.0)),
                    0.0,
                ),
                axis=1,
            )
            if log_nodes > 0:
                entropy /= log_nodes
            signatures[:, offset] = entropy
            squared_mass = np.square(renormalized).sum(axis=1)
            inverse_participation = np.zeros(num_channels, dtype=np.float64)
            inverse_participation[valid] = 1.0 / (
                num_nodes * squared_mass[valid]
            )
            signatures[:, offset + 1] = inverse_participation
            if upper.nnz:
                signatures[:, offset + 2] = np.abs(
                    renormalized[:, upper.row] - renormalized[:, upper.col]
                ).sum(axis=1)
            ecf_offset = offset + 3
            for frequency_index, frequency in enumerate(frequencies):
                angles = frequency * scaled
                signatures[:, ecf_offset + 2 * frequency_index] = np.cos(
                    angles
                ).mean(axis=1)
                signatures[:, ecf_offset + 2 * frequency_index + 1] = np.sin(
                    angles
                ).mean(axis=1)
            # ECF of an all-zero field would otherwise be one; retain zero as the
            # explicit empty-support representation across every feature block.
            signatures[~active_mask, start : start + width_per_time] = 0.0

    if not np.isfinite(signatures).all():
        raise FloatingPointError("impact-diffusion signatures are non-finite")
    block_names = {
        "signatures": "fused_normalized_impact_diffusion",
        "magnitudes": "magnitude_only_total_impact",
        "active_mask": "support_activity_indicator",
    }
    metadata = {
        "representation": "impact_diffusion_signature",
        "feature_block": "topology_conditioned_normalized_effect_field",
        "uses_impact_values": True,
        "uses_total_impact_magnitude_in_signatures": False,
        "eligible_for_topology_only_gate": False,
        "magnitude_returned_separately": True,
        "scope_graph": "induced_by_scope_mask",
        "scope_size": int(scope_indices.size),
        "times": list(diffusion_times),
        "quantiles": list(distribution_quantiles),
        "characteristic_frequencies": list(frequencies),
        "laplacian": "symmetric_normalized_with_zero_rows_for_isolates",
        "normalization": "input_and_each_diffusion_row_l1",
        "distribution_value_scaling": "scope_size_times_probability_mass",
        "minimum_mass": float(minimum_mass),
        "empty_support": "zero_signature_with_inactive_mask",
    }
    return ImpactDiffusionSignatures(
        signatures=signatures,
        feature_names=tuple(feature_names),
        magnitudes=magnitudes,
        active_mask=active_mask,
        l1_norms_by_time=l1_norms,
        block_names=block_names,
        metadata=metadata,
    )
