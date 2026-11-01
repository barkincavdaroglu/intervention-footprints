"""Deterministic graph and neuron-label nulls for topology Gate 1."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

from gnn_repair.structural import canonical_edge_sha256
from gnn_repair.topology_gate1 import degree_preserving_rewire


def _stable_seed(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _canonical_adjacency(adjacency: sp.spmatrix) -> sp.csr_matrix:
    if not sp.issparse(adjacency):
        raise TypeError("adjacency must be a SciPy sparse matrix")
    graph = adjacency.tocsr(copy=True).astype(np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be square")
    graph.eliminate_zeros()
    if graph.nnz:
        graph.data = np.ones(graph.nnz, dtype=np.float64)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph.sum_duplicates()
    graph.sort_indices()
    return graph


def complete_degree_preserving_rewires(
    adjacency: sp.spmatrix,
    *,
    repetitions: int,
    swaps_per_edge: float,
    random_seed: int,
    max_attempt_multiplier: int = 20,
    retry_multiplier: int = 5,
) -> tuple[tuple[sp.csr_matrix, ...], dict[str, Any]]:
    """Return only complete double-edge-swap nulls, with failed attempts audited."""

    graph = _canonical_adjacency(adjacency)
    if repetitions < 0:
        raise ValueError("repetitions must be non-negative")
    if not math.isfinite(swaps_per_edge) or swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be finite and non-negative")
    if max_attempt_multiplier < 1 or retry_multiplier < 1:
        raise ValueError("attempt and retry multipliers must be positive")
    edge_count = graph.nnz // 2
    requested_swaps = int(math.ceil(edge_count * swaps_per_edge))
    maximum_trials = repetitions * retry_multiplier
    accepted: list[sp.csr_matrix] = []
    attempts: list[dict[str, Any]] = []
    for trial in range(maximum_trials):
        if len(accepted) >= repetitions:
            break
        seed = _stable_seed(random_seed, "complete-degree-rewire", trial)
        rewired, metadata = degree_preserving_rewire(
            graph,
            swaps=requested_swaps,
            random_seed=seed,
            max_attempts=max(100, requested_swaps * max_attempt_multiplier),
        )
        complete = metadata["successful_swaps"] == metadata["requested_swaps"]
        metadata = {
            **metadata,
            "trial": trial,
            "accepted": bool(complete),
            "rejection_reason": (
                None if complete else "requested edge swaps were not completed"
            ),
        }
        attempts.append(metadata)
        if complete:
            accepted.append(rewired)
    return tuple(accepted), {
        "method": "complete simple-undirected double-edge swaps",
        "original_edge_sha256": canonical_edge_sha256(graph),
        "edge_count": int(edge_count),
        "requested_repetitions": int(repetitions),
        "valid_repetitions": len(accepted),
        "evaluable": len(accepted) == repetitions,
        "requested_swaps_per_repetition": requested_swaps,
        "swaps_per_edge": float(swaps_per_edge),
        "max_attempt_multiplier": int(max_attempt_multiplier),
        "retry_multiplier": int(retry_multiplier),
        "attempts": attempts,
        "not_evaluable_reason": (
            None
            if len(accepted) == repetitions
            else "insufficient complete degree-preserving rewires"
        ),
    }


def deterministic_edge_dropout(
    adjacency: sp.spmatrix,
    *,
    fraction: float,
    random_seed: int,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Drop an exact deterministic fraction of undirected edges."""

    graph = _canonical_adjacency(adjacency)
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("fraction must be finite and in [0, 1)")
    upper = sp.triu(graph, k=1).tocoo()
    pairs = np.column_stack((upper.row, upper.col)).astype(np.int64, copy=False)
    edge_count = int(pairs.shape[0])
    drop_count = int(round(fraction * edge_count))
    generator = np.random.default_rng(random_seed)
    dropped = (
        np.sort(generator.choice(edge_count, size=drop_count, replace=False))
        if drop_count
        else np.empty(0, dtype=np.int64)
    )
    keep = np.ones(edge_count, dtype=bool)
    keep[dropped] = False
    remaining = pairs[keep]
    if remaining.size:
        rows = np.concatenate((remaining[:, 0], remaining[:, 1]))
        columns = np.concatenate((remaining[:, 1], remaining[:, 0]))
        result = sp.csr_matrix(
            (np.ones(rows.size, dtype=np.float64), (rows, columns)),
            shape=graph.shape,
        )
    else:
        result = sp.csr_matrix(graph.shape, dtype=np.float64)
    result.sort_indices()
    return result, {
        "method": "uniform edge dropout without replacement",
        "random_seed": int(random_seed),
        "fraction": float(fraction),
        "original_edge_count": edge_count,
        "dropped_edge_count": drop_count,
        "remaining_edge_count": int(result.nnz // 2),
        "canonical_edge_sha256": canonical_edge_sha256(result),
    }


def _validate_kernel(kernel: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(kernel, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains a non-finite value")
    if not np.allclose(values, values.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    return values


def _normalized_rank_vector(kernel: np.ndarray) -> np.ndarray | None:
    upper = np.triu_indices(kernel.shape[0], k=1)
    values = kernel[upper]
    if values.size < 2 or np.ptp(values) == 0:
        return None
    ranks = rankdata(values, method="average").astype(np.float64)
    ranks -= ranks.mean()
    norm = float(np.linalg.norm(ranks))
    return ranks / norm if norm > 0 else None


def paired_incremental_permutation_test(
    control_kernels: Mapping[str, np.ndarray],
    candidate_kernels: Mapping[str, np.ndarray],
    effect_kernel: np.ndarray,
    *,
    repetitions: int,
    random_seed: int,
    channel_indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Test candidate-minus-best-control alignment using common neuron permutations."""

    if not control_kernels or not candidate_kernels:
        raise ValueError("At least one control and candidate kernel are required")
    if repetitions < 0:
        raise ValueError("repetitions must be non-negative")
    effect = _validate_kernel(effect_kernel, "effect_kernel")
    width = effect.shape[0]
    indices = (
        np.arange(width, dtype=np.int64)
        if channel_indices is None
        else np.asarray(channel_indices, dtype=np.int64)
    )
    if indices.ndim != 1 or np.unique(indices).size != indices.size:
        raise ValueError("channel_indices must be a unique vector")
    if indices.size and (indices.min() < 0 or indices.max() >= width):
        raise ValueError("channel_indices contains an invalid channel")
    if indices.size < 3:
        return {
            "evaluable": False,
            "not_evaluable_reason": "fewer than three channels",
            "repetitions": repetitions,
            "random_seed": random_seed,
            "controls": {},
            "candidates": {},
        }

    def subset(mapping: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = {}
        for name, kernel in sorted(mapping.items()):
            checked = _validate_kernel(kernel, name)
            if checked.shape != (width, width):
                raise ValueError(f"{name} does not align with effect_kernel")
            result[str(name)] = checked[np.ix_(indices, indices)]
        return result

    controls = subset(control_kernels)
    candidates = subset(candidate_kernels)
    scoped_effect = effect[np.ix_(indices, indices)]
    effect_vector = _normalized_rank_vector(scoped_effect)
    control_vectors = {
        name: _normalized_rank_vector(kernel) for name, kernel in controls.items()
    }
    candidate_vectors = {
        name: _normalized_rank_vector(kernel) for name, kernel in candidates.items()
    }
    valid_controls = {
        name: vector for name, vector in control_vectors.items() if vector is not None
    }
    valid_candidates = {
        name: vector for name, vector in candidate_vectors.items() if vector is not None
    }
    if effect_vector is None or not valid_controls or not valid_candidates:
        return {
            "evaluable": False,
            "not_evaluable_reason": "effect, control, or candidate ranks are constant",
            "repetitions": repetitions,
            "random_seed": random_seed,
            "controls": {},
            "candidates": {},
        }

    control_names = tuple(valid_controls)
    candidate_names = tuple(valid_candidates)
    control_matrix = np.stack([valid_controls[name] for name in control_names])
    candidate_matrix = np.stack([valid_candidates[name] for name in candidate_names])
    observed_controls = control_matrix @ effect_vector
    observed_candidates = candidate_matrix @ effect_vector
    best_index = int(np.argmax(observed_controls))
    best_name = control_names[best_index]
    best_value = float(observed_controls[best_index])
    observed_increment = observed_candidates - best_value

    upper = np.triu_indices(indices.size, k=1)
    effect_ranks = np.zeros_like(scoped_effect, dtype=np.float64)
    raw_ranks = rankdata(scoped_effect[upper], method="average").astype(np.float64)
    effect_ranks[upper] = raw_ranks
    effect_ranks[(upper[1], upper[0])] = raw_ranks
    generator = np.random.default_rng(random_seed)
    null_increment = np.empty((repetitions, len(candidate_names)), dtype=np.float64)
    for repeat in range(repetitions):
        permutation = generator.permutation(indices.size)
        values = effect_ranks[np.ix_(permutation, permutation)][upper]
        values = values - values.mean()
        norm = float(np.linalg.norm(values))
        if norm == 0:
            null_increment[repeat] = 0.0
            continue
        normalized = values / norm
        control_alignment = control_matrix @ normalized
        candidate_alignment = candidate_matrix @ normalized
        null_increment[repeat] = candidate_alignment - float(
            control_alignment.max()
        )

    candidate_records: dict[str, Any] = {}
    for position, name in enumerate(candidate_names):
        null = null_increment[:, position]
        observed = float(observed_candidates[position])
        increment = float(observed_increment[position])
        candidate_records[name] = {
            "observed_alignment": observed,
            "best_control": best_name,
            "best_control_alignment": best_value,
            "incremental_alignment": increment,
            "p_increment_greater": (
                float((1 + np.count_nonzero(null >= increment)) / (repetitions + 1))
                if repetitions
                else None
            ),
            "null_increment_mean": float(null.mean()) if repetitions else None,
            "null_increment_std": float(null.std()) if repetitions else None,
        }
    return {
        "evaluable": True,
        "not_evaluable_reason": None,
        "statistic": "Spearman kernel alignment minus strongest control",
        "permutation_scheme": (
            "one common simultaneous neuron row/column permutation per null draw"
        ),
        "channel_count": int(indices.size),
        "repetitions": int(repetitions),
        "random_seed": int(random_seed),
        "controls": {
            name: {"observed_alignment": float(observed_controls[position])}
            for position, name in enumerate(control_names)
        },
        "candidates": candidate_records,
    }

