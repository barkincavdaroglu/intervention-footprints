"""Graph dataset loading, validation, splitting, and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split


CITATIONFULL_CITESEER_EXPECTED = {
    "num_nodes": 4_230,
    "num_directed_edges": 10_674,
    "num_features": 602,
    "num_classes": 6,
}

@dataclass(frozen=True)
class GraphData:
    """Tensors and provenance required for a full-batch node-classification run."""

    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    adjacency: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    metadata: dict[str, Any]

    def to(self, device: torch.device | str) -> "GraphData":
        return GraphData(
            x=self.x.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            adjacency=self.adjacency.to(device),
            train_mask=self.train_mask.to(device),
            val_mask=self.val_mask.to(device),
            test_mask=self.test_mask.to(device),
            metadata=self.metadata,
        )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(url: str, destination: Path) -> None:
    """Download a dataset atomically, leaving no partial file on failure."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "gnn-neuron-repair/0.1"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(destination)
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the dataset from {url}. Place the public archive "
            f"at {destination} and re-run with --no-download if this environment "
            "blocks network access."
        ) from error


def _csr_from_npz(payload: Any, prefix: str) -> sp.csr_matrix:
    required = [f"{prefix}_data", f"{prefix}_indices", f"{prefix}_indptr", f"{prefix}_shape"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Dataset is missing {prefix!r} CSR arrays: {missing}")
    return sp.csr_matrix(
        (
            payload[f"{prefix}_data"],
            payload[f"{prefix}_indices"],
            payload[f"{prefix}_indptr"],
        ),
        shape=tuple(payload[f"{prefix}_shape"]),
    )


def _to_undirected(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    combined_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    adjacency = torch.sparse_coo_tensor(
        combined_edge_index,
        torch.ones(combined_edge_index.shape[1], dtype=torch.float32),
        size=(num_nodes, num_nodes),
    ).coalesce()
    return adjacency.indices()


def load_sparse_csr_npz(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load the sparse CSR NPZ format used by CitationFull and Amazon."""

    with np.load(path, allow_pickle=False) as payload:
        adjacency = _csr_from_npz(payload, "adj")
        attributes = _csr_from_npz(payload, "attr")
        if "labels" not in payload:
            raise ValueError("Dataset is missing the 'labels' array")
        labels = np.asarray(payload["labels"], dtype=np.int64)

    adjacency = adjacency.tocoo()
    attributes = attributes.astype(np.float32)
    x = torch.from_numpy(np.asarray(attributes.toarray(), dtype=np.float32))
    y = torch.from_numpy(labels)
    raw_edge_index = torch.from_numpy(
        np.vstack([adjacency.row, adjacency.col]).astype(np.int64, copy=False)
    )
    # PyTorch Geometric's CitationFull loader calls read_npz(...,
    # to_undirected=True). Reproduce that preprocessing explicitly instead of
    # treating the one-directional raw archive as the training graph.
    edge_index = _to_undirected(raw_edge_index, x.shape[0])

    metadata = {
        "source_file": str(path),
        "sha256": _file_sha256(path),
        "num_nodes": int(x.shape[0]),
        "num_raw_directed_edges": int(raw_edge_index.shape[1]),
        "num_directed_edges": int(edge_index.shape[1]),
        "num_features": int(x.shape[1]),
        "num_classes": int(torch.unique(y).numel()),
    }
    return x, y, edge_index, metadata


def load_citation_npz(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Backward-compatible name for the shared sparse CSR NPZ loader."""

    return load_sparse_csr_npz(path)


def load_heterophilous_npz(
    path: Path,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dict[str, Any],
]:
    """Load the Yandex heterophilous-graphs NPZ format and fixed splits."""

    required = {
        "node_features",
        "node_labels",
        "edges",
        "train_masks",
        "val_masks",
        "test_masks",
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"Dataset is missing heterophilous graph arrays: {missing}")
        x = torch.from_numpy(np.asarray(payload["node_features"], dtype=np.float32))
        y = torch.from_numpy(np.asarray(payload["node_labels"], dtype=np.int64))
        raw_edges = np.asarray(payload["edges"], dtype=np.int64)
        if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
            raise ValueError("The 'edges' array must have shape [num_edges, 2]")
        raw_edge_index = torch.from_numpy(raw_edges.T.copy())
        split_masks = tuple(
            torch.from_numpy(np.asarray(payload[key], dtype=bool))
            for key in ("train_masks", "val_masks", "test_masks")
        )

    if any(mask.ndim != 2 or mask.shape[1] != x.shape[0] for mask in split_masks):
        raise ValueError("Predefined split masks must have shape [num_splits, num_nodes]")
    num_splits = split_masks[0].shape[0]
    if any(mask.shape[0] != num_splits for mask in split_masks):
        raise ValueError("Train, validation, and test arrays have different split counts")

    edge_index = _to_undirected(raw_edge_index, x.shape[0])
    metadata = {
        "source_file": str(path),
        "sha256": _file_sha256(path),
        "num_nodes": int(x.shape[0]),
        "num_raw_directed_edges": int(raw_edge_index.shape[1]),
        "num_directed_edges": int(edge_index.shape[1]),
        "num_features": int(x.shape[1]),
        "num_classes": int(torch.unique(y).numel()),
        "num_predefined_splits": int(num_splits),
    }
    return x, y, edge_index, split_masks, metadata


def row_normalize_features(x: torch.Tensor) -> torch.Tensor:
    row_sum = x.sum(dim=1, keepdim=True)
    safe_denominator = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    return x / safe_denominator


def normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Construct D^-1/2 (A + I) D^-1/2 as a sparse COO tensor."""

    loops = torch.arange(num_nodes, dtype=torch.long)
    loop_index = torch.stack([loops, loops])
    indices = torch.cat([edge_index.cpu(), loop_index], dim=1)
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    adjacency = torch.sparse_coo_tensor(
        indices, values, size=(num_nodes, num_nodes), dtype=torch.float32
    ).coalesce()
    row, col = adjacency.indices()
    values = adjacency.values()
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, row, values)
    inverse_sqrt_degree = degree.clamp_min(1).pow(-0.5)
    normalized_values = inverse_sqrt_degree[row] * values * inverse_sqrt_degree[col]
    return torch.sparse_coo_tensor(
        adjacency.indices(),
        normalized_values,
        size=adjacency.shape,
        dtype=torch.float32,
    ).coalesce()


def stratified_masks(
    y: torch.Tensor,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive")

    indices = np.arange(y.numel())
    y_numpy = y.cpu().numpy()
    train_indices, remainder = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
        stratify=y_numpy,
    )
    relative_val_fraction = val_fraction / (val_fraction + test_fraction)
    val_indices, test_indices = train_test_split(
        remainder,
        train_size=relative_val_fraction,
        random_state=seed + 1,
        shuffle=True,
        stratify=y_numpy[remainder],
    )

    def mask_from(split_indices: np.ndarray) -> torch.Tensor:
        mask = torch.zeros(y.numel(), dtype=torch.bool)
        mask[torch.from_numpy(split_indices)] = True
        return mask

    return (
        mask_from(train_indices),
        mask_from(val_indices),
        mask_from(test_indices),
    )


def _validate_masks(
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_nodes: int,
) -> None:
    masks = (train_mask, val_mask, test_mask)
    if any(mask.dtype != torch.bool or mask.shape != (num_nodes,) for mask in masks):
        raise ValueError("Each split mask must be a boolean vector with one value per node")
    if torch.any(train_mask & val_mask) or torch.any(train_mask & test_mask):
        raise ValueError("Training nodes overlap another split")
    if torch.any(val_mask & test_mask):
        raise ValueError("Validation and test nodes overlap")
    if not torch.all(train_mask | val_mask | test_mask):
        raise ValueError("Predefined splits must assign every node exactly once")


def _verify_expected_stats(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    missing = sorted(set(expected).difference(metadata))
    if missing:
        raise ValueError(f"Expected-stat keys are not available from this loader: {missing}")
    mismatches = {
        key: {"observed": metadata[key], "expected": value}
        for key, value in expected.items()
        if metadata[key] != value
    }
    if mismatches:
        raise ValueError(f"The dataset archive does not match its specification: {mismatches}")


def prepare_dataset(
    data_config: dict[str, Any],
    *,
    seed: int,
    allow_download: bool = True,
) -> GraphData:
    """Prepare any configured single-graph node-classification dataset."""

    path = Path(data_config["root"]) / str(data_config["filename"])
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Re-run without --no-download or place it there."
            )
        download_dataset(str(data_config["url"]), path)

    loader = str(data_config.get("loader", "sparse_csr_npz"))
    predefined_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if loader == "sparse_csr_npz":
        x, y, edge_index, metadata = load_sparse_csr_npz(path)
    elif loader == "heterophilous_npz":
        x, y, edge_index, predefined_masks, metadata = load_heterophilous_npz(path)
    else:
        raise ValueError(
            f"Unknown data loader {loader!r}; expected 'sparse_csr_npz' or "
            "'heterophilous_npz'"
        )

    expected = data_config.get("expected_stats")
    if bool(data_config.get("verify_expected_stats", True)):
        if expected is None:
            raise ValueError("verify_expected_stats requires data.expected_stats")
        _verify_expected_stats(metadata, dict(expected))

    normalize_features = bool(data_config.get("normalize_features", True))
    if normalize_features:
        x = row_normalize_features(x)

    split_mode = str(data_config.get("split_mode", "stratified_random"))
    if split_mode == "stratified_random":
        train_mask, val_mask, test_mask = stratified_masks(
            y,
            seed=seed,
            train_fraction=float(data_config["train_fraction"]),
            val_fraction=float(data_config["val_fraction"]),
            test_fraction=float(data_config["test_fraction"]),
        )
        split_metadata: dict[str, Any] = {
            "split_mode": split_mode,
            "split_seed": seed,
        }
    elif split_mode == "predefined":
        if predefined_masks is None:
            raise ValueError("The selected loader does not provide predefined splits")
        split_index = int(data_config.get("split_index", seed))
        if not 0 <= split_index < predefined_masks[0].shape[0]:
            raise ValueError(
                f"Split index {split_index} is outside [0, {predefined_masks[0].shape[0]})"
            )
        train_mask, val_mask, test_mask = (
            masks[split_index].clone() for masks in predefined_masks
        )
        split_metadata = {
            "split_mode": split_mode,
            "split_index": split_index,
            "training_seed": seed,
        }
    else:
        raise ValueError(
            f"Unknown split mode {split_mode!r}; expected 'stratified_random' or 'predefined'"
        )

    _validate_masks(train_mask, val_mask, test_mask, x.shape[0])
    test_label_access = str(data_config.get("test_label_access", "available"))
    if test_label_access not in {"available", "withheld"}:
        raise ValueError("data.test_label_access must be 'available' or 'withheld'")
    if test_label_access == "withheld":
        y = y.clone()
        y[test_mask] = -1
    metadata = {
        **metadata,
        "dataset_name": str(data_config.get("name", path.stem)),
        "loader": loader,
        "feature_normalization": "row_l1" if normalize_features else "none",
        "test_label_status": (
            "blinded" if test_label_access == "withheld" else "available"
        ),
        **split_metadata,
        "split_sizes": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
    }
    return GraphData(
        x=x,
        y=y,
        edge_index=edge_index,
        adjacency=normalized_adjacency(edge_index, x.shape[0]),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        metadata=metadata,
    )


def prepare_citeseer(
    *,
    root: Path,
    filename: str,
    url: str,
    seed: int,
    normalize_features: bool,
    verify_expected_stats: bool,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    allow_download: bool = True,
) -> GraphData:
    return prepare_dataset(
        {
            "name": "citationfull_citeseer",
            "loader": "sparse_csr_npz",
            "root": str(root),
            "filename": filename,
            "url": url,
            "normalize_features": normalize_features,
            "verify_expected_stats": verify_expected_stats,
            "expected_stats": CITATIONFULL_CITESEER_EXPECTED,
            "split_mode": "stratified_random",
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
        },
        seed=seed,
        allow_download=allow_download,
    )
