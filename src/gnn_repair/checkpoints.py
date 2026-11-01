"""Validated loading of trained models, exact splits, and checkpoint provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gnn_repair.data import GraphData, prepare_dataset
from gnn_repair.model import build_model
from gnn_repair.protocol import (
    blind_test_labels,
    configured_num_classes,
    evaluation_splits,
    study_record,
)


@dataclass(frozen=True)
class LoadedRun:
    """A trained model paired with the exact data split recorded at training time."""

    run_path: Path
    result: dict[str, Any]
    checkpoint: dict[str, Any]
    model: torch.nn.Module
    data: GraphData


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(checkpoint: dict[str, Any]) -> str:
    """Hash checkpoint tensor names, shapes, dtypes, and exact values."""

    return _state_dict_sha256(checkpoint["model_state_dict"])


def module_state_sha256(model: torch.nn.Module) -> str:
    """Hash the live module state so frozen-backbone checks detect mutation."""

    return _state_dict_sha256(model.state_dict())


def split_indices_sha256(data: GraphData) -> str:
    """Hash the exact ordered membership of every train/validation/test split."""

    digest = hashlib.sha256()
    for name, mask in (
        ("train", data.train_mask),
        ("validation", data.val_mask),
        ("test", data.test_mask),
    ):
        indices = mask.nonzero(as_tuple=False).flatten().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(indices.shape)).encode("ascii"))
        digest.update(indices.numpy().tobytes())
    return digest.hexdigest()


def source_code_sha256(filenames: tuple[str, ...]) -> str:
    """Hash named source modules that define an experimental protocol."""

    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for filename in filenames:
        digest.update(filename.encode("utf-8"))
        with (source_root / filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_recorded_path(recorded: str, run_path: Path) -> Path:
    path = Path(recorded)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    for parent in run_path.resolve().parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Recorded artifact does not exist: {recorded}")


def _mask_indices(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).flatten().cpu()


def load_trained_run(
    run_path: Path,
    *,
    device: torch.device | str = "cpu",
    allow_download: bool = False,
) -> LoadedRun:
    """Load a run only after its config, dataset hash, and split indices agree."""

    run_path = run_path.resolve()
    with run_path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    checkpoint_path = _resolve_recorded_path(str(result["checkpoint"]), run_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    config = result["config"]
    if checkpoint.get("config") != config:
        raise RuntimeError(f"Checkpoint config does not match run record {run_path}")
    seed = int(result["seed"])
    if int(checkpoint.get("seed", -1)) != seed:
        raise RuntimeError(f"Checkpoint seed does not match run record {run_path}")

    data = prepare_dataset(
        config["data"],
        seed=seed,
        allow_download=allow_download,
    )
    selected_splits = evaluation_splits(
        result.get("evaluation_splits", study_record(config)["evaluation_splits"])
    )
    data = blind_test_labels(data, selected_splits).to(device)
    if checkpoint.get("data_metadata", {}).get("sha256") != data.metadata.get("sha256"):
        raise RuntimeError(f"Dataset hash does not match checkpoint for {run_path}")
    expected_splits = checkpoint["split_indices"]
    observed_splits = {
        "train": _mask_indices(data.train_mask),
        "validation": _mask_indices(data.val_mask),
        "test": _mask_indices(data.test_mask),
    }
    for split_name, observed in observed_splits.items():
        expected = expected_splits[split_name].cpu()
        if not torch.equal(observed, expected):
            raise RuntimeError(
                f"{split_name.capitalize()} split does not match checkpoint for {run_path}"
            )

    num_classes = configured_num_classes(config["data"], data)
    model = build_model(
        config["model"],
        in_channels=int(data.x.shape[1]),
        out_channels=num_classes,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return LoadedRun(
        run_path=run_path,
        result=result,
        checkpoint=checkpoint,
        model=model,
        data=data,
    )
