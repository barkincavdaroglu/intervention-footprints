"""Shared safeguards for held-out evaluation splits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import hashlib
import json
from typing import Any

import torch

from gnn_repair.data import GraphData


HELD_OUT_SPLITS = ("validation", "test")
STUDY_PHASES = ("exploratory", "confirmatory_blind")


def canonical_payload_sha256(payload: Any) -> str:
    """Hash a JSON-compatible effective protocol independent of YAML formatting."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluation_splits(
    values: Iterable[str] | None,
    *,
    default: tuple[str, ...] = HELD_OUT_SPLITS,
) -> tuple[str, ...]:
    """Validate and canonicalize the held-out splits a report may inspect."""

    selected = tuple(default if values is None else (str(value) for value in values))
    if not selected:
        raise ValueError("evaluation_splits must contain at least one held-out split")
    if len(selected) != len(set(selected)):
        raise ValueError("evaluation_splits must not contain duplicates")
    unknown = sorted(set(selected).difference(HELD_OUT_SPLITS))
    if unknown:
        raise ValueError(
            f"Unknown evaluation splits {unknown}; expected validation and/or test"
        )
    if "validation" not in selected:
        raise ValueError("validation must remain available for model selection")
    return tuple(split for split in HELD_OUT_SPLITS if split in selected)


def split_masks(
    data: GraphData,
    selected: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Return only explicitly authorized held-out masks."""

    names = evaluation_splits(selected)
    if "test" in names and data.metadata.get("test_label_status") == "blinded":
        raise RuntimeError(
            "Test split access is disabled while test labels are blinded"
        )
    masks = {
        "validation": data.val_mask,
        "test": data.test_mask,
    }
    return {name: masks[name] for name in names}


def test_status(selected: Iterable[str] | None) -> str:
    """Describe whether test outcomes are intentionally absent from a report."""

    return "unblinded" if "test" in evaluation_splits(selected) else "blinded"


def study_record(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the study phase and its authorized evaluation splits."""

    study = config.get("study", {})
    if not isinstance(study, dict):
        raise ValueError("study must be a mapping")
    phase = str(study.get("phase", "exploratory"))
    if phase not in STUDY_PHASES:
        raise ValueError(f"Unknown study phase {phase!r}")
    study_id = str(study.get("id", "exploratory-development"))
    if not study_id:
        raise ValueError("study.id must not be empty")
    selected = evaluation_splits(config.get("evaluation_splits"))
    if phase == "confirmatory_blind" and selected != ("validation",):
        raise ValueError(
            "confirmatory_blind requires evaluation_splits: [validation]"
        )
    if (
        phase == "confirmatory_blind"
        and "data" in config
        and str(config["data"].get("split_mode")) != "predefined"
    ):
        raise ValueError("confirmatory_blind requires published predefined splits")
    if (
        phase == "confirmatory_blind"
        and "data" in config
        and str(config["data"].get("test_label_access")) != "withheld"
    ):
        raise ValueError("confirmatory_blind requires data.test_label_access: withheld")
    return {
        "id": study_id,
        "phase": phase,
        "evaluation_splits": list(selected),
        "test_status": test_status(selected),
    }


def blind_test_labels(
    data: GraphData,
    selected: Iterable[str] | None,
) -> GraphData:
    """Replace sealed test labels by a sentinel that fails accidental scoring."""

    names = evaluation_splits(selected)
    if "test" in names:
        return data
    labels = data.y.clone()
    labels[data.test_mask] = -1
    metadata = {
        **data.metadata,
        "test_label_status": "blinded",
    }
    return replace(data, y=labels, metadata=metadata)


def configured_num_classes(data_config: dict[str, Any], data: GraphData) -> int:
    """Resolve output width from public/frozen metadata, never sealed labels."""

    if "num_classes" in data_config:
        value = int(data_config["num_classes"])
    else:
        expected = data_config.get("expected_stats", {})
        value = int(expected.get("num_classes", data.metadata["num_classes"]))
    if value < 2:
        raise ValueError("Configured class count must be at least two")
    return value
