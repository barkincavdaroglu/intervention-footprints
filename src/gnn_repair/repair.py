"""Sparse activation-adapter repair with validation-only model selection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from gnn_repair.checkpoints import (
    LoadedRun,
    load_trained_run,
    model_state_sha256,
    module_state_sha256,
    source_code_sha256,
    split_indices_sha256,
)
from gnn_repair.interventions import (
    GROUPED_PROTOCOL_CODE_FILES,
    intervention_effects,
)
from gnn_repair.localization import (
    LOCALIZATION_PROTOCOL_CODE_FILES,
    prediction_profile,
)
from gnn_repair.model import GCN, MLP, ResidualGCN
from gnn_repair.protocol import (
    canonical_payload_sha256,
    evaluation_splits,
    split_masks,
    study_record,
    test_status,
)
from gnn_repair.reproducibility import set_deterministic


REPAIR_PROTOCOL_VERSION = 3
PROTOCOL_CODE_FILES = (
    "repair.py",
    "model.py",
    "interventions.py",
    "localization.py",
    "metrics.py",
    "data.py",
    "checkpoints.py",
    "protocol.py",
    "reproducibility.py",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_frozen_logits(
    path: Path,
    base_logits: torch.Tensor,
    modified_logits: torch.Tensor,
) -> dict[str, Any]:
    """Atomically save fixed all-node logits for later score-only unblinding."""

    if base_logits.shape != modified_logits.shape:
        raise ValueError("Base and modified logits must have identical shapes")
    values = torch.stack((base_logits, modified_logits)).detach().cpu().numpy()
    if values.dtype != np.float32:
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError("Frozen logits contain a nonfinite value")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "layout": ["base", "modified"],
    }


def validate_frozen_logits(record: dict[str, Any]) -> None:
    """Reject a missing or changed prediction sidecar before cache reuse."""

    path = Path(str(record["path"]))
    if not path.exists() or _file_sha256(path) != record.get("sha256"):
        raise RuntimeError(f"Frozen-logit artifact is missing or changed: {path}")
    with path.open("rb") as stream:
        values = np.load(stream, allow_pickle=False)
    if record.get("layout") != ["base", "modified"]:
        raise RuntimeError(f"Frozen-logit layout is invalid for {path}")
    if (
        values.ndim != 3
        or values.shape[0] != 2
        or values.shape[1] < 1
        or values.shape[2] < 2
        or values.dtype != np.float32
        or list(values.shape) != record.get("shape")
        or record.get("dtype") != "float32"
    ):
        raise RuntimeError(f"Frozen-logit metadata does not match {path}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"Frozen-logit artifact contains a nonfinite value: {path}")


def _json_payload_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class SelectedSiteAffine(nn.Module):
    """Identity-initialized affine deltas for selected channels in one layer."""

    def __init__(
        self,
        indices: list[int],
        training_mean: torch.Tensor,
        training_std: torch.Tensor,
    ) -> None:
        super().__init__()
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Selected neuron indices must be non-empty and unique")
        sorted_indices = sorted(int(index) for index in indices)
        width = int(training_mean.numel())
        if sorted_indices[0] < 0 or sorted_indices[-1] >= width:
            raise ValueError("Selected neuron index is outside the activation width")
        index = torch.tensor(
            sorted_indices,
            dtype=torch.long,
            device=training_mean.device,
        )
        self.register_buffer("indices", index)
        self.register_buffer("mean", training_mean[index].detach().clone())
        self.register_buffer(
            "std",
            training_std[index].detach().clone().clamp_min(1e-6),
        )
        self.delta_scale = nn.Parameter(
            torch.zeros(len(sorted_indices), device=index.device)
        )
        self.delta_shift = nn.Parameter(
            torch.zeros(len(sorted_indices), device=index.device)
        )

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        selected = activation.index_select(1, self.indices)
        adjustment = (selected - self.mean) * self.delta_scale + self.std * self.delta_shift
        return activation.index_add(1, self.indices, adjustment)


class ActivationAffineAdapter(nn.Module):
    """Layer-indexed sparse adapters compatible with every repository backbone."""

    def __init__(
        self,
        sites: dict[int, list[int]],
        training_means: list[torch.Tensor],
        training_stds: list[torch.Tensor],
    ) -> None:
        super().__init__()
        if not sites:
            raise ValueError("At least one selected layer is required")
        if len(training_means) != len(training_stds):
            raise ValueError("Activation means and standard deviations do not align")
        modules = {}
        for layer, indices in sorted(sites.items()):
            if not 0 <= layer < len(training_means):
                raise ValueError(f"Selected layer {layer} is outside the backbone")
            modules[str(layer)] = SelectedSiteAffine(
                indices,
                training_means[layer],
                training_stds[layer],
            )
        self.layers = nn.ModuleDict(modules)

    def forward(self, layer: int, activation: torch.Tensor) -> torch.Tensor:
        key = str(layer)
        return activation if key not in self.layers else self.layers[key](activation)

    @property
    def selected_sites(self) -> int:
        return sum(module.indices.numel() for module in self.layers.values())


class TargetLogitBias(nn.Module):
    """One-parameter class-prior calibration control."""

    def __init__(self, target_class: int) -> None:
        super().__init__()
        self.target_class = int(target_class)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        class_indicator = F.one_hot(
            torch.tensor(self.target_class, device=logits.device),
            num_classes=logits.shape[-1],
        ).to(dtype=logits.dtype)
        return logits + self.bias * class_indicator


@torch.no_grad()
def activation_statistics(loaded: LoadedRun) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    _, activations = loaded.model(
        loaded.data.x,
        loaded.data.adjacency,
        return_activations=True,
    )
    means = [activation[loaded.data.train_mask].mean(dim=0) for activation in activations]
    stds = [
        activation[loaded.data.train_mask].std(dim=0, unbiased=False)
        for activation in activations
    ]
    return means, stds


def _forward_repaired(
    loaded: LoadedRun,
    repair: ActivationAffineAdapter | TargetLogitBias,
) -> torch.Tensor:
    if isinstance(repair, TargetLogitBias):
        return repair(loaded.model(loaded.data.x, loaded.data.adjacency))
    return loaded.model(
        loaded.data.x,
        loaded.data.adjacency,
        activation_intervention=repair,
    )


def _non_target_macro(profile: dict[str, Any], target_class: int) -> float:
    return mean(
        float(metrics["f1"])
        for class_id, metrics in profile["per_class"].items()
        if int(class_id) != target_class
    )


def _validation_record(
    base_profile: dict[str, Any],
    modified_profile: dict[str, Any],
    target_class: int,
    *,
    penalty: float,
    max_non_target_drop: float,
    max_single_non_target_drop: float,
    max_accuracy_drop: float,
) -> dict[str, Any]:
    target_delta = (
        float(modified_profile["per_class"][str(target_class)]["f1"])
        - float(base_profile["per_class"][str(target_class)]["f1"])
    )
    non_target_delta = _non_target_macro(
        modified_profile, target_class
    ) - _non_target_macro(base_profile, target_class)
    non_target_class_deltas = {
        class_id: (
            float(modified_profile["per_class"][class_id]["f1"])
            - float(metrics["f1"])
        )
        for class_id, metrics in base_profile["per_class"].items()
        if int(class_id) != target_class
    }
    worst_non_target_delta = min(non_target_class_deltas.values())
    accuracy_delta = float(modified_profile["accuracy"]) - float(
        base_profile["accuracy"]
    )
    utility = target_delta - penalty * max(0.0, -non_target_delta)
    return {
        "target_f1_delta": target_delta,
        "non_target_macro_f1_delta": non_target_delta,
        "worst_non_target_class_f1_delta": worst_non_target_delta,
        "accuracy_delta": accuracy_delta,
        "utility": utility,
        "eligible": (
            non_target_delta >= -max_non_target_drop
            and worst_non_target_delta >= -max_single_non_target_drop
            and accuracy_delta >= -max_accuracy_drop
        ),
    }


def _training_objective(
    logits: torch.Tensor,
    base_logits: torch.Tensor,
    loaded: LoadedRun,
    target_class: int,
    identity_l2: torch.Tensor,
    training: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    data = loaded.data
    target_mask = data.train_mask & (data.y == target_class)
    non_target_mask = data.train_mask & (data.y != target_class)
    if not bool(target_mask.any()) or not bool(non_target_mask.any()):
        raise ValueError("Training split lacks a target or non-target group")
    target_ce = F.cross_entropy(logits[target_mask], data.y[target_mask])
    non_target_ce = F.cross_entropy(logits[non_target_mask], data.y[non_target_mask])
    non_target_kl = F.kl_div(
        F.log_softmax(logits[non_target_mask], dim=-1),
        F.softmax(base_logits[non_target_mask], dim=-1),
        reduction="batchmean",
    )
    loss = (
        float(training.get("target_ce_weight", 1.0)) * target_ce
        + float(training.get("non_target_ce_weight", 1.0)) * non_target_ce
        + float(training.get("non_target_kl_weight", 1.0)) * non_target_kl
        + float(training.get("identity_l2_weight", 0.001)) * identity_l2
    )
    return loss, {
        "target_ce": float(target_ce.detach().item()),
        "non_target_ce": float(non_target_ce.detach().item()),
        "non_target_kl": float(non_target_kl.detach().item()),
        "identity_l2": float(identity_l2.detach().item()),
        "loss": float(loss.detach().item()),
    }


def _elementwise_mean_square(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Return one parameter-count-weighted mean square across tensors."""

    if not tensors:
        raise ValueError("Mean-square regularization requires at least one tensor")
    return sum(tensor.square().sum() for tensor in tensors) / sum(
        tensor.numel() for tensor in tensors
    )


def fit_repair(
    loaded: LoadedRun,
    *,
    target_class: int,
    sites: dict[int, list[int]] | None,
    training: dict[str, Any],
    logit_bias: bool = False,
    selected_splits: tuple[str, ...] = ("validation", "test"),
    frozen_logits_path: Path | None = None,
) -> dict[str, Any]:
    """Fit one repair independently from an exact frozen base checkpoint."""

    set_deterministic(
        int(loaded.result["seed"]),
        num_threads=int(training.get("num_threads", 1)),
    )
    learning_rate = float(training["learning_rate"])
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    if not learning_rate > 0:
        raise ValueError("Repair learning rate must be positive")
    if max_epochs < 1 or patience < 1:
        raise ValueError("Repair max_epochs and patience must be positive")

    model = loaded.model
    model.eval()
    model.requires_grad_(False)
    checkpoint_hash = model_state_sha256(loaded.checkpoint)
    base_hash = module_state_sha256(model)
    if base_hash != checkpoint_hash:
        raise RuntimeError("Loaded backbone does not match its checkpoint")
    with torch.no_grad():
        base_logits = model(loaded.data.x, loaded.data.adjacency).detach()
        base_validation_profile = prediction_profile(
            base_logits, loaded.data.y, loaded.data.val_mask
        )

    if logit_bias:
        repair: ActivationAffineAdapter | TargetLogitBias = TargetLogitBias(
            target_class
        ).to(loaded.data.x.device)
        selected_sites = 0
    else:
        if sites is None:
            raise ValueError("Activation repair requires selected sites")
        means, stds = activation_statistics(loaded)
        repair = ActivationAffineAdapter(sites, means, stds).to(loaded.data.x.device)
        selected_sites = repair.selected_sites

    with torch.no_grad():
        identity_logits = _forward_repaired(loaded, repair)
    if not torch.equal(identity_logits, base_logits):
        raise RuntimeError("Repair is not exactly identity-initialized")

    parameters = list(repair.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=learning_rate,
        weight_decay=0.0,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in parameters}:
        raise RuntimeError("Optimizer parameter set does not exactly match the repair")

    penalty = float(training.get("utility_penalty", 1.0))
    max_non_target_drop = float(training.get("max_non_target_f1_drop", 0.01))
    max_single_non_target_drop = float(
        training.get("max_single_non_target_f1_drop", 0.05)
    )
    max_accuracy_drop = float(training.get("max_accuracy_drop", 0.01))
    min_delta = float(training.get("min_utility_delta", 1e-6))
    best_state = copy.deepcopy(repair.state_dict())
    best_epoch = 0
    best_validation = {
        "target_f1_delta": 0.0,
        "non_target_macro_f1_delta": 0.0,
        "worst_non_target_class_f1_delta": 0.0,
        "accuracy_delta": 0.0,
        "utility": 0.0,
        "eligible": True,
    }
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_repaired(loaded, repair)
        loss, objective = _training_objective(
            logits,
            base_logits,
            loaded,
            target_class,
            _elementwise_mean_square(parameters),
            training,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Repair loss became nonfinite at epoch {epoch}")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Frozen backbone received gradients during repair")
        if any(
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise FloatingPointError(
                f"Repair gradient became missing or nonfinite at epoch {epoch}"
            )
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
            raise FloatingPointError(
                f"Repair parameter became nonfinite at epoch {epoch}"
            )

        with torch.no_grad():
            validation_logits = _forward_repaired(loaded, repair)
            validation_profile = prediction_profile(
                validation_logits,
                loaded.data.y,
                loaded.data.val_mask,
            )
        validation = _validation_record(
            base_validation_profile,
            validation_profile,
            target_class,
            penalty=penalty,
            max_non_target_drop=max_non_target_drop,
            max_single_non_target_drop=max_single_non_target_drop,
            max_accuracy_drop=max_accuracy_drop,
        )
        history.append({"epoch": epoch, **objective, "validation": validation})
        if (
            validation["eligible"]
            and validation["utility"] > best_validation["utility"] + min_delta
        ):
            best_state = copy.deepcopy(repair.state_dict())
            best_epoch = epoch
            best_validation = validation
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    repair.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        repaired_logits = _forward_repaired(loaded, repair)
        effects = intervention_effects(
            base_logits,
            repaired_logits,
            loaded.data,
            [target_class],
            selected_splits=selected_splits,
        )[target_class]
        final_diagnostics = {}
        for split_name, mask in split_masks(
            loaded.data, selected_splits
        ).items():
            base_profile = prediction_profile(
                base_logits,
                loaded.data.y,
                mask,
            )
            repaired_profile = prediction_profile(
                repaired_logits,
                loaded.data.y,
                mask,
            )
            final_diagnostics[split_name] = _validation_record(
                base_profile,
                repaired_profile,
                target_class,
                penalty=penalty,
                max_non_target_drop=max_non_target_drop,
                max_single_non_target_drop=max_single_non_target_drop,
                max_accuracy_drop=max_accuracy_drop,
            )
    if best_epoch == 0 and not torch.equal(repaired_logits, base_logits):
        raise RuntimeError("Epoch-zero selection did not restore the exact no-op state")
    if module_state_sha256(model) != base_hash:
        raise RuntimeError("Frozen backbone changed during repair")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen backbone retains a gradient after repair")

    state = {
        name: tensor.detach().cpu().tolist()
        for name, tensor in repair.state_dict().items()
        if name.endswith("delta_scale")
        or name.endswith("delta_shift")
        or name == "bias"
    }
    frozen_logits = (
        save_frozen_logits(frozen_logits_path, base_logits, repaired_logits)
        if frozen_logits_path is not None
        else None
    )
    return {
        "target_class": target_class,
        "selected_sites": selected_sites,
        "num_trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation": best_validation,
        "effects": effects,
        "final_diagnostics": final_diagnostics,
        "repair_state": state,
        "history": history,
        "base_model_sha256": base_hash,
        "evaluation_splits": list(evaluation_splits(selected_splits)),
        "test_status": test_status(selected_splits),
        "frozen_logits": frozen_logits,
    }


def finetune_parameter_names(model: nn.Module, scope: str) -> list[str]:
    """Return the exact conventional fine-tuning parameter set."""

    all_names = [name for name, _ in model.named_parameters()]
    if scope == "full_model":
        return all_names
    if scope != "last_layer":
        raise ValueError("Fine-tuning scope must be 'last_layer' or 'full_model'")
    if isinstance(model, GCN):
        prefix = f"convolutions.{len(model.convolutions) - 1}.linear."
    elif isinstance(model, MLP):
        prefix = f"linears.{len(model.linears) - 1}."
    elif isinstance(model, ResidualGCN):
        prefix = "output_linear."
    else:
        raise TypeError(f"Unsupported fine-tuning backbone {type(model).__name__}")
    selected = [name for name in all_names if name.startswith(prefix)]
    if not selected:
        raise RuntimeError(f"No parameters matched output-layer prefix {prefix!r}")
    return selected


def _fine_tune_learning_rate(training: dict[str, Any], scope: str) -> float:
    key = f"{scope}_learning_rate"
    return float(
        training.get(key, training.get("finetune_learning_rate", training["learning_rate"]))
    )


def fit_finetune_control(
    loaded: LoadedRun,
    *,
    target_class: int,
    scope: str,
    training: dict[str, Any],
    selected_splits: tuple[str, ...] = ("validation", "test"),
    frozen_logits_path: Path | None = None,
) -> dict[str, Any]:
    """Fit an optimization-cap-matched head-only or full-model control."""

    set_deterministic(
        int(loaded.result["seed"]),
        num_threads=int(training.get("num_threads", 1)),
    )
    learning_rate = _fine_tune_learning_rate(training, scope)
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    if learning_rate <= 0 or max_epochs < 1 or patience < 1:
        raise ValueError("Fine-tuning learning rate, max_epochs, and patience must be positive")

    base_model = loaded.model
    base_model.eval()
    base_model.requires_grad_(False)
    checkpoint_hash = model_state_sha256(loaded.checkpoint)
    base_hash = module_state_sha256(base_model)
    if base_hash != checkpoint_hash:
        raise RuntimeError("Loaded backbone does not match its checkpoint")
    with torch.no_grad():
        base_logits = base_model(loaded.data.x, loaded.data.adjacency).detach()
        base_validation_profile = prediction_profile(
            base_logits, loaded.data.y, loaded.data.val_mask
        )

    control_model = copy.deepcopy(base_model)
    control_model.eval()
    control_model.requires_grad_(False)
    selected_names = finetune_parameter_names(control_model, scope)
    named_parameters = dict(control_model.named_parameters())
    for name in selected_names:
        named_parameters[name].requires_grad_(True)
    parameters = [named_parameters[name] for name in selected_names]
    initial_state = copy.deepcopy(control_model.state_dict())
    initial_trainable = {
        name: named_parameters[name].detach().clone() for name in selected_names
    }
    with torch.no_grad():
        identity_logits = control_model(loaded.data.x, loaded.data.adjacency)
    if not torch.equal(identity_logits, base_logits):
        raise RuntimeError("Fine-tuning control is not exactly identity-initialized")

    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in parameters}:
        raise RuntimeError("Fine-tuning optimizer does not match selected parameters")

    penalty = float(training.get("utility_penalty", 1.0))
    max_non_target_drop = float(training.get("max_non_target_f1_drop", 0.01))
    max_single_non_target_drop = float(
        training.get("max_single_non_target_f1_drop", 0.05)
    )
    max_accuracy_drop = float(training.get("max_accuracy_drop", 0.01))
    min_delta = float(training.get("min_utility_delta", 1e-6))
    best_state = copy.deepcopy(initial_state)
    best_epoch = 0
    best_validation = {
        "target_f1_delta": 0.0,
        "non_target_macro_f1_delta": 0.0,
        "worst_non_target_class_f1_delta": 0.0,
        "accuracy_delta": 0.0,
        "utility": 0.0,
        "eligible": True,
    }
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        control_model.eval()
        optimizer.zero_grad(set_to_none=True)
        logits = control_model(loaded.data.x, loaded.data.adjacency)
        identity_l2 = _elementwise_mean_square(
            [
                named_parameters[name] - initial_trainable[name]
                for name in selected_names
            ]
        )
        loss, objective = _training_objective(
            logits,
            base_logits,
            loaded,
            target_class,
            identity_l2,
            training,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"{scope} fine-tuning loss became nonfinite at epoch {epoch}"
            )
        loss.backward()
        if any(parameter.grad is not None for parameter in base_model.parameters()):
            raise RuntimeError("Original backbone received fine-tuning gradients")
        if any(
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise FloatingPointError(
                f"{scope} fine-tuning gradient became missing or nonfinite"
            )
        if any(
            parameter.grad is not None
            for name, parameter in named_parameters.items()
            if name not in selected_names
        ):
            raise RuntimeError("A nonselected fine-tuning parameter received gradients")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
            raise FloatingPointError(
                f"{scope} fine-tuning parameter became nonfinite at epoch {epoch}"
            )

        with torch.no_grad():
            validation_logits = control_model(
                loaded.data.x, loaded.data.adjacency
            )
            validation_profile = prediction_profile(
                validation_logits, loaded.data.y, loaded.data.val_mask
            )
        validation = _validation_record(
            base_validation_profile,
            validation_profile,
            target_class,
            penalty=penalty,
            max_non_target_drop=max_non_target_drop,
            max_single_non_target_drop=max_single_non_target_drop,
            max_accuracy_drop=max_accuracy_drop,
        )
        history.append({"epoch": epoch, **objective, "validation": validation})
        if (
            validation["eligible"]
            and validation["utility"] > best_validation["utility"] + min_delta
        ):
            best_state = copy.deepcopy(control_model.state_dict())
            best_epoch = epoch
            best_validation = validation
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    control_model.load_state_dict(best_state, strict=True)
    control_model.eval()
    with torch.no_grad():
        modified_logits = control_model(loaded.data.x, loaded.data.adjacency)
        effects = intervention_effects(
            base_logits,
            modified_logits,
            loaded.data,
            [target_class],
            selected_splits=selected_splits,
        )[target_class]
        final_diagnostics = {}
        for split_name, mask in split_masks(
            loaded.data, selected_splits
        ).items():
            final_diagnostics[split_name] = _validation_record(
                prediction_profile(base_logits, loaded.data.y, mask),
                prediction_profile(modified_logits, loaded.data.y, mask),
                target_class,
                penalty=penalty,
                max_non_target_drop=max_non_target_drop,
                max_single_non_target_drop=max_single_non_target_drop,
                max_accuracy_drop=max_accuracy_drop,
            )

    if best_epoch == 0 and not torch.equal(modified_logits, base_logits):
        raise RuntimeError("Epoch-zero fine-tuning did not restore exact base logits")
    for name, value in control_model.state_dict().items():
        if name not in selected_names and not torch.equal(value, initial_state[name]):
            raise RuntimeError(f"Nonselected state changed during {scope}: {name}")
    if module_state_sha256(base_model) != base_hash:
        raise RuntimeError("Original backbone changed during fine-tuning control")
    if any(parameter.grad is not None for parameter in base_model.parameters()):
        raise RuntimeError("Original backbone retains a fine-tuning gradient")

    final_named_parameters = dict(control_model.named_parameters())
    delta_l2 = float(
        _elementwise_mean_square(
            [
                final_named_parameters[name] - initial_trainable[name]
                for name in selected_names
            ]
        ).item()
    )
    frozen_logits = (
        save_frozen_logits(frozen_logits_path, base_logits, modified_logits)
        if frozen_logits_path is not None
        else None
    )
    return {
        "target_class": target_class,
        "selected_sites": 0,
        "scope": scope,
        "learning_rate": learning_rate,
        "trainable_parameter_names": selected_names,
        "num_trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation": best_validation,
        "effects": effects,
        "final_diagnostics": final_diagnostics,
        "repair_state": {
            "scope": scope,
            "state_sha256": module_state_sha256(control_model),
            "delta_l2": delta_l2,
        },
        "history": history,
        "base_model_sha256": base_hash,
        "evaluation_splits": list(evaluation_splits(selected_splits)),
        "test_status": test_status(selected_splits),
        "frozen_logits": frozen_logits,
    }


def _actual_budget(width: int, fraction: float) -> int:
    return min(max(1, round(width * fraction)), width)


def _selection_utility(record: dict[str, Any], *, penalty: float) -> float:
    target_delta = record["delta_target_f1"]["mean"]
    non_target_delta = record["delta_non_target_macro_f1"]["mean"]
    return target_delta - penalty * max(0.0, -non_target_delta)


def select_grouped_choice(
    grouped_backbone: dict[str, Any],
    *,
    target_class: int,
    method: str,
    budget_fraction: float,
    utility_penalty: float = 1.0,
    max_non_target_drop: float = 0.01,
) -> dict[str, Any]:
    """Select one layer from aggregate validation outcomes at a fixed fraction."""

    candidates = []
    target_summary = grouped_backbone["summary"][str(target_class)]
    for layer, methods in target_summary.items():
        if method not in methods:
            raise ValueError(f"Grouped report is missing method {method!r}")
        budgets = methods[method]
        width = next(
            int(record["width"])
            for run in grouped_backbone["runs"]
            for record in run["records"]
            if int(record["target_class"]) == target_class
            and int(record["layer"]) == int(layer)
        )
        budget = _actual_budget(width, budget_fraction)
        validation = budgets[str(budget)]["validation"]
        non_target_delta = float(
            validation["delta_non_target_macro_f1"]["mean"]
        )
        eligible = non_target_delta >= -max_non_target_drop
        score = (
            _selection_utility(validation, penalty=utility_penalty)
            if method == "mean_ablation_bottom"
            else float(validation["delta_target_loss"]["mean"])
        )
        candidates.append(
            (
                float(score),
                -int(layer),
                int(layer),
                budget,
                width,
                non_target_delta,
                eligible,
            )
        )
    if method == "mean_ablation_bottom":
        eligible_candidates = [candidate for candidate in candidates if candidate[-1]]
        if eligible_candidates:
            candidates = eligible_candidates
    score, _, layer, budget, width, non_target_delta, eligible = max(candidates)
    return {
        "layer": layer,
        "budget": budget,
        "width": width,
        "budget_fraction": budget_fraction,
        "validation_selection_score": score,
        "validation_non_target_macro_f1_delta": non_target_delta,
        "validation_collateral_eligible": eligible,
        "selection_metric": (
            "validation_repair_utility"
            if method == "mean_ablation_bottom"
            else "validation_target_loss_increase"
        ),
    }


def select_validation_targets(
    grouped_backbone: dict[str, Any],
    *,
    target_count: int,
    budget_fraction: float,
    utility_penalty: float,
    max_non_target_drop: float,
) -> tuple[list[int], dict[str, Any]]:
    """Rank prelocalized classes using aggregate validation bottom-ablation utility."""

    candidate_targets = [
        int(value) for value in grouped_backbone["target_classes"]
    ]
    if not 1 <= target_count <= len(candidate_targets):
        raise ValueError("Repair target count is outside the grouped candidate set")
    priorities = {
        target: select_grouped_choice(
            grouped_backbone,
            target_class=target,
            method="mean_ablation_bottom",
            budget_fraction=budget_fraction,
            utility_penalty=utility_penalty,
            max_non_target_drop=max_non_target_drop,
        )
        for target in candidate_targets
    }
    ranked = sorted(
        candidate_targets,
        key=lambda target: (
            -int(priorities[target]["validation_collateral_eligible"]),
            -float(priorities[target]["validation_selection_score"]),
            target,
        ),
    )
    if sum(
        bool(priorities[target]["validation_collateral_eligible"])
        for target in ranked
    ) < target_count:
        raise RuntimeError(
            "Too few validation target candidates satisfy the collateral cap"
        )
    return ranked[:target_count], {
        str(target): priorities[target] for target in ranked
    }


def _index_unique_runs(
    runs: list[dict[str, Any]], *, source: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for run in runs:
        seed = int(run["seed"])
        if seed in indexed:
            raise RuntimeError(f"Duplicate seed {seed} in {source}")
        indexed[seed] = run
    return indexed


def _validate_upstream_run(
    loaded: LoadedRun,
    run: dict[str, Any],
    *,
    source: str,
) -> None:
    expected = {
        "experiment": str(loaded.result["experiment"]),
        "seed": int(loaded.result["seed"]),
        "checkpoint": str(loaded.result["checkpoint"]),
        "checkpoint_model_sha256": model_state_sha256(loaded.checkpoint),
        "dataset_sha256": str(loaded.data.metadata["sha256"]),
        "split_indices_sha256": split_indices_sha256(loaded.data),
    }
    observed = {key: run.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"{source} provenance does not match loaded run: "
            f"expected {expected}, observed {observed}"
        )


def _ranking_for(
    localization_run: dict[str, Any],
    *,
    target_class: int,
    method: str,
    layer: int,
) -> list[int]:
    layer_records = [
        record
        for record in localization_run["classes"][str(target_class)]["layers"]
        if int(record["layer"]) == layer
    ]
    if len(layer_records) != 1:
        raise RuntimeError(
            f"Expected exactly one localization record for layer {layer}"
        )
    layer_record = layer_records[0]
    source_method = "mean_ablation" if method == "mean_ablation_bottom" else method
    source = layer_record["methods"][source_method]
    width = int(layer_record["width"])
    if method == "mean_ablation_bottom":
        scores = [float(value) for value in source["scores"]]
        ranking = sorted(range(width), key=lambda index: (scores[index], index))
    else:
        ranking = [int(index) for index in source["ranking"]]
    if sorted(ranking) != list(range(width)):
        raise RuntimeError("Localization ranking is not a complete permutation")
    return ranking


def _random_ranking(
    width: int,
    experiment: str,
    seed: int,
    target_class: int,
    layer: int,
    repeat: int,
) -> list[int]:
    material = (
        f"repair:{experiment}:{seed}:{target_class}:{layer}:{repeat}"
    ).encode("utf-8")
    random_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return np.random.default_rng(random_seed).permutation(width).tolist()


def _cache_identity(
    loaded: LoadedRun,
    *,
    target_class: int,
    method: str,
    repeat: int | None,
    choice: dict[str, Any] | None,
    neurons: list[int] | None,
    training: dict[str, Any],
    source_payload_sha256: str,
    upstream_artifacts: dict[str, Any],
    protocol_code_sha256: str,
    selected_splits: tuple[str, ...] = ("validation", "test"),
    frozen_logits_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "version": REPAIR_PROTOCOL_VERSION,
        "protocol_code_sha256": protocol_code_sha256,
        "source_payload_sha256": source_payload_sha256,
        "upstream_artifacts": upstream_artifacts,
        "experiment": loaded.result["experiment"],
        "seed": int(loaded.result["seed"]),
        "checkpoint_model_sha256": model_state_sha256(loaded.checkpoint),
        "dataset_sha256": loaded.data.metadata["sha256"],
        "split_indices_sha256": split_indices_sha256(loaded.data),
        "target_class": target_class,
        "method": method,
        "repeat": repeat,
        "choice": choice,
        "neurons": neurons,
        "training": training,
        "evaluation_splits": list(evaluation_splits(selected_splits)),
        "study": study_record(loaded.result["config"]),
        "frozen_logits_path": (
            str(frozen_logits_path) if frozen_logits_path is not None else None
        ),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }


def _fit_with_cache(
    loaded: LoadedRun,
    *,
    target_class: int,
    method: str,
    repeat: int | None,
    choice: dict[str, Any] | None,
    neurons: list[int] | None,
    training: dict[str, Any],
    source_payload_sha256: str,
    upstream_artifacts: dict[str, Any],
    protocol_code_sha256: str,
    cache_path: Path,
    resume: bool,
    selected_splits: tuple[str, ...] = ("validation", "test"),
) -> dict[str, Any]:
    frozen_logits_path = cache_path.with_name(f"{cache_path.stem}.logits.npy")
    identity = _cache_identity(
        loaded,
        target_class=target_class,
        method=method,
        repeat=repeat,
        choice=choice,
        neurons=neurons,
        training=training,
        source_payload_sha256=source_payload_sha256,
        upstream_artifacts=upstream_artifacts,
        protocol_code_sha256=protocol_code_sha256,
        selected_splits=selected_splits,
        frozen_logits_path=frozen_logits_path,
    )
    if resume and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("identity") != identity:
            raise RuntimeError(f"Repair cache {cache_path} differs from active protocol")
        validate_frozen_logits(cached["candidate"]["frozen_logits"])
        return cached["candidate"]

    if method == "target_logit_bias":
        fitted = fit_repair(
            loaded,
            target_class=target_class,
            sites=None,
            training=training,
            logit_bias=True,
            selected_splits=selected_splits,
            frozen_logits_path=frozen_logits_path,
        )
    elif method in {"last_layer_finetune", "full_model_finetune"}:
        scope = method.removesuffix("_finetune")
        fitted = fit_finetune_control(
            loaded,
            target_class=target_class,
            scope=scope,
            training=training,
            selected_splits=selected_splits,
            frozen_logits_path=frozen_logits_path,
        )
    else:
        if choice is None or neurons is None:
            raise ValueError("Activation repair candidate is missing its selected sites")
        fitted = fit_repair(
            loaded,
            target_class=target_class,
            sites={int(choice["layer"]): neurons},
            training=training,
            selected_splits=selected_splits,
            frozen_logits_path=frozen_logits_path,
        )
    candidate = {
        "method": method,
        "repeat": repeat,
        "choice": choice,
        "neurons": neurons,
        **fitted,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            {"identity": identity, "candidate": candidate},
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(cache_path)
    return candidate


def _summary(values: list[float]) -> dict[str, Any]:
    sample_standard_deviation = stdev(values) if len(values) > 1 else 0.0
    standard_error = sample_standard_deviation / len(values) ** 0.5
    return {
        "mean": mean(values),
        "population_std": pstdev(values) if len(values) > 1 else 0.0,
        "sample_std": sample_standard_deviation,
        "standard_error": standard_error,
        "normal_95_ci": [
            mean(values) - 1.96 * standard_error,
            mean(values) + 1.96 * standard_error,
        ],
        "values": values,
    }


def _aggregate_candidates(runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_seed: dict[tuple[int, str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        seed = int(run["seed"])
        target = int(run["target_class"])
        for candidate in run["candidates"]:
            method = str(candidate["method"])
            for split_name, split in candidate["effects"].items():
                for state_name in ("base", "modified"):
                    for metric, value in split[state_name].items():
                        per_seed[
                            (target, method, f"{split_name}.{state_name}_{metric}")
                        ][seed].append(float(value))
                for metric, value in split["delta"].items():
                    per_seed[(target, method, f"{split_name}.delta_{metric}")][
                        seed
                    ].append(float(value))
                for metric, value in split["transitions"].items():
                    per_seed[(target, method, f"{split_name}.{metric}")][seed].append(
                        float(value)
                    )
            for split_name, diagnostics in candidate["final_diagnostics"].items():
                for metric, value in diagnostics.items():
                    if isinstance(value, bool):
                        value = float(value)
                    per_seed[(target, method, f"{split_name}.{metric}")][seed].append(
                        float(value)
                    )
            per_seed[(target, method, "best_epoch")][seed].append(
                float(candidate["best_epoch"])
            )
            per_seed[(target, method, "selected_no_op")][seed].append(
                float(candidate["best_epoch"] == 0)
            )
            per_seed[(target, method, "selected_sites")][seed].append(
                float(candidate["selected_sites"])
            )
            per_seed[(target, method, "num_trainable_parameters")][seed].append(
                float(candidate["num_trainable_parameters"])
            )

    summary: dict[str, Any] = {}
    for (target, method, metric), by_seed in per_seed.items():
        values = [mean(repeats) for _, repeats in sorted(by_seed.items())]
        summary.setdefault(str(target), {}).setdefault(method, {})[metric] = _summary(
            values
        )
    return summary


def _paired_primary_vs_random(
    runs: list[dict[str, Any]],
    *,
    primary_method: str,
) -> dict[str, Any]:
    metrics = (
        "target_f1_delta",
        "non_target_macro_f1_delta",
        "accuracy_delta",
        "utility",
    )
    paired: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for run in runs:
        target = int(run["target_class"])
        primary = [
            candidate
            for candidate in run["candidates"]
            if candidate["method"] == primary_method
        ]
        random = [
            candidate
            for candidate in run["candidates"]
            if candidate["method"] == "matched_random"
        ]
        if len(primary) != 1 or not random:
            raise RuntimeError("Primary-versus-random candidates are incomplete")
        split_names = tuple(primary[0]["final_diagnostics"])
        if any(tuple(candidate["final_diagnostics"]) != split_names for candidate in random):
            raise RuntimeError("Primary and random evaluation splits differ")
        for split_name in split_names:
            for metric in metrics:
                primary_value = float(
                    primary[0]["final_diagnostics"][split_name][metric]
                )
                random_value = mean(
                    float(candidate["final_diagnostics"][split_name][metric])
                    for candidate in random
                )
                paired[(target, split_name, metric)].append(
                    primary_value - random_value
                )

    report: dict[str, Any] = {}
    for (target, split_name, metric), values in paired.items():
        report.setdefault(str(target), {}).setdefault(split_name, {})[metric] = (
            _summary(values)
        )
    return report


def _paired_primary_vs_controls(
    runs: list[dict[str, Any]],
    *,
    primary_method: str,
    controls: list[str],
) -> dict[str, Any]:
    metrics = (
        "target_f1_delta",
        "non_target_macro_f1_delta",
        "accuracy_delta",
        "utility",
    )
    paired: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    for run in runs:
        target = int(run["target_class"])
        candidates = run["candidates"]
        primary = [value for value in candidates if value["method"] == primary_method]
        if len(primary) != 1:
            raise RuntimeError("Primary control comparison is incomplete")
        split_names = tuple(primary[0]["final_diagnostics"])
        for control in controls:
            matched = [value for value in candidates if value["method"] == control]
            if len(matched) != 1 or tuple(matched[0]["final_diagnostics"]) != split_names:
                raise RuntimeError(f"Control comparison is incomplete for {control}")
            for split_name in split_names:
                for metric in metrics:
                    paired[(target, control, split_name, metric)].append(
                        float(primary[0]["final_diagnostics"][split_name][metric])
                        - float(matched[0]["final_diagnostics"][split_name][metric])
                    )
    report: dict[str, Any] = {}
    for (target, control, split_name, metric), values in paired.items():
        report.setdefault(str(target), {}).setdefault(control, {}).setdefault(
            split_name, {}
        )[metric] = _summary(values)
    return report


def _run_paths(root: Path) -> list[Path]:
    paths = list((root / "runs").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No run records found under {root / 'runs'}")

    def seed(path: Path) -> int:
        with path.open("r", encoding="utf-8") as stream:
            return int(json.load(stream)["seed"])

    return sorted(paths, key=seed)


def build_repair_report(
    config: dict[str, Any],
    *,
    allow_download: bool = False,
    progress: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    localization_path = Path(config["localization_report"])
    grouped_path = Path(config["grouped_report"])
    with localization_path.open("r", encoding="utf-8") as stream:
        localization = json.load(stream)
    with grouped_path.open("r", encoding="utf-8") as stream:
        grouped = json.load(stream)
    budget_fraction = float(config.get("budget_fraction", 0.0625))
    random_repeats = int(config.get("random_repeats", 2))
    controls = [
        str(value)
        for value in config.get("controls", ["target_logit_bias"])
    ]
    allowed_controls = {
        "target_logit_bias",
        "last_layer_finetune",
        "full_model_finetune",
    }
    if len(controls) != len(set(controls)) or not set(controls).issubset(
        allowed_controls
    ):
        raise ValueError(
            "controls must be unique target_logit_bias, last_layer_finetune, "
            "and/or full_model_finetune entries"
        )
    training = dict(config["training"])
    study = study_record(config)
    selected_splits = tuple(study["evaluation_splits"])
    if study["phase"] == "confirmatory_blind":
        for source_name, source_report in (
            ("localization", localization),
            ("grouped", grouped),
        ):
            if source_report.get("protocol", {}).get("study") != study:
                raise RuntimeError(
                    f"{source_name.capitalize()} report does not match the "
                    "confirmatory study"
                )
    cache_root = Path(config.get("cache_root", "artifacts/repair_runs"))
    if not 0 < budget_fraction <= 1:
        raise ValueError("Repair budget_fraction must be in (0, 1]")
    if random_repeats < 1:
        raise ValueError("Repair random_repeats must be positive")

    utility_penalty = float(training.get("utility_penalty", 1.0))
    max_non_target_drop = float(training.get("max_non_target_f1_drop", 0.01))
    protocol_code_sha256 = source_code_sha256(PROTOCOL_CODE_FILES)
    upstream_artifacts = {
        "localization_report": str(localization_path),
        "localization_report_sha256": _file_sha256(localization_path),
        "grouped_report": str(grouped_path),
        "grouped_report_sha256": _file_sha256(grouped_path),
    }

    report: dict[str, Any] = {
        "protocol": {
            "version": REPAIR_PROTOCOL_VERSION,
            "study": study,
            "config_sha256": canonical_payload_sha256(config),
            "study_status": (
                "confirmatory protocol with test outcomes sealed"
                if study["phase"] == "confirmatory_blind"
                else "exploratory development protocol"
            ),
            "fit_labels": "training only",
            "selection_labels": "validation only, with epoch zero as no-op candidate",
            "evaluation_splits": list(selected_splits),
            "test_status": test_status(selected_splits),
            "test_labels": (
                "replaced by an invalid sentinel and absent from all development metrics"
                if "test" not in selected_splits
                else "evaluated only in the declared unblinded phase"
            ),
            "target_selection": (
                "highest aggregate validation bottom-ablation repair utility at the "
                "fixed budget among each backbone's prelocalized low-F1 classes"
            ),
            "layer_selection": (
                "one layer selected by aggregate validation target-loss increase for "
                "the dataset's primary localization method, then shared by every "
                "ranking and matched-random control"
            ),
            "repair": "training-mean-centered sparse affine activation adapter",
            "parameters_per_selected_neuron": 2,
            "budget_fraction": budget_fraction,
            "random_repeats": random_repeats,
            "random_control": (
                "deterministic random neurons at the exact shared layer, width, site "
                "count, parameter count, objective, and stopping rule"
            ),
            "calibration_control": (
                "one-parameter target-logit bias; not parameter-matched"
            ),
            "fine_tuning_controls": {
                "methods": controls,
                "matching": (
                    "same checkpoint, objective, train/validation labels, optimizer "
                    "family, update cap, early stopping, epoch-zero candidate, and "
                    "collateral-harm constraints; parameter counts are disclosed and "
                    "are intentionally not matched"
                ),
                "last_layer": "output linear weight and bias only",
                "full_model": "every backbone parameter",
            },
            "experimental_unit": (
                "checkpoint seed or published split; random repeats are averaged "
                "within experimental unit"
            ),
            "training": training,
            "protocol_code_sha256": protocol_code_sha256,
            "upstream_artifacts": upstream_artifacts,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        },
        "datasets": {},
    }
    for dataset_name, dataset_config in config["datasets"].items():
        dataset_report = {
            "primary_method": str(dataset_config["primary_method"]),
            "primary_method_rationale": str(
                dataset_config["primary_method_rationale"]
            ),
            "backbones": {},
        }
        methods = [str(value) for value in dataset_config["methods"]]
        primary_method = str(dataset_config["primary_method"])
        if len(methods) != len(set(methods)) or primary_method not in methods:
            raise ValueError(
                f"Dataset {dataset_name!r} requires unique methods including its primary"
        )
        for backbone_name, backbone_config in dataset_config["backbones"].items():
            root = Path(backbone_config["root"])
            configured_targets = backbone_config.get("targets")
            if configured_targets is None:
                target_count = int(backbone_config.get("target_count", 1))
                if target_count < 1:
                    raise ValueError("Repair target_count must be positive")
            else:
                target_classes = [int(value) for value in configured_targets]
                target_count = len(target_classes)
            localized_runs = _index_unique_runs(
                localization["datasets"][dataset_name]["backbones"][backbone_name][
                    "runs"
                ],
                source=f"{dataset_name}/{backbone_name} localization report",
            )
            grouped_backbone = grouped["datasets"][dataset_name]["backbones"][
                backbone_name
            ]
            grouped_runs = _index_unique_runs(
                grouped_backbone["runs"],
                source=f"{dataset_name}/{backbone_name} grouped report",
            )
            expected_targets, target_priorities = select_validation_targets(
                grouped_backbone,
                target_count=target_count,
                budget_fraction=budget_fraction,
                utility_penalty=utility_penalty,
                max_non_target_drop=max_non_target_drop,
            )
            if configured_targets is None:
                target_classes = expected_targets
            elif target_classes != expected_targets:
                raise RuntimeError(
                    f"Configured repair targets {target_classes} do not match the "
                    f"validation-only rule {expected_targets} for "
                    f"{dataset_name}/{backbone_name}"
                )
            choices = {}
            for target in target_classes:
                shared_choice = select_grouped_choice(
                    grouped_backbone,
                    target_class=target,
                    method=primary_method,
                    budget_fraction=budget_fraction,
                    utility_penalty=utility_penalty,
                    max_non_target_drop=max_non_target_drop,
                )
                shared_choice.update(
                    {
                        "selected_by_method": primary_method,
                        "shared_across_ranking_methods": True,
                    }
                )
                choices[str(target)] = shared_choice

            runs = []
            paths = _run_paths(root)
            path_seeds = []
            for path in paths:
                with path.open("r", encoding="utf-8") as stream:
                    path_seeds.append(int(json.load(stream)["seed"]))
            if len(path_seeds) != len(set(path_seeds)):
                raise RuntimeError(f"Duplicate checkpoint seeds under {root}")
            expected_seed_set = set(path_seeds)
            if set(localized_runs) != expected_seed_set or set(grouped_runs) != expected_seed_set:
                raise RuntimeError(
                    f"Checkpoint, localization, and grouped seed sets differ for "
                    f"{dataset_name}/{backbone_name}"
                )

            for run_number, run_path in enumerate(paths, start=1):
                loaded = load_trained_run(run_path, allow_download=allow_download)
                if study_record(loaded.result["config"]) != study:
                    raise RuntimeError(
                        f"Study protocol mismatch for {dataset_name}/{backbone_name}"
                    )
                seed = int(loaded.result["seed"])
                localization_run = localized_runs[seed]
                grouped_run = grouped_runs[seed]
                _validate_upstream_run(
                    loaded,
                    localization_run,
                    source="Localization run",
                )
                _validate_upstream_run(
                    loaded,
                    grouped_run,
                    source="Grouped-intervention run",
                )
                if localization_run.get(
                    "localization_code_sha256"
                ) != source_code_sha256(LOCALIZATION_PROTOCOL_CODE_FILES):
                    raise RuntimeError("Localization implementation hash is stale")
                if grouped_run.get("grouped_code_sha256") != source_code_sha256(
                    GROUPED_PROTOCOL_CODE_FILES
                ):
                    raise RuntimeError("Grouped-intervention implementation hash is stale")
                localization_classes_sha256 = _json_payload_sha256(
                    localization_run["classes"]
                )
                if (
                    grouped_run.get("localization_classes_sha256")
                    != localization_classes_sha256
                ):
                    raise RuntimeError(
                        "Grouped run was not generated from the active localization "
                        f"payload for seed {seed}"
                    )
                for target_class in target_classes:
                    if target_class not in [
                        int(value) for value in grouped_run["target_classes"]
                    ]:
                        raise RuntimeError(
                            f"Target class {target_class} is absent from grouped run {seed}"
                        )
                    if str(target_class) not in localization_run["classes"]:
                        raise RuntimeError(
                            f"Target class {target_class} is absent from localization run {seed}"
                        )
                    if progress:
                        print(
                            f"[{dataset_name}/{backbone_name}] repair run "
                            f"{run_number}/{len(paths)} seed {seed} class {target_class}",
                            flush=True,
                        )
                    candidates = []
                    choice = choices[str(target_class)]
                    source_payload_sha256 = _json_payload_sha256(
                        {
                            "localization_class": localization_run["classes"][
                                str(target_class)
                            ],
                            "grouped_target_summary": grouped_backbone["summary"][
                                str(target_class)
                            ],
                            "target_priority": target_priorities[str(target_class)],
                            "shared_choice": choice,
                        }
                    )
                    for method in methods:
                        if progress:
                            print(f"  fitting {method}", flush=True)
                        ranking = _ranking_for(
                            localization_run,
                            target_class=target_class,
                            method=method,
                            layer=int(choice["layer"]),
                        )
                        if len(ranking) != int(choice["width"]):
                            raise RuntimeError(
                                f"Localization width does not match shared choice for "
                                f"seed {seed}, class {target_class}, method {method}"
                            )
                        neurons = ranking[: int(choice["budget"])]
                        if len(neurons) != int(choice["budget"]):
                            raise RuntimeError("Repair neuron budget is incomplete")
                        candidates.append(
                            _fit_with_cache(
                                loaded,
                                target_class=target_class,
                                method=method,
                                repeat=None,
                                choice=choice,
                                neurons=neurons,
                                training=training,
                                source_payload_sha256=source_payload_sha256,
                                upstream_artifacts=upstream_artifacts,
                                protocol_code_sha256=protocol_code_sha256,
                                cache_path=(
                                    cache_root
                                    / dataset_name
                                    / backbone_name
                                    / str(seed)
                                    / str(target_class)
                                    / f"{method}.json"
                                ),
                                resume=resume,
                                selected_splits=selected_splits,
                            )
                        )

                    for repeat in range(random_repeats):
                        if progress:
                            print(
                                f"  fitting matched_random repeat {repeat + 1}/"
                                f"{random_repeats}",
                                flush=True,
                            )
                        ranking = _random_ranking(
                            int(choice["width"]),
                            str(loaded.result["experiment"]),
                            seed,
                            target_class,
                            int(choice["layer"]),
                            repeat,
                        )
                        neurons = ranking[: int(choice["budget"])]
                        candidates.append(
                            _fit_with_cache(
                                loaded,
                                target_class=target_class,
                                method="matched_random",
                                repeat=repeat,
                                choice=choice,
                                neurons=neurons,
                                training=training,
                                source_payload_sha256=source_payload_sha256,
                                upstream_artifacts=upstream_artifacts,
                                protocol_code_sha256=protocol_code_sha256,
                                cache_path=(
                                    cache_root
                                    / dataset_name
                                    / backbone_name
                                    / str(seed)
                                    / str(target_class)
                                    / f"matched_random_{repeat}.json"
                                ),
                                resume=resume,
                                selected_splits=selected_splits,
                            )
                        )
                    for control in controls:
                        if progress:
                            print(f"  fitting {control}", flush=True)
                        candidates.append(
                            _fit_with_cache(
                                loaded,
                                target_class=target_class,
                                method=control,
                                repeat=None,
                                choice=None,
                                neurons=None,
                                training=training,
                                source_payload_sha256=source_payload_sha256,
                                upstream_artifacts=upstream_artifacts,
                                protocol_code_sha256=protocol_code_sha256,
                                cache_path=(
                                    cache_root
                                    / dataset_name
                                    / backbone_name
                                    / str(seed)
                                    / str(target_class)
                                    / f"{control}.json"
                                ),
                                resume=resume,
                                selected_splits=selected_splits,
                            )
                        )
                    runs.append(
                        {
                            "experiment": loaded.result["experiment"],
                            "seed": seed,
                            "target_class": target_class,
                            "candidates": candidates,
                        }
                    )
            dataset_report["backbones"][backbone_name] = {
                "targets": target_classes,
                "target_validation_priority": target_priorities,
                "shared_choices": choices,
                "runs": runs,
                "summary": _aggregate_candidates(runs),
                "paired_primary_vs_matched_random": _paired_primary_vs_random(
                    runs,
                    primary_method=primary_method,
                ),
                "paired_primary_vs_controls": _paired_primary_vs_controls(
                    runs,
                    primary_method=primary_method,
                    controls=controls,
                ),
            }
        report["datasets"][dataset_name] = dataset_report
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/repair_pilot.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = build_repair_report(
        config,
        allow_download=args.allow_download,
        progress=True,
        resume=args.resume,
    )
    output = args.output or Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Wrote repair report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
