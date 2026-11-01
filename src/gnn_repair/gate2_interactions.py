"""Validation-only infrastructure for Gate 2 channel-interaction experiments.

The functions in this module deliberately stop short of running a development
study.  They define the continuous repair utility, identity-initialized
singleton/pair adapters, matched same-versus-different-cluster sampling, the
predeclared interaction contrasts, and coherent-versus-MMR set construction.
No function reads test masks or reports test-set outcomes.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from gnn_repair.checkpoints import (
    LoadedRun,
    model_state_sha256,
    module_state_sha256,
)
from gnn_repair.reproducibility import set_deterministic


GATE2_BUDGETS = (1, 2, 4, 8)
MATCHING_FEATURE_NAMES = (
    "singleton_utility_sum",
    "gradient_score_sum",
    "activation_magnitude_sum",
    "target_selectivity_sum",
    "footprint_size_sum",
)


def _stable_key(value: Hashable) -> tuple[Any, ...]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return ("number", "integer", int(value))
    if isinstance(value, str):
        return ("text", "string", value)
    if isinstance(value, tuple):
        return ("sequence", "tuple", tuple(_stable_key(item) for item in value))
    value_type = type(value)
    return (value_type.__module__, value_type.__qualname__, repr(value))


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ContinuousUtility:
    """Components of target benefit minus continuous non-target harm."""

    target_ce_reduction: float
    non_target_kl: float
    collateral_ce_increase: float
    utility: float
    target_count: int
    non_target_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "target_ce_reduction": self.target_ce_reduction,
            "non_target_kl": self.non_target_kl,
            "collateral_ce_increase": self.collateral_ce_increase,
            "utility": self.utility,
            "target_count": self.target_count,
            "non_target_count": self.non_target_count,
        }


def continuous_validation_utility(
    base_logits: torch.Tensor,
    modified_logits: torch.Tensor,
    labels: torch.Tensor,
    validation_mask: torch.Tensor,
    target_class: int,
    *,
    non_target_kl_weight: float = 1.0,
    collateral_ce_weight: float = 1.0,
) -> ContinuousUtility:
    """Score one candidate using only explicitly supplied validation nodes.

    The target term is the reduction in cross-entropy relative to the frozen
    checkpoint.  Non-target drift is measured by ``KL(base || modified)`` and
    collateral damage is the positive part of the non-target cross-entropy
    increase.  Thus the exact no-op has utility zero.
    """

    if base_logits.ndim != 2 or modified_logits.shape != base_logits.shape:
        raise ValueError("Base and modified logits must be aligned matrices")
    num_nodes, num_classes = base_logits.shape
    if labels.shape != (num_nodes,) or validation_mask.shape != (num_nodes,):
        raise ValueError("Labels and validation mask must align with the logits")
    if validation_mask.dtype != torch.bool:
        raise TypeError("validation_mask must be a boolean tensor")
    if not 0 <= int(target_class) < num_classes:
        raise ValueError("target_class is outside the logit class dimension")
    if base_logits.device != modified_logits.device:
        raise ValueError("Base and modified logits must be on the same device")
    if (
        labels.device != base_logits.device
        or validation_mask.device != base_logits.device
    ):
        raise ValueError("Logits, labels, and validation mask must share a device")
    kl_weight = _finite_float(non_target_kl_weight, "non_target_kl_weight")
    collateral_weight = _finite_float(collateral_ce_weight, "collateral_ce_weight")
    if kl_weight < 0 or collateral_weight < 0:
        raise ValueError("Continuous-utility penalty weights must be non-negative")

    validation_labels = labels[validation_mask]
    if validation_labels.numel() == 0:
        raise ValueError("Validation mask is empty")
    if bool((validation_labels < 0).any()) or bool(
        (validation_labels >= num_classes).any()
    ):
        raise ValueError("Validation labels contain an invalid or blinded class")
    target_mask = validation_mask & (labels == int(target_class))
    non_target_mask = validation_mask & (labels != int(target_class))
    if not bool(target_mask.any()) or not bool(non_target_mask.any()):
        raise ValueError("Validation split lacks a target or non-target group")
    if not bool(torch.isfinite(base_logits[validation_mask]).all()) or not bool(
        torch.isfinite(modified_logits[validation_mask]).all()
    ):
        raise FloatingPointError("Validation logits contain a non-finite value")

    target_count = int(target_mask.sum().item())
    non_target_count = int(non_target_mask.sum().item())
    if torch.equal(base_logits[validation_mask], modified_logits[validation_mask]):
        return ContinuousUtility(
            target_ce_reduction=0.0,
            non_target_kl=0.0,
            collateral_ce_increase=0.0,
            utility=0.0,
            target_count=target_count,
            non_target_count=non_target_count,
        )

    with torch.no_grad():
        base_target_ce = F.cross_entropy(base_logits[target_mask], labels[target_mask])
        modified_target_ce = F.cross_entropy(
            modified_logits[target_mask], labels[target_mask]
        )
        base_non_target_ce = F.cross_entropy(
            base_logits[non_target_mask], labels[non_target_mask]
        )
        modified_non_target_ce = F.cross_entropy(
            modified_logits[non_target_mask], labels[non_target_mask]
        )
        non_target_kl = torch.clamp_min(
            F.kl_div(
                F.log_softmax(modified_logits[non_target_mask], dim=-1),
                F.softmax(base_logits[non_target_mask], dim=-1),
                reduction="batchmean",
            ),
            0.0,
        )
        target_reduction = base_target_ce - modified_target_ce
        collateral_increase = torch.clamp_min(
            modified_non_target_ce - base_non_target_ce, 0.0
        )
        utility = (
            target_reduction
            - kl_weight * non_target_kl
            - collateral_weight * collateral_increase
        )
    return ContinuousUtility(
        target_ce_reduction=float(target_reduction.item()),
        non_target_kl=float(non_target_kl.item()),
        collateral_ce_increase=float(collateral_increase.item()),
        utility=float(utility.item()),
        target_count=target_count,
        non_target_count=non_target_count,
    )


def singleton_utilities(
    base_logits: torch.Tensor,
    candidate_logits: Mapping[Hashable, torch.Tensor],
    labels: torch.Tensor,
    validation_mask: torch.Tensor,
    target_class: int,
    *,
    non_target_kl_weight: float = 1.0,
    collateral_ce_weight: float = 1.0,
) -> dict[Hashable, ContinuousUtility]:
    """Compute validation-only utility records for singleton candidates."""

    if not candidate_logits:
        raise ValueError("At least one singleton candidate is required")
    return {
        candidate: continuous_validation_utility(
            base_logits,
            logits,
            labels,
            validation_mask,
            target_class,
            non_target_kl_weight=non_target_kl_weight,
            collateral_ce_weight=collateral_ce_weight,
        )
        for candidate, logits in candidate_logits.items()
    }


class _LayerSparseAffineAdapter(nn.Module):
    """Identity-initialized affine deltas applied at one hidden layer."""

    def __init__(
        self,
        layer: int,
        channels: Sequence[int],
        training_mean: torch.Tensor,
        training_std: torch.Tensor,
        *,
        tied: bool,
    ) -> None:
        super().__init__()
        if int(layer) < 0:
            raise ValueError("Adapter layer must be non-negative")
        if training_mean.ndim != 1 or training_std.shape != training_mean.shape:
            raise ValueError("Training mean and standard deviation must align")
        selected = sorted(int(channel) for channel in channels)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("Selected channels must be non-empty and unique")
        if selected[0] < 0 or selected[-1] >= int(training_mean.numel()):
            raise ValueError("Selected channel is outside the activation width")
        indices = torch.tensor(
            selected,
            dtype=torch.long,
            device=training_mean.device,
        )
        self.layer = int(layer)
        self.tied = bool(tied)
        self.register_buffer("indices", indices)
        self.register_buffer("mean", training_mean[indices].detach().clone())
        self.register_buffer(
            "std", training_std[indices].detach().clone().clamp_min(1e-6)
        )
        parameter_width = 1 if tied else len(selected)
        self.delta_scale = nn.Parameter(
            torch.zeros(parameter_width, device=training_mean.device)
        )
        self.delta_shift = nn.Parameter(
            torch.zeros(parameter_width, device=training_mean.device)
        )

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.indices.detach().cpu().tolist())

    @property
    def selected_sites(self) -> int:
        return int(self.indices.numel())

    def forward(self, layer: int, activation: torch.Tensor) -> torch.Tensor:
        if int(layer) != self.layer:
            return activation
        if activation.ndim != 2 or int(activation.shape[1]) <= int(self.indices[-1]):
            raise ValueError("Activation matrix does not contain selected channels")
        selected = activation.index_select(1, self.indices)
        adjustment = (
            selected - self.mean
        ) * self.delta_scale + self.std * self.delta_shift
        return activation.index_add(1, self.indices, adjustment)


class SingletonAffineAdapter(_LayerSparseAffineAdapter):
    """Independent scale and shift deltas for one channel (two parameters)."""

    def __init__(
        self,
        layer: int,
        channel: int,
        training_mean: torch.Tensor,
        training_std: torch.Tensor,
    ) -> None:
        super().__init__(
            layer,
            [channel],
            training_mean,
            training_std,
            tied=False,
        )


class IndependentPairAffineAdapter(_LayerSparseAffineAdapter):
    """Separate scale/shift deltas for two channels (four parameters)."""

    def __init__(
        self,
        layer: int,
        channels: Sequence[int],
        training_mean: torch.Tensor,
        training_std: torch.Tensor,
    ) -> None:
        if len(channels) != 2:
            raise ValueError("An independent pair adapter requires two channels")
        super().__init__(
            layer,
            channels,
            training_mean,
            training_std,
            tied=False,
        )


class TiedPairAffineAdapter(_LayerSparseAffineAdapter):
    """One shared scale/shift delta for two channels (two parameters)."""

    def __init__(
        self,
        layer: int,
        channels: Sequence[int],
        training_mean: torch.Tensor,
        training_std: torch.Tensor,
    ) -> None:
        if len(channels) != 2:
            raise ValueError("A tied pair adapter requires two channels")
        super().__init__(
            layer,
            channels,
            training_mean,
            training_std,
            tied=True,
        )


@torch.no_grad()
def gate2_activation_statistics(
    loaded: LoadedRun,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute per-layer activation statistics from training nodes only."""

    _, activations = loaded.model(
        loaded.data.x,
        loaded.data.adjacency,
        return_activations=True,
    )
    means = [value[loaded.data.train_mask].mean(dim=0) for value in activations]
    stds = [
        value[loaded.data.train_mask].std(dim=0, unbiased=False)
        for value in activations
    ]
    return means, stds


def select_validation_epoch(
    validation_utilities: Sequence[ContinuousUtility | float],
    *,
    min_delta: float = 0.0,
) -> int:
    """Select an epoch against an implicit identity/no-op candidate at epoch 0."""

    threshold = _finite_float(min_delta, "min_delta")
    if threshold < 0:
        raise ValueError("min_delta must be non-negative")
    best_epoch = 0
    best_utility = 0.0
    for epoch, record in enumerate(validation_utilities, start=1):
        value = (
            record.utility if isinstance(record, ContinuousUtility) else float(record)
        )
        value = _finite_float(value, f"validation utility at epoch {epoch}")
        if value > best_utility + threshold:
            best_epoch = epoch
            best_utility = value
    return best_epoch


def _adapter_forward(
    loaded: LoadedRun,
    adapter: _LayerSparseAffineAdapter,
) -> torch.Tensor:
    return loaded.model(
        loaded.data.x,
        loaded.data.adjacency,
        activation_intervention=adapter,
    )


def _parameter_mean_square(parameters: Sequence[torch.Tensor]) -> torch.Tensor:
    if not parameters:
        raise ValueError("Adapter has no trainable parameters")
    return sum(parameter.square().sum() for parameter in parameters) / sum(
        parameter.numel() for parameter in parameters
    )


def fit_gate2_adapter(
    loaded: LoadedRun,
    adapter: _LayerSparseAffineAdapter,
    *,
    target_class: int,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit an adapter on training nodes and select only on validation utility.

    Epoch zero is always the exactly identity-initialized no-op candidate.  The
    function intentionally neither accepts a test split nor computes test
    metrics; downstream study code must freeze Gate 2 before any fresh reveal.
    The supplied adapter is left in its selected state.
    """

    learning_rate = _finite_float(training["learning_rate"], "learning_rate")
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    min_delta = _finite_float(training.get("min_utility_delta", 0.0), "min_delta")
    target_ce_weight = _finite_float(
        training.get("target_ce_weight", 1.0), "target_ce_weight"
    )
    non_target_kl_weight = _finite_float(
        training.get("non_target_kl_weight", 1.0), "non_target_kl_weight"
    )
    collateral_ce_weight = _finite_float(
        training.get("collateral_ce_weight", 1.0), "collateral_ce_weight"
    )
    identity_l2_weight = _finite_float(
        training.get("identity_l2_weight", 0.001), "identity_l2_weight"
    )
    if learning_rate <= 0 or max_epochs < 1 or patience < 1:
        raise ValueError("Learning rate, max_epochs, and patience must be positive")
    if min_delta < 0 or any(
        value < 0
        for value in (
            target_ce_weight,
            non_target_kl_weight,
            collateral_ce_weight,
            identity_l2_weight,
        )
    ):
        raise ValueError("Training and selection weights must be non-negative")

    set_deterministic(
        int(loaded.result["seed"]),
        num_threads=int(training.get("num_threads", 1)),
    )
    model = loaded.model
    model.eval()
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    checkpoint_hash = model_state_sha256(loaded.checkpoint)
    base_hash = module_state_sha256(model)
    if base_hash != checkpoint_hash:
        raise RuntimeError("Loaded backbone does not match its checkpoint")

    device = loaded.data.x.device
    adapter.to(device)
    with torch.no_grad():
        base_logits = model(loaded.data.x, loaded.data.adjacency).detach()
        identity_logits = _adapter_forward(loaded, adapter)
    if not torch.equal(identity_logits, base_logits):
        raise RuntimeError("Gate 2 adapter is not exactly identity-initialized")

    labels = loaded.data.y
    train_target = loaded.data.train_mask & (labels == int(target_class))
    train_non_target = loaded.data.train_mask & (labels != int(target_class))
    if not bool(train_target.any()) or not bool(train_non_target.any()):
        raise ValueError("Training split lacks a target or non-target group")
    with torch.no_grad():
        base_non_target_ce = F.cross_entropy(
            base_logits[train_non_target], labels[train_non_target]
        )
        no_op_validation = continuous_validation_utility(
            base_logits,
            base_logits,
            labels,
            loaded.data.val_mask,
            target_class,
            non_target_kl_weight=non_target_kl_weight,
            collateral_ce_weight=collateral_ce_weight,
        )
    if no_op_validation.utility != 0.0:
        raise RuntimeError("Identity candidate does not have zero validation utility")

    parameters = list(adapter.parameters())
    if not parameters:
        raise ValueError("Gate 2 adapter has no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in parameters}:
        raise RuntimeError("Optimizer parameter set does not match the adapter")

    initial_state = copy.deepcopy(adapter.state_dict())
    best_state = copy.deepcopy(initial_state)
    best_validation = no_op_validation
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    validation_records: list[ContinuousUtility] = []
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = _adapter_forward(loaded, adapter)
        target_ce = F.cross_entropy(logits[train_target], labels[train_target])
        non_target_ce = F.cross_entropy(
            logits[train_non_target], labels[train_non_target]
        )
        non_target_kl = F.kl_div(
            F.log_softmax(logits[train_non_target], dim=-1),
            F.softmax(base_logits[train_non_target], dim=-1),
            reduction="batchmean",
        )
        collateral_ce_increase = torch.clamp_min(
            non_target_ce - base_non_target_ce, 0.0
        )
        identity_l2 = _parameter_mean_square(parameters)
        loss = (
            target_ce_weight * target_ce
            + non_target_kl_weight * non_target_kl
            + collateral_ce_weight * collateral_ce_increase
            + identity_l2_weight * identity_l2
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Gate 2 loss became non-finite at epoch {epoch}")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Frozen backbone received a Gate 2 gradient")
        if any(
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise FloatingPointError(
                f"Gate 2 adapter gradient is invalid at epoch {epoch}"
            )
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
            raise FloatingPointError(
                f"Gate 2 adapter parameter is non-finite at epoch {epoch}"
            )

        with torch.no_grad():
            validation_logits = _adapter_forward(loaded, adapter)
            validation = continuous_validation_utility(
                base_logits,
                validation_logits,
                labels,
                loaded.data.val_mask,
                target_class,
                non_target_kl_weight=non_target_kl_weight,
                collateral_ce_weight=collateral_ce_weight,
            )
        validation_records.append(validation)
        history.append(
            {
                "epoch": epoch,
                "training": {
                    "target_ce": float(target_ce.detach().item()),
                    "non_target_ce": float(non_target_ce.detach().item()),
                    "non_target_kl": float(non_target_kl.detach().item()),
                    "collateral_ce_increase": float(
                        collateral_ce_increase.detach().item()
                    ),
                    "identity_l2": float(identity_l2.detach().item()),
                    "loss": float(loss.detach().item()),
                },
                "validation": validation.as_dict(),
            }
        )
        selected_epoch = select_validation_epoch(
            validation_records,
            min_delta=min_delta,
        )
        if selected_epoch == epoch:
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())
            best_validation = validation
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    adapter.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        selected_logits = _adapter_forward(loaded, adapter)
    if best_epoch == 0 and not torch.equal(selected_logits, base_logits):
        raise RuntimeError("Epoch-zero selection did not restore the exact no-op")
    if module_state_sha256(model) != base_hash:
        raise RuntimeError("Frozen backbone changed during Gate 2 fitting")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen backbone retains a Gate 2 gradient")

    state = {
        name: tensor.detach().cpu().tolist()
        for name, tensor in adapter.state_dict().items()
        if name.endswith("delta_scale") or name.endswith("delta_shift")
    }
    return {
        "target_class": int(target_class),
        "adapter": type(adapter).__name__,
        "layer": adapter.layer,
        "channels": list(adapter.channels),
        "selected_sites": adapter.selected_sites,
        "num_trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation": best_validation.as_dict(),
        "history": history,
        "repair_state": state,
        "selection_split": "validation",
        "evaluation_splits": ["validation"],
        "base_model_sha256": base_hash,
    }


def pair_interaction(
    singleton_first: float,
    singleton_second: float,
    independent_pair_utility: float,
) -> float:
    """Return ``I_ij = u_ij_ind - u_i - u_j``."""

    first = _finite_float(singleton_first, "singleton_first")
    second = _finite_float(singleton_second, "singleton_second")
    independent = _finite_float(independent_pair_utility, "independent_pair_utility")
    return independent - first - second


def tied_sharing_gain(
    tied_pair_utility: float,
    independent_pair_utility: float,
) -> float:
    """Return the pair-level tied-minus-independent utility difference."""

    tied = _finite_float(tied_pair_utility, "tied_pair_utility")
    independent = _finite_float(independent_pair_utility, "independent_pair_utility")
    return tied - independent


@dataclass(frozen=True)
class PairUtilityRecord:
    """Utilities required for one same- or different-cluster pair."""

    pair: tuple[Hashable, Hashable]
    same_cluster: bool
    singleton_first: float
    singleton_second: float
    independent_pair_utility: float
    tied_pair_utility: float

    def __post_init__(self) -> None:
        if len(self.pair) != 2 or self.pair[0] == self.pair[1]:
            raise ValueError("PairUtilityRecord requires two distinct channels")
        for name in (
            "singleton_first",
            "singleton_second",
            "independent_pair_utility",
            "tied_pair_utility",
        ):
            _finite_float(getattr(self, name), name)

    @property
    def interaction(self) -> float:
        return pair_interaction(
            self.singleton_first,
            self.singleton_second,
            self.independent_pair_utility,
        )

    @property
    def sharing_gain(self) -> float:
        return tied_sharing_gain(
            self.tied_pair_utility,
            self.independent_pair_utility,
        )


@dataclass(frozen=True)
class InteractionContrasts:
    """Checkpoint-level same-minus-different Gate 2 estimands."""

    same_count: int
    different_count: int
    same_mean_interaction: float
    different_mean_interaction: float
    d_int: float
    same_mean_sharing_gain: float
    different_mean_sharing_gain: float
    gamma: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "same_count": self.same_count,
            "different_count": self.different_count,
            "same_mean_interaction": self.same_mean_interaction,
            "different_mean_interaction": self.different_mean_interaction,
            "d_int": self.d_int,
            "same_mean_sharing_gain": self.same_mean_sharing_gain,
            "different_mean_sharing_gain": self.different_mean_sharing_gain,
            "gamma": self.gamma,
        }


def interaction_contrasts(
    records: Sequence[PairUtilityRecord],
) -> InteractionContrasts:
    """Compute checkpoint-level ``D_int`` and ``Gamma`` from matched pairs."""

    same = [record for record in records if record.same_cluster]
    different = [record for record in records if not record.same_cluster]
    if not same or not different:
        raise ValueError("Interaction contrasts require both pair groups")
    same_interaction = fmean(record.interaction for record in same)
    different_interaction = fmean(record.interaction for record in different)
    same_sharing = fmean(record.sharing_gain for record in same)
    different_sharing = fmean(record.sharing_gain for record in different)
    return InteractionContrasts(
        same_count=len(same),
        different_count=len(different),
        same_mean_interaction=same_interaction,
        different_mean_interaction=different_interaction,
        d_int=same_interaction - different_interaction,
        same_mean_sharing_gain=same_sharing,
        different_mean_sharing_gain=different_sharing,
        gamma=same_sharing - different_sharing,
    )


@dataclass(frozen=True)
class ChannelMatchingCovariates:
    """Predeclared non-structural covariates for pair matching."""

    singleton_utility: float
    gradient_score: float
    activation_magnitude: float
    target_selectivity: float
    layer: int
    footprint_size: float

    def __post_init__(self) -> None:
        if int(self.layer) < 0:
            raise ValueError("Matching layer must be non-negative")
        for name in (
            "singleton_utility",
            "gradient_score",
            "activation_magnitude",
            "target_selectivity",
            "footprint_size",
        ):
            _finite_float(getattr(self, name), name)


@dataclass(frozen=True)
class MatchedClusterPair:
    """One same-cluster pair and its cross-cluster covariate match."""

    same_pair: tuple[Hashable, Hashable]
    different_pair: tuple[Hashable, Hashable]
    layer: int
    standardized_distance: float
    same_match_vector: tuple[float, ...]
    different_match_vector: tuple[float, ...]


def _ordered_pair(
    first: Hashable,
    second: Hashable,
) -> tuple[Hashable, Hashable]:
    if first == second:
        raise ValueError("A channel cannot be paired with itself")
    return (
        (first, second) if _stable_key(first) < _stable_key(second) else (second, first)
    )


def _pair_match_vector(
    pair: tuple[Hashable, Hashable],
    covariates: Mapping[Hashable, ChannelMatchingCovariates],
) -> np.ndarray:
    first, second = (covariates[pair[0]], covariates[pair[1]])
    if int(first.layer) != int(second.layer):
        raise ValueError("Matched pair candidates must remain within one layer")
    return np.asarray(
        [
            first.singleton_utility + second.singleton_utility,
            first.gradient_score + second.gradient_score,
            first.activation_magnitude + second.activation_magnitude,
            first.target_selectivity + second.target_selectivity,
            first.footprint_size + second.footprint_size,
        ],
        dtype=np.float64,
    )


def match_same_different_cluster_pairs(
    cluster_by_channel: Mapping[Hashable, Hashable],
    covariates: Mapping[Hashable, ChannelMatchingCovariates],
    *,
    seed: int = 0,
    max_pairs: int | None = None,
) -> tuple[MatchedClusterPair, ...]:
    """Match same-cluster pairs to cross-cluster pairs without replacement.

    Matching is exact on layer.  Within each layer a minimum-cost bipartite
    assignment uses standardized sums of the five preregistered continuous
    covariates.  Structural similarity is intentionally absent from the API.
    ``seed`` breaks exact cost ties reproducibly; it does not change covariates.
    """

    if not cluster_by_channel:
        raise ValueError("Pair matching requires clustered channels")
    if set(cluster_by_channel) != set(covariates):
        raise ValueError("Cluster labels and matching covariates must share keys")
    if max_pairs is not None and int(max_pairs) < 1:
        raise ValueError("max_pairs must be positive when supplied")
    for cluster in cluster_by_channel.values():
        try:
            hash(cluster)
        except TypeError as error:
            raise TypeError("Cluster labels must be hashable") from error

    channels_by_layer: dict[int, list[Hashable]] = {}
    for channel, values in covariates.items():
        channels_by_layer.setdefault(int(values.layer), []).append(channel)
    rng = np.random.default_rng(int(seed))
    matches: list[MatchedClusterPair] = []
    for layer in sorted(channels_by_layer):
        channels = sorted(channels_by_layer[layer], key=_stable_key)
        same_pairs: list[tuple[Hashable, Hashable]] = []
        different_pairs: list[tuple[Hashable, Hashable]] = []
        for first_index, first in enumerate(channels):
            for second in channels[first_index + 1 :]:
                pair = _ordered_pair(first, second)
                if cluster_by_channel[first] == cluster_by_channel[second]:
                    same_pairs.append(pair)
                else:
                    different_pairs.append(pair)
        if not same_pairs or not different_pairs:
            continue
        same_values = np.vstack(
            [_pair_match_vector(pair, covariates) for pair in same_pairs]
        )
        different_values = np.vstack(
            [_pair_match_vector(pair, covariates) for pair in different_pairs]
        )
        combined = np.vstack((same_values, different_values))
        scale = combined.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        distances = np.linalg.norm(
            (same_values[:, None, :] - different_values[None, :, :]) / scale,
            axis=2,
        )
        tie_jitter = rng.uniform(0.0, 1e-12, size=distances.shape)
        same_indices, different_indices = linear_sum_assignment(distances + tie_jitter)
        for same_index, different_index in zip(
            same_indices.tolist(), different_indices.tolist(), strict=True
        ):
            matches.append(
                MatchedClusterPair(
                    same_pair=same_pairs[same_index],
                    different_pair=different_pairs[different_index],
                    layer=layer,
                    standardized_distance=float(distances[same_index, different_index]),
                    same_match_vector=tuple(
                        float(value) for value in same_values[same_index]
                    ),
                    different_match_vector=tuple(
                        float(value) for value in different_values[different_index]
                    ),
                )
            )
    if not matches:
        raise ValueError(
            "No layer contains both same-cluster and different-cluster pair candidates"
        )
    matches.sort(
        key=lambda match: (
            match.standardized_distance,
            match.layer,
            tuple(_stable_key(value) for value in match.same_pair),
            tuple(_stable_key(value) for value in match.different_pair),
        )
    )
    if max_pairs is not None:
        matches = matches[: int(max_pairs)]
    return tuple(matches)


def _validated_functional_scores(
    functional_scores: Mapping[Hashable, float],
) -> dict[Hashable, float]:
    if not functional_scores:
        raise ValueError("Selection requires at least one functional candidate")
    return {
        candidate: _finite_float(score, f"functional score for {candidate!r}")
        for candidate, score in functional_scores.items()
    }


def _ranked_candidates(
    functional_scores: Mapping[Hashable, float],
) -> list[Hashable]:
    return sorted(
        functional_scores,
        key=lambda candidate: (-functional_scores[candidate], _stable_key(candidate)),
    )


def coherent_selection(
    functional_scores: Mapping[Hashable, float],
    cluster_by_channel: Mapping[Hashable, Hashable],
    k: int,
) -> tuple[Hashable, ...]:
    """Choose the highest-total-score top-``k`` set contained in one cluster."""

    scores = _validated_functional_scores(functional_scores)
    if set(scores) != set(cluster_by_channel):
        raise ValueError("Functional scores and cluster labels must share keys")
    if int(k) < 1:
        raise ValueError("Selection budget must be positive")
    groups: dict[Hashable, list[Hashable]] = {}
    for candidate, cluster in cluster_by_channel.items():
        groups.setdefault(cluster, []).append(candidate)
    candidates: list[tuple[float, tuple[Hashable, ...]]] = []
    for members in groups.values():
        ranked = _ranked_candidates({member: scores[member] for member in members})
        if len(ranked) >= int(k):
            selected = tuple(ranked[: int(k)])
            candidates.append((sum(scores[value] for value in selected), selected))
    if not candidates:
        raise ValueError("No structural cluster can supply the requested budget")
    candidates.sort(
        key=lambda value: (
            -value[0],
            tuple(_stable_key(candidate) for candidate in value[1]),
        )
    )
    return candidates[0][1]


Similarity = (
    np.ndarray
    | Mapping[tuple[Hashable, Hashable], float]
    | Callable[[Hashable, Hashable], float]
)


def _structural_similarity(
    similarity: Similarity,
    first: Hashable,
    second: Hashable,
) -> float:
    if callable(similarity):
        value = similarity(first, second)
    elif isinstance(similarity, np.ndarray):
        if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
            raise ValueError("Structural-similarity array must be square")
        if not isinstance(first, (int, np.integer)) or not isinstance(
            second, (int, np.integer)
        ):
            raise TypeError(
                "Array structural similarity requires integer channel identifiers"
            )
        if not (
            0 <= int(first) < similarity.shape[0]
            and 0 <= int(second) < similarity.shape[1]
        ):
            raise IndexError("Channel identifier is outside the similarity array")
        value = similarity[int(first), int(second)]
    elif (first, second) in similarity:
        value = similarity[(first, second)]
    elif (second, first) in similarity:
        value = similarity[(second, first)]
    else:
        raise KeyError(f"Missing structural similarity for {(first, second)!r}")
    return _finite_float(value, f"structural similarity for {(first, second)!r}")


def diverse_mmr_selection(
    functional_scores: Mapping[Hashable, float],
    structural_similarity: Similarity,
    k: int,
    *,
    diversity_penalty: float,
) -> tuple[Hashable, ...]:
    """Greedily maximize ``f_i - lambda * max_j K_struct(i, j)``."""

    scores = _validated_functional_scores(functional_scores)
    budget = int(k)
    penalty = _finite_float(diversity_penalty, "diversity_penalty")
    if budget < 1 or budget > len(scores):
        raise ValueError("MMR budget must be between one and the candidate count")
    if penalty < 0:
        raise ValueError("diversity_penalty must be non-negative")
    selected = [_ranked_candidates(scores)[0]]
    while len(selected) < budget:
        remaining = [candidate for candidate in scores if candidate not in selected]
        mmr_scores = {
            candidate: scores[candidate]
            - penalty
            * max(
                _structural_similarity(structural_similarity, candidate, chosen)
                for chosen in selected
            )
            for candidate in remaining
        }
        selected.append(_ranked_candidates(mmr_scores)[0])
    return tuple(selected)


def coherent_vs_diverse_selections(
    functional_scores: Mapping[Hashable, float],
    cluster_by_channel: Mapping[Hashable, Hashable],
    structural_similarity: Similarity,
    *,
    diversity_penalty: float,
    budgets: Sequence[int] = GATE2_BUDGETS,
) -> dict[int, dict[str, tuple[Hashable, ...]]]:
    """Build parameter-count-matched coherent and MMR sets where feasible."""

    requested = tuple(int(value) for value in budgets)
    if not requested or any(value < 1 for value in requested):
        raise ValueError("Gate 2 budgets must be positive")
    if len(set(requested)) != len(requested):
        raise ValueError("Gate 2 budgets must be unique")
    scores = _validated_functional_scores(functional_scores)
    selections: dict[int, dict[str, tuple[Hashable, ...]]] = {}
    for budget in requested:
        if budget > len(scores):
            continue
        try:
            coherent = coherent_selection(scores, cluster_by_channel, budget)
        except ValueError:
            continue
        diverse = diverse_mmr_selection(
            scores,
            structural_similarity,
            budget,
            diversity_penalty=diversity_penalty,
        )
        if budget == 1 and coherent != diverse:
            raise RuntimeError("Coherent and diverse selection must coincide at k=1")
        selections[budget] = {
            "coherent": coherent,
            "diverse_mmr": diverse,
        }
    return selections
