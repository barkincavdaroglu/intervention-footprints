"""Native sparse-PyTorch node-classification backbones."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


ActivationIntervention = Callable[[int, torch.Tensor], torch.Tensor]


def _intervene(
    layer_index: int,
    activation: torch.Tensor,
    intervention: ActivationIntervention | None,
) -> torch.Tensor:
    return activation if intervention is None else intervention(layer_index, activation)


class GraphConvolution(nn.Module):
    """A GCN layer operating on a pre-normalized sparse adjacency matrix."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # A @ (X @ W) is equivalent to (A @ X) @ W, but projecting first is
        # much cheaper when the input feature dimension exceeds the hidden
        # dimension. Add the bias after aggregation, matching GCNConv.
        projected = F.linear(x, self.linear.weight, bias=None)
        aggregated = torch.sparse.mm(adjacency, projected)
        if self.linear.bias is not None:
            aggregated = aggregated + self.linear.bias
        return aggregated


class GCN(nn.Module):
    """Three-layer full-batch GCN with dataset-dependent input and output widths."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: tuple[int, int] | list[int],
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(hidden_channels) != 2:
            raise ValueError("The reference GCN requires exactly two hidden layers")
        hidden_1, hidden_2 = hidden_channels
        self.convolutions = nn.ModuleList(
            [
                GraphConvolution(in_channels, hidden_1),
                GraphConvolution(hidden_1, hidden_2),
                GraphConvolution(hidden_2, out_channels),
            ]
        )
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        return_activations: bool = False,
        activation_intervention: ActivationIntervention | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden_1 = F.relu(self.convolutions[0](x, adjacency))
        hidden_1 = _intervene(0, hidden_1, activation_intervention)
        x = F.dropout(hidden_1, p=self.dropout, training=self.training)
        hidden_2 = F.relu(self.convolutions[1](x, adjacency))
        hidden_2 = _intervene(1, hidden_2, activation_intervention)
        x = F.dropout(hidden_2, p=self.dropout, training=self.training)
        logits = self.convolutions[2](x, adjacency)
        if return_activations:
            return logits, (hidden_1, hidden_2)
        return logits


class MLP(nn.Module):
    """Graph-free control matched to the reference GCN's linear layers."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: tuple[int, int] | list[int],
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(hidden_channels) != 2:
            raise ValueError("The matched MLP requires exactly two hidden layers")
        hidden_1, hidden_2 = hidden_channels
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_channels, hidden_1),
                nn.Linear(hidden_1, hidden_2),
                nn.Linear(hidden_2, out_channels),
            ]
        )
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        return_activations: bool = False,
        activation_intervention: ActivationIntervention | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        del adjacency
        hidden_1 = F.relu(self.linears[0](x))
        hidden_1 = _intervene(0, hidden_1, activation_intervention)
        x = F.dropout(hidden_1, p=self.dropout, training=self.training)
        hidden_2 = F.relu(self.linears[1](x))
        hidden_2 = _intervene(1, hidden_2, activation_intervention)
        x = F.dropout(hidden_2, p=self.dropout, training=self.training)
        logits = self.linears[2](x)
        if return_activations:
            return logits, (hidden_1, hidden_2)
        return logits


class FeedForward(nn.Module):
    """Two-layer GELU feed-forward submodule used inside residual blocks."""

    def __init__(self, channels: int, multiplier: float, dropout: float) -> None:
        super().__init__()
        expanded_channels = int(channels * multiplier)
        if expanded_channels < 1:
            raise ValueError("The feed-forward multiplier produces zero channels")
        self.linear_1 = nn.Linear(channels, expanded_channels)
        self.linear_2 = nn.Linear(expanded_channels, channels)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.gelu(x)
        x = self.linear_2(x)
        return F.dropout(x, p=self.dropout, training=self.training)


class ResidualGCNBlock(nn.Module):
    """Pre-normalized GCN/MLP residual block from the heterophily benchmark."""

    def __init__(self, channels: int, multiplier: float, dropout: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)
        self.feed_forward = FeedForward(channels, multiplier, dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        update = self.normalization(x)
        update = torch.sparse.mm(adjacency, update)
        update = self.feed_forward(update)
        return x + update


class ResidualGCN(nn.Module):
    """Residual, layer-normalized GCN inspired by Platonov et al. (2023)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        hidden_multiplier: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("ResidualGCN requires at least one block")
        self.input_linear = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                ResidualGCNBlock(hidden_channels, hidden_multiplier, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.output_normalization = nn.LayerNorm(hidden_channels)
        self.output_linear = nn.Linear(hidden_channels, out_channels)
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        return_activations: bool = False,
        activation_intervention: ActivationIntervention | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.input_linear(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.gelu(x)
        activations: list[torch.Tensor] = []
        for layer_index, block in enumerate(self.blocks):
            x = block(x, adjacency)
            x = _intervene(layer_index, x, activation_intervention)
            if return_activations:
                activations.append(x)
        logits = self.output_linear(self.output_normalization(x))
        if return_activations:
            return logits, tuple(activations)
        return logits


def build_model(
    model_config: dict[str, Any], *, in_channels: int, out_channels: int
) -> nn.Module:
    """Construct a configured backbone behind a shared training interface."""

    model_type = str(model_config.get("type", "gcn"))
    if model_type in {"gcn", "mlp"}:
        model_class = GCN if model_type == "gcn" else MLP
        return model_class(
            in_channels=in_channels,
            hidden_channels=list(model_config["hidden_channels"]),
            out_channels=out_channels,
            dropout=float(model_config["dropout"]),
        )
    if model_type == "residual_gcn":
        return ResidualGCN(
            in_channels=in_channels,
            hidden_channels=int(model_config["hidden_channels"]),
            out_channels=out_channels,
            num_blocks=int(model_config["num_blocks"]),
            hidden_multiplier=float(model_config.get("hidden_multiplier", 1.0)),
            dropout=float(model_config["dropout"]),
        )
    raise ValueError(
        f"Unknown model type {model_type!r}; expected 'gcn', 'mlp', or 'residual_gcn'"
    )
