"""Classification metrics used by every baseline and later intervention."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(
    logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    if mask.dtype != torch.bool or mask.shape != y.shape:
        raise ValueError("Metric mask must be a boolean vector aligned with labels")
    if not bool(mask.any()):
        raise ValueError("Cannot score an empty split")
    selected_labels = y[mask]
    if bool((selected_labels < 0).any()):
        raise RuntimeError(
            "Attempted to score blinded labels; use the auditable unblinding command"
        )
    if bool((selected_labels >= logits.shape[-1]).any()):
        raise ValueError("A scored label is outside the model output range")
    predictions = logits[mask].argmax(dim=-1).detach().cpu().numpy()
    truth = selected_labels.detach().cpu().numpy()
    labels = np.arange(logits.shape[-1])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        truth, predictions, average="macro", zero_division=0
    )
    _, _, micro_f1, _ = precision_recall_fscore_support(
        truth, predictions, average="micro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "per_class": {
            str(label): {
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in labels
        },
    }
