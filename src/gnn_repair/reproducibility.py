"""Helpers for deterministic and auditable experiments."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np
import sklearn
import scipy
import torch


def set_deterministic(seed: int, *, num_threads: int = 1) -> None:
    """Seed supported RNGs and request deterministic PyTorch operations.

    The baseline intentionally defaults to one CPU thread. This is slower than
    an unconstrained run but prevents thread scheduling from adding avoidable
    variation while the reference implementation is being established.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def environment_metadata() -> dict[str, Any]:
    """Return version and hardware facts saved with every experiment."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
