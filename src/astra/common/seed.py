"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - some libraries still draw from the legacy global RNG
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # pyright: ignore[reportMissingImports]  - optional `dl` extra
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
