"""Device selection and GPU memory monitoring with CPU fallback."""

from __future__ import annotations

import os
from typing import Any

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """`prefer` or env NEBULA_DEVICE in {"cuda", "cpu"}; defaults to CUDA when available."""
    choice = (prefer or os.environ.get("NEBULA_DEVICE", "")).lower()
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_memory() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "free_mb": round(free / 1e6),
        "total_mb": round(total / 1e6),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    }


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
