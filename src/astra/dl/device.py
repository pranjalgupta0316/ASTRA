"""Device selection and GPU memory monitoring with CPU fallback."""

from __future__ import annotations

import os
from typing import Any

try:  # torch is the optional `dl` extra; CI and non-DL installs run without it
    import torch  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - exercised only without the extra
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise ImportError("PyTorch is not installed; install the `dl` extra (uv sync --extra dl)")
    return torch


def get_device(prefer: str | None = None) -> Any:
    """`prefer` or env ASTRA_DEVICE in {"cuda", "cpu"}; defaults to CUDA when available."""
    th = _require_torch()
    choice = (prefer or os.environ.get("ASTRA_DEVICE", "")).lower()
    if choice == "cpu":
        return th.device("cpu")
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def gpu_memory() -> dict[str, Any]:
    if torch is None:
        return {"cuda": False, "torch": False}
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
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
