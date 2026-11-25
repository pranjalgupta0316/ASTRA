"""Environment / resource report (hardware, GPU, library versions)."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
from typing import Any

PACKAGES = [
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "duckdb",
    "scipy",
    "statsmodels",
    "arch",
    "scikit-learn",
    "xgboost",
    "cvxpy",
    "yfinance",
    "exchange-calendars",
    "pydantic",
    "streamlit",
    "plotly",
    "torch",
]


def environment_report() -> dict[str, Any]:
    rep: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore[import-not-found]

        vm = psutil.virtual_memory()
        rep["ram_total_gb"] = round(vm.total / 1e9, 1)
        rep["ram_available_gb"] = round(vm.available / 1e9, 1)
    except ImportError:
        rep["ram_total_gb"] = "psutil not installed"
    du = shutil.disk_usage(".")
    rep["disk_free_gb"] = round(du.free / 1e9, 1)
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    rep["packages"] = versions
    try:
        import torch  # pyright: ignore[reportMissingImports]  - optional `dl` extra

        rep["torch_cuda_available"] = torch.cuda.is_available()
        rep["torch_cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            rep["gpu"] = props.name
            rep["gpu_vram_gb"] = round(props.total_memory / 1e9, 2)
            free, _total = torch.cuda.mem_get_info()
            rep["gpu_vram_free_gb"] = round(free / 1e9, 2)
    except ImportError:
        rep["torch_cuda_available"] = "torch not installed"
    return rep
