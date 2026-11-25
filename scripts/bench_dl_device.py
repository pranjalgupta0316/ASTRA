"""Benchmark CPU vs CUDA for ASTRA's small PyTorch classifiers on real walk-forward folds.

Usage: uv run python scripts/bench_dl_device.py [SYMBOL]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from astra.common.config import get_config
from astra.dl.device import gpu_memory, reset_peak
from astra.ml.classical import walk_forward_from_splits
from astra.ml.datasets import load_features, model_frame
from astra.ml.validation import PurgedWalkForward
from astra.research.ml_study import SEQ_LEN, _inputs, model_factory


def main(symbol: str = "AAPL") -> None:
    ppy = get_config().market_of(symbol).periods_per_year
    df = model_frame(load_features(symbol), SEQ_LEN)
    y = df["target_up"].to_numpy()
    cv = PurgedWalkForward(min_train=2 * ppy, test_size=ppy // 4, label_horizon=2, embargo=5)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"{symbol}: rows={len(df)} folds={cv.n_splits(len(df))}")
    for model in ["mlp", "gru"]:
        probas = {}
        for dev in devices:
            reset_peak()
            t = time.perf_counter()
            p, _, _ = walk_forward_from_splits(
                _inputs(df, model), y, model_factory(model, dev), cv.split(len(df))
            )
            probas[dev] = p
            mem = gpu_memory().get("max_allocated_mb") if dev == "cuda" else "-"
            print(f"  {model:4s} {dev:5s} {time.perf_counter() - t:7.1f}s  peak_vram_mb={mem}")
        if len(probas) == 2:
            m = ~np.isnan(probas["cpu"])
            print(
                f"  {model} max |p_cpu - p_cuda| = {np.abs(probas['cpu'][m] - probas['cuda'][m]).max():.2e}"
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
