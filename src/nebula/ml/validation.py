"""Chronological cross-validation with purging and embargo (positional, sorted data)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedWalkForward:
    """Walk-forward splits over n time-ordered samples.

    - Test blocks of `test_size` follow an initial `min_train` window.
    - `label_horizon` h: sample i's label uses data up to i+h, so training samples within h of the test
      start are purged (their labels overlap the test period).
    - `embargo`: extra samples dropped after the purge (guards against serial correlation).
    - `expanding=True` uses all history; otherwise a rolling window of `min_train`.
    """

    min_train: int
    test_size: int
    label_horizon: int = 1
    embargo: int = 0
    expanding: bool = True

    def split(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        start = self.min_train
        gap = self.label_horizon + self.embargo
        while start < n:
            stop = min(start + self.test_size, n)
            train_end = start - gap  # exclusive
            train_start = 0 if self.expanding else max(0, train_end - self.min_train)
            if train_end - train_start > 0:
                yield np.arange(train_start, train_end), np.arange(start, stop)
            start = stop

    def n_splits(self, n: int) -> int:
        return sum(1 for _ in self.split(n))
