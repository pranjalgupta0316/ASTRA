from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from nebula.dl.device import get_device
from nebula.dl.models import TorchClassifier
from nebula.features.technical import FEATURE_COLUMNS, build_features
from nebula.ml.classical import walk_forward_from_splits
from nebula.ml.datasets import date_folds, lag_columns, lagged_matrix, model_frame


def test_device_cpu_fallback_forced():
    assert get_device("cpu").type == "cpu"


@pytest.mark.parametrize("kind", ["mlp", "gru"])
def test_torch_classifier_learns_separable_signal(kind):
    rng = np.random.default_rng(0)
    n, f, L = 1500, 3, 4
    x = rng.normal(size=(n, f))
    y = (x[:, 0] > 0).astype(float)
    X = np.repeat(x, L, axis=1).reshape(n, L * f) if kind == "gru" else x  # last step carries signal
    clf = TorchClassifier(kind=kind, n_features=f, seq_len=L, max_epochs=60, device="cpu").fit(
        X[:1000], y[:1000]
    )
    p = clf.predict_proba(X[1000:])[:, 1]
    assert ((p > 0.5) == y[1000:]).mean() > 0.9


def test_torch_classifier_deterministic():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(400, 5)), (rng.random(400) > 0.5).astype(float)
    a = TorchClassifier(seed=3, max_epochs=5, device="cpu").fit(X, y).predict_proba(X)
    b = TorchClassifier(seed=3, max_epochs=5, device="cpu").fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_torch_classifier_runs_on_cuda():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(300, 5)), (rng.random(300) > 0.5).astype(float)
    clf = TorchClassifier(max_epochs=3, device="cuda").fit(X, y)
    assert clf.device_.type == "cuda" and clf.predict_proba(X).shape == (300, 2)


def test_lagged_matrix_is_causal(us_bars):
    f = build_features(us_bars)
    lag = lagged_matrix(f, FEATURE_COLUMNS, 5)
    # row t, newest block equals features at t; oldest block equals features at t-4
    t = 150
    np.testing.assert_allclose(
        lag.loc[t, [f"{c}_lag0" for c in FEATURE_COLUMNS]].to_numpy(dtype=float),
        f.loc[t, FEATURE_COLUMNS].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        lag.loc[t, [f"{c}_lag4" for c in FEATURE_COLUMNS]].to_numpy(dtype=float),
        f.loc[t - 4, FEATURE_COLUMNS].to_numpy(dtype=float),
    )
    assert lag_columns(5)[: len(FEATURE_COLUMNS)] == [f"{c}_lag4" for c in FEATURE_COLUMNS]  # oldest first


def test_model_frame_has_no_nans_and_targets(us_bars):
    f = build_features(us_bars)
    f["symbol"] = "TEST"
    df = model_frame(f, 10)
    assert not df.isna().any().any() and set(df["target_up"].unique()) <= {0.0, 1.0}


def test_date_folds_purge_gap_and_no_overlap():
    dates = pd.Series(np.repeat(pd.date_range("2020-01-01", periods=100, freq="D"), 3))
    folds = list(date_folds(dates, min_train_dates=40, test_dates=20, gap_dates=7))
    assert len(folds) == 3
    for tr, te in folds:
        last_train, first_test = dates.iloc[tr].max(), dates.iloc[te].min()
        assert (first_test - last_train).days >= 8  # 7 purged dates in between
        assert len(set(tr) & set(te)) == 0
        # every row of a test date is in the same fold
        assert dates.iloc[te].value_counts().eq(3).all()


def test_walk_forward_from_splits_baseline_is_ex_ante():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    X = np.zeros((8, 1))
    splits = [(np.arange(0, 4), np.arange(4, 8))]

    class Const:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

    _, base, _ = walk_forward_from_splits(X, y, Const, splits)
    assert np.all(base[4:] == 1.0)  # train-fold base rate, not the test-period rate (0.0)
