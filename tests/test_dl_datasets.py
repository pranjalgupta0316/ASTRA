from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # optional `dl` extra; CI runs without it

from nebula.dl.device import get_device
from nebula.dl.models import TorchClassifier


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
