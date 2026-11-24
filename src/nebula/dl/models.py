"""Small PyTorch classifiers with a scikit-learn style interface (fit / predict_proba).

Inputs are 2-D arrays in time order. For the GRU, each row is a flattened window of `seq_len` days x
`n_features` (oldest first), built causally by `nebula.ml.datasets.lagged_matrix`.
Standardization is fit inside `fit` on the training rows only. Early stopping uses the LAST
`val_frac` of the training rows (chronological), never the test rows.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from nebula.dl.device import get_device


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...] = (32, 16), dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, n_features: int, seq_len: int, hidden: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.n_features, self.seq_len = n_features, seq_len
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.view(x.shape[0], self.seq_len, self.n_features)
        _, h = self.gru(seq)
        return self.head(self.drop(h[-1])).squeeze(-1)


@dataclass
class TorchClassifier:
    kind: str = "mlp"  # "mlp" | "gru"
    n_features: int = 13
    seq_len: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 200
    patience: int = 15
    batch_size: int = 256
    val_frac: float = 0.15
    val_gap: int = 2
    seed: int = 0
    device: str | None = None
    history: dict[str, float] = field(default_factory=dict)

    def _build(self, n_in: int) -> nn.Module:
        if self.kind == "mlp":
            return MLP(n_in)
        if self.kind == "gru":
            return GRUNet(self.n_features, self.seq_len)
        raise ValueError(self.kind)

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchClassifier:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)  # noqa: NPY002
        dev = get_device(self.device)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_val = max(1, int(len(X) * self.val_frac))
        # purge `val_gap` rows between the fit part and the validation tail (labels span 2 bars)
        cut = len(X) - n_val
        Xtr, ytr = X[: max(1, cut - self.val_gap)], y[: max(1, cut - self.val_gap)]
        Xva, yva = X[cut:], y[cut:]
        self.mu_ = Xtr.mean(axis=0)
        self.sd_ = Xtr.std(axis=0) + 1e-8
        tr_x = torch.from_numpy((Xtr - self.mu_) / self.sd_).to(dev)
        tr_y = torch.from_numpy(ytr).to(dev)
        va_x = torch.from_numpy((Xva - self.mu_) / self.sd_).to(dev)
        va_y = torch.from_numpy(yva).to(dev)
        model = self._build(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        best, best_state, bad, epochs = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(len(tr_x), generator=gen).to(dev)
            for i in range(0, len(tr_x), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(tr_x[idx]), tr_y[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = float(loss_fn(model(va_x), va_y))
            epochs = epoch + 1
            if v < best - 1e-5:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        model.load_state_dict(best_state)
        self.model_, self.device_ = model, dev
        self.history = {"epochs": epochs, "best_val_loss": best}
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x = torch.from_numpy((np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(x)).cpu().numpy()
        return np.column_stack([1 - p, p])
