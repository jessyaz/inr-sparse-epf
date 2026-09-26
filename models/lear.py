import warnings

import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV, LassoLarsIC
from tqdm import tqdm

from data.loader import interpolate
from models.base import Forecaster, day_ahead_features


def l2_normalize(X, stats=None):
    if stats is None:
        mean = X.mean(axis=0)
        norm = np.linalg.norm(X - mean, axis=0)
        norm[norm == 0] = 1.0
        stats = (mean, norm)
    mean, norm = stats
    return (X - mean) / norm, stats


def fit_lasso(X, Y, max_iter):
    models = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for h in range(Y.shape[1]):
            if X.shape[0] > X.shape[1]:
                alpha = LassoLarsIC(criterion="bic", max_iter=max_iter).fit(X, Y[:, h]).alpha_
            else:
                alpha = LassoCV(cv=5, max_iter=max_iter, n_jobs=-1).fit(X, Y[:, h]).alpha_
            models.append(Lasso(alpha=alpha, max_iter=max_iter).fit(X, Y[:, h]))
    return models


class LEAR(Forecaster):

    def __init__(self, cfg):
        super().__init__(cfg)
        c = cfg.model
        self.max_iter = c.max_iter
        self.recalib_every = c.recalib_every
        self.calibration_windows = list(c.calibration_windows)
        self.starts = None
        self.recal = None

    def _features(self, batch):
        P = interpolate(batch["P_look"].cpu(), batch["mask"].cpu())
        return day_ahead_features(P, batch["X_look"].cpu(), batch["X_fut"].cpu(),
                                  batch["dow"].cpu(), self.use_lookback).numpy()

    def _collect(self, loader):
        X, Y, T = [], [], []
        for batch in loader:
            X.append(self._features(batch))
            Y.append(batch["Y"].numpy())
            T.append(batch["t"].numpy())
        X, Y, T = np.concatenate(X), np.concatenate(Y), np.concatenate(T)
        order = np.argsort(T)
        return X[order], Y[order], T[order]

    def _calibrate(self, X, Y, window):
        if window > 0:
            X, Y = X[-window:], Y[-window:]
        Xn, stats = l2_normalize(X)
        return fit_lasso(Xn, Y, self.max_iter), stats

    def fit(self, train_loader, val_loader, test_loader):
        X, Y, T = self._collect(train_loader)
        Xv, Yv, _ = self._collect(val_loader)
        Xte, Yte, _ = self._collect(test_loader)
        n_hist = len(X) + len(Xv)
        X_all = np.concatenate([X, Xv, Xte])
        Y_all = np.concatenate([Y, Yv, Yte])

        self.starts = np.arange(0, len(Xte), self.recalib_every)
        self.recal = {}
        with tqdm(total=len(self.starts) * len(self.calibration_windows),
                  desc="LEAR recalibration") as bar:
            for s in self.starts:
                end = n_hist + s
                self.recal[int(s)] = []
                for w in self.calibration_windows:
                    self.recal[int(s)].append(self._calibrate(X_all[:end], Y_all[:end], w))
                    bar.update(1)

    def forward_step(self, batch, device):
        X = self._features(batch)
        days = batch["t"].numpy() // 24
        starts = self.starts[np.searchsorted(self.starts, days, side="right") - 1]

        acc = np.zeros((X.shape[0], self.horizon), dtype=np.float64)
        for s in np.unique(starts):
            sel = starts == s
            for models, stats in self.recal[int(s)]:
                xn, _ = l2_normalize(X[sel], stats)
                for h in range(self.horizon):
                    acc[sel, h] += models[h].predict(xn)
        acc /= len(self.calibration_windows)
        return torch.as_tensor(acc, dtype=torch.float32, device=device)

    def save(self, path):
        joblib.dump({"starts": self.starts, "recal": self.recal}, path)

    def load(self, path):
        d = joblib.load(path)
        self.starts, self.recal = d["starts"], d["recal"]
        return self
