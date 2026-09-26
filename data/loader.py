import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

MARKETS = ["NP", "PJM", "BE", "FR", "DE"]


def mcar_mask(length, rate, rng):
    if rate <= 0.0:
        return np.ones(length, dtype=bool)
    if rate >= 1.0:
        return np.zeros(length, dtype=bool)
    return rng.random(length) >= rate


class WindowDataset(Dataset):

    def __init__(self, series, dates, lookback, horizon, stride, rate, seed):
        self.s = torch.as_tensor(np.ascontiguousarray(series), dtype=torch.float32)
        self.dow = pd.DatetimeIndex(dates).dayofweek.values.astype(np.int64)
        self.L, self.H = lookback, horizon
        self.rate, self.seed = rate, seed
        self.starts = np.arange(0, len(self.s) - lookback - horizon + 1, stride,
                                dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = int(self.starts[i])
        w = self.s[t: t + self.L + self.H]
        rng = np.random.default_rng([self.seed, t])
        return {
            "P_look": w[:self.L, 0].clone(),
            "mask": torch.from_numpy(mcar_mask(self.L, self.rate, rng)),
            "X_look": w[:self.L, 1:],
            "X_fut": w[self.L:, 1:],
            "Y": w[self.L:, 0],
            "t": torch.tensor(t),
            "dow": torch.tensor(self.dow[t + self.L]),
        }


def interpolate(P, mask):
    x = P.clone()
    x[~mask] = float("nan")
    df = pd.DataFrame(x.numpy().T).interpolate(limit_direction="both").fillna(0.0)
    return torch.tensor(df.values.T, dtype=torch.float32)


def load_market(market, processed_dir):
    with open(f"{processed_dir}/{market}_series.pkl", "rb") as f:
        return pickle.load(f)


def build_loader(data, split, cfg, stride, rate=0.0, seed=0, shuffle=False):
    ds = WindowDataset(data[split], data[f"dates_{split}"],
                       cfg.window.lookback, cfg.window.horizon, stride, rate, seed)
    n = cfg.dataset.num_workers
    return DataLoader(ds, batch_size=cfg.dataset.batch_size, shuffle=shuffle,
                      num_workers=n, pin_memory=True, persistent_workers=n > 0)


def inverse_price(scaler, p):
    flat = np.asarray(p).reshape(-1)
    tmp = np.zeros((len(flat), 3), dtype=np.float32)
    tmp[:, 0] = flat
    return scaler.inverse_transform(tmp)[:, 0].reshape(np.shape(p))
