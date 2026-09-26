import torch
import torch.nn as nn

DAY_SLICES = {1: slice(144, 168), 2: slice(120, 144),
              3: slice(96, 120), 7: slice(0, 24)}
PRICE_LAGS = (1, 2, 3, 7)
EXOG_LAGS = (0, 1, 7)


class Forecaster(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lookback = cfg.window.lookback
        self.horizon = cfg.window.horizon
        self.use_lookback = cfg.model.get("use_lookback", True)

    def loss(self, batch, device):
        pred = self.forward_step(batch, device)
        return ((pred - batch["Y"].to(device)) ** 2).mean()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        return self


def day_ahead_features(P, X_look, X_fut, dow, use_lookback):
    B = P.shape[0]
    feats = []
    if use_lookback:
        feats += [P[:, DAY_SLICES[lag]] for lag in PRICE_LAGS]
    for lag in EXOG_LAGS:
        if lag == 0:
            feats.append(X_fut.reshape(B, -1))
        elif use_lookback:
            feats.append(X_look[:, DAY_SLICES[lag]].reshape(B, -1))
    dummies = torch.zeros(B, 7, dtype=P.dtype, device=P.device)
    dummies[torch.arange(B, device=P.device), dow] = 1.0
    feats.append(dummies)
    return torch.cat(feats, dim=-1)


def n_day_ahead_features(exog_dim, use_lookback):
    n = 24 * exog_dim + 7
    if use_lookback:
        n += 24 * len(PRICE_LAGS) + 24 * exog_dim * (len(EXOG_LAGS) - 1)
    return n
