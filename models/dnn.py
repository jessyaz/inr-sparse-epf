import torch.nn as nn

from data.loader import interpolate
from models.base import Forecaster, day_ahead_features, n_day_ahead_features


class DNN(Forecaster):

    def __init__(self, cfg):
        super().__init__(cfg)
        c = cfg.model
        n_in = n_day_ahead_features(c.exog_dim, self.use_lookback)
        self.net = nn.Sequential(
            nn.Linear(n_in, c.hidden_1), nn.GELU(), nn.Dropout(c.dropout),
            nn.Linear(c.hidden_1, c.hidden_2), nn.GELU(), nn.Dropout(c.dropout),
            nn.Linear(c.hidden_2, self.horizon),
        )

    def forward_step(self, batch, device):
        P = interpolate(batch["P_look"].cpu(), batch["mask"].cpu()).to(device)
        x = day_ahead_features(P, batch["X_look"].to(device),
                               batch["X_fut"].to(device),
                               batch["dow"].to(device), self.use_lookback)
        return self.net(x)
