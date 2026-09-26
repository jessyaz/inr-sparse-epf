import math

import torch
import torch.nn as nn

from data.loader import interpolate
from models.base import Forecaster


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[: x.size(1)].unsqueeze(0))


class DailyElectricTransformer(nn.Module):

    def __init__(self, embedding_dim, num_heads, dim_feedforward, num_layers,
                 dropout, exog_dim):
        super().__init__()
        self.values_embeddings = nn.Sequential(nn.Linear(24, embedding_dim), nn.ReLU())
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward,
                                           dropout, batch_first=True, norm_first=False,
                                           activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers)
        self.features_embeddings = nn.Sequential(
            nn.Linear(24 * exog_dim, embedding_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 24),
        )

    def forward(self, values, features):
        B = values.size(0)
        enc = self.transformer_encoder(
            self.positional_encoding(self.values_embeddings(values.reshape(B, -1, 24))))
        feat = self.features_embeddings(features.reshape(B, enc.size(1), -1))
        return self.head(torch.cat([feat, enc], dim=2))


class EPFTransformer(Forecaster):

    def __init__(self, cfg):
        super().__init__(cfg)
        c = cfg.model
        self.net = DailyElectricTransformer(c.embedding_dim, c.num_heads,
                                            c.dim_feedforward, c.num_layers,
                                            c.dropout, c.exog_dim)

    def _inputs(self, batch, device):
        P = interpolate(batch["P_look"].cpu(), batch["mask"].cpu()).to(device)
        if not self.use_lookback:
            P = torch.zeros_like(P)
        X = torch.cat([batch["X_look"].to(device), batch["X_fut"].to(device)],
                      dim=1)[:, self.horizon:]
        return P, X

    def loss(self, batch, device):
        values, features = self._inputs(batch, device)
        full = torch.cat([batch["P_look"].to(device), batch["Y"].to(device)], dim=1)
        targets = full[:, self.horizon:].reshape(values.size(0), -1, self.horizon)
        return ((self.net(values, features) - targets) ** 2).mean()

    def forward_step(self, batch, device):
        values, features = self._inputs(batch, device)
        return self.net(values, features)[:, -1]
