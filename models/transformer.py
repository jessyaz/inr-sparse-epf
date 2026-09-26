import math

import torch
import torch.nn as nn

from data.loader import interpolate
from models.base import Forecaster


class SinusoidalPE(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class Transformer(Forecaster):

    def __init__(self, cfg):
        super().__init__(cfg)
        c = cfg.model
        self.masked_attention = c.masked_attention
        d = c.embedding_dim

        self.value_embed = nn.Linear(2, d)
        self.pos_enc = SinusoidalPE(d, max_len=self.lookback + 1)
        layer = nn.TransformerEncoderLayer(d, c.num_heads, c.dim_feedforward, c.dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, c.num_layers)
        self.exog_embed = nn.Sequential(nn.Linear(self.horizon * c.exog_dim, d), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, c.dim_feedforward),
            nn.Dropout(c.dropout),
            nn.GELU(),
            nn.Linear(c.dim_feedforward, self.horizon),
        )
        self.register_buffer(
            "t", torch.arange(self.lookback, dtype=torch.float32) / self.lookback * 2.0 - 1.0,
            persistent=False)

    def _encode(self, P, mask):
        t = self.t.unsqueeze(0).expand(P.shape[0], -1)
        x = self.pos_enc(self.value_embed(torch.stack([P, t], dim=-1)))

        if not self.masked_attention:
            return self.encoder(x).mean(dim=1)

        pad = ~mask
        empty = pad.all(dim=1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
        enc = self.encoder(x, src_key_padding_mask=pad)

        m = mask.unsqueeze(-1).to(enc.dtype)
        z = (enc * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return torch.where(mask.any(dim=1, keepdim=True), z, torch.zeros_like(z))

    def forward_step(self, batch, device):
        mask = batch["mask"].to(device)
        X_fut = batch["X_fut"].to(device)
        B = X_fut.shape[0]

        if self.masked_attention:
            P = batch["P_look"].to(device)
        else:
            P = interpolate(batch["P_look"].cpu(), batch["mask"].cpu()).to(device)

        if self.use_lookback:
            z = self._encode(P, mask)
        else:
            z = torch.zeros(B, self.value_embed.out_features, device=device, dtype=P.dtype)

        e = self.exog_embed(X_fut.reshape(B, -1))
        return self.head(torch.cat([e, z], dim=-1))
