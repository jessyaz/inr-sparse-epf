import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Forecaster

FREQ_MIN = 0.875
FREQ_MAX = 84.0


class FourierFeatures(nn.Module):

    def __init__(self, num_frequencies):
        super().__init__()
        freqs = torch.logspace(torch.log2(torch.tensor(FREQ_MIN)),
                               torch.log2(torch.tensor(FREQ_MAX)),
                               num_frequencies, base=2.0)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        angles = t.unsqueeze(-1) * self.freqs * 2.0
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class INRNetwork(nn.Module):

    def __init__(self, num_frequencies, hidden_dim, num_layers):
        super().__init__()
        self.fourier = FourierFeatures(num_frequencies)
        dims = [2 * num_frequencies] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                     for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, t, film):
        gamma, beta = film
        x = self.fourier(t)
        for i, layer in enumerate(self.layers):
            x = F.gelu(self.norms[i](gamma[:, i] * layer(x) + beta[:, i]))
        return self.output_layer(x)


class FiLMGenerator(nn.Module):

    def __init__(self, z_dim, hidden_dim, num_layers, layer_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2 * layer_dim)
                                    for _ in range(num_layers)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.bias[:layer_dim] = 1.0

    def forward(self, z):
        h = self.shared(z)
        out = torch.stack([head(h) for head in self.heads], dim=1)
        return out.chunk(2, dim=-1)


class DeepSetsEncoder(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(2 * hidden_dim)

    def forward(self, elements, mask):
        e = self.phi(elements)
        m = mask.unsqueeze(-1).to(e.dtype)
        n_obs = m.sum(dim=1)
        mean_pool = (e * m).sum(dim=1) / n_obs.clamp(min=1.0)
        max_pool = e.masked_fill(m == 0, float("-inf")).max(dim=1).values
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)
        z = torch.cat([mean_pool, max_pool], dim=-1)
        return torch.where(n_obs > 0, self.norm(z), z)


class INR(Forecaster):

    def __init__(self, cfg):
        super().__init__(cfg)
        c = cfg.model
        self.use_exog = c.use_exog
        self.static_exog = c.static_exog

        self.deepsets_encoder = DeepSetsEncoder(c.set_dim)
        z_dim = 2 * c.set_dim if self.use_lookback else 0
        if self.use_exog:
            self.lstm_past = nn.LSTM(c.exog_dim, c.lstm_dim, batch_first=True)
            self.lstm_future = nn.LSTM(c.exog_dim, c.lstm_dim, batch_first=True)
            z_dim += c.lstm_dim

        self.inr = INRNetwork(c.num_frequencies, c.hidden_dim, c.num_layers)
        self.film_generator = FiLMGenerator(z_dim, c.film_dim, c.num_layers, c.hidden_dim)

        t = torch.arange(-self.lookback, self.horizon, dtype=torch.float32) / self.lookback * torch.pi
        self.register_buffer("t_past", t[:self.lookback], persistent=False)
        self.register_buffer("t_future", t[self.lookback:], persistent=False)

    def forward_step(self, batch, device):
        P = batch["P_look"].to(device)
        mask = batch["mask"].to(device)
        B, H = P.shape[0], self.horizon

        parts = []
        if self.use_lookback:
            elems = torch.stack([self.t_past.unsqueeze(0).expand(B, -1), P], dim=-1)
            parts.append(self.deepsets_encoder(elems, mask).unsqueeze(1).expand(-1, H, -1))
        if self.use_exog:
            _, state = self.lstm_past(batch["X_look"].to(device))
            h_fut, _ = self.lstm_future(batch["X_fut"].to(device), state)
            if self.static_exog:
                h_fut = h_fut[:, -1:].expand(-1, H, -1)
            parts.append(h_fut)
        z = torch.cat(parts, dim=-1)

        films = self.film_generator(z.reshape(B * H, -1))
        return self.inr(self.t_future.repeat(B), films).reshape(B, H)
