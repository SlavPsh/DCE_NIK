# nik_model.py
import math
import numpy as np
import torch
import torch.nn as nn

class FourierFeatures(nn.Module):
    def __init__(self, in_dim, n_freq=64, sigma=6.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2.0 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = w0
        self.is_first = is_first
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class NIK_SIREN(nn.Module):
    def __init__(self, n_coils, k_freq=96, t_freq=16, k_sigma=6.0, t_sigma=3.0,
                 coil_emb=16, hidden=256, depth=7, w0=30.0):
        super().__init__()
        self.ff_k = FourierFeatures(3, n_freq=k_freq, sigma=k_sigma)
        self.ff_t = FourierFeatures(1, n_freq=t_freq, sigma=t_sigma)
        self.coil_emb = nn.Embedding(n_coils, coil_emb)

        in_dim = 2 * k_freq + 2 * t_freq + coil_emb
        layers = [SineLayer(in_dim, hidden, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)  # [log_mag, phase_raw]

    def forward(self, kxyz_t, coil_idx):
        k = kxyz_t[:, :3]
        t = kxyz_t[:, 3:4]
        zk = self.ff_k(k)
        zt = self.ff_t(t)
        ec = self.coil_emb(coil_idx)
        h = torch.cat([zk, zt, ec], dim=-1)
        h = self.backbone(h)
        out = self.head(h)
        log_mag = out[:, 0:1]
        phase  = np.pi * torch.tanh(out[:, 1:2])
        return log_mag, phase


def magphase_to_ri(log_mag, phase):
    mag = torch.exp(log_mag)
    re = mag * torch.cos(phase)
    im = mag * torch.sin(phase)
    return torch.cat([re, im], dim=1)  # (B,2)


class NIK_SIREN_REIM(NIK_SIREN):
    """
    Same as NIK_SIREN,  output is interpreted as (Re, Im) 
    """
    def forward(self, kxyz_t, coil_idx):
        k = kxyz_t[:, :3]
        t = kxyz_t[:, 3:4]
        zk = self.ff_k(k)
        zt = self.ff_t(t)
        ec = self.coil_emb(coil_idx)
        h = torch.cat([zk, zt, ec], dim=-1)
        h = self.backbone(h)
        out = self.head(h)          # (B,2)  -> (Re, Im)
        return out


class ZEncoder(nn.Module):
    """
    Separate z encoding. z_norm is a scalar in [-1,1].
    Modes:
      - "linear": small MLP on z (low bandwidth)
      - "ff": Fourier features on z with small sigma (low bandwidth)
    """
    def __init__(self, mode="linear", z_dim=16, z_freq=8, z_sigma=1.0):
        super().__init__()
        self.mode = mode

        if mode == "linear":
            self.net = nn.Sequential(
                nn.Linear(1, z_dim),
                nn.SiLU(),
                nn.Linear(z_dim, z_dim),
            )
            self.out_dim = z_dim

        elif mode == "ff":
            self.ff = FourierFeatures(1, n_freq=z_freq, sigma=z_sigma)
            self.proj = nn.Linear(2 * z_freq, z_dim)
            self.out_dim = z_dim

        else:
            raise ValueError("ZEncoder mode must be 'linear' or 'ff'")

    def forward(self, z_norm):
        # z_norm: (B,1)
        if self.mode == "linear":
            return self.net(z_norm)
        else:
            return self.proj(self.ff(z_norm))


class NIK_SIREN2D_REIM(nn.Module):
    """
    Input x: (B,4) [kx, ky, z_norm, t_norm]
    Output: (B,2) [Re, Im]
    """
    def __init__(
        self,
        n_coils: int,
        *,
        k_freq=64,
        k_sigma=3.0,
        t_freq=16,
        t_sigma=3.0,
        z_mode="linear",   # "linear" or "ff"
        z_dim=16,
        z_freq=8,
        z_sigma=1.0,
        coil_emb=16,
        hidden=256,
        depth=7,
        w0=30.0,
    ):
        super().__init__()
        self.ff_kxy = FourierFeatures(2, n_freq=k_freq, sigma=k_sigma)
        self.ff_t = FourierFeatures(1, n_freq=t_freq, sigma=t_sigma)
        self.z_enc = ZEncoder(mode=z_mode, z_dim=z_dim, z_freq=z_freq, z_sigma=z_sigma)

        self.coil_emb = nn.Embedding(n_coils, coil_emb)

        in_dim = 2 * k_freq + 2 * t_freq + self.z_enc.out_dim + coil_emb

        layers = [SineLayer(in_dim, hidden, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)  # (Re, Im)

    def forward(self, x, coil_idx):
        # x: (B,4) = [kx, ky, z_norm, t_norm]
        kxy = x[:, 0:2]
        z   = x[:, 2:3]
        t   = x[:, 3:4]

        zk  = self.ff_kxy(kxy)
        zz  = self.z_enc(z)
        zt  = self.ff_t(t)
        ec  = self.coil_emb(coil_idx)

        h = torch.cat([zk, zz, zt, ec], dim=-1)
        h = self.backbone(h)
        return self.head(h)



