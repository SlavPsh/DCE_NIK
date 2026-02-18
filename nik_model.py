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
    def __init__(self, in_features, out_features, w0=15.0, is_first=False):
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
            self.linear.bias.zero_()

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
    Same as NIK_SIREN,  output is (Re, Im) 
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
      - "linear":  MLP on z 
      - "ff": Fourier features on z 
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
    


class NIK_SIREN_KXY_FF_REIM(nn.Module):
    """
    Input:  x (B,2) = [kx, ky]
    Output: (B,2)   = [Re, Im]
    """
    def __init__(
        self,
        *,
        x_dim = 2,
        k_freq=32,
        k_sigma=3.0,
        hidden=256,
        depth=7,
        w0=15,
    ):
        super().__init__()
        self.ff_kxy = FourierFeatures(x_dim, n_freq=k_freq, sigma=k_sigma)

        in_dim = 2 * k_freq
        layers = [SineLayer(in_dim, hidden, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)  # (Re, Im)

    def forward(self, x):
        # x: (B,2) [kx, ky]
        zk = self.ff_kxy(x)
        h = self.backbone(zk)
        return self.head(h)  # (B,2) [Re, Im]


class NIK_SIREN_KXY_REIM(nn.Module):
    """
    Input:  x (B,2) = [kx, ky]
    Output: (B,2)   = [Re, Im]
    """
    def __init__(
        self,
        *,
        in_dim = 2,
        hidden=256,
        depth=7,
        w0=15,
    ):
        super().__init__()

        layers = [SineLayer(in_dim, hidden, w0=4*w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)  # (Re, Im)

    def forward(self, x):
        # x: (B,2) [kx, ky]
        h = self.backbone(x)
        return self.head(h)  # (B,2) [Re, Im]



def cart_to_s_sincos_foldpi(x_kxy: torch.Tensor, eps: float = 1e-12):
    """
    x_kxy: (B,2) [kx, ky]

    Returns:
      x_ssc: (B,3) [s, sin(theta0), cos(theta0)]
    where:
      theta0 is the spoke direction folded to (-pi/2, pi/2] (mod pi),
      s is signed coordinate along that direction (negative -> positive across center).
    """
    kx = x_kxy[:, 0]
    ky = x_kxy[:, 1]

    theta = torch.atan2(ky, kx)  # (-pi, pi]

    # fold angle modulo pi to represent spoke direction, not ray direction
    # map to (-pi/2, pi/2]
    theta0 = torch.remainder(theta + 0.5 * np.pi, np.pi) - 0.5 * np.pi

    c = torch.cos(theta0)
    sng = torch.sin(theta0)

    # signed coordinate along the spoke direction
    s_coord = kx * c + ky * sng  # can be negative

    x_ssc = torch.stack([s_coord, sng, c], dim=-1)
    return x_ssc

def normalize_s(x_ssc: torch.Tensor, s_max: float = None, eps: float = 1e-12):
    s = x_ssc[:, 0]
    if s_max is None:
        s_max = s.abs().max().clamp_min(eps)
    s = s / s_max
    return torch.stack([s, x_ssc[:, 1], x_ssc[:, 2]], dim=-1), s_max


class SignedSpokeAdapter:
    """
    Converts (kx, ky) -> (s, sin(theta0), cos(theta0))

    where:
        theta0 = spoke direction modulo pi
        s      = signed coordinate along that spoke direction

    s is normalized by s_max (max |s| seen during training).
    """

    def __init__(self, s_max: float, eps: float = 1e-12):
        self.s_max = float(s_max)
        self.eps = eps

    def __call__(self, x_kxy: torch.Tensor) -> torch.Tensor:
        kx = x_kxy[:, 0]
        ky = x_kxy[:, 1]

        # full angle (-pi, pi]
        theta = torch.atan2(ky, kx)

        # fold to spoke direction modulo pi
        # map to (-pi/2, pi/2]
        theta0 = torch.remainder(theta + 0.5 * np.pi, np.pi) - 0.5 * np.pi

        c = torch.cos(theta0)
        sng = torch.sin(theta0)

        # signed coordinate along spoke direction
        s_coord = kx * c + ky * sng

        # normalize signed coordinate
        s_coord = s_coord / max(self.s_max, self.eps)

        return torch.stack([s_coord, sng, c], dim=-1)

def compute_theta0_per_spoke(x_all_kxy: torch.Tensor, spoke_id_all: torch.Tensor):
    # returns dict: spoke_id -> theta0 (float)
    theta0 = {}
    for sp in torch.unique(spoke_id_all).tolist():
        m = (spoke_id_all == sp)
        xk = x_all_kxy[m][:, :2]
        r = torch.sqrt((xk**2).sum(dim=1))
        j = torch.argmax(r)
        th = torch.atan2(xk[j,1], xk[j,0])
        th0 = torch.remainder(th + 0.5*np.pi, np.pi) - 0.5*np.pi
        theta0[int(sp)] = float(th0)
    return theta0

def loss_function(y_pred, y):
    
    res = y_pred - y
    mag = torch.sqrt(y[:,0]**2 + y[:,1]**2 + 1e-12)
    w = 1.0 / (mag + 0.1)
    loss = (w * (res[:,0]**2 + res[:,1]**2)).mean()

    return loss
