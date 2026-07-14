"""nik models, siren wire ff gp polar"""
import math
import numpy as np
import torch
import torch.nn as nn

class FourierFeatures(nn.Module):
    def __init__(self, in_dim, n_freq=64, sigma=6.0, seed=None):
        super().__init__()
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
            B = torch.randn(in_dim, n_freq, generator=gen) * sigma
        else:
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


class NIKMRISineLayer(nn.Module):
    """nikmri sine layer"""

    def __init__(self, in_features, out_features, *, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1.0 / self.in_features,
                    1.0 / self.in_features,
                )
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class GaborLayer(nn.Module):
    """gabor wavelet, wire activation"""
    def __init__(self, in_features, out_features, w0=20.0, s0=10.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = w0
        self.s0 = s0
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
        h = self.linear(x)
        gauss = torch.exp(-0.5 * (self.s0 * h) ** 2)
        # complex gabor reim
        real = gauss * torch.cos(self.w0 * h)
        imag = gauss * torch.sin(self.w0 * h)
        return torch.cat([real, imag], dim=-1)


class WIRE_KXY_REIM(nn.Module):
    """wire kspace, reim"""
    def __init__(
        self,
        *,
        in_dim=2,
        hidden=64,
        depth=8,
        w0=20.0,
        s0=10.0,
        out_dim=2,
        dropout=0.0,
    ):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        super().__init__()
        self.out_dim = out_dim
        self.dropout = float(dropout)

        layers = [GaborLayer(in_dim, hidden, w0=w0, s0=s0, is_first=True)]
        for _ in range(depth - 2):
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(GaborLayer(2 * hidden, hidden, w0=w0, s0=s0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(2 * hidden, out_dim)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)


class WIRE_KXY_COIL_T_REIM(nn.Module):
    """wire kspace with coil embedding and time, reim output

    forward signature:
        kcoords:  (N, 2)        kx, ky in radial-scaled units
        t:        (N,)          normalized time in [-1, 1]
        coil_idx: (N,) long     coil index in [0, n_coils)

    returns:  (N, 2)  Re/Im of predicted single-coil k-space value
    """
    def __init__(
        self,
        *,
        n_coils: int,
        coil_embed_dim: int = 8,
        hidden: int = 64,
        depth: int = 8,
        w0: float = 20.0,
        s0: float = 10.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_coils < 1:
            raise ValueError(f"n_coils must be >= 1, got {n_coils}")
        self.n_coils = int(n_coils)
        self.coil_embed_dim = int(coil_embed_dim)

        self.coil_embed = nn.Embedding(self.n_coils, self.coil_embed_dim)
        nn.init.uniform_(self.coil_embed.weight, -1.0, 1.0)

        in_dim = 2 + 1 + self.coil_embed_dim   # kx, ky, t, coil_embed
        self.backbone_model = WIRE_KXY_REIM(
            in_dim=in_dim, hidden=hidden, depth=depth,
            w0=w0, s0=s0, out_dim=2, dropout=dropout,
        )

    def forward(self, kcoords: torch.Tensor, t: torch.Tensor, coil_idx: torch.Tensor) -> torch.Tensor:
        if kcoords.ndim != 2 or kcoords.shape[-1] != 2:
            raise ValueError(f"kcoords must be (N, 2), got {tuple(kcoords.shape)}")
        N = kcoords.shape[0]
        t = t.reshape(N, 1).to(kcoords.dtype)
        ce = self.coil_embed(coil_idx.long())          # (N, coil_embed_dim)
        x = torch.cat([kcoords, t, ce], dim=-1)        # (N, 2 + 1 + coil_embed_dim)
        return self.backbone_model(x)


class WIRE_KXYZ_COIL_T_REIM(nn.Module):
    """wire kspace with kz coordinate, coil embedding and time, reim output

    3D stack-of-stars variant of WIRE_KXY_COIL_T_REIM. takes kz as a network input
    (no separable kz to z ifft beforehand), so it fits rotated stacks where the
    in-plane angle depends on the partition.

    forward signature:
        kcoords:  (N, 3)        kx, ky, kz in normalized units (each in [-0.5, 0.5])
        t:        (N,)          normalized time (pipeline uses [-1, 1])
        coil_idx: (N,) long     coil index in [0, n_coils)

    returns:  (N, 2)  Re/Im of predicted single-coil k-space value
    """
    def __init__(
        self,
        *,
        n_coils: int,
        coil_embed_dim: int = 8,
        hidden: int = 64,
        depth: int = 8,
        w0: float = 20.0,
        s0: float = 10.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_coils < 1:
            raise ValueError(f"n_coils must be >= 1, got {n_coils}")
        self.n_coils = int(n_coils)
        self.coil_embed_dim = int(coil_embed_dim)

        self.coil_embed = nn.Embedding(self.n_coils, self.coil_embed_dim)
        nn.init.uniform_(self.coil_embed.weight, -1.0, 1.0)

        in_dim = 3 + 1 + self.coil_embed_dim   # kx, ky, kz, t, coil_embed
        self.backbone_model = WIRE_KXY_REIM(
            in_dim=in_dim, hidden=hidden, depth=depth,
            w0=w0, s0=s0, out_dim=2, dropout=dropout,
        )

    def forward(self, kcoords: torch.Tensor, t: torch.Tensor, coil_idx: torch.Tensor) -> torch.Tensor:
        if kcoords.ndim != 2 or kcoords.shape[-1] != 3:
            raise ValueError(f"kcoords must be (N, 3), got {tuple(kcoords.shape)}")
        N = kcoords.shape[0]
        t = t.reshape(N, 1).to(kcoords.dtype)
        ce = self.coil_embed(coil_idx.long())          # (N, coil_embed_dim)
        x = torch.cat([kcoords, t, ce], dim=-1)        # (N, 3 + 1 + coil_embed_dim)
        return self.backbone_model(x)


class NIK_SIREN(nn.Module):
    def __init__(self, n_coils, k_freq=96, t_freq=16, k_sigma=6.0, t_sigma=3.0,
                 coil_emb=16, hidden=256, depth=7, w0=30.0):
        super().__init__()
        self.ff_k = FourierFeatures(3, n_freq=k_freq, sigma=k_sigma)
        self.ff_t = FourierFeatures(1, n_freq=t_freq, sigma=t_sigma)
        self.coil_emb = nn.Embedding(n_coils, coil_emb)

        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        in_dim = 2 * k_freq + 2 * t_freq + coil_emb
        layers = [SineLayer(in_dim, hidden, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)

    def _encode(self, kxyz_t, coil_idx):
        k = kxyz_t[:, :3]
        t = kxyz_t[:, 3:4]
        zk = self.ff_k(k)
        zt = self.ff_t(t)
        ec = self.coil_emb(coil_idx)
        h = torch.cat([zk, zt, ec], dim=-1)
        return self.backbone(h)

    def forward(self, kxyz_t, coil_idx):
        out = self.head(self._encode(kxyz_t, coil_idx))
        log_mag = out[:, 0:1]
        phase  = np.pi * torch.tanh(out[:, 1:2])
        return log_mag, phase


def magphase_to_ri(log_mag, phase):
    mag = torch.exp(log_mag)
    re = mag * torch.cos(phase)
    im = mag * torch.sin(phase)
    return torch.cat([re, im], dim=-1)


class NIK_SIREN_REIM(NIK_SIREN):
    """siren reim, incompatible checkpoints"""
    def forward(self, kxyz_t, coil_idx):
        return self.head(self._encode(kxyz_t, coil_idx))


class ZEncoder(nn.Module):
    """z encoder, linear or ff"""
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
        if self.mode == "linear":
            return self.net(z_norm)
        else:
            return self.proj(self.ff(z_norm))


class NIK_SIREN2D_REIM(nn.Module):
    """siren 2d, kxy z t"""
    def __init__(
        self,
        n_coils: int,
        *,
        k_freq=64,
        k_sigma=3.0,
        t_freq=16,
        t_sigma=3.0,
        z_mode="linear",
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

        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        in_dim = 2 * k_freq + 2 * t_freq + self.z_enc.out_dim + coil_emb

        layers = [SineLayer(in_dim, hidden, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, coil_idx):
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
    


class NIK_MRI_SIREN_REIM(nn.Module):
    """nikmri siren, t coil kxy"""

    def __init__(
        self,
        *,
        coord_dim=4,
        feature_dim=512,
        num_layers=8,
        out_dim=1,
        omega_0=30.0,
        ff_scale=1.0,
        ff_seed=None,
        dropout=0.0,
    ):
        super().__init__()

        if feature_dim % 2 != 0:
            raise ValueError(f"feature_dim must be even, got {feature_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        if ff_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(ff_seed)
            B = torch.randn(coord_dim, feature_dim // 2, generator=gen) * ff_scale
        else:
            B = torch.randn(coord_dim, feature_dim // 2) * ff_scale
        self.register_buffer("B", B)

        self.out_dim = out_dim
        self.dropout = float(dropout)
        layers = [
            NIKMRISineLayer(
                feature_dim,
                feature_dim,
                omega_0=omega_0,
                is_first=True,
            )
        ]
        for _ in range(num_layers - 1):
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(
                NIKMRISineLayer(
                    feature_dim,
                    feature_dim,
                    omega_0=omega_0,
                    is_first=False,
                )
            )
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(feature_dim, out_dim * 2)
        with torch.no_grad():
            bound = np.sqrt(6.0 / feature_dim) / omega_0
            self.head.weight.uniform_(-bound, bound)

    def encode(self, coords):
        proj = coords @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, coords):
        features = self.encode(coords)
        hidden = self.backbone(features)
        output = self.head(hidden)
        return torch.complex(
            output[..., :self.out_dim],
            output[..., self.out_dim:],
        )


class NIK_SIREN_KXY_FF_REIM(nn.Module):
    """siren ff, kxy reim"""
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

        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        in_dim = 2 * k_freq
        layers = [SineLayer(in_dim, hidden, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        zk = self.ff_kxy(x)
        h = self.backbone(zk)
        return self.head(h)


class NIK_SIREN_KXY_REIM(nn.Module):
    """siren kxy reim"""
    def __init__(
        self,
        *,
        in_dim = 2,
        hidden=256,
        depth=7,
        w0=15,
    ):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        super().__init__()

        layers = [SineLayer(in_dim, hidden, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)


class ReLU_MLP_KXY_REIM(nn.Module):
    """relu mlp baseline"""
    def __init__(
        self,
        *,
        in_dim=2,
        hidden=64,
        depth=8,
    ):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU(inplace=True))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)


class ELU_MLP_KXY_REIM(nn.Module):
    """elu mlp baseline"""
    def __init__(
        self,
        *,
        in_dim=2,
        hidden=64,
        depth=8,
    ):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ELU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ELU(inplace=True))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # leaky relu kaiming
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)


class FF_ReLU_MLP_KXY_REIM(nn.Module):
    """ff relu mlp"""
    def __init__(
        self,
        *,
        in_dim=2,
        k_freq=64,
        k_sigma=6.0,
        hidden=256,
        depth=7,
        ff_seed=None,
    ):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        super().__init__()

        self.ff = FourierFeatures(in_dim, n_freq=k_freq, sigma=k_sigma, seed=ff_seed)

        ff_out_dim = 2 * k_freq
        layers = []
        layers.append(nn.Linear(ff_out_dim, hidden))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU(inplace=True))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.ff(x)
        h = self.backbone(h)
        return self.head(h)


class FF_ELU_MLP_KXY_REIM(nn.Module):
    """ff elu mlp"""
    def __init__(
        self,
        *,
        in_dim=2,
        k_freq=64,
        k_sigma=6.0,
        hidden=256,
        depth=7,
        ff_seed=None,
    ):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        super().__init__()

        self.ff = FourierFeatures(in_dim, n_freq=k_freq, sigma=k_sigma, seed=ff_seed)

        ff_out_dim = 2 * k_freq
        layers = []
        layers.append(nn.Linear(ff_out_dim, hidden))
        layers.append(nn.ELU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ELU(inplace=True))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.ff(x)
        h = self.backbone(h)
        return self.head(h)


def cart_to_s_sincos_foldpi(x_kxy: torch.Tensor):
    """signed s, folded angle"""
    kx = x_kxy[:, 0]
    ky = x_kxy[:, 1]

    theta = torch.atan2(ky, kx)

    # fold modulo pi
    theta0 = torch.remainder(theta + 0.5 * np.pi, np.pi) - 0.5 * np.pi

    c = torch.cos(theta0)
    sin_theta0 = torch.sin(theta0)

    # signed spoke s
    s_coord = kx * c + ky * sin_theta0

    x_ssc = torch.stack([s_coord, sin_theta0, c], dim=-1)
    return x_ssc

def normalize_s(x_ssc: torch.Tensor, s_max: float = None, eps: float = 1e-12):
    s = x_ssc[:, 0]
    if s_max is None:
        s_max = s.abs().max().clamp_min(eps)
    s = s / s_max
    return torch.stack([s, x_ssc[:, 1], x_ssc[:, 2]], dim=-1), s_max


class SignedSpokeAdapter:
    """kxy to signed spoke"""

    def __init__(self, s_max: float, eps: float = 1e-12):
        self.s_max = float(s_max)
        self.eps = eps

    def __call__(self, x_kxy: torch.Tensor) -> torch.Tensor:
        kx = x_kxy[:, 0]
        ky = x_kxy[:, 1]

        # full angle
        theta = torch.atan2(ky, kx)

        # spoke fold
        theta0 = torch.remainder(theta + 0.5 * np.pi, np.pi) - 0.5 * np.pi

        c = torch.cos(theta0)
        sin_theta0 = torch.sin(theta0)

        # signed s
        s_coord = kx * c + ky * sin_theta0

        # normalized s
        s_coord = s_coord / max(self.s_max, self.eps)

        return torch.stack([s_coord, sin_theta0, c], dim=-1)

def compute_theta0_per_spoke(x_all_kxy: torch.Tensor, spoke_id_all: torch.Tensor):
    # spoke theta0 dict
    theta0 = {}
    for sp in torch.unique(spoke_id_all).tolist():
        m = (spoke_id_all == sp)
        xk = x_all_kxy[m][:, :2]
        r = torch.sqrt((xk**2).sum(dim=1))
        r_max = r.max()
        if r_max == 0:
            # dc only spoke
            theta0[int(sp)] = 0.0
            continue
        j = torch.argmax(r)
        th = torch.atan2(xk[j,1], xk[j,0])
        th0 = torch.remainder(th + 0.5*np.pi, np.pi) - 0.5*np.pi
        theta0[int(sp)] = float(th0)
    return theta0

# gp kernels

class RBFKernel(nn.Module):
    """rbf kernel, log params"""
    def __init__(self, lengthscale: float = 0.1, outputscale: float = 1.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.tensor(float(lengthscale)).log())
        self.log_outputscale = nn.Parameter(torch.tensor(float(outputscale)).log())

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.log_lengthscale.exp()

    @property
    def outputscale(self) -> torch.Tensor:
        return self.log_outputscale.exp()

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """rbf gram"""
        sq_dist = torch.cdist(X1, X2).pow(2)
        return self.outputscale.pow(2) * torch.exp(-0.5 * sq_dist / self.lengthscale.pow(2))


class Matern32Kernel(nn.Module):
    """matern 3/2 kernel"""
    def __init__(self, lengthscale: float = 0.1, outputscale: float = 1.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.tensor(float(lengthscale)).log())
        self.log_outputscale = nn.Parameter(torch.tensor(float(outputscale)).log())

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.log_lengthscale.exp()

    @property
    def outputscale(self) -> torch.Tensor:
        return self.log_outputscale.exp()

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """matern gram"""
        r = torch.cdist(X1, X2).clamp_min(0.0)
        sqrt3_r_over_l = math.sqrt(3) * r / self.lengthscale
        return self.outputscale.pow(2) * (1.0 + sqrt3_r_over_l) * torch.exp(-sqrt3_r_over_l)


# classical gp regression

class GP_REIM(nn.Module):
    """gp regression, kxy reim"""

    def __init__(
        self,
        kernel: str | nn.Module = "rbf",
        lengthscale: float = 0.1,
        outputscale: float = 1.0,
        noise: float = 1e-3,
    ):
        super().__init__()

        if isinstance(kernel, str):
            if kernel == "rbf":
                self.kernel = RBFKernel(lengthscale=lengthscale, outputscale=outputscale)
            elif kernel in ("matern32", "matern"):
                self.kernel = Matern32Kernel(lengthscale=lengthscale, outputscale=outputscale)
            else:
                raise ValueError(f"Unknown kernel '{kernel}'. Choose 'rbf' or 'matern32'.")
        else:
            self.kernel = kernel

        self.log_noise = nn.Parameter(torch.tensor(float(noise)).log())

        # fit cache
        self._X_train: torch.Tensor | None = None
        self._alpha:   torch.Tensor | None = None
        self._L:       torch.Tensor | None = None

    @property
    def noise(self) -> torch.Tensor:
        return self.log_noise.exp()

    # internal helpers

    def _build_K(self, X: torch.Tensor) -> torch.Tensor:
        """gram plus noise"""
        K = self.kernel(X, X)
        K.diagonal().add_(self.noise.pow(2))
        return K

    # public api

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """cholesky fit"""
        with torch.no_grad():
            K = self._build_K(X_train)
            L = torch.linalg.cholesky(K)
            # cholesky solve
            alpha = torch.cholesky_solve(y_train, L)

        self._X_train = X_train.detach()
        self._alpha   = alpha.detach()
        self._L       = L.detach()

    def log_marginal_likelihood(
        self, X_train: torch.Tensor, y_train: torch.Tensor
    ) -> torch.Tensor:
        """log marginal likelihood"""
        K   = self._build_K(X_train)
        L   = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y_train, L)

        # data fit term
        data_fit = -0.5 * (y_train * alpha).sum()

        # log det term
        log_det_term = -L.diagonal().log().sum() * y_train.shape[1]

        N = X_train.shape[0]
        const = -0.5 * N * y_train.shape[1] * math.log(2 * math.pi)

        return data_fit + log_det_term + const

    @torch.no_grad()
    def forward(self, X_test: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
        """posterior mean, chunked"""
        if self._alpha is None or self._X_train is None:
            raise RuntimeError("GP_REIM.fit() must be called before forward().")

        out_chunks = []
        for i in range(0, X_test.shape[0], chunk_size):
            K_star = self.kernel(X_test[i : i + chunk_size], self._X_train)
            out_chunks.append(K_star @ self._alpha)
        return torch.cat(out_chunks, dim=0)

    @torch.no_grad()
    def posterior_variance(
        self, X_test: torch.Tensor, chunk_size: int = 8192
    ) -> torch.Tensor:
        """posterior std, chunked"""
        if self._L is None or self._X_train is None:
            raise RuntimeError("GP_REIM.fit() must be called before posterior_variance().")

        var_chunks = []
        k_diag = self.kernel.outputscale.pow(2)
        for i in range(0, X_test.shape[0], chunk_size):
            K_star = self.kernel(X_test[i : i + chunk_size], self._X_train)
            # triangular solve
            v = torch.linalg.solve_triangular(self._L, K_star.mT, upper=False)
            var_chunks.append((k_diag - v.pow(2).sum(dim=0)).clamp_min(0.0))
        return torch.cat(var_chunks, dim=0).sqrt()


# 1d radial nets

class SIREN_1D(nn.Module):
    """1d siren, polar radial"""
    def __init__(self, out_dim=2, hidden=128, depth=4, w0=30.0):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        super().__init__()
        layers = [SineLayer(1, hidden, w0=w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


class WIRE_1D(nn.Module):
    """1d wire, polar radial"""
    def __init__(self, out_dim=2, hidden=64, depth=4, w0=20.0, s0=10.0):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        super().__init__()
        layers = [GaborLayer(1, hidden, w0=w0, s0=s0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(GaborLayer(2 * hidden, hidden, w0=w0, s0=s0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(2 * hidden, out_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


# polar kspace net

class PolarKSpaceNet(nn.Module):
    """polar kspace, radial angular"""
    def __init__(
        self,
        n_angular_modes: int = 16,
        radial_depth: int = 4,
        radial_width: int = 128,
        radial_type: str = "wire",
        omega_0: float = 30.0,
        s_0: float = 10.0,
        s_max: float = 1.0,
    ):
        super().__init__()
        self.N = n_angular_modes
        self.s_max = s_max
        n_modes = 2 * n_angular_modes + 1
        # reim per mode
        out_dim = 2 * n_modes

        if radial_type == "wire":
            self.radial = WIRE_1D(
                out_dim=out_dim, hidden=radial_width, depth=radial_depth,
                w0=omega_0, s0=s_0,
            )
        elif radial_type == "siren":
            self.radial = SIREN_1D(
                out_dim=out_dim, hidden=radial_width, depth=radial_depth,
                w0=omega_0,
            )
        else:
            raise ValueError(f"radial_type must be 'wire' or 'siren', got {radial_type!r}")

        # mode indices
        self.register_buffer(
            "mode_n", torch.arange(-n_angular_modes, n_angular_modes + 1, dtype=torch.float32)
        )

    def _to_polar(self, k_coords: torch.Tensor):
        """kxy to s, theta"""
        kx = k_coords[:, 0]
        ky = k_coords[:, 1]
        theta = torch.atan2(ky, kx)

        # signed s, folded
        theta0 = torch.remainder(theta + 0.5 * np.pi, np.pi) - 0.5 * np.pi
        c = torch.cos(theta0)
        sin_t = torch.sin(theta0)
        s = kx * c + ky * sin_t

        s = s / max(self.s_max, 1e-12)
        return s, theta

    def forward(self, k_coords: torch.Tensor) -> torch.Tensor:
        """polar forward"""
        s, theta = self._to_polar(k_coords)

        # radial coeffs
        coeffs = self.radial(s.unsqueeze(-1))

        n_modes = 2 * self.N + 1
        # reim coeff splits
        c_re = coeffs[:, :n_modes]
        c_im = coeffs[:, n_modes:]

        # angular fourier
        n_theta = self.mode_n.unsqueeze(0) * theta.unsqueeze(1)
        cos_basis = torch.cos(n_theta)
        sin_basis = torch.sin(n_theta)

        # mode sums
        out_re = (c_re * cos_basis - c_im * sin_basis).sum(dim=1)
        out_im = (c_re * sin_basis + c_im * cos_basis).sum(dim=1)

        return torch.stack([out_re, out_im], dim=1)

    def dc_predictions(self, n_theta: int = 64) -> torch.Tensor:
        """dc predictions, angles"""
        device = self.mode_n.device
        thetas = torch.linspace(-np.pi, np.pi, n_theta + 1, device=device)[:-1]
        # tiny radius
        eps = 1e-8
        k_coords = torch.stack([eps * torch.cos(thetas), eps * torch.sin(thetas)], dim=1)
        return self.forward(k_coords)


# multi coil models

class CoilFiLMModulation(nn.Module):
    """coil film, gamma beta"""
    def __init__(self, n_coils: int, coil_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.coil_embed = nn.Embedding(n_coils, coil_embed_dim)
        self.gamma_layer = nn.Linear(coil_embed_dim, hidden_dim)
        self.beta_layer = nn.Linear(coil_embed_dim, hidden_dim)

        # identity init
        nn.init.zeros_(self.gamma_layer.weight)
        nn.init.zeros_(self.gamma_layer.bias)
        nn.init.zeros_(self.beta_layer.weight)
        nn.init.zeros_(self.beta_layer.bias)

    def forward(self, coil_idx: torch.Tensor):
        """gamma beta"""
        c = self.coil_embed(coil_idx)
        gamma = self.gamma_layer(c) + 1.0
        beta = self.beta_layer(c)
        return gamma, beta


class MultiCoilWIRE(nn.Module):
    """multicoil wire, film"""
    def __init__(
        self,
        *,
        in_dim=2,
        hidden=64,
        depth=8,
        w0=20.0,
        s0=10.0,
        n_coils=8,
        coil_embed_dim=32,
    ):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        super().__init__()

        self.n_coils = n_coils
        self.hidden = hidden
        self.depth = depth

        # first layer
        self.first_layer = GaborLayer(in_dim, hidden, w0=w0, s0=s0, is_first=True)
        self.first_film = CoilFiLMModulation(n_coils, coil_embed_dim, hidden)

        # hidden layers
        self.hidden_linears = nn.ModuleList()
        self.hidden_films = nn.ModuleList()
        for _ in range(depth - 2):
            self.hidden_linears.append(nn.Linear(2 * hidden, hidden))
            self.hidden_films.append(CoilFiLMModulation(n_coils, coil_embed_dim, hidden))

        # wire init
        for lin in self.hidden_linears:
            bound = math.sqrt(6 / lin.in_features) / w0
            nn.init.uniform_(lin.weight, -bound, bound)
            nn.init.zeros_(lin.bias)

        self.w0 = w0
        self.s0 = s0

        # output head
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, coords: torch.Tensor, coil_idx: torch.Tensor) -> torch.Tensor:
        """multicoil forward"""
        # film before gabor
        h = self.first_layer.linear(coords)
        gamma, beta = self.first_film(coil_idx)
        h = gamma * h + beta
        # gabor activation
        gauss = torch.exp(-0.5 * (self.s0 * h) ** 2)
        real = gauss * torch.cos(self.w0 * h)
        imag = gauss * torch.sin(self.w0 * h)
        h = torch.cat([real, imag], dim=-1)

        # hidden layers
        for lin, film in zip(self.hidden_linears, self.hidden_films):
            h = lin(h)
            gamma, beta = film(coil_idx)
            h = gamma * h + beta
            gauss = torch.exp(-0.5 * (self.s0 * h) ** 2)
            real = gauss * torch.cos(self.w0 * h)
            imag = gauss * torch.sin(self.w0 * h)
            h = torch.cat([real, imag], dim=-1)

        return self.head(h)


class MultiCoilSIREN(nn.Module):
    """multicoil siren, film"""
    def __init__(
        self,
        *,
        in_dim=2,
        hidden=64,
        depth=8,
        w0=15.0,
        n_coils=8,
        coil_embed_dim=32,
    ):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        super().__init__()

        self.n_coils = n_coils
        self.hidden = hidden
        self.w0 = w0

        # first layer
        self.first_linear = nn.Linear(in_dim, hidden)
        self.first_film = CoilFiLMModulation(n_coils, coil_embed_dim, hidden)
        # siren init
        nn.init.uniform_(self.first_linear.weight, -1 / in_dim, 1 / in_dim)
        nn.init.zeros_(self.first_linear.bias)

        # hidden layers
        self.hidden_linears = nn.ModuleList()
        self.hidden_films = nn.ModuleList()
        for _ in range(depth - 2):
            lin = nn.Linear(hidden, hidden)
            bound = math.sqrt(6 / hidden) / w0
            nn.init.uniform_(lin.weight, -bound, bound)
            nn.init.zeros_(lin.bias)
            self.hidden_linears.append(lin)
            self.hidden_films.append(CoilFiLMModulation(n_coils, coil_embed_dim, hidden))

        self.head = nn.Linear(hidden, 2)

    def forward(self, coords: torch.Tensor, coil_idx: torch.Tensor) -> torch.Tensor:
        h = self.first_linear(coords)
        gamma, beta = self.first_film(coil_idx)
        h = torch.sin(self.w0 * (gamma * h + beta))

        for lin, film in zip(self.hidden_linears, self.hidden_films):
            h = lin(h)
            gamma, beta = film(coil_idx)
            h = torch.sin(self.w0 * (gamma * h + beta))

        return self.head(h)


class MultiCoilConcat(nn.Module):
    """multicoil concat"""
    def __init__(
        self,
        backbone_family: str = "wire",
        backbone_kwargs: dict = None,
        n_coils: int = 8,
        coil_embed_dim: int = 16,
    ):
        super().__init__()
        self.coil_embed = nn.Embedding(n_coils, coil_embed_dim)
        self.n_coils = n_coils

        kwargs = dict(backbone_kwargs or {})
        kwargs["in_dim"] = 2 + coil_embed_dim

        if backbone_family == "wire":
            self.backbone = WIRE_KXY_REIM(**kwargs)
        elif backbone_family == "siren":
            self.backbone = NIK_SIREN_KXY_REIM(**kwargs)
        else:
            raise ValueError(f"Unknown backbone_family: {backbone_family!r}")

    def forward(self, coords: torch.Tensor, coil_idx: torch.Tensor) -> torch.Tensor:
        """concat forward"""
        c_emb = self.coil_embed(coil_idx)
        x = torch.cat([coords, c_emb], dim=-1)
        return self.backbone(x)


def loss_function(y_pred, y, mag_eps: float = 1e-12, mag_reg: float = 0.1):
    """inverse magnitude mse"""
    res = y_pred - y
    mag = torch.sqrt(y[:,0]**2 + y[:,1]**2 + mag_eps)
    w = 1.0 / (mag + mag_reg)
    return (w * (res[:,0]**2 + res[:,1]**2)).mean()


# ---------------------------------------------------------------------------
# Winning real-data recipe (ported from nik-autoresearch run nik_2901994).
# Ablation E1-E8: WIRE+FF+residual, depth 12, hidden 512, k_freq 256/k_sigma 2.5,
# t_freq 32/t_sigma 1.5. Held-out 0.323, swing 51%, nav-corr 0.885 on slice 13.
# ---------------------------------------------------------------------------
class WIRE_FF_KXY_COIL_T_REIM(nn.Module):
    """WIRE (Gabor) backbone with Fourier-feature encoding of (kx,ky) and t.
    forward(kcoords(N,2), t(N,), coil_idx(N,)) -> (N,2) Re/Im."""
    def __init__(self, n_coils, coil_embed_dim=8, hidden=384, depth=8, w0=62.0, s0=15.0,
                 k_freq=128, k_sigma=2.5, t_freq=16, t_sigma=1.5, ff_seed=0, dropout=0.0):
        super().__init__()
        self.coil_embed = nn.Embedding(int(n_coils), int(coil_embed_dim))
        nn.init.uniform_(self.coil_embed.weight, -1.0, 1.0)
        self.ff_k = FourierFeatures(2, n_freq=int(k_freq), sigma=k_sigma, seed=ff_seed)
        self.ff_t = FourierFeatures(1, n_freq=int(t_freq), sigma=t_sigma, seed=ff_seed)
        in_dim = 2 * int(k_freq) + 2 * int(t_freq) + int(coil_embed_dim)
        self.backbone_model = WIRE_KXY_REIM(in_dim=in_dim, hidden=hidden, depth=depth,
                                            w0=w0, s0=s0, out_dim=2, dropout=dropout)

    def forward(self, kcoords, t, coil_idx):
        tt = t.view(-1, 1)
        ce = self.coil_embed(coil_idx.long())
        x = torch.cat([self.ff_k(kcoords), self.ff_t(tt), ce], dim=-1)
        return self.backbone_model(x)


class WIRE_FF_RES_KXY_COIL_T_REIM(nn.Module):
    """WIRE_FF with RESIDUAL (skip) connections between the intermediate Gabor blocks.
    h <- h + GaborBlock(h). All intermediate states are 2*hidden, so skips are
    dimension-matched. THE WINNING MODEL (depth 12). forward sig unchanged."""
    def __init__(self, n_coils, coil_embed_dim=8, hidden=384, depth=8, w0=62.0, s0=15.0,
                 k_freq=128, k_sigma=2.5, t_freq=16, t_sigma=1.5, ff_seed=0, dropout=0.0):
        super().__init__()
        self.coil_embed = nn.Embedding(int(n_coils), int(coil_embed_dim))
        nn.init.uniform_(self.coil_embed.weight, -1.0, 1.0)
        self.ff_k = FourierFeatures(2, n_freq=int(k_freq), sigma=k_sigma, seed=ff_seed)
        self.ff_t = FourierFeatures(1, n_freq=int(t_freq), sigma=t_sigma, seed=ff_seed)
        in_dim = 2 * int(k_freq) + 2 * int(t_freq) + int(coil_embed_dim)
        self.first = GaborLayer(in_dim, hidden, w0=w0, s0=s0, is_first=True)          # -> 2*hidden
        self.blocks = nn.ModuleList([GaborLayer(2 * hidden, hidden, w0=w0, s0=s0)     # 2*hidden -> 2*hidden
                                     for _ in range(max(0, depth - 2))])
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, kcoords, t, coil_idx):
        tt = t.view(-1, 1); ec = self.coil_embed(coil_idx.long())
        h = self.first(torch.cat([self.ff_k(kcoords), self.ff_t(tt), ec], dim=-1))
        for blk in self.blocks:
            h = h + blk(h)                                                            # residual skip
        return self.head(h)
