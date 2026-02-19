# nik_model.py
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
        self.head = nn.Linear(hidden, 2)  # [log_mag, phase_raw]

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
    return torch.cat([re, im], dim=-1)  # (B,2)


class NIK_SIREN_REIM(NIK_SIREN):
    """
    Same as NIK_SIREN, but outputs (Re, Im) directly instead of (log_mag, phase).
    Note: checkpoints are NOT interchangeable with NIK_SIREN despite sharing architecture.
    """
    def forward(self, kxyz_t, coil_idx):
        return self.head(self._encode(kxyz_t, coil_idx))  # (B,2) -> (Re, Im)


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

        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
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

        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
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
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        super().__init__()

        # First layer uses 4*w0 to boost high-frequency sensitivity on raw (kx,ky) input.
        # Note: _init_weights for is_first=True does NOT use w0, so only the forward
        # activation is scaled; verify this is intentional if training is unstable.
        layers = [SineLayer(in_dim, hidden, w0=4*w0, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)  # (Re, Im)

    def forward(self, x):
        # x: (B,2) [kx, ky]
        h = self.backbone(x)
        return self.head(h)  # (B,2) [Re, Im]



def cart_to_s_sincos_foldpi(x_kxy: torch.Tensor):
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
    sin_theta0 = torch.sin(theta0)

    # signed coordinate along the spoke direction
    s_coord = kx * c + ky * sin_theta0  # can be negative

    x_ssc = torch.stack([s_coord, sin_theta0, c], dim=-1)
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
        sin_theta0 = torch.sin(theta0)

        # signed coordinate along spoke direction
        s_coord = kx * c + ky * sin_theta0

        # normalize signed coordinate
        s_coord = s_coord / max(self.s_max, self.eps)

        return torch.stack([s_coord, sin_theta0, c], dim=-1)

def compute_theta0_per_spoke(x_all_kxy: torch.Tensor, spoke_id_all: torch.Tensor):
    # returns dict: spoke_id -> theta0 (float)
    theta0 = {}
    for sp in torch.unique(spoke_id_all).tolist():
        m = (spoke_id_all == sp)
        xk = x_all_kxy[m][:, :2]
        r = torch.sqrt((xk**2).sum(dim=1))
        r_max = r.max()
        if r_max == 0:
            # all points at DC origin; spoke direction is undefined — default to 0
            theta0[int(sp)] = 0.0
            continue
        j = torch.argmax(r)
        th = torch.atan2(xk[j,1], xk[j,0])
        th0 = torch.remainder(th + 0.5*np.pi, np.pi) - 0.5*np.pi
        theta0[int(sp)] = float(th0)
    return theta0

# ---------------------------------------------------------------------------
# GP kernels
# ---------------------------------------------------------------------------

class RBFKernel(nn.Module):
    """
    Squared-exponential (RBF) kernel:
        k(x, x') = σ_f² exp( -‖x - x'‖² / (2 ℓ²) )

    Parameters are stored as log values for unconstrained optimization.
    """
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
        """
        X1: (N, D)
        X2: (M, D)
        Returns: (N, M)
        """
        sq_dist = torch.cdist(X1, X2).pow(2)
        return self.outputscale.pow(2) * torch.exp(-0.5 * sq_dist / self.lengthscale.pow(2))


class Matern32Kernel(nn.Module):
    """
    Matérn 3/2 kernel:
        k(x, x') = σ_f² (1 + √3 r/ℓ) exp(-√3 r/ℓ),   r = ‖x - x'‖

    Produces once-differentiable sample paths — less smooth than RBF.
    Useful when the k-space signal has sharper features.
    """
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
        """
        X1: (N, D)
        X2: (M, D)
        Returns: (N, M)
        """
        r = torch.cdist(X1, X2).clamp_min(0.0)
        sqrt3_r_over_l = math.sqrt(3) * r / self.lengthscale
        return self.outputscale.pow(2) * (1.0 + sqrt3_r_over_l) * torch.exp(-sqrt3_r_over_l)


# ---------------------------------------------------------------------------
# Classical GP regression (Rasmussen & Williams 2006, §2.2)
# ---------------------------------------------------------------------------

class GP_REIM(nn.Module):
    """
    Gaussian Process regression for 2D k-space (kx, ky) → (Re, Im).

    Re and Im are modelled as two independent GPs sharing the same kernel and
    noise hyperparameters (learnable via log-marginal-likelihood).

    Workflow
    --------
    1.  Instantiate:   model = GP_REIM(kernel="rbf")
    2.  Fit:           model.fit(X_train, y_train)   # O(N²) mem, O(N³) compute
    3.  Predict:       y_pred = model(X_test)         # (M, 2) [Re, Im]

    Hyperparameter optimisation (optional, done inside fit_gp):
        model.log_marginal_likelihood(X, y)  # differentiable, use Adam

    Notes
    -----
    - X coordinates are assumed to be normalised to roughly [-1, 1] (same
      convention as the rest of the NIK pipeline).
    - For N ≈ 22 k the full kernel matrix requires ~1.8 GB of GPU memory.
    - forward() evaluates K_* @ α in chunks to bound peak GPU memory.
    """

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
            self.kernel = kernel  # accept a pre-built kernel module

        self.log_noise = nn.Parameter(torch.tensor(float(noise)).log())

        # Set by fit(); None until then
        self._X_train: torch.Tensor | None = None
        self._alpha:   torch.Tensor | None = None  # (N, 2)  K⁻¹ y
        self._L:       torch.Tensor | None = None  # (N, N)  lower Cholesky of K

    @property
    def noise(self) -> torch.Tensor:
        return self.log_noise.exp()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_K(self, X: torch.Tensor) -> torch.Tensor:
        """K_XX + σ_n² I  — (N, N)"""
        K = self.kernel(X, X)
        K.diagonal().add_(self.noise.pow(2))
        return K

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """
        Compute and store the Cholesky factor L and dual variables α = K⁻¹ y.

        Parameters
        ----------
        X_train : (N, D) float tensor on the model's device
        y_train : (N, 2) float tensor [Re, Im]
        """
        with torch.no_grad():
            K = self._build_K(X_train)                        # (N, N)
            L = torch.linalg.cholesky(K)                      # lower triangular
            # cholesky_solve(B, L, upper=False) solves (L Lᵀ) x = B
            alpha = torch.cholesky_solve(y_train, L)          # (N, 2)

        self._X_train = X_train.detach()
        self._alpha   = alpha.detach()
        self._L       = L.detach()

    def log_marginal_likelihood(
        self, X_train: torch.Tensor, y_train: torch.Tensor
    ) -> torch.Tensor:
        """
        Differentiable log marginal likelihood (Rasmussen eq. 2.30):

            log p(y|X,θ) = -½ yᵀ K⁻¹ y  - ½ log|K|  - N/2 log(2π)
                         = -½ tr(αᵀ y)   - Σ log diag(L)  + const

        Averages over the two output channels (Re, Im).
        """
        K   = self._build_K(X_train)
        L   = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y_train, L)               # (N, 2)

        # -½ yᵀ α  (sum over both channels, then average)
        data_fit = -0.5 * (y_train * alpha).sum()

        # -½ log|K| = -Σ log diag(L)  (factor of 2 from both channels)
        log_det_term = -L.diagonal().log().sum() * y_train.shape[1]

        N = X_train.shape[0]
        const = -0.5 * N * y_train.shape[1] * math.log(2 * math.pi)

        return data_fit + log_det_term + const

    @torch.no_grad()
    def forward(self, X_test: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
        """
        Predict posterior mean at X_test.

            μ* = K(X*, X_train) @ α

        Evaluated in chunks to bound peak GPU memory.

        Parameters
        ----------
        X_test     : (M, D) coordinates
        chunk_size : rows of K_star computed at once

        Returns
        -------
        (M, 2) [Re, Im] predicted k-space values
        """
        if self._alpha is None or self._X_train is None:
            raise RuntimeError("GP_REIM.fit() must be called before forward().")

        out_chunks = []
        for i in range(0, X_test.shape[0], chunk_size):
            K_star = self.kernel(X_test[i : i + chunk_size], self._X_train)  # (chunk, N)
            out_chunks.append(K_star @ self._alpha)
        return torch.cat(out_chunks, dim=0)  # (M, 2)

    @torch.no_grad()
    def posterior_variance(
        self, X_test: torch.Tensor, chunk_size: int = 8192
    ) -> torch.Tensor:
        """
        Diagonal of the posterior covariance (same for both channels):

            σ²*(x) = k(x,x) - K(x,X) K⁻¹ K(X,x)

        Returns
        -------
        (M,) posterior standard deviation at each test point
        """
        if self._L is None or self._X_train is None:
            raise RuntimeError("GP_REIM.fit() must be called before posterior_variance().")

        var_chunks = []
        k_diag = self.kernel.outputscale.pow(2)          # k(x,x) for RBF/Matérn
        for i in range(0, X_test.shape[0], chunk_size):
            K_star = self.kernel(X_test[i : i + chunk_size], self._X_train)  # (chunk, N)
            # v = L⁻¹ K_starᵀ  →  ‖v‖² = K_star K⁻¹ K_starᵀ diagonal
            v = torch.linalg.solve_triangular(self._L, K_star.mT, upper=False)  # (N, chunk)
            var_chunks.append((k_diag - v.pow(2).sum(dim=0)).clamp_min(0.0))
        return torch.cat(var_chunks, dim=0).sqrt()        # (M,) posterior std


def loss_function(y_pred, y, mag_eps: float = 1e-12, mag_reg: float = 0.1):
    """
    Inverse-magnitude weighted MSE on (Re, Im) predictions.

    Args:
        y_pred: (B,2) predicted [Re, Im]
        y:      (B,2) target    [Re, Im]
        mag_eps: small constant inside sqrt to avoid NaN at exactly zero magnitude
        mag_reg: additive regulariser in weight denominator; controls how aggressively
                 near-zero-magnitude (DC/low-signal) samples are up-weighted
    """
    res = y_pred - y
    mag = torch.sqrt(y[:,0]**2 + y[:,1]**2 + mag_eps)
    w = 1.0 / (mag + mag_reg)
    return (w * (res[:,0]**2 + res[:,1]**2)).mean()
