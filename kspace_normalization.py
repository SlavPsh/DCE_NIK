"""
kspace_normalization.py — K-space normalization for INR training on radial MRI.

Provides:
  - compute_radius: radial distance in k-space
  - compute_dcf_radial: density compensation weights
  - estimate_radial_envelope: smooth magnitude decay a(r)
  - compute_global_scale: robust global scale after envelope correction
  - KSpaceNormalizer: fit on radial, apply to radial+Cartesian
  - prepare_training_targets / prepare_validation_targets: convenience API

Normalization rule:
    y_tilde(k) = y(k) / (a(r) * s)

where a(r) is the radial envelope and s is the global scale.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal


# =========================================================================
# 1. Radial radius computation
# =========================================================================

def compute_radius(kcoords: torch.Tensor) -> torch.Tensor:
    """Compute radial distance: r = sqrt(kx^2 + ky^2).

    Args:
        kcoords: (N, 2) tensor of (kx, ky) coordinates.

    Returns:
        r: (N,) tensor of radial distances.
    """
    return torch.sqrt(kcoords[:, 0] ** 2 + kcoords[:, 1] ** 2)


# =========================================================================
# 2. Density compensation weights
# =========================================================================

def compute_dcf_radial(
    kcoords: torch.Tensor,
    method: str = "simple_ramp",
    eps: float = 1e-8,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute density compensation weights for radial k-space.

    Radial sampling has density ~ 1/r, so we weight by r to compensate.
    Weights are normalized to mean=1 so loss magnitudes stay interpretable.

    Args:
        kcoords: (N, 2) tensor of (kx, ky).
        method: "simple_ramp" (weight = r) or "power_ramp" (weight = r^gamma).
        eps: floor to avoid zero weights at DC.
        gamma: exponent for power_ramp method.

    Returns:
        dcf: (N,) tensor, mean-normalized to 1.
    """
    r = compute_radius(kcoords)

    if method == "simple_ramp":
        dcf = torch.clamp(r, min=eps)
    elif method == "power_ramp":
        dcf = torch.clamp(r, min=eps) ** gamma
    else:
        raise ValueError(f"Unknown DCF method: {method}")

    dcf = dcf / (dcf.mean() + eps)
    return dcf


# =========================================================================
# 3. Radial envelope estimation
# =========================================================================

@dataclass
class RadialEnvelope:
    """Stores a smoothed radial magnitude envelope a(r).

    The envelope describes the average k-space magnitude as a function of
    radial distance. Used to flatten the dynamic range before global scaling.
    """
    bin_centers: torch.Tensor
    raw_shell_values: torch.Tensor
    smoothed_shell_values: torch.Tensor
    floor_value: float
    r_max: float
    statistic: str
    smooth_method: str

    def evaluate(self, r_query: torch.Tensor) -> torch.Tensor:
        """Interpolate the envelope at arbitrary radii.

        Uses linear interpolation between bin centers, clamped to [0, r_max].
        Values below the floor are replaced by the floor.

        Args:
            r_query: (M,) tensor of query radii.

        Returns:
            a: (M,) tensor of envelope values (>= floor_value).
        """
        device = r_query.device
        centers = self.bin_centers.to(device)
        values = self.smoothed_shell_values.to(device)

        # Normalize query radii to [0, n_bins-1] for grid_sample-style interp
        n_bins = centers.shape[0]
        r_clamped = torch.clamp(r_query, 0.0, self.r_max)
        # Map r to bin index (continuous)
        idx = r_clamped / (self.r_max + 1e-12) * (n_bins - 1)
        idx_lo = torch.clamp(idx.long(), 0, n_bins - 2)
        idx_hi = idx_lo + 1
        frac = idx - idx_lo.float()

        a = values[idx_lo] * (1.0 - frac) + values[idx_hi] * frac
        a = torch.clamp(a, min=self.floor_value)
        return a


def estimate_radial_envelope(
    kcoords: torch.Tensor,
    y: torch.Tensor,
    dcf: Optional[torch.Tensor] = None,
    n_bins: int = 128,
    statistic: str = "weighted_rms",
    smooth_method: str = "moving_average",
    smooth_width: int = 5,
    floor_fraction: float = 1e-3,
    eps: float = 1e-8,
) -> RadialEnvelope:
    """Estimate smooth radial magnitude envelope a(r).

    Bins k-space samples by radius, computes a magnitude statistic per shell,
    smooths across shells, and returns an interpolatable envelope.

    Args:
        kcoords: (N, 2) tensor of (kx, ky).
        y: (N,) complex tensor of k-space values.
        dcf: (N,) optional density weights (used as shell weights).
        n_bins: number of radial shells.
        statistic: "weighted_rms" — shell magnitude statistic.
        smooth_method: "moving_average" — smoothing across shells.
        smooth_width: kernel width for smoothing.
        floor_fraction: minimum envelope value as fraction of envelope max.
        eps: numerical stability.

    Returns:
        RadialEnvelope object with evaluate(r_query) method.
    """
    device = kcoords.device
    r = compute_radius(kcoords)
    mag = torch.abs(y)
    r_max = float(r.max().item())

    if dcf is None:
        w = torch.ones_like(r)
    else:
        w = dcf

    # Bin edges and centers
    bin_edges = torch.linspace(0, r_max, n_bins + 1, device=device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Compute per-shell statistic
    raw_shell = torch.zeros(n_bins, device=device)
    bin_idx = torch.clamp(
        ((r / (r_max + eps)) * n_bins).long(), 0, n_bins - 1
    )

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            raw_shell[b] = 0.0
            continue

        if statistic == "weighted_rms":
            wb = w[mask]
            mb = mag[mask]
            raw_shell[b] = torch.sqrt((wb * mb ** 2).sum() / (wb.sum() + eps))
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    # Fill empty bins by nearest-neighbor interpolation
    filled = raw_shell.clone()
    valid = filled > 0
    if valid.any() and not valid.all():
        valid_idx = torch.where(valid)[0]
        for b in range(n_bins):
            if not valid[b]:
                distances = (valid_idx - b).abs()
                nearest = valid_idx[distances.argmin()]
                filled[b] = filled[nearest]

    # Smooth
    if smooth_method == "moving_average":
        kernel = torch.ones(smooth_width, device=device) / smooth_width
        # Pad and convolve
        pad = smooth_width // 2
        padded = F.pad(filled.unsqueeze(0).unsqueeze(0), (pad, pad), mode="replicate")
        smoothed = F.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0)).squeeze()
    else:
        raise ValueError(f"Unknown smooth method: {smooth_method}")

    # Floor: fraction of max smoothed value
    floor_value = float(floor_fraction * smoothed.max().item())
    smoothed = torch.clamp(smoothed, min=floor_value)

    return RadialEnvelope(
        bin_centers=bin_centers.cpu(),
        raw_shell_values=raw_shell.cpu(),
        smoothed_shell_values=smoothed.cpu(),
        floor_value=floor_value,
        r_max=r_max,
        statistic=statistic,
        smooth_method=smooth_method,
    )


# =========================================================================
# 4. Global robust scale
# =========================================================================

def compute_global_scale(
    y_normed: torch.Tensor,
    dcf: Optional[torch.Tensor] = None,
    method: str = "weighted_rms",
    eps: float = 1e-8,
) -> float:
    """Compute global scale of envelope-corrected k-space.

    Applied after radial envelope correction. Returns a scalar.

    Args:
        y_normed: (N,) complex tensor (after dividing by envelope).
        dcf: (N,) optional density weights.
        method: "weighted_rms" or "weighted_quantile".
        eps: numerical stability.

    Returns:
        Scalar scale value (float).
    """
    mag = torch.abs(y_normed)

    if dcf is None:
        w = torch.ones_like(mag)
    else:
        w = dcf

    if method == "weighted_rms":
        scale = torch.sqrt((w * mag ** 2).sum() / (w.sum() + eps))
    elif method == "weighted_quantile":
        # Approximate: sort by magnitude, find 0.99 quantile
        sorted_mag, _ = torch.sort(mag)
        idx_99 = int(0.99 * len(sorted_mag))
        scale = sorted_mag[idx_99]
    else:
        raise ValueError(f"Unknown global scale method: {method}")

    return float(torch.clamp(scale, min=eps).item())


# =========================================================================
# 5. KSpaceNormalizer
# =========================================================================

def _to_complex(y: torch.Tensor) -> torch.Tensor:
    """Convert (N, 2) or (N, 2*C) real, or (N,) complex to complex.

    For single coil (N, 2): returns (N,) complex.
    For multicoil (N, 2*C): returns (N, C) complex.
    For already complex: returns as-is.
    """
    if torch.is_complex(y):
        return y
    if y.ndim == 2 and y.shape[-1] == 2:
        return torch.complex(y[:, 0], y[:, 1])
    if y.ndim == 2 and y.shape[-1] > 2 and y.shape[-1] % 2 == 0:
        C = y.shape[-1] // 2
        return torch.complex(y[:, 0::2], y[:, 1::2])  # (N, C) complex
    raise ValueError(f"Cannot convert shape {y.shape} to complex")


def _rss_magnitude(y: torch.Tensor) -> torch.Tensor:
    """Compute RSS magnitude for normalizer fitting.

    Single coil (N,) complex → (N,) magnitude.
    Multicoil (N, C) complex → (N,) RSS magnitude.
    """
    if y.ndim == 1:
        return torch.abs(y)
    return torch.sqrt((torch.abs(y) ** 2).sum(dim=-1))


def _from_complex(y: torch.Tensor, as_real: bool = False) -> torch.Tensor:
    """Convert complex to real format.

    (N,) complex → (N, 2) real.
    (N, C) complex → (N, 2*C) real [Re_c0, Im_c0, Re_c1, Im_c1, ...].
    """
    if as_real:
        if y.ndim == 1:
            return torch.stack([y.real, y.imag], dim=-1)
        # (N, C) → (N, 2*C) interleaved
        parts = []
        for c in range(y.shape[-1]):
            parts.append(y[:, c].real)
            parts.append(y[:, c].imag)
        return torch.stack(parts, dim=-1)
    return y


class KSpaceNormalizer:
    """Fit normalization on radial data, apply to any k-space coordinates.

    Normalization: y_tilde = y / (a(r) * s)
    where a(r) is the radial envelope and s is the global scale.
    Both are fitted on radial training data and applied identically
    to radial training and Cartesian validation targets.
    """

    def __init__(self):
        self.envelope: Optional[RadialEnvelope] = None
        self.global_scale: float = 1.0
        self._fitted = False

    def fit(
        self,
        kcoords_radial: torch.Tensor,
        y_radial: torch.Tensor,
        dcf: Optional[torch.Tensor] = None,
        envelope_bins: int = 128,
        envelope_statistic: str = "weighted_rms",
        envelope_smooth_method: str = "moving_average",
        envelope_smooth_width: int = 5,
        envelope_floor_fraction: float = 1e-3,
        global_scale_method: str = "weighted_rms",
        eps: float = 1e-8,
    ):
        """Fit normalizer on radial training data.

        Supports single coil (N, 2) or multicoil (N, 2*C) real format.
        For multicoil, the envelope is fitted on RSS magnitude across coils.
        The same per-point scale a(r)*s is applied to all coils.

        Args:
            kcoords_radial: (N, 2) radial coordinates.
            y_radial: (N, 2) or (N, 2*C) real, or (N,) complex k-space values.
            dcf: (N,) optional density compensation weights.
        """
        y_c = _to_complex(y_radial)
        # For envelope fitting, use RSS magnitude across coils
        y_mag_for_envelope = _rss_magnitude(y_c)  # (N,)
        # Create a dummy complex with this magnitude for the envelope estimator
        y_mag_complex = y_mag_for_envelope.to(torch.complex64)

        # Step 1: estimate radial envelope a(r) from RSS magnitude
        self.envelope = estimate_radial_envelope(
            kcoords_radial, y_mag_complex, dcf=dcf,
            n_bins=envelope_bins,
            statistic=envelope_statistic,
            smooth_method=envelope_smooth_method,
            smooth_width=envelope_smooth_width,
            floor_fraction=envelope_floor_fraction,
            eps=eps,
        )

        # Step 2: divide by envelope
        r = compute_radius(kcoords_radial)
        a_r = self.envelope.evaluate(r).to(y_c.device)
        y_env_corrected_mag = y_mag_for_envelope / (a_r + eps)

        # Step 3: global scale from envelope-corrected RSS magnitude
        self.global_scale = compute_global_scale(
            y_env_corrected_mag.to(torch.complex64), dcf=dcf,
            method=global_scale_method, eps=eps,
        )

        self._fitted = True

    def normalize(
        self, kcoords: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Normalize k-space: y_tilde = y / (a(r) * s).

        Args:
            kcoords: (N, 2) coordinates.
            y: (N,) complex or (N, 2) real k-space values.

        Returns:
            y_tilde: same format as input y.
        """
        assert self._fitted, "Call fit() first"
        as_real = (not torch.is_complex(y)) and y.ndim == 2
        y_c = _to_complex(y)

        r = compute_radius(kcoords)
        a_r = self.envelope.evaluate(r).to(y_c.device)
        pointwise_scale = a_r * self.global_scale  # (N,)

        # Broadcast across coils if multicoil (N, C)
        if y_c.ndim == 2:
            pointwise_scale = pointwise_scale.unsqueeze(-1)

        y_tilde = y_c / (pointwise_scale + 1e-8)
        return _from_complex(y_tilde, as_real=as_real)

    def denormalize(
        self, kcoords: torch.Tensor, y_tilde: torch.Tensor
    ) -> torch.Tensor:
        """Denormalize: y = y_tilde * a(r) * s.

        Args:
            kcoords: (N, 2) coordinates.
            y_tilde: (N,) complex or (N, 2) or (N, 2*C) real normalized values.

        Returns:
            y: same format as input y_tilde.
        """
        assert self._fitted, "Call fit() first"
        as_real = (not torch.is_complex(y_tilde)) and y_tilde.ndim == 2
        y_c = _to_complex(y_tilde)

        r = compute_radius(kcoords)
        a_r = self.envelope.evaluate(r).to(y_c.device)
        pointwise_scale = a_r * self.global_scale  # (N,)

        # Broadcast across coils if multicoil (N, C)
        if y_c.ndim == 2:
            pointwise_scale = pointwise_scale.unsqueeze(-1)

        y = y_c * pointwise_scale
        return _from_complex(y, as_real=as_real)

    def get_pointwise_scale(self, kcoords: torch.Tensor) -> torch.Tensor:
        """Get per-point normalization scale a(r) * s.

        Args:
            kcoords: (N, 2) coordinates.

        Returns:
            scale: (N,) tensor.
        """
        assert self._fitted, "Call fit() first"
        r = compute_radius(kcoords)
        a_r = self.envelope.evaluate(r).to(kcoords.device)
        return a_r * self.global_scale


# =========================================================================
# 8. Training-time recommendation API
# =========================================================================

def prepare_training_targets(
    kcoords_radial: torch.Tensor,
    y_radial: torch.Tensor,
    normalizer: Optional[KSpaceNormalizer] = None,
    dcf_method: str = "simple_ramp",
    dcf_gamma: float = 1.0,
    eps: float = 1e-8,
    **normalizer_kwargs,
) -> dict:
    """Prepare radial training data: DCF, normalizer, normalized targets.

    Args:
        kcoords_radial: (N, 2) radial coordinates.
        y_radial: (N,) complex or (N, 2) real k-space values.
        normalizer: existing normalizer, or None to create+fit one.
        dcf_method: method for compute_dcf_radial.
        dcf_gamma: gamma for power_ramp DCF.
        normalizer_kwargs: passed to normalizer.fit().

    Returns:
        dict with: normalizer, dcf, y_radial_norm, r_radial.
    """
    dcf = compute_dcf_radial(kcoords_radial, method=dcf_method, gamma=dcf_gamma, eps=eps)
    r = compute_radius(kcoords_radial)

    if normalizer is None:
        normalizer = KSpaceNormalizer()
        normalizer.fit(kcoords_radial, y_radial, dcf=dcf, eps=eps, **normalizer_kwargs)

    y_norm = normalizer.normalize(kcoords_radial, y_radial)

    return {
        "normalizer": normalizer,
        "dcf": dcf,
        "y_radial_norm": y_norm,
        "r_radial": r,
    }


def prepare_validation_targets(
    kcoords_cart: torch.Tensor,
    y_cart: torch.Tensor,
    normalizer: KSpaceNormalizer,
) -> dict:
    """Prepare Cartesian validation data using the SAME normalizer from training.

    Args:
        kcoords_cart: (M, 2) Cartesian grid coordinates.
        y_cart: (M,) complex or (M, 2) real k-space values.
        normalizer: fitted KSpaceNormalizer (from prepare_training_targets).

    Returns:
        dict with: y_cart_norm, r_cart.
    """
    assert normalizer._fitted, "Normalizer must be fitted on radial data first"
    y_norm = normalizer.normalize(kcoords_cart, y_cart)
    r = compute_radius(kcoords_cart)

    return {
        "y_cart_norm": y_norm,
        "r_cart": r,
    }
