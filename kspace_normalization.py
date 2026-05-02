"""kspace normalization, dcf and envelope"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal


# radial radius

def compute_radius(kcoords: torch.Tensor) -> torch.Tensor:
    """radial distance"""
    return torch.sqrt(kcoords[:, 0] ** 2 + kcoords[:, 1] ** 2)


# dcf weights

def compute_dcf_radial(
    kcoords: torch.Tensor,
    method: str = "simple_ramp",
    eps: float = 1e-8,
    gamma: float = 1.0,
) -> torch.Tensor:
    """radial dcf, mean normed"""
    r = compute_radius(kcoords)

    if method == "simple_ramp":
        dcf = torch.clamp(r, min=eps)
    elif method == "power_ramp":
        dcf = torch.clamp(r, min=eps) ** gamma
    else:
        raise ValueError(f"Unknown DCF method: {method}")

    dcf = dcf / (dcf.mean() + eps)
    return dcf


# radial envelope

@dataclass
class RadialEnvelope:
    """smoothed radial envelope"""
    bin_centers: torch.Tensor
    raw_shell_values: torch.Tensor
    smoothed_shell_values: torch.Tensor
    floor_value: float
    r_max: float
    statistic: str
    smooth_method: str

    def evaluate(self, r_query: torch.Tensor) -> torch.Tensor:
        """envelope interp"""
        device = r_query.device
        centers = self.bin_centers.to(device)
        values = self.smoothed_shell_values.to(device)

        # bin index
        n_bins = centers.shape[0]
        r_clamped = torch.clamp(r_query, 0.0, self.r_max)
        # continuous index
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
    """estimate envelope"""
    device = kcoords.device
    r = compute_radius(kcoords)
    mag = torch.abs(y)
    r_max = float(r.max().item())

    if dcf is None:
        w = torch.ones_like(r)
    else:
        w = dcf

    # bin edges, centers
    bin_edges = torch.linspace(0, r_max, n_bins + 1, device=device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # per shell statistic
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

    # nearest neighbor fill
    filled = raw_shell.clone()
    valid = filled > 0
    if valid.any() and not valid.all():
        valid_idx = torch.where(valid)[0]
        for b in range(n_bins):
            if not valid[b]:
                distances = (valid_idx - b).abs()
                nearest = valid_idx[distances.argmin()]
                filled[b] = filled[nearest]

    # smooth
    if smooth_method == "moving_average":
        kernel = torch.ones(smooth_width, device=device) / smooth_width
        # pad, convolve
        pad = smooth_width // 2
        padded = F.pad(filled.unsqueeze(0).unsqueeze(0), (pad, pad), mode="replicate")
        smoothed = F.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0)).squeeze()
    elif smooth_method == "power_law":
        # a r^p, weighted log log lstsq
        counts = torch.bincount(bin_idx, minlength=n_bins).float()
        valid_fit = (filled > 0) & (bin_centers > 0)
        if int(valid_fit.sum().item()) < 2:
            smoothed = filled.clone()
        else:
            log_r = torch.log(bin_centers[valid_fit] + eps)
            log_a = torch.log(filled[valid_fit] + eps)
            wts = counts[valid_fit].clamp_min(eps)
            wsum = wts.sum()
            log_r_bar = (wts * log_r).sum() / wsum
            log_a_bar = (wts * log_a).sum() / wsum
            num = (wts * (log_r - log_r_bar) * (log_a - log_a_bar)).sum()
            den = (wts * (log_r - log_r_bar) ** 2).sum().clamp_min(eps)
            p = num / den
            log_A = log_a_bar - p * log_r_bar
            r_pos = bin_centers[bin_centers > 0]
            r_min_positive = float(r_pos.min().item()) if r_pos.numel() else eps
            r_eval = torch.clamp(bin_centers, min=r_min_positive)
            smoothed = torch.exp(log_A + p * torch.log(r_eval + eps))
    else:
        raise ValueError(f"Unknown smooth method: {smooth_method}")

    # floor fraction
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


# global scale

def compute_global_scale(
    y_normed: torch.Tensor,
    dcf: Optional[torch.Tensor] = None,
    method: str = "weighted_rms",
    eps: float = 1e-8,
) -> float:
    """robust global scale"""
    mag = torch.abs(y_normed)

    if dcf is None:
        w = torch.ones_like(mag)
    else:
        w = dcf

    if method == "weighted_rms":
        scale = torch.sqrt((w * mag ** 2).sum() / (w.sum() + eps))
    elif method == "weighted_quantile":
        # approx, p99 quantile
        sorted_mag, _ = torch.sort(mag)
        idx_99 = int(0.99 * len(sorted_mag))
        scale = sorted_mag[idx_99]
    else:
        raise ValueError(f"Unknown global scale method: {method}")

    return float(torch.clamp(scale, min=eps).item())


# normalizer

def _to_complex(y: torch.Tensor) -> torch.Tensor:
    """real to complex"""
    if torch.is_complex(y):
        return y
    if y.ndim == 2 and y.shape[-1] == 2:
        return torch.complex(y[:, 0], y[:, 1])
    if y.ndim == 2 and y.shape[-1] > 2 and y.shape[-1] % 2 == 0:
        C = y.shape[-1] // 2
        return torch.complex(y[:, 0::2], y[:, 1::2])
    raise ValueError(f"Cannot convert shape {y.shape} to complex")


def _rss_magnitude(y: torch.Tensor) -> torch.Tensor:
    """rss magnitude"""
    if y.ndim == 1:
        return torch.abs(y)
    return torch.sqrt((torch.abs(y) ** 2).sum(dim=-1))


def _from_complex(y: torch.Tensor, as_real: bool = False) -> torch.Tensor:
    """complex to real"""
    if as_real:
        if y.ndim == 1:
            return torch.stack([y.real, y.imag], dim=-1)
        # interleaved reim
        parts = []
        for c in range(y.shape[-1]):
            parts.append(y[:, c].real)
            parts.append(y[:, c].imag)
        return torch.stack(parts, dim=-1)
    return y


class KSpaceNormalizer:
    """envelope, global scale"""

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
        """fit normalizer"""
        y_c = _to_complex(y_radial)
        # rss for envelope
        y_mag_for_envelope = _rss_magnitude(y_c)
        # dummy complex
        y_mag_complex = y_mag_for_envelope.to(torch.complex64)

        # step 1, envelope
        self.envelope = estimate_radial_envelope(
            kcoords_radial, y_mag_complex, dcf=dcf,
            n_bins=envelope_bins,
            statistic=envelope_statistic,
            smooth_method=envelope_smooth_method,
            smooth_width=envelope_smooth_width,
            floor_fraction=envelope_floor_fraction,
            eps=eps,
        )

        # step 2, envelope divide
        r = compute_radius(kcoords_radial)
        a_r = self.envelope.evaluate(r).to(y_c.device)
        y_env_corrected_mag = y_mag_for_envelope / (a_r + eps)

        # step 3, global scale
        self.global_scale = compute_global_scale(
            y_env_corrected_mag.to(torch.complex64), dcf=dcf,
            method=global_scale_method, eps=eps,
        )

        self._fitted = True

    def normalize(
        self, kcoords: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """normalize"""
        assert self._fitted, "Call fit() first"
        as_real = (not torch.is_complex(y)) and y.ndim == 2
        y_c = _to_complex(y)

        r = compute_radius(kcoords)
        a_r = self.envelope.evaluate(r).to(y_c.device)
        pointwise_scale = a_r * self.global_scale

        # multicoil broadcast
        if y_c.ndim == 2:
            pointwise_scale = pointwise_scale.unsqueeze(-1)

        y_tilde = y_c / (pointwise_scale + 1e-8)
        return _from_complex(y_tilde, as_real=as_real)

    def denormalize(
        self, kcoords: torch.Tensor, y_tilde: torch.Tensor
    ) -> torch.Tensor:
        """denormalize"""
        assert self._fitted, "Call fit() first"
        as_real = (not torch.is_complex(y_tilde)) and y_tilde.ndim == 2
        y_c = _to_complex(y_tilde)

        r = compute_radius(kcoords)
        a_r = self.envelope.evaluate(r).to(y_c.device)
        pointwise_scale = a_r * self.global_scale

        # multicoil broadcast
        if y_c.ndim == 2:
            pointwise_scale = pointwise_scale.unsqueeze(-1)

        y = y_c * pointwise_scale
        return _from_complex(y, as_real=as_real)

    def get_pointwise_scale(self, kcoords: torch.Tensor) -> torch.Tensor:
        """pointwise scale"""
        assert self._fitted, "Call fit() first"
        r = compute_radius(kcoords)
        a_r = self.envelope.evaluate(r).to(kcoords.device)
        return a_r * self.global_scale


# training api

def prepare_training_targets(
    kcoords_radial: torch.Tensor,
    y_radial: torch.Tensor,
    normalizer: Optional[KSpaceNormalizer] = None,
    dcf_method: str = "simple_ramp",
    dcf_gamma: float = 1.0,
    eps: float = 1e-8,
    **normalizer_kwargs,
) -> dict:
    """radial training prep"""
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
    """cartesian val prep"""
    assert normalizer._fitted, "Normalizer must be fitted on radial data first"
    y_norm = normalizer.normalize(kcoords_cart, y_cart)
    r = compute_radius(kcoords_cart)

    return {
        "y_cart_norm": y_norm,
        "r_cart": r,
    }
