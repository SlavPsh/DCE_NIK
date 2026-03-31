"""
losses.py — Complex-valued k-space losses with density weighting.

Provides:
  - weighted_complex_mse: DCF-weighted complex MSE with power tempering
  - shell_balanced_complex_mse: equal weighting per radial shell
"""

import torch
from kspace_normalization import compute_radius


def weighted_complex_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor = None,
    power: float = 0.7,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Density-weighted complex MSE loss.

    Computes pointwise |pred - target|^2, optionally weighted by DCF^power.
    Power tempering (default 0.7) avoids over-emphasizing noisy outer k-space.

    Args:
        pred: (N,) complex or (N, 2) real.
        target: (N,) complex or (N, 2) real.
        weights: (N,) optional DCF weights (should have mean~1).
        power: exponent on weights. 1.0 = full DCF, 0.0 = uniform.
        normalize: if True, normalize by sum of effective weights.
        eps: numerical stability.

    Returns:
        Scalar loss tensor.
    """
    # Compute pointwise squared error
    if torch.is_complex(pred):
        err_sq = (pred - target).abs() ** 2
    elif pred.ndim == 2 and pred.shape[-1] == 2:
        diff = pred - target
        err_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2
    else:
        raise ValueError(f"Unsupported pred shape: {pred.shape}")

    if weights is not None:
        w_eff = torch.clamp(weights, min=eps) ** power
        if normalize:
            loss = (w_eff * err_sq).sum() / (w_eff.sum() + eps)
        else:
            loss = (w_eff * err_sq).mean()
    else:
        loss = err_sq.mean()

    return loss


def shell_balanced_complex_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    radii: torch.Tensor,
    n_bins: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Shell-balanced complex MSE: equal weighting per radial shell.

    Splits samples into radial bins, computes mean MSE per non-empty shell,
    then averages shell losses equally. Useful when outer k-space is underfit.

    Args:
        pred: (N,) complex or (N, 2) real.
        target: (N,) complex or (N, 2) real.
        radii: (N,) radial distances.
        n_bins: number of radial shells.
        eps: numerical stability.

    Returns:
        Scalar loss tensor.
    """
    if torch.is_complex(pred):
        err_sq = (pred - target).abs() ** 2
    elif pred.ndim == 2 and pred.shape[-1] == 2:
        diff = pred - target
        err_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2
    else:
        raise ValueError(f"Unsupported pred shape: {pred.shape}")

    r_max = radii.max().item()
    bin_idx = torch.clamp(
        (radii / (r_max + eps) * n_bins).long(), 0, n_bins - 1
    )

    shell_losses = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            shell_losses.append(err_sq[mask].mean())

    if len(shell_losses) == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.stack(shell_losses).mean()
