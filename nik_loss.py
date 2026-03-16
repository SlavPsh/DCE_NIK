"""
nik_loss.py -- Loss functions for NIK k-space fitting.

All loss functions take (y_pred, y_true) with shape (N, 2) [Re, Im]
and return a scalar loss.
"""
import torch
import torch.nn.functional as F


def mse_loss(y_pred, y_true):
    """Plain MSE — equal weight on every k-space sample."""
    return F.mse_loss(y_pred, y_true)


def wmse_sqrt_loss(y_pred, y_true, eps=1e-8):
    """Weighted MSE with w = 1 / sqrt(|y| + eps).

    Soft down-weighting of high-magnitude (center) samples.
    """
    mag = torch.sqrt(y_true[:, 0] ** 2 + y_true[:, 1] ** 2 + eps)
    w = 1.0 / torch.sqrt(mag + eps)
    return (w.unsqueeze(1) * (y_pred - y_true) ** 2).mean()


def wmse_inv_loss(y_pred, y_true, eps=1e-8):
    """Weighted MSE with w = 1 / (|y| + eps).

    Stronger down-weighting of center than sqrt variant.
    """
    mag = torch.sqrt(y_true[:, 0] ** 2 + y_true[:, 1] ** 2 + eps)
    w = 1.0 / (mag + eps)
    return (w.unsqueeze(1) * (y_pred - y_true) ** 2).mean()


def relative_mse_loss(y_pred, y_true, eps=1e-8):
    """Relative MSE: |pred - true|^2 / (|true|^2 + eps).

    Each sample's error is relative to its own magnitude.
    Most aggressive equalisation.
    """
    mag_sq = y_true[:, 0] ** 2 + y_true[:, 1] ** 2 + eps
    err_sq = (y_pred - y_true) ** 2
    return (err_sq.sum(dim=1) / mag_sq).mean()


def log_mse_loss(y_pred, y_true, eps=1e-8):
    """MSE in log-magnitude space (phase fitted linearly).

    Compresses the dynamic range of k-space magnitudes.
    Loss = MSE(log|pred|, log|true|) + MSE(angle(pred), angle(true))
    """
    pred_mag = torch.sqrt(y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2 + eps)
    true_mag = torch.sqrt(y_true[:, 0] ** 2 + y_true[:, 1] ** 2 + eps)
    mag_loss = F.mse_loss(torch.log(pred_mag), torch.log(true_mag))

    pred_phase = torch.atan2(y_pred[:, 1], y_pred[:, 0])
    true_phase = torch.atan2(y_true[:, 1], y_true[:, 0])
    # Wrap-aware phase difference
    phase_diff = pred_phase - true_phase
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_loss = (phase_diff ** 2).mean()

    return mag_loss + phase_loss


# ---------------------------------------------------------------------------
# Density-weighted loss
# ---------------------------------------------------------------------------

def density_weighted_mse_loss(y_pred, y_true, k_coords, eps=1e-8):
    """Density-weighted MSE: weight = |kr| = sqrt(kx^2 + ky^2).

    Up-weights outer k-space (high frequency) samples, which are sparser
    in radial trajectories and contribute more to image detail.

    Args:
        y_pred:   (N, 2) predicted [Re, Im]
        y_true:   (N, 2) target    [Re, Im]
        k_coords: (N, 2) [kx, ky] coordinates
    """
    kr = torch.sqrt(k_coords[:, 0] ** 2 + k_coords[:, 1] ** 2 + eps)
    err_sq = (y_pred - y_true) ** 2  # (N, 2)
    return (kr.unsqueeze(1) * err_sq).mean()


# ---------------------------------------------------------------------------
# DC consistency loss (for PolarKSpaceNet)
# ---------------------------------------------------------------------------

def dc_consistency_loss(model, n_theta=64):
    """Variance of predictions at DC (k=0) across different angles.

    At the center of k-space (s=0), the signal should be the same
    regardless of the spoke angle. This loss penalizes angular variation
    at DC.

    Args:
        model: PolarKSpaceNet instance (must have dc_predictions method)
        n_theta: number of angles to sample
    Returns:
        scalar loss = Var(Re predictions at DC) + Var(Im predictions at DC)
    """
    dc_preds = model.dc_predictions(n_theta)  # (n_theta, 2)
    return dc_preds.var(dim=0).sum()


# ---------------------------------------------------------------------------
# Conjugate symmetry loss
# ---------------------------------------------------------------------------

def conjugate_symmetry_loss(y_pred, y_pred_neg, eps=1e-8):
    """Enforce S(-k) = S*(k) for real-valued images.

    For each sample, compares the prediction at k with the complex conjugate
    of the prediction at -k:
        loss = |S(k) - conj(S(-k))|^2

    S*(k) = (Re, -Im), so S(-k) should equal (Re(k), -Im(k)).

    Args:
        y_pred:     (N, 2) predictions at k = (kx, ky)
        y_pred_neg: (N, 2) predictions at -k = (-kx, -ky)
    """
    # conj(S(-k)) = (Re(-k), -Im(-k))
    conj_neg = torch.stack([y_pred_neg[:, 0], -y_pred_neg[:, 1]], dim=1)
    return F.mse_loss(y_pred, conj_neg)


def conjugate_symmetry_loss_from_model(model, k_coords, eps=1e-8):
    """Compute conjugate symmetry loss by evaluating model at k and -k.

    Convenience wrapper that handles the forward passes.

    Args:
        model: network with forward(k_coords) → (N, 2)
        k_coords: (N, 2) [kx, ky]
    """
    y_pred = model(k_coords)
    y_pred_neg = model(-k_coords)
    return conjugate_symmetry_loss(y_pred, y_pred_neg)


# ---------------------------------------------------------------------------
# Combined polar loss
# ---------------------------------------------------------------------------

def polar_kspace_loss(
    y_pred, y_true, k_coords,
    model=None,
    base_loss="mse",
    density_weight=1.0,
    dc_weight=0.01,
    conj_weight=0.01,
    n_dc_theta=64,
):
    """Combined loss for PolarKSpaceNet training.

    L = L_base + density_weight * L_density + dc_weight * L_dc + conj_weight * L_conj

    Args:
        y_pred:   (N, 2) predicted [Re, Im]
        y_true:   (N, 2) target [Re, Im]
        k_coords: (N, 2) [kx, ky]
        model:    PolarKSpaceNet (needed for DC and conjugate losses)
        base_loss: base loss type ('mse', 'wmse_sqrt', etc.)
        density_weight: weight for density-weighted MSE term
        dc_weight: weight for DC consistency term
        conj_weight: weight for conjugate symmetry term
        n_dc_theta: number of angles for DC consistency evaluation
    """
    # Base reconstruction loss
    base_fn = LOSS_FNS[base_loss]
    loss = base_fn(y_pred, y_true)

    # Density-weighted loss
    if density_weight > 0:
        loss = loss + density_weight * density_weighted_mse_loss(y_pred, y_true, k_coords)

    # DC consistency (requires model)
    if dc_weight > 0 and model is not None and hasattr(model, "dc_predictions"):
        loss = loss + dc_weight * dc_consistency_loss(model, n_dc_theta)

    # Conjugate symmetry (requires model)
    if conj_weight > 0 and model is not None:
        loss = loss + conj_weight * conjugate_symmetry_loss_from_model(model, k_coords)

    return loss


# Registry mapping config string → function
LOSS_FNS = {
    "mse": mse_loss,
    "wmse_sqrt": wmse_sqrt_loss,
    "wmse_inv": wmse_inv_loss,
    "relative_mse": relative_mse_loss,
    "log_mse": log_mse_loss,
    "density_mse": density_weighted_mse_loss,
}


def get_loss_fn(name: str):
    """Return loss function by config name."""
    if name not in LOSS_FNS:
        raise ValueError(f"Unknown loss_type '{name}'. Choose from: {list(LOSS_FNS.keys())}")
    return LOSS_FNS[name]
