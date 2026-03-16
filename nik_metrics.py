# nik_metrics.py
"""
Image-space quality metrics for NIK SIREN baseline.

Provides PSNR, SSIM, and NRMSE between reconstructed and reference images.
All functions accept 2-D numpy arrays (single-image) or batches via the
convenience wrapper ``compute_image_metrics``.

Also provides k-space validation metrics broken down by spoke and frame.
"""
import numpy as np
import torch
from scipy.ndimage import uniform_filter


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(img_pred: np.ndarray, img_ref: np.ndarray, data_range: float = None) -> float:
    """
    Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(data_range^2 / MSE)

    Parameters
    ----------
    img_pred : 2-D array  – reconstructed image (magnitude).
    img_ref  : 2-D array  – reference / ground-truth image (magnitude).
    data_range : float, optional
        Dynamic range of the reference image.  When *None* it is set to
        ``img_ref.max() - img_ref.min()``.

    Returns
    -------
    float – PSNR in dB.  Returns ``inf`` when images are identical.
    """
    img_pred = np.asarray(img_pred, dtype=np.float64)
    img_ref = np.asarray(img_ref, dtype=np.float64)

    if data_range is None:
        data_range = float(img_ref.max() - img_ref.min())

    mse = np.mean((img_pred - img_ref) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


# ---------------------------------------------------------------------------
# SSIM  (Wang et al. 2004, simplified single-scale)
# ---------------------------------------------------------------------------

def ssim(
    img_pred: np.ndarray,
    img_ref: np.ndarray,
    data_range: float = None,
    win_size: int = 7,
) -> float:
    """
    Structural Similarity Index (mean SSIM over the image).

    Uses a uniform (box) filter of size ``win_size`` for the local statistics,
    following the Wang et al. 2004 formulation:

        SSIM(x,y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
                    / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))

    Parameters
    ----------
    img_pred, img_ref : 2-D arrays of same shape.
    data_range : float, optional – defaults to max − min of img_ref.
    win_size : int – side length of the averaging window (must be odd).

    Returns
    -------
    float – mean SSIM in [−1, 1].
    """
    img_pred = np.asarray(img_pred, dtype=np.float64)
    img_ref = np.asarray(img_ref, dtype=np.float64)

    if data_range is None:
        data_range = float(img_ref.max() - img_ref.min())

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    uf = lambda x: uniform_filter(x, size=win_size, mode="reflect")

    mu_x = uf(img_pred)
    mu_y = uf(img_ref)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = uf(img_pred ** 2) - mu_x_sq
    sigma_y_sq = uf(img_ref ** 2) - mu_y_sq
    sigma_xy = uf(img_pred * img_ref) - mu_xy

    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = num / den

    # Crop border (half-window) to avoid edge effects before averaging
    pad = win_size // 2
    if pad > 0:
        ssim_map = ssim_map[pad:-pad, pad:-pad]

    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# NRMSE
# ---------------------------------------------------------------------------

def nrmse(
    img_pred: np.ndarray,
    img_ref: np.ndarray,
    normalization: str = "euclidean",
) -> float:
    """
    Normalized Root Mean Square Error.

    Parameters
    ----------
    img_pred, img_ref : 2-D arrays of same shape.
    normalization : str
        ``"euclidean"`` – RMSE / ||img_ref||_2  (norm of the reference).
        ``"min-max"``   – RMSE / (max − min) of the reference.
        ``"mean"``      – RMSE / mean of the reference.

    Returns
    -------
    float – NRMSE (lower is better, 0 = identical).
    """
    img_pred = np.asarray(img_pred, dtype=np.float64)
    img_ref = np.asarray(img_ref, dtype=np.float64)

    rmse_val = np.sqrt(np.mean((img_pred - img_ref) ** 2))

    if normalization == "euclidean":
        denom = np.sqrt(np.mean(img_ref ** 2))
    elif normalization == "min-max":
        denom = float(img_ref.max() - img_ref.min())
    elif normalization == "mean":
        denom = float(np.abs(img_ref.mean()))
    else:
        raise ValueError(f"Unknown normalization: {normalization!r}")

    if denom == 0:
        return 0.0 if rmse_val == 0 else float("inf")

    return float(rmse_val / denom)


# ---------------------------------------------------------------------------
# Convenience: compute all three metrics at once
# ---------------------------------------------------------------------------

def compute_image_metrics(
    img_pred: np.ndarray,
    img_ref: np.ndarray,
    data_range: float = None,
    ssim_win_size: int = 7,
    nrmse_norm: str = "euclidean",
) -> dict:
    """
    Compute PSNR, SSIM, and NRMSE between two 2-D images.

    Parameters
    ----------
    img_pred : (H, W) magnitude image from the model.
    img_ref  : (H, W) ground-truth / reference magnitude image.
    data_range : float, optional – defaults to max − min of img_ref.
    ssim_win_size : int – SSIM window size (default 7).
    nrmse_norm : str – NRMSE normalisation mode (default ``"euclidean"``).

    Returns
    -------
    dict with keys ``"psnr_db"``, ``"ssim"``, ``"nrmse"``.
    """
    img_pred = np.asarray(img_pred, dtype=np.float64)
    img_ref = np.asarray(img_ref, dtype=np.float64)

    if data_range is None:
        data_range = float(img_ref.max() - img_ref.min())

    return {
        "psnr_db": psnr(img_pred, img_ref, data_range=data_range),
        "ssim": ssim(img_pred, img_ref, data_range=data_range, win_size=ssim_win_size),
        "nrmse": nrmse(img_pred, img_ref, normalization=nrmse_norm),
    }


# ---------------------------------------------------------------------------
# K-space validation metrics broken down by spoke / frame
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_spoke_mse(
    model,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    spoke_id_all: torch.Tensor,
    idx: torch.Tensor = None,
) -> dict:
    """Compute MSE broken down by spoke index.

    Args:
        model: network with forward(x) → (N, 2)
        x_all:        (N, D) coordinates (first 2 cols used as kx, ky)
        y_all:        (N, 2) target [Re, Im]
        spoke_id_all: (N,) spoke index per point
        idx:          optional subset indices (e.g., val_idx)

    Returns:
        dict with:
            "per_spoke": {spoke_id: mse_float, ...}
            "mean":      mean over all spokes
            "std":       std over all spokes
            "worst_spoke": spoke_id with highest MSE
            "best_spoke":  spoke_id with lowest MSE
    """
    if idx is not None:
        x = x_all[idx][:, :2]
        y = y_all[idx]
        spokes = spoke_id_all[idx]
    else:
        x = x_all[:, :2]
        y = y_all
        spokes = spoke_id_all

    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    spokes = spokes.to(device)

    y_pred = model(x)
    err_sq = (y_pred - y).pow(2).sum(dim=1)  # (N,)

    unique_spokes = torch.unique(spokes)
    per_spoke = {}
    for sp in unique_spokes:
        mask = spokes == sp
        per_spoke[int(sp.item())] = float(err_sq[mask].mean().item())

    mse_values = list(per_spoke.values())
    mse_arr = np.array(mse_values)

    return {
        "per_spoke": per_spoke,
        "mean": float(mse_arr.mean()),
        "std": float(mse_arr.std()),
        "worst_spoke": max(per_spoke, key=per_spoke.get),
        "best_spoke": min(per_spoke, key=per_spoke.get),
    }


@torch.no_grad()
def per_frame_mse(
    model,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    frame_id_all: torch.Tensor,
    idx: torch.Tensor = None,
) -> dict:
    """Compute MSE broken down by time frame.

    Args:
        model: network with forward(x) → (N, 2)
        x_all:         (N, D) coordinates (first 2 cols used as kx, ky)
        y_all:         (N, 2) target [Re, Im]
        frame_id_all:  (N,) frame index per point
        idx:           optional subset indices (e.g., val_idx)

    Returns:
        dict with:
            "per_frame": {frame_id: mse_float, ...}
            "mean":      mean over all frames
            "std":       std over all frames
            "worst_frame": frame_id with highest MSE
            "best_frame":  frame_id with lowest MSE
    """
    if idx is not None:
        x = x_all[idx][:, :2]
        y = y_all[idx]
        frames = frame_id_all[idx]
    else:
        x = x_all[:, :2]
        y = y_all
        frames = frame_id_all

    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    frames = frames.to(device)

    y_pred = model(x)
    err_sq = (y_pred - y).pow(2).sum(dim=1)  # (N,)

    unique_frames = torch.unique(frames)
    per_frame = {}
    for fr in unique_frames:
        mask = frames == fr
        per_frame[int(fr.item())] = float(err_sq[mask].mean().item())

    mse_values = list(per_frame.values())
    mse_arr = np.array(mse_values)

    return {
        "per_frame": per_frame,
        "mean": float(mse_arr.mean()),
        "std": float(mse_arr.std()),
        "worst_frame": max(per_frame, key=per_frame.get),
        "best_frame": min(per_frame, key=per_frame.get),
    }


@torch.no_grad()
def spoke_angle_error_distribution(
    model,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    spoke_id_all: torch.Tensor,
    idx: torch.Tensor = None,
    n_angle_bins: int = 36,
) -> dict:
    """Compute error distribution as a function of spoke angle.

    Bins spokes by their angle theta and reports MSE per bin.
    Useful for diagnosing angular bias in the model.

    Args:
        model: network with forward(x) → (N, 2)
        x_all:        (N, D) coordinates (first 2 cols = kx, ky)
        y_all:        (N, 2) target [Re, Im]
        spoke_id_all: (N,) spoke index per point
        idx:          optional subset indices
        n_angle_bins: number of angular bins over (-pi, pi]

    Returns:
        dict with:
            "bin_edges":  (n_bins+1,) bin edges in radians
            "bin_mse":    (n_bins,) mean MSE per angular bin
            "bin_counts": (n_bins,) number of points per bin
    """
    if idx is not None:
        x = x_all[idx][:, :2]
        y = y_all[idx]
    else:
        x = x_all[:, :2]
        y = y_all

    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)

    y_pred = model(x)
    err_sq = (y_pred - y).pow(2).sum(dim=1).cpu().numpy()  # (N,)

    # Compute angle for each point
    kx = x[:, 0].cpu().numpy()
    ky = x[:, 1].cpu().numpy()
    theta = np.arctan2(ky, kx)  # (-pi, pi]

    bin_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    bin_mse = np.zeros(n_angle_bins)
    bin_counts = np.zeros(n_angle_bins, dtype=int)

    bin_idx = np.digitize(theta, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_angle_bins - 1)

    for b in range(n_angle_bins):
        mask = bin_idx == b
        bin_counts[b] = int(mask.sum())
        if bin_counts[b] > 0:
            bin_mse[b] = float(err_sq[mask].mean())

    return {
        "bin_edges": bin_edges,
        "bin_mse": bin_mse,
        "bin_counts": bin_counts,
    }
