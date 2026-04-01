# nik_metrics.py
"""
Image-space quality metrics for NIK SIREN baseline.

Provides PSNR, SSIM, and NRMSE between reconstructed and reference images.
All functions accept 2-D numpy arrays (single-image) or batches via the
convenience wrapper ``compute_image_metrics``.

Also provides perceptual image quality metrics (DISTS, HaarPSI, VSI, LPIPS)
validated for MRI via the ``piq`` library (Kastryulin et al. 2022).

Also provides k-space validation metrics broken down by spoke and frame.
"""
import numpy as np
import torch
from scipy.ndimage import uniform_filter

_PIQ_DISTS_METRIC = None


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
# Perceptual image quality metrics (piq library)
# ---------------------------------------------------------------------------

def _to_piq_tensor(img: np.ndarray, n_channels: int = 1) -> torch.Tensor:
    """Convert 2-D numpy array to (1, C, H, W) float32 tensor on CPU."""
    t = torch.from_numpy(np.asarray(img, dtype=np.float32))
    t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    if n_channels == 3:
        t = t.repeat(1, 3, 1, 1)
    return t


@torch.no_grad()
def compute_perceptual_metrics(
    img_pred: np.ndarray,
    img_ref: np.ndarray,
) -> dict:
    """
    Evaluation-only perceptual image quality metrics. NOT for use as training loss.

    Computed with no_grad on CPU. Do not backpropagate through these.

    Metrics chosen based on Kastryulin et al. 2022 — "Image Quality Assessment
    for Magnetic Resonance Imaging":
      - DISTS (best for MRI, SRCC=0.76 on artifacts) — lower is better
      - HaarPSI (second best, wavelet-based) — higher is better
      - VSI (third best, saliency-based) — higher is better
      - PSNR, SSIM from piq as baselines

    Parameters
    ----------
    img_pred, img_ref : 2-D numpy arrays (magnitude images, any value range).

    Returns
    -------
    dict with metric names as keys and float values.
    """
    import piq
    global _PIQ_DISTS_METRIC

    img_pred = np.asarray(img_pred, dtype=np.float64)
    img_ref = np.asarray(img_ref, dtype=np.float64)

    # Normalize both to [0, 1] using the same scale
    vmax = max(img_pred.max(), img_ref.max())
    if vmax == 0:
        return {
            "PSNR": 0.0, "SSIM": 0.0, "DISTS": 1.0,
            "HaarPSI": 0.0, "VSI": 0.0,
        }
    img_pred_n = img_pred / vmax
    img_ref_n = img_ref / vmax

    # Build tensors — all on CPU
    pred_1ch = _to_piq_tensor(img_pred_n, n_channels=1)
    ref_1ch = _to_piq_tensor(img_ref_n, n_channels=1)
    pred_3ch = _to_piq_tensor(img_pred_n, n_channels=3)
    ref_3ch = _to_piq_tensor(img_ref_n, n_channels=3)

    results = {}

    # PSNR (higher is better)
    results["PSNR"] = float(piq.psnr(pred_1ch, ref_1ch, data_range=1.0).item())

    # SSIM (higher is better)
    results["SSIM"] = float(piq.ssim(pred_1ch, ref_1ch, data_range=1.0).item())

    # HaarPSI (higher is better) — works on grayscale natively
    results["HaarPSI"] = float(piq.haarpsi(pred_1ch, ref_1ch, data_range=1.0).item())

    # DISTS (lower is better) — needs 3-channel
    if _PIQ_DISTS_METRIC is None:
        _PIQ_DISTS_METRIC = piq.DISTS()
    results["DISTS"] = float(_PIQ_DISTS_METRIC(pred_3ch, ref_3ch).item())

    # VSI (higher is better) — needs 3-channel
    results["VSI"] = float(piq.vsi(pred_3ch, ref_3ch, data_range=1.0).item())

    return results


# ---------------------------------------------------------------------------
# Formatting and comparison utilities
# ---------------------------------------------------------------------------

# Direction indicators: True = higher is better
_METRIC_DIRECTION = {
    "psnr_db": True, "ssim": True, "nrmse": False,
    "PSNR": True, "SSIM": True, "DISTS": False,
    "HaarPSI": True, "VSI": True,
}


def format_metrics_table(metrics_dict: dict, label: str = "") -> str:
    """
    Format a metrics dict into a readable string with direction arrows.

    Parameters
    ----------
    metrics_dict : dict
        May contain keys from compute_image_metrics (psnr_db, ssim, nrmse)
        and/or compute_perceptual_metrics (PSNR, SSIM, DISTS, HaarPSI, VSI).
    label : str, optional
        Context label, e.g. "vs NUFFT proxy (not ground truth)" or
        "vs Ground Truth".

    Returns
    -------
    str — formatted table.
    """
    basic_keys = ["psnr_db", "ssim", "nrmse"]
    perceptual_keys = ["DISTS", "HaarPSI", "VSI", "PSNR", "SSIM"]

    lines = []

    basic_present = [k for k in basic_keys if k in metrics_dict]
    if basic_present:
        header = f"Basic Metrics{f' ({label})' if label else ''}:"
        lines.append(header)
        for k in basic_present:
            arrow = "\u2191" if _METRIC_DIRECTION.get(k, True) else "\u2193"
            v = metrics_dict[k]
            unit = " dB" if k == "psnr_db" else ""
            lines.append(f"  {k:8s} {arrow}  {v:.4f}{unit}")

    perc_present = [k for k in perceptual_keys if k in metrics_dict]
    if perc_present:
        header = f"Perceptual Metrics{f' ({label})' if label else ''}:"
        lines.append(header)
        for k in perc_present:
            arrow = "\u2191" if _METRIC_DIRECTION.get(k, True) else "\u2193"
            v = metrics_dict[k]
            unit = " dB" if k == "PSNR" else ""
            lines.append(f"  {k:8s} {arrow}  {v:.4f}{unit}")

    return "\n".join(lines)


def compare_reconstructions(
    metrics_list: list,
    labels: list,
    csv_path: str = None,
) -> str:
    """
    Compare multiple reconstruction methods side-by-side.

    Parameters
    ----------
    metrics_list : list of dicts — one per method.
    labels : list of str — method names.
    csv_path : optional path to save CSV.

    Returns
    -------
    str — formatted comparison table with best values highlighted.
    """
    all_keys = []
    for m in metrics_list:
        for k in m:
            if k not in all_keys:
                all_keys.append(k)

    # Column widths
    label_w = max(len(l) for l in labels)
    col_w = max(12, label_w + 2)

    header = f"{'Metric':>10s} | " + " | ".join(f"{l:>{col_w}s}" for l in labels)
    sep = "-" * len(header)

    lines = [header, sep]
    for k in all_keys:
        vals = [m.get(k) for m in metrics_list]
        higher_better = _METRIC_DIRECTION.get(k, True)

        # Find best index
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        best_i = None
        if valid:
            best_i = (max if higher_better else min)(valid, key=lambda x: x[1])[0]

        row_parts = []
        for i, v in enumerate(vals):
            if v is None:
                s = "--"
            else:
                s = f"{v:.4f}"
                if i == best_i:
                    s = f"*{s}*"
            row_parts.append(f"{s:>{col_w}s}")

        arrow = "\u2191" if higher_better else "\u2193"
        lines.append(f"{k:>8s} {arrow} | " + " | ".join(row_parts))

    table = "\n".join(lines)

    if csv_path is not None:
        with open(csv_path, "w") as f:
            f.write("metric," + ",".join(labels) + "\n")
            for k in all_keys:
                vals = [str(m.get(k, "")) for m in metrics_list]
                f.write(k + "," + ",".join(vals) + "\n")

    return table


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
