"""composable k-space loss, focal weighting

envelope normalization is assumed to be applied at the data level
(targets and predictions are already in envelope-divided units),
so this module only handles dcf and focal weighting on the residual.
"""

import torch


def _residual_magsq(y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """|r|^2 per sample, complex or (N, 2*C) reim-interleaved"""
    if torch.is_complex(y_pred):
        return (y_pred - y_target).abs() ** 2
    if y_pred.ndim == 2 and y_pred.shape[-1] % 2 == 0:
        diff = y_pred - y_target
        return (diff ** 2).sum(dim=-1)
    raise ValueError(f"unsupported y_pred shape: {y_pred.shape}")


def composable_kspace_loss(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
    *,
    dcf: torch.Tensor = None,
    use_dcf: bool = False,
    dcf_power: float = 0.0,
    use_focal: bool = False,
    focal_alpha: float = 1.0,
    focal_normalize: bool = True,
    focal_log_matrix: bool = False,
    focal_warmup_progress: float = 1.0,
    return_diagnostics: bool = False,
    eps: float = 1e-8,
):
    """composable dcf, focal weighted loss on complex k-space residual

    reduction is sum/sum (matches existing weighted_complex_mse), which
    collapses to plain mse when all weights = 1.

    focal weights are detached from the graph. warmup ramps linearly from
    uniform weights (progress=0) to |r|^alpha weights (progress=1).
    """
    r_magsq = _residual_magsq(y_pred, y_target)

    # focal weights, content adaptive, no gradient
    if use_focal:
        r_mag = r_magsq.detach().sqrt()
        base = r_mag ** focal_alpha
        w_focal_target = torch.log1p(base) if focal_log_matrix else base
        if focal_normalize:
            w_focal_target = w_focal_target / (w_focal_target.mean() + eps)
        p = float(min(max(focal_warmup_progress, 0.0), 1.0))
        if p < 1.0:
            w_focal = (1.0 - p) + p * w_focal_target
        else:
            w_focal = w_focal_target
    else:
        w_focal = torch.ones_like(r_magsq)

    # dcf weights, geometry
    if use_dcf and dcf is not None and dcf_power != 0.0:
        w_dcf = torch.clamp(dcf, min=eps) ** dcf_power
    else:
        w_dcf = torch.ones_like(r_magsq)

    w = (w_focal * w_dcf).detach()
    weighted = w * r_magsq
    loss = weighted.sum() / (w.sum() + eps)

    if not return_diagnostics:
        return loss

    with torch.no_grad():
        diag = {
            "w_focal_mean": float(w_focal.mean()),
            "w_focal_max":  float(w_focal.max()),
            "w_focal_min":  float(w_focal.min()),
            "w_focal_p99":  float(torch.quantile(w_focal, 0.99)),
        }
        n = weighted.numel()
        if n > 0:
            top_n = max(1, n // 100)
            top_vals, _ = weighted.topk(top_n)
            total = weighted.sum() + eps
            diag["top1pct_loss_frac"] = float(top_vals.sum() / total)
    return loss, diag


def split_residual_norm_by_k(r_magsq: torch.Tensor, kcoords: torch.Tensor):
    """|r| split into low and high |k|, median cut"""
    with torch.no_grad():
        r = torch.sqrt(kcoords[:, 0] ** 2 + kcoords[:, 1] ** 2)
        med = torch.median(r)
        lo = r <= med
        hi = ~lo
        return {
            "resid_norm_lowk":  float(r_magsq[lo].mean().sqrt()) if lo.any() else 0.0,
            "resid_norm_highk": float(r_magsq[hi].mean().sqrt()) if hi.any() else 0.0,
            "k_median": float(med),
        }
