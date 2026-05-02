"""nik kspace losses"""
import torch
import torch.nn.functional as F


def _ensure_complex(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Expected last dimension of size 2 for real-imag tensor, got {x.shape}")
    return torch.view_as_complex(x.contiguous())


class HDRLossFF(torch.nn.Module):
    """nikmri hdr loss, gaussian ff regularizer"""

    def __init__(self, sigma=1.0, eps=1e-2, factor=0.0):
        super().__init__()
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.factor = float(factor)

    def forward(self, input, target, kcoords, weights=None, reduce=True):
        input_c = _ensure_complex(input)
        target_c = _ensure_complex(target)

        if kcoords.shape[-1] < 4:
            raise ValueError(
                "HDRLossFF expects coordinates ordered as [t, coil, kx, ky]"
            )

        dist_to_center2 = kcoords[..., 2] ** 2 + kcoords[..., 3] ** 2
        filter_value = torch.exp(-dist_to_center2 / (2 * self.sigma ** 2))
        while filter_value.ndim < input_c.ndim:
            filter_value = filter_value.unsqueeze(-1)

        denom = input_c.detach().abs() + self.eps
        error = input_c - target_c
        loss = (error.abs() / denom) ** 2

        if weights is not None:
            while weights.ndim < loss.ndim:
                weights = weights.unsqueeze(-1)
            loss = loss * weights

        reg_error = input_c - input_c * filter_value
        reg = self.factor * (reg_error.abs() / denom) ** 2

        if reduce:
            return loss.mean() + reg.mean(), reg.mean()
        return loss, reg


def mse_loss(y_pred, y_true):
    """plain mse"""
    return F.mse_loss(y_pred, y_true)


def wmse_sqrt_loss(y_pred, y_true, eps=1e-8):
    """sqrt magnitude weighted mse"""
    mag = torch.sqrt(y_true[:, 0] ** 2 + y_true[:, 1] ** 2 + eps)
    w = 1.0 / torch.sqrt(mag + eps)
    return (w.unsqueeze(1) * (y_pred - y_true) ** 2).mean()


def wmse_inv_loss(y_pred, y_true, eps=1e-8):
    """inverse magnitude weighted mse"""
    mag = torch.sqrt(y_true[:, 0] ** 2 + y_true[:, 1] ** 2 + eps)
    w = 1.0 / (mag + eps)
    return (w.unsqueeze(1) * (y_pred - y_true) ** 2).mean()


def relative_mse_loss(y_pred, y_true, eps=1e-8):
    """relative mse"""
    mag_sq = y_true[:, 0] ** 2 + y_true[:, 1] ** 2 + eps
    err_sq = (y_pred - y_true) ** 2
    return (err_sq.sum(dim=1) / mag_sq).mean()


def log_mse_loss(y_pred, y_true, eps=1e-8):
    """log magnitude mse, phase mse"""
    pred_mag = torch.sqrt(y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2 + eps)
    true_mag = torch.sqrt(y_true[:, 0] ** 2 + y_true[:, 1] ** 2 + eps)
    mag_loss = F.mse_loss(torch.log(pred_mag), torch.log(true_mag))

    pred_phase = torch.atan2(y_pred[:, 1], y_pred[:, 0])
    true_phase = torch.atan2(y_true[:, 1], y_true[:, 0])
    # wrap aware phase
    phase_diff = pred_phase - true_phase
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_loss = (phase_diff ** 2).mean()

    return mag_loss + phase_loss


# density weighted loss

def density_weighted_mse_loss(y_pred, y_true, k_coords, eps=1e-8):
    """kr density weighted mse"""
    kr = torch.sqrt(k_coords[:, 0] ** 2 + k_coords[:, 1] ** 2 + eps)
    err_sq = (y_pred - y_true) ** 2
    return (kr.unsqueeze(1) * err_sq).mean()


# dc consistency loss, polar net

def dc_consistency_loss(model, n_theta=64):
    """dc angular variance"""
    dc_preds = model.dc_predictions(n_theta)
    return dc_preds.var(dim=0).sum()


# conjugate symmetry loss

def conjugate_symmetry_loss(y_pred, y_pred_neg, eps=1e-8):
    """real image conjugate symmetry"""
    # conj of negative kspace
    conj_neg = torch.stack([y_pred_neg[:, 0], -y_pred_neg[:, 1]], dim=1)
    return F.mse_loss(y_pred, conj_neg)


def conjugate_symmetry_loss_from_model(model, k_coords, eps=1e-8):
    """model wrapper, conjugate symmetry"""
    y_pred = model(k_coords)
    y_pred_neg = model(-k_coords)
    return conjugate_symmetry_loss(y_pred, y_pred_neg)


# polar combined loss

def polar_kspace_loss(
    y_pred, y_true, k_coords,
    model=None,
    base_loss="mse",
    density_weight=1.0,
    dc_weight=0.01,
    conj_weight=0.01,
    n_dc_theta=64,
):
    """combined polar loss"""
    # base loss
    base_fn = LOSS_FNS[base_loss]
    loss = base_fn(y_pred, y_true)

    # density term
    if density_weight > 0:
        loss = loss + density_weight * density_weighted_mse_loss(y_pred, y_true, k_coords)

    # dc term
    if dc_weight > 0 and model is not None and hasattr(model, "dc_predictions"):
        loss = loss + dc_weight * dc_consistency_loss(model, n_dc_theta)

    # conjugate term
    if conj_weight > 0 and model is not None:
        loss = loss + conj_weight * conjugate_symmetry_loss_from_model(model, k_coords)

    return loss


# loss registry
LOSS_FNS = {
    "mse": mse_loss,
    "wmse_sqrt": wmse_sqrt_loss,
    "wmse_inv": wmse_inv_loss,
    "relative_mse": relative_mse_loss,
    "log_mse": log_mse_loss,
    "density_mse": density_weighted_mse_loss,
}


def get_loss_fn(name: str):
    """loss lookup"""
    if name not in LOSS_FNS:
        raise ValueError(f"Unknown loss_type '{name}'. Choose from: {list(LOSS_FNS.keys())}")
    return LOSS_FNS[name]
