"""nikmri style helpers"""
import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors


class DynamicSliceDataset(Dataset):
    """nikmri pointwise dataset"""

    def __init__(self, coords: torch.Tensor, targets: torch.Tensor):
        if coords.ndim != 2 or coords.shape[-1] != 4:
            raise ValueError(f"coords must have shape (N, 4), got {coords.shape}")
        if targets.ndim != 2 or targets.shape[-1] != 1:
            raise ValueError(f"targets must have shape (N, 1), got {targets.shape}")
        self.coords = coords.contiguous()
        self.targets = targets.contiguous()

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, index):
        return {
            "coords": self.coords[index],
            "targets": self.targets[index],
        }


def normalized_time_coords(n_frames, *, device=None, dtype=torch.float32):
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if n_frames == 1:
        return torch.zeros(1, device=device, dtype=dtype)
    return torch.linspace(
        -1.0 + 1.0 / n_frames,
        1.0 - 1.0 / n_frames,
        n_frames,
        device=device,
        dtype=dtype,
    )


def build_nik_mri_dce_dataset(
    k_img_space,
    traj_t,
    scales,
    *,
    z_slice_idx,
    n_slices=None,
    target_device="cpu",
):
    """multicoil dynamic slice"""

    if k_img_space.ndim != 5:
        raise ValueError(
            f"k_img_space must have shape (T, S_z, C, Z, RO), got {k_img_space.shape}"
        )

    sx, sy, _ = scales
    T, n_ro_per_slice, C, n_z_slices, RO = k_img_space.shape
    S = traj_t.shape[1]
    if n_slices is None:
        n_slices = n_z_slices

    z_slice_idx = int(max(0, min(int(z_slice_idx), n_z_slices - 1)))
    traj_device = traj_t.device
    data_device = k_img_space.device
    target_device = torch.device(target_device)

    indices = torch.arange(0, S, n_slices, device=traj_device)
    if indices.numel() != n_ro_per_slice:
        raise ValueError(
            f"Expected {n_ro_per_slice} readouts per slice, got {indices.numel()}"
        )

    time_coords = normalized_time_coords(T, device=traj_device, dtype=traj_t.dtype)
    coil_coords = torch.linspace(-1.0, 1.0, C, device=traj_device, dtype=traj_t.dtype)

    points_per_time_coil = n_ro_per_slice * RO
    n_total = T * C * points_per_time_coil
    coords = torch.empty((n_total, 4), dtype=torch.float32, device=data_device)
    targets = torch.empty((n_total, 1), dtype=k_img_space.dtype, device=data_device)

    offset = 0
    for t_idx in range(T):
        kx = (traj_t[t_idx, indices, 0, :] / sx).reshape(-1).to(data_device)
        ky = (traj_t[t_idx, indices, 1, :] / sy).reshape(-1).to(data_device)
        t_val = float(time_coords[t_idx].item())

        for coil_idx in range(C):
            sl = slice(offset, offset + points_per_time_coil)
            coords[sl, 0] = t_val
            coords[sl, 1] = float(coil_coords[coil_idx].item())
            coords[sl, 2] = kx
            coords[sl, 3] = ky
            targets[sl, 0] = k_img_space[t_idx, :, coil_idx, z_slice_idx, :].reshape(-1)
            offset += points_per_time_coil

    scale = torch.abs(targets).amax().clamp_min(1e-8)
    targets = targets / scale

    coords = coords.to(target_device)
    targets = targets.to(target_device)

    meta = {
        "z_slice_idx": z_slice_idx,
        "n_frames": T,
        "n_coils": C,
        "n_slices": n_z_slices,
        "n_ro_per_slice": n_ro_per_slice,
        "points_per_time_coil": points_per_time_coil,
        "n_total": n_total,
        "scale": float(scale.item()),
        "time_coords": time_coords.detach().cpu(),
        "coil_coords": coil_coords.detach().cpu(),
    }
    return coords, targets, meta


def prepare_coil_sensitivity_maps(coil_maps, z_slice_idx, *, n_coils=None):
    """zslice coil maps, rss norm"""

    if coil_maps is None:
        return None

    sens = np.asarray(coil_maps)
    if sens.ndim == 4:
        if n_coils is not None and sens.shape[0] == n_coils:
            z_slice_idx = int(max(0, min(int(z_slice_idx), sens.shape[1] - 1)))
            sens = sens[:, z_slice_idx, :, :]
        elif n_coils is not None and sens.shape[1] == n_coils:
            z_slice_idx = int(max(0, min(int(z_slice_idx), sens.shape[0] - 1)))
            sens = sens[z_slice_idx, :, :, :]
        elif sens.shape[0] <= sens.shape[1]:
            z_slice_idx = int(max(0, min(int(z_slice_idx), sens.shape[1] - 1)))
            sens = sens[:, z_slice_idx, :, :]
        else:
            z_slice_idx = int(max(0, min(int(z_slice_idx), sens.shape[0] - 1)))
            sens = sens[z_slice_idx, :, :, :]
    elif sens.ndim != 3:
        raise ValueError(f"Unsupported coil map shape: {sens.shape}")

    if n_coils is not None and sens.shape[0] != n_coils:
        if sens.shape[-1] == n_coils:
            sens = np.moveaxis(sens, -1, 0)
        else:
            raise ValueError(
                f"Could not infer coil dimension for coil maps with shape {sens.shape}"
            )

    sens = sens.astype(np.complex64, copy=False)
    rss = np.sqrt(np.sum(np.abs(sens) ** 2, axis=0, keepdims=True)) + 1e-8
    return sens / rss


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(kspace, dim=(-2, -1)),
            norm="ortho",
        ),
        dim=(-2, -1),
    )


def coil_combine(coil_images: torch.Tensor, sensitivity_maps=None) -> torch.Tensor:
    if sensitivity_maps is None:
        return torch.sqrt((coil_images.abs() ** 2).sum(dim=1))

    sens = torch.as_tensor(
        sensitivity_maps,
        device=coil_images.device,
        dtype=coil_images.dtype,
    )
    if sens.ndim == 3:
        sens = sens.unsqueeze(0)
    return torch.sum(coil_images * torch.conj(sens), dim=1)


@torch.no_grad()
def predict_cartesian_kspace(
    model,
    *,
    nt,
    nc,
    nx,
    ny,
    device,
    chunk_size=131072,
    time_coords=None,
    coil_coords=None,
):
    """cartesian grid recon"""

    device = torch.device(device)
    if time_coords is None:
        time_coords = normalized_time_coords(nt, device=device)
    else:
        time_coords = torch.as_tensor(time_coords, device=device, dtype=torch.float32)

    if coil_coords is None:
        coil_coords = torch.linspace(-1.0, 1.0, nc, device=device, dtype=torch.float32)
    else:
        coil_coords = torch.as_tensor(coil_coords, device=device, dtype=torch.float32)

    kxs = torch.linspace(-1.0, 1.0 - 2.0 / nx, nx, device=device)
    kys = torch.linspace(-1.0, 1.0 - 2.0 / ny, ny, device=device)
    kx_grid, ky_grid = torch.meshgrid(kxs, kys, indexing="ij")
    flat_kx = kx_grid.reshape(-1)
    flat_ky = ky_grid.reshape(-1)
    mask = (kx_grid.square() + ky_grid.square()) < 1.0

    model_was_training = model.training
    model.eval()

    kpred = torch.empty((nt, nc, nx, ny), dtype=torch.complex64, device=device)
    n_points = flat_kx.numel()

    for t_idx, t_val in enumerate(time_coords):
        t_col = torch.full((n_points,), float(t_val.item()), device=device)
        for c_idx, c_val in enumerate(coil_coords):
            c_col = torch.full((n_points,), float(c_val.item()), device=device)
            coords = torch.stack([t_col, c_col, flat_kx, flat_ky], dim=1)

            preds = []
            for start in range(0, n_points, chunk_size):
                end = min(start + chunk_size, n_points)
                preds.append(model(coords[start:end]).squeeze(-1))

            pred = torch.cat(preds, dim=0).reshape(nx, ny)
            pred[~mask] = 0
            kpred[t_idx, c_idx] = pred

    if model_was_training:
        model.train()

    return kpred


def _to_uint8(arr, *, log_scale=False):
    arr = np.asarray(arr)
    if log_scale:
        arr = np.log(np.abs(arr) + 1e-4)
    else:
        arr = np.abs(arr)

    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def make_recon_videos(kpred: torch.Tensor, combined: torch.Tensor):
    """wandb videos, kspace and image"""

    kpred_np = kpred.detach().cpu().numpy()
    combined_np = combined.detach().cpu().numpy()
    center_coil = min(kpred_np.shape[1] // 2, kpred_np.shape[1] - 1)

    k_mag = _to_uint8(kpred_np[:, center_coil], log_scale=True)[:, None, :, :]
    combined_mag = _to_uint8(combined_np)[:, None, :, :]

    phase = np.angle(combined_np)
    mapper = cm.ScalarMappable(
        norm=colors.Normalize(vmin=-np.pi, vmax=np.pi),
        cmap="viridis",
    )
    phase_rgb = np.stack(
        [mapper.to_rgba(frame, bytes=True)[..., :3] for frame in phase],
        axis=0,
    )
    phase_rgb = np.moveaxis(phase_rgb, -1, 1).astype(np.uint8)

    return {
        "k_mag": k_mag,
        "combined_mag": combined_mag,
        "combined_phase": phase_rgb,
    }
