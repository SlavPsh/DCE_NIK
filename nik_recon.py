"""nik recon, datasets and nufft"""
import numpy as np
import torch
import cupy as cp
import cufinufft

from nik_metrics import compute_image_metrics, compute_perceptual_metrics


def make_fixed_frame_zslice_coil_dataset(
    k_img_space,
    traj_t,
    scales,
    dims,
    *,
    y_scale,
    t_fixed: int = 0,
    coil_fixed: int = 0,
    z_slice_idx: int = 0,
    n_slices: int = None,
    compute_device: str = "cuda",
):
    """single frame coil zslice"""
    sx, sy, _ = scales
    T, S, C, RO = dims
    dev_data = k_img_space.device
    dev_compute = torch.device(compute_device)

    assert 0 <= t_fixed < T
    assert 0 <= coil_fixed < C

    if n_slices is None:
        kz_vals = traj_t[t_fixed, :, 2, 0]
        n_slices = len(torch.unique(kz_vals))

    z_slice_idx = int(max(0, min(int(z_slice_idx), int(n_slices - 1))))
    n_ro_per_slice = int(S // n_slices)

    # interleaved readouts
    indices = torch.arange(0, S, n_slices, device=traj_t.device)

    # spoke ro ids
    spoke_ids = torch.arange(n_ro_per_slice, device=dev_data, dtype=torch.long)[:, None].expand(n_ro_per_slice, RO)
    spoke_id_all = spoke_ids.reshape(-1)

    ro_ids = torch.arange(RO, device=dev_data, dtype=torch.long)[None, :].expand(n_ro_per_slice, RO)
    ro_id_all = ro_ids.reshape(-1)


    # kxy per readout
    kx = traj_t[t_fixed, indices, 0, :] / sx
    ky = traj_t[t_fixed, indices, 1, :] / sy

    ro_mid = RO // 2
    theta_sp = torch.atan2(ky[:, ro_mid], kx[:, ro_mid])

    # normalized z, t
    z_norm = (torch.tensor(z_slice_idx, device=dev_data, dtype=traj_t.dtype) / (n_slices - 1 + 1e-8)) * 2.0 - 1.0
    t_norm = (torch.tensor(t_fixed, device=dev_data, dtype=traj_t.dtype) / (T - 1 + 1e-8)) * 2.0 - 1.0
    z_col = torch.full((indices.numel(), RO), z_norm, device=dev_data, dtype=traj_t.dtype)
    t_col = torch.full((indices.numel(), RO), t_norm, device=dev_data, dtype=traj_t.dtype)

    # flatten
    kx_all = kx.reshape(-1)
    ky_all = ky.reshape(-1)
    z_all = z_col.reshape(-1)
    t_all = t_col.reshape(-1)

    x_all = torch.stack([kx_all, ky_all, z_all, t_all], dim=1).float()

    # measured kspace
    y = k_img_space[t_fixed, :, coil_fixed, z_slice_idx, :].reshape(-1)
    y_ri = torch.view_as_real(y).float()

    non_block = (dev_data.type == "cuda" and dev_compute.type == "cuda")
    x_all = x_all.to(dev_compute, non_blocking=non_block)
    y_ri = y_ri.to(dev_compute, non_blocking=non_block)
    kx_all = kx_all.to(dev_compute, non_blocking=non_block)
    ky_all = ky_all.to(dev_compute, non_blocking=non_block)
    spoke_id_all = spoke_id_all.to(dev_compute, non_blocking=non_block)
    ro_id_all = ro_id_all.to(dev_compute, non_blocking=non_block)

    y_ri = y_ri / y_scale

    meta = {
        "t_fixed": t_fixed,
        "coil_fixed": coil_fixed,
        "z_slice_idx": z_slice_idx,
        "n_slices": int(n_slices),
        "n_ro_per_slice": int(indices.numel()),
        "sp_idx_global" : indices.detach().cpu(),
        "theta_sp" : theta_sp.detach().cpu(),
        "N": int(x_all.shape[0]),
        "y_scale": float(y_scale.item()) if isinstance(y_scale, torch.Tensor) else float(y_scale),
    }
    return x_all, y_ri, kx_all, ky_all, spoke_id_all, ro_id_all, meta


def make_fixed_frame_zslice_multicoil_dataset(
    k_img_space,
    traj_t,
    scales,
    dims,
    *,
    y_scale,
    t_fixed: int = 0,
    z_slice_idx: int = 0,
    n_slices: int = None,
    compute_device: str = "cuda",
):
    """single frame zslice, all coils"""
    sx, sy, _ = scales
    T, S, C, RO = dims
    dev_data = k_img_space.device
    dev_compute = torch.device(compute_device)

    assert 0 <= t_fixed < T

    if n_slices is None:
        kz_vals = traj_t[t_fixed, :, 2, 0]
        n_slices = len(torch.unique(kz_vals))

    z_slice_idx = int(max(0, min(int(z_slice_idx), int(n_slices - 1))))
    n_ro_per_slice = int(S // n_slices)

    # interleaved readouts
    indices = torch.arange(0, S, n_slices, device=traj_t.device)

    # single coil ids
    spoke_ids_1c = torch.arange(n_ro_per_slice, device=dev_data, dtype=torch.long)[:, None].expand(n_ro_per_slice, RO).reshape(-1)
    ro_ids_1c = torch.arange(RO, device=dev_data, dtype=torch.long)[None, :].expand(n_ro_per_slice, RO).reshape(-1)

    # shared kxy
    kx = traj_t[t_fixed, indices, 0, :] / sx
    ky = traj_t[t_fixed, indices, 1, :] / sy

    # normalized z, t
    z_norm = (torch.tensor(z_slice_idx, device=dev_data, dtype=traj_t.dtype) / (n_slices - 1 + 1e-8)) * 2.0 - 1.0
    t_norm = (torch.tensor(t_fixed, device=dev_data, dtype=traj_t.dtype) / (T - 1 + 1e-8)) * 2.0 - 1.0
    z_col = torch.full((indices.numel(), RO), z_norm, device=dev_data, dtype=traj_t.dtype)
    t_col = torch.full((indices.numel(), RO), t_norm, device=dev_data, dtype=traj_t.dtype)

    kx_all_1c = kx.reshape(-1)
    ky_all_1c = ky.reshape(-1)
    z_all_1c = z_col.reshape(-1)
    t_all_1c = t_col.reshape(-1)

    x_1c = torch.stack([kx_all_1c, ky_all_1c, z_all_1c, t_all_1c], dim=1).float()
    N_per_coil = x_1c.shape[0]

    # tile per coil
    x_all = x_1c.repeat(C, 1)

    # coil index
    coil_id_all = torch.arange(C, device=dev_data, dtype=torch.long).repeat_interleave(N_per_coil)

    # ids per coil
    spoke_id_all = spoke_ids_1c.repeat(C)
    ro_id_all = ro_ids_1c.repeat(C)

    # concat coils
    y_parts = []
    for c in range(C):
        y_c = k_img_space[t_fixed, :, c, z_slice_idx, :].reshape(-1)
        y_parts.append(torch.view_as_real(y_c).float())
    y_all = torch.cat(y_parts, dim=0)

    # to compute device
    non_block = (dev_data.type == "cuda" and dev_compute.type == "cuda")
    x_all = x_all.to(dev_compute, non_blocking=non_block)
    y_all = y_all.to(dev_compute, non_blocking=non_block)
    coil_id_all = coil_id_all.to(dev_compute, non_blocking=non_block)
    spoke_id_all = spoke_id_all.to(dev_compute, non_blocking=non_block)
    ro_id_all = ro_id_all.to(dev_compute, non_blocking=non_block)

    y_all = y_all / y_scale

    meta = {
        "t_fixed": t_fixed,
        "z_slice_idx": z_slice_idx,
        "n_slices": int(n_slices),
        "n_coils": C,
        "n_ro_per_slice": int(indices.numel()),
        "N_per_coil": N_per_coil,
        "N_total": N_per_coil * C,
        "y_scale": float(y_scale.item()) if isinstance(y_scale, torch.Tensor) else float(y_scale),
    }
    return x_all, y_all, coil_id_all, spoke_id_all, ro_id_all, meta


def split_points_by_spokes(spoke_id_all, *, val_frac=0.2, seed=0, mode="random"):
    """random spoke split"""
    device = spoke_id_all.device
    S_kz = int(spoke_id_all.max().item()) + 1
    n_val = max(1, int(round(S_kz * val_frac)))

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    if mode == "random":
        perm = torch.randperm(S_kz, generator=g, device=device)
        val_spokes = perm[:n_val]
        train_spokes = perm[n_val:]
    else:
        raise ValueError("mode must be 'random' ")

    # point indices
    train_mask = torch.isin(spoke_id_all, train_spokes)
    val_mask = torch.isin(spoke_id_all, val_spokes)

    train_idx = torch.where(train_mask)[0]
    val_idx   = torch.where(val_mask)[0]
    return train_idx, val_idx, train_spokes, val_spokes


def split_points_by_angular_sector(
    spoke_id_all,
    theta_sp,
    *,
    n_sectors=4,
    val_sector=0,
):
    """angular sector holdout"""
    device = spoke_id_all.device
    theta_sp = theta_sp.to(device).float()

    # fold to 0,pi
    theta_folded = torch.remainder(theta_sp, torch.tensor(np.pi, device=device))

    sector_width = np.pi / n_sectors
    lo = val_sector * sector_width
    hi = (val_sector + 1) * sector_width

    val_spoke_mask = (theta_folded >= lo) & (theta_folded < hi)
    val_spoke_ids = torch.where(val_spoke_mask)[0].to(device)

    val_mask = torch.isin(spoke_id_all, val_spoke_ids)
    train_mask = ~val_mask

    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    return train_idx, val_idx, val_spoke_ids


def verify_spoke_holdout(spoke_id_all, val_idx, train_idx, n_coils=None, coil_id_all=None, RO=None):
    """spoke holdout check"""
    val_spokes = torch.unique(spoke_id_all[val_idx])
    train_spokes = torch.unique(spoke_id_all[train_idx])

    # no overlap
    overlap = torch.isin(val_spokes, train_spokes)
    assert not overlap.any(), f"Spokes appear in both train and val: {val_spokes[overlap].tolist()}"

    # whole spoke check
    val_mask_expected = torch.isin(spoke_id_all, val_spokes)
    val_mask_actual = torch.zeros(spoke_id_all.shape[0], dtype=torch.bool, device=spoke_id_all.device)
    val_mask_actual[val_idx] = True
    mismatch = (val_mask_expected != val_mask_actual).sum().item()
    assert mismatch == 0, f"Whole-spoke violation: {mismatch} points mismatched"

    # multicoil check
    if n_coils is not None and coil_id_all is not None and n_coils > 1:
        val_spokes_ref = None
        for c in range(n_coils):
            coil_mask = coil_id_all == c
            coil_val_spokes = torch.unique(spoke_id_all[coil_mask & val_mask_actual])
            if val_spokes_ref is None:
                val_spokes_ref = coil_val_spokes
            else:
                assert torch.equal(coil_val_spokes, val_spokes_ref), \
                    f"Coil {c} has different val spokes than coil 0"

    return True


def verify_multicoil_data(x_all, y_all, coil_id_all, spoke_id_all, val_idx, n_coils, N_per_coil):
    """multicoil checks"""
    # shared coords
    coords_ref = x_all[:N_per_coil, :2]
    for c in range(1, n_coils):
        start = c * N_per_coil
        end = (c + 1) * N_per_coil
        coords_c = x_all[start:end, :2]
        assert torch.allclose(coords_c, coords_ref, atol=1e-6), \
            f"Coil {c} has different coordinates than coil 0!"

    # same val spokes
    val_mask = torch.zeros(x_all.shape[0], dtype=torch.bool, device=x_all.device)
    val_mask[val_idx] = True
    val_spokes_ref = torch.unique(spoke_id_all[:N_per_coil][val_mask[:N_per_coil]])
    for c in range(1, n_coils):
        start = c * N_per_coil
        end = (c + 1) * N_per_coil
        val_spokes_c = torch.unique(spoke_id_all[start:end][val_mask[start:end]])
        assert torch.equal(val_spokes_c, val_spokes_ref), \
            f"Coil {c} has different val spokes than coil 0!"

    # different targets
    y_ref = y_all[:N_per_coil]
    for c in range(1, n_coils):
        start = c * N_per_coil
        end = (c + 1) * N_per_coil
        assert not torch.allclose(y_all[start:end], y_ref, atol=1e-8), \
            f"Coil {c} has identical targets to coil 0!"

    # total points
    assert x_all.shape[0] == N_per_coil * n_coils, \
        f"Expected {N_per_coil * n_coils} total points, got {x_all.shape[0]}"

    return True


def reconstruct_from_kspace(k_t, traj_t, t_frame, coil_idx, z_slice_idx, scales,
                            img_size=(128, 128)):
    """xcat eric recon"""
    sx, sy, sz = scales
    T, S, C, RO = k_t.shape

    # zslice count
    kz_vals = traj_t[t_frame, :, 2, 0]
    unique_kz = torch.unique(kz_vals)
    n_slices = len(unique_kz)

    # interleave check
    n_ro_per_slice = S // n_slices
    if S % n_slices != 0:
        print(f"Warning: S={S} not divisible by n_slices={n_slices}")

    # reorganize interleaved
    k_slices_org = torch.zeros(
        (T, n_ro_per_slice, C, n_slices, RO),
        dtype=k_t.dtype,
        device=k_t.device,
    )
    for sl in range(n_slices):
        indices = torch.arange(sl, S, n_slices, device=k_t.device)
        k_slices_org[:, :, :, sl, :] = k_t[:, indices, :, :]

    # kz sort
    kz_first_readouts = traj_t[t_frame, :n_slices, 2, 0]
    _, kz_sort_order = torch.sort(kz_first_readouts)

    k_slices_org = k_slices_org[:, :, :, kz_sort_order]

    # kz to z ifft
    k_img_space = torch.fft.ifft(k_slices_org, dim=3)

    # z fftshift
    k_img_space = torch.fft.fftshift(k_img_space, dim=3)

    # one zslice
    z_slice_idx = min(z_slice_idx, n_slices - 1)
    k_slice = k_img_space[t_frame, :, coil_idx, z_slice_idx, :]

    # frame trajectory
    kx = traj_t[t_frame, :, 0, :] / sx
    ky = traj_t[t_frame, :, 1, :] / sy

    # interleaved subset
    indices = torch.arange(0, S, n_slices, device=traj_t.device)
    kx_slice = kx[indices, :]
    ky_slice = ky[indices, :]

    # flatten
    kx_flat = kx_slice.reshape(-1).cpu().numpy()
    ky_flat = ky_slice.reshape(-1).cpu().numpy()

    # cufinufft pi range
    kx_pi = kx_flat * np.pi
    ky_pi = ky_flat  * np.pi

    # radial dcf
    density = np.sqrt(kx_flat**2 + ky_flat**2) + 1e-8

    # flat kspace
    k_flat = k_slice.reshape(-1).cpu().numpy()

    # density weighted
    k_weighted = k_flat * density

    # nufft adjoint
    k_weighted_cu = cp.asarray(k_weighted, dtype=cp.complex64)
    kx_cu = cp.asarray(kx_pi, dtype=cp.float32)
    ky_cu = cp.asarray(ky_pi, dtype=cp.float32)

    # cufinufft type1
    plan = cufinufft.Plan(nufft_type=1, n_modes=img_size, eps=1e-6, dtype=np.complex64)
    plan.setpts(kx_cu, ky_cu)
    img_cu = plan.execute(k_weighted_cu)

    # magnitude
    img = np.abs(cp.asnumpy(img_cu))

    return img


def ifft1d_kz_to_z(k_t, traj_t, t_frame):
    """interleaved kz to z"""
    T, S, C, RO = k_t.shape

    kz_vals = traj_t[t_frame, :, 2, 0]
    unique_kz = torch.unique(kz_vals)
    n_slices = len(unique_kz)

    n_ro_per_slice = S // n_slices
    if S % n_slices != 0:
        print(f"Warning: S={S} not divisible by n_slices={n_slices}")

    k_slices_org = torch.zeros(
        (T, n_ro_per_slice, C, n_slices, RO),
        dtype=k_t.dtype,
        device=k_t.device,
    )
    for sl in range(n_slices):
        indices = torch.arange(sl, S, n_slices, device=k_t.device)
        k_slices_org[:, :, :, sl, :] = k_t[:, indices, :, :]

    kz_first_readouts = traj_t[t_frame, :n_slices, 2, 0]
    _, kz_sort_order = torch.sort(kz_first_readouts)
    k_slices_org = k_slices_org[:, :, :, kz_sort_order]

    k_img_space = torch.fft.ifft(k_slices_org, dim=3)
    k_img_space = torch.fft.fftshift(k_img_space, dim=3)

    return k_img_space, n_slices, n_ro_per_slice, kz_sort_order


def nufft2d_recon(k_img_space, traj_t, t_frame, coil_idx, z_slice_idx, scales,
                 img_size=(128, 128), n_slices=None, return_complex=False):
    """nufft adjoint, single zslice"""
    sx, sy, _ = scales
    _, S, _, _ = traj_t.shape

    if n_slices is None:
        kz_vals = traj_t[t_frame, :, 2, 0]
        n_slices = len(torch.unique(kz_vals))

    z_slice_idx = min(z_slice_idx, n_slices - 1)
    k_slice = k_img_space[t_frame, :, coil_idx, z_slice_idx, :]

    kx = traj_t[t_frame, :, 0, :] / sx
    ky = traj_t[t_frame, :, 1, :] / sy

    indices = torch.arange(0, S, n_slices, device=traj_t.device)
    kx_slice = kx[indices, :]
    ky_slice = ky[indices, :]

    kx_flat = kx_slice.reshape(-1).cpu().numpy()
    ky_flat = ky_slice.reshape(-1).cpu().numpy()

    kx_pi = kx_flat * np.pi
    ky_pi = ky_flat * np.pi

    density = np.sqrt(kx_flat**2 + ky_flat**2) + 1e-8

    k_flat = k_slice.reshape(-1).cpu().numpy()
    k_weighted = k_flat * density

    k_weighted_cu = cp.asarray(k_weighted, dtype=cp.complex64)
    kx_cu = cp.asarray(kx_pi, dtype=cp.float32)
    ky_cu = cp.asarray(ky_pi, dtype=cp.float32)

    plan = cufinufft.Plan(nufft_type=1, n_modes=img_size, eps=1e-6, dtype=np.complex64)
    plan.setpts(kx_cu, ky_cu)
    img_cu = plan.execute(k_weighted_cu)

    img = cp.asnumpy(img_cu)
    if return_complex:
        return img
    return np.abs(img)


def nufft2d_recon_multicoil_sos(
    model,
    *,
    x_all,
    coil_id_all,
    y_scale,
    k_img_space,
    traj_t,
    scales,
    t_frame,
    z_slice_idx,
    n_z_slices,
    n_ro_per_slice,
    n_coils,
    N_per_coil,
    RO,
    img_size=(128, 128),
):
    """multicoil sos recon"""
    device = next(model.parameters()).device
    model.eval()

    ys = float(y_scale.item()) if isinstance(y_scale, torch.Tensor) else float(y_scale)

    # shared coords
    x_1c = x_all[:N_per_coil, :2].to(device)

    sos_pred_sq = np.zeros(img_size, dtype=np.float64)
    sos_meas_sq = np.zeros(img_size, dtype=np.float64)
    imgs_per_coil = []

    for c in range(n_coils):
        # predicted kspace
        coil_idx_tensor = torch.full((N_per_coil,), c, device=device, dtype=torch.long)
        with torch.no_grad():
            y_pred = model(x_1c, coil_idx_tensor) * ys
        k_pred = torch.complex(y_pred[:, 0], y_pred[:, 1])
        k_pred_slice = k_pred.reshape(n_ro_per_slice, RO)

        # per coil tensor
        k_img_space_pred = torch.zeros_like(k_img_space)
        k_img_space_pred[t_frame, :, c, z_slice_idx, :] = k_pred_slice

        # predicted recon
        img_c = nufft2d_recon(
            k_img_space_pred, traj_t,
            t_frame=t_frame, coil_idx=c,
            z_slice_idx=z_slice_idx,
            scales=scales, img_size=img_size, n_slices=n_z_slices,
        )
        imgs_per_coil.append(img_c)
        sos_pred_sq += img_c.astype(np.float64) ** 2

        # measured recon
        img_meas_c = nufft2d_recon(
            k_img_space, traj_t,
            t_frame=t_frame, coil_idx=c,
            z_slice_idx=z_slice_idx,
            scales=scales, img_size=img_size, n_slices=n_z_slices,
        )
        sos_meas_sq += img_meas_c.astype(np.float64) ** 2

    img_sos_pred = np.sqrt(sos_pred_sq)
    img_sos_measured = np.sqrt(sos_meas_sq)

    return {
        "img_sos_pred": img_sos_pred,
        "img_sos_measured": img_sos_measured,
        "imgs_per_coil": imgs_per_coil,
    }


def fft2d_uniform(k_xy, axes=(-2, -1), shift=True, return_magnitude=False):
    if shift:
        k_xy = torch.fft.ifftshift(k_xy, dim=axes)
    img_xy = torch.fft.ifft2(k_xy, dim=axes)
    if shift:
        img_xy = torch.fft.fftshift(img_xy, dim=axes)
    if return_magnitude:
        img_xy = torch.abs(img_xy)
    return img_xy


def to_plot(x):
    if torch.is_tensor(x):
        x = x.detach()
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    x = np.asarray(x)
    if np.iscomplexobj(x):
        x = np.abs(x)
    return x

def norm_img(img, p=99):
    img = np.asarray(img)
    s = np.percentile(img, p)
    return img / (s + 1e-12)


@torch.no_grad()
def evaluate_image_metrics(
    model,
    *,
    x_all,
    y_scale,
    k_img_space,
    traj_t,
    scales,
    t_frame,
    coil_idx,
    z_slice_idx,
    n_z_slices,
    n_ro_per_slice,
    RO,
    img_size=(128, 128),
    gt_img=None,
    ssim_win_size=7,
    nrmse_norm="euclidean",
    compute_perceptual=False,
):
    """end to end image metrics"""
    device = next(model.parameters()).device
    model.eval()

    ys = float(y_scale.item()) if isinstance(y_scale, torch.Tensor) else float(y_scale)

    x_in = x_all[:, :2].to(device)
    y_pred_all = model(x_in) * ys
    k_pred = torch.complex(y_pred_all[:, 0], y_pred_all[:, 1])
    k_pred_slice = k_pred.reshape(n_ro_per_slice, RO)

    k_img_space_pred = torch.zeros_like(k_img_space)
    k_img_space_pred[t_frame, :, coil_idx, z_slice_idx, :] = k_pred_slice

    img_pred = nufft2d_recon(
        k_img_space_pred, traj_t,
        t_frame=t_frame, coil_idx=coil_idx,
        z_slice_idx=z_slice_idx,
        scales=scales, img_size=img_size, n_slices=n_z_slices,
    )

    img_measured = nufft2d_recon(
        k_img_space, traj_t,
        t_frame=t_frame, coil_idx=coil_idx,
        z_slice_idx=z_slice_idx,
        scales=scales, img_size=img_size, n_slices=n_z_slices,
    )

    # nufft proxy, relative only
    metrics_vs_nufft_proxy = compute_image_metrics(
        img_pred, img_measured,
        ssim_win_size=ssim_win_size, nrmse_norm=nrmse_norm,
    )

    metrics_vs_ground_truth = None
    img_pred_r = img_pred
    if gt_img is not None:
        gt_slice = np.asarray(gt_img[t_frame, z_slice_idx, :, :], dtype=np.float64)
        if img_pred.shape != gt_slice.shape:
            from scipy.ndimage import zoom
            zf = (gt_slice.shape[0] / img_pred.shape[0],
                  gt_slice.shape[1] / img_pred.shape[1])
            img_pred_r = zoom(img_pred, zf, order=3)
        metrics_vs_ground_truth = compute_image_metrics(
            img_pred_r, gt_slice,
            ssim_win_size=ssim_win_size, nrmse_norm=nrmse_norm,
        )

    # perceptual, eval only
    perceptual_vs_nufft_proxy = None
    perceptual_vs_ground_truth = None
    if compute_perceptual:
        perceptual_vs_nufft_proxy = compute_perceptual_metrics(img_pred, img_measured)
        if metrics_vs_ground_truth is not None:
            perceptual_vs_ground_truth = compute_perceptual_metrics(
                img_pred_r, gt_slice,
            )

    return {
        "img_pred": img_pred,
        "img_nufft_proxy": img_measured,
        "metrics_vs_nufft_proxy": metrics_vs_nufft_proxy,
        "metrics_vs_ground_truth": metrics_vs_ground_truth,
        "perceptual_vs_nufft_proxy": perceptual_vs_nufft_proxy,
        "perceptual_vs_ground_truth": perceptual_vs_ground_truth,
    }


# cartesian eval utils

def ifft1d_kz_to_z_cartesian(k_cart):
    """cart kz to z ifft"""
    # radial aligned roll
    n_kz = k_cart.shape[2]
    k_z_space = torch.fft.ifft(k_cart, dim=2)
    k_z_space = torch.fft.fftshift(k_z_space, dim=2)
    k_z_space = torch.roll(k_z_space, shifts=-(n_kz // 2), dims=2)
    return k_z_space


def make_cartesian_eval_dataset(
    k_cart_z,
    *,
    t_fixed: int = 0,
    coil_fixed: int = 0,
    z_slice_idx: int = 0,
    scales_radial,
    y_scale,
    compute_device: str = "cuda",
):
    """cartesian eval dataset"""
    sx, sy, _ = scales_radial
    dev = torch.device(compute_device)

    T, C, nz, nky, nkx = k_cart_z.shape
    z_slice_idx = int(max(0, min(int(z_slice_idx), nz - 1)))

    # single slice
    k_slice = k_cart_z[t_fixed, coil_fixed, z_slice_idx, :, :]

    # centered fft bins
    kx_lin = torch.fft.fftshift(
        torch.fft.fftfreq(nkx, device=dev, dtype=torch.float32)
    )
    ky_lin = torch.fft.fftshift(
        torch.fft.fftfreq(nky, device=dev, dtype=torch.float32)
    )
    KY, KX = torch.meshgrid(ky_lin, kx_lin, indexing="ij")

    # radial scale norm
    sx_f = float(sx.item()) if torch.is_tensor(sx) else float(sx)
    sy_f = float(sy.item()) if torch.is_tensor(sy) else float(sy)
    KX_norm = KX / sx_f
    KY_norm = KY / sy_f

    # flatten
    x_cart = torch.stack([KX_norm.reshape(-1), KY_norm.reshape(-1)], dim=1).float().to(dev)

    # reim values
    k_flat = k_slice.reshape(-1)
    y_cart = torch.view_as_real(k_flat).float().to(dev)

    ys = float(y_scale.item()) if torch.is_tensor(y_scale) else float(y_scale)
    y_cart = y_cart / ys

    meta_cart = {
        "nky": nky,
        "nkx": nkx,
        "nz": nz,
        "t_fixed": t_fixed,
        "coil_fixed": coil_fixed,
        "z_slice_idx": z_slice_idx,
        "N": nky * nkx,
    }
    return x_cart, y_cart, meta_cart


# multicoil utils

def make_multicoil_radial_dataset(
    k_img_space,
    traj_t,
    scales,
    dims,
    *,
    t_fixed: int = 0,
    z_slice_idx: int = 0,
    n_slices: int = None,
    compute_device: str = "cuda",
):
    """multicoil radial, one frame zslice"""
    sx, sy, _ = scales
    T, S, C, RO = dims
    dev_data = k_img_space.device
    dev_compute = torch.device(compute_device)

    if n_slices is None:
        kz_vals = traj_t[t_fixed, :, 2, 0]
        n_slices = len(torch.unique(kz_vals))

    z_slice_idx = int(max(0, min(int(z_slice_idx), int(n_slices - 1))))
    n_ro_per_slice = int(S // n_slices)

    indices = torch.arange(0, S, n_slices, device=traj_t.device)

    spoke_ids = torch.arange(n_ro_per_slice, device=dev_data, dtype=torch.long)[:, None].expand(n_ro_per_slice, RO)
    spoke_id_all = spoke_ids.reshape(-1)
    ro_ids = torch.arange(RO, device=dev_data, dtype=torch.long)[None, :].expand(n_ro_per_slice, RO)
    ro_id_all = ro_ids.reshape(-1)

    kx = traj_t[t_fixed, indices, 0, :] / sx
    ky = traj_t[t_fixed, indices, 1, :] / sy

    z_norm = (torch.tensor(z_slice_idx, device=dev_data, dtype=traj_t.dtype) / (n_slices - 1 + 1e-8)) * 2.0 - 1.0
    t_norm = (torch.tensor(t_fixed, device=dev_data, dtype=traj_t.dtype) / (T - 1 + 1e-8)) * 2.0 - 1.0
    z_col = torch.full((indices.numel(), RO), z_norm, device=dev_data, dtype=traj_t.dtype)
    t_col = torch.full((indices.numel(), RO), t_norm, device=dev_data, dtype=traj_t.dtype)

    kx_all = kx.reshape(-1)
    ky_all = ky.reshape(-1)
    x_all = torch.stack([kx_all, ky_all, z_col.reshape(-1), t_col.reshape(-1)], dim=1).float()

    # concat all coils
    y_parts = []
    for c in range(C):
        y_c = k_img_space[t_fixed, :, c, z_slice_idx, :].reshape(-1)
        y_parts.append(torch.view_as_real(y_c).float())
    y_all = torch.cat(y_parts, dim=1)

    non_block = (dev_data.type == "cuda" and dev_compute.type == "cuda")
    x_all = x_all.to(dev_compute, non_blocking=non_block)
    y_all = y_all.to(dev_compute, non_blocking=non_block)
    spoke_id_all = spoke_id_all.to(dev_compute, non_blocking=non_block)
    ro_id_all = ro_id_all.to(dev_compute, non_blocking=non_block)

    meta = {
        "t_fixed": t_fixed,
        "z_slice_idx": z_slice_idx,
        "n_slices": int(n_slices),
        "n_coils": C,
        "n_ro_per_slice": int(indices.numel()),
        "N": int(x_all.shape[0]),
        "RO": RO,
    }
    return x_all, y_all, spoke_id_all, ro_id_all, meta


def make_multicoil_cartesian_dataset(
    k_cart_z,
    *,
    t_fixed: int = 0,
    z_slice_idx: int = 0,
    scales_radial,
    compute_device: str = "cuda",
):
    """multicoil cart eval"""
    sx, sy, _ = scales_radial
    dev = torch.device(compute_device)

    T, C, nz, nky, nkx = k_cart_z.shape
    z_slice_idx = int(max(0, min(int(z_slice_idx), nz - 1)))

    kx_lin = torch.linspace(-0.5, 0.5, nkx)
    ky_lin = torch.linspace(-0.5, 0.5, nky)
    KY, KX = torch.meshgrid(ky_lin, kx_lin, indexing="ij")

    sx_f = float(sx.item()) if torch.is_tensor(sx) else float(sx)
    sy_f = float(sy.item()) if torch.is_tensor(sy) else float(sy)
    x_cart = torch.stack([KX.reshape(-1) / sx_f, KY.reshape(-1) / sy_f], dim=1).float().to(dev)

    # concat coils
    y_parts = []
    for c in range(C):
        k_slice = k_cart_z[t_fixed, c, z_slice_idx, :, :].reshape(-1)
        y_parts.append(torch.view_as_real(k_slice).float())
    y_cart = torch.cat(y_parts, dim=1).to(dev)

    meta = {
        "nky": nky, "nkx": nkx, "nz": nz, "n_coils": C,
        "t_fixed": t_fixed, "z_slice_idx": z_slice_idx,
        "N": nky * nkx,
    }
    return x_cart, y_cart, meta


def coil_combine_sense(coil_images, sensitivity_maps):
    """sense coil combine"""
    S = sensitivity_maps
    if not np.iscomplexobj(coil_images):
        # magnitude rss
        return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))

    numerator = np.sum(np.conj(S) * coil_images, axis=0)
    denominator = np.sum(np.abs(S) ** 2, axis=0) + 1e-10
    return np.abs(numerator / denominator)


def coil_combine_rss(coil_images):
    """rss combine"""
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))
