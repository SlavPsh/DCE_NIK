# nik_recon.py
import numpy as np
import torch
import cupy as cp
import cufinufft


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
    """
    Build a deterministic dataset for ONE frame, ONE z-slice, ONE coil
    after kz->z IFFT.

    returns:
      x_all   : (N,4) float  [kx, ky, z_norm, t_norm]
      y_all   : (N,2) float  [Re, Im]
      kx_all, ky_all : (N,) float (for plotting)
    """
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

    # Interleaved readouts for one slice
    indices = torch.arange(0, S, n_slices, device=traj_t.device)

    # kx, ky for those readouts
    kx = traj_t[t_fixed, indices, 0, :] / sx
    ky = traj_t[t_fixed, indices, 1, :] / sy

    # Fixed z and t (normalized to [-1, 1])
    z_norm = (torch.tensor(z_slice_idx, device=dev_data, dtype=traj_t.dtype) / (n_slices - 1 + 1e-8)) * 2.0 - 1.0
    t_norm = (torch.tensor(t_fixed, device=dev_data, dtype=traj_t.dtype) / (T - 1 + 1e-8)) * 2.0 - 1.0
    z_col = torch.full((indices.numel(), RO), z_norm, device=dev_data, dtype=traj_t.dtype)
    t_col = torch.full((indices.numel(), RO), t_norm, device=dev_data, dtype=traj_t.dtype)

    # Flatten to (N,)
    kx_all = kx.reshape(-1)
    ky_all = ky.reshape(-1)
    z_all = z_col.reshape(-1)
    t_all = t_col.reshape(-1)

    x_all = torch.stack([kx_all, ky_all, z_all, t_all], dim=1).float()

    # Measured k-space for this coil/slice: (n_ro_per_slice, RO) -> (N,)
    y = k_img_space[t_fixed, :, coil_fixed, z_slice_idx, :].reshape(-1)
    y_ri = torch.view_as_real(y).float()

    non_block = (dev_data.type == "cuda" and dev_compute.type == "cuda")
    x_all = x_all.to(dev_compute, non_blocking=non_block)
    y_ri = y_ri.to(dev_compute, non_blocking=non_block)
    kx_all = kx_all.to(dev_compute, non_blocking=non_block)
    ky_all = ky_all.to(dev_compute, non_blocking=non_block)

    y_ri = y_ri / y_scale

    meta = {
        "t_fixed": t_fixed,
        "coil_fixed": coil_fixed,
        "z_slice_idx": z_slice_idx,
        "n_slices": int(n_slices),
        "n_ro_per_slice": int(indices.numel()),
        "N": int(x_all.shape[0]),
        "y_scale": float(y_scale.item()) if isinstance(y_scale, torch.Tensor) else float(y_scale),
    }
    return x_all, y_ri, kx_all, ky_all, meta


def reconstruct_from_kspace(k_t, traj_t, t_frame, coil_idx, z_slice_idx, scales, 
                            img_size=(128, 128)):
    """
    Reconstruct 2D image from k-space following XCAT-ERIC approach.
    
    Uses interleaved readout pattern where readouts are distributed across z-slices.
    
    Steps:
    1. Identify number of z-slices from unique kz values
    2. Reorganize k-space by interleaving: slice i gets readouts at indices i, i+nSlcs, i+2*nSlcs, ...
    3. Sort by kz coordinate
    4. 1D IFFT in z-direction to convert kz frequency → z image space
    5. Extract one z-slice
    6. Extract trajectory (kx, ky) for this frame
    7. Apply radial density compensation: sqrt(kx^2 + ky^2)
    8. 2D NUFFT adjoint (gridding)
    9. Return magnitude
    
    Args:
        k_t: (T,S,C,RO) k-space tensor (complex) where S is interleaved readouts
        traj_t: (T,S,3,RO) trajectory tensor
        t_frame: frame index
        coil_idx: which coil to use
        z_slice_idx: which z-slice to reconstruct
        scales: (sx, sy, sz) trajectory scales
        img_size: (Nx, Ny) output image size
    
    Returns:
        img: (Nx, Ny) reconstructed image magnitude
    """
    sx, sy, sz = scales
    T, S, C, RO = k_t.shape
    
    # ============================================================
    # Step 1: Get number of z-slices from unique kz values
    # ============================================================
    kz_vals = traj_t[t_frame, :, 2, 0]  # (S,) - kz value for each spoke
    unique_kz = torch.unique(kz_vals)
    n_slices = len(unique_kz)
    
    # Verify interleaving: total readouts should be divisible by n_slices
    n_ro_per_slice = S // n_slices
    if S % n_slices != 0:
        print(f"Warning: S={S} not divisible by n_slices={n_slices}")
    
    # ============================================================
    # Step 2: Reorganize k-space using interleaving pattern
    # ============================================================
    # Extract every n_slices-th readout for each slice
    # Slice 0: readouts 0, n_slices, 2*n_slices, ...
    # Slice 1: readouts 1, n_slices+1, 2*n_slices+1, ...
    # etc.
    k_slices_org = torch.zeros(
        (T, n_ro_per_slice, C, n_slices, RO),
        dtype=k_t.dtype,
        device=k_t.device,
    )
    for sl in range(n_slices):
        indices = torch.arange(sl, S, n_slices, device=k_t.device)
        k_slices_org[:, :, :, sl, :] = k_t[:, indices, :, :]
    
    # ============================================================
    # Step 3: Sort by kz coordinate
    # ============================================================
    # Get kz values for first readout of each interleaved group
    kz_first_readouts = traj_t[t_frame, :n_slices, 2, 0]  # (n_slices,)
    _, kz_sort_order = torch.sort(kz_first_readouts)
    
    # Reorder slices according to sorted kz
    k_slices_org = k_slices_org[:, :, :, kz_sort_order]
    
    # ============================================================
    # Step 4: 1D IFFT in z-direction
    # ============================================================
    # k_slices_org is (T, n_ro_per_slice, C, n_slices)
    # IFFT along dim=3 (z-dimension) to convert kz frequency → z image space
    k_img_space = torch.fft.ifft(k_slices_org, dim=3)
    
    # Apply fftshift in z-dimension for proper slice ordering
    k_img_space = torch.fft.fftshift(k_img_space, dim=3)
    
    # ============================================================
    # Step 5: Extract one z-slice
    # ============================================================
    z_slice_idx = min(z_slice_idx, n_slices - 1)
    k_slice = k_img_space[t_frame, :, coil_idx, z_slice_idx, :]  # (n_ro_per_slice, RO)
    
    # ============================================================
    # Step 6: Get trajectory (kx, ky) for this frame
    # ============================================================
    # Use all interleaved readouts (not just one z-plane's spokes)
    # because we've already separated them by z via IFFT
    kx = traj_t[t_frame, :, 0, :] / sx  # (S, RO)
    ky = traj_t[t_frame, :, 1, :] / sy  # (S, RO)
    
    # Extract only for the readouts that correspond to this slice (interleaved pattern)
    indices = torch.arange(0, S, n_slices, device=traj_t.device)
    kx_slice = kx[indices, :]  # (n_ro_per_slice, RO)
    ky_slice = ky[indices, :]  # (n_ro_per_slice, RO)
    
    # Flatten to (N,)
    kx_flat = kx_slice.reshape(-1).cpu().numpy()
    ky_flat = ky_slice.reshape(-1).cpu().numpy()
    
    # Scale to [-pi, pi] for cufinufft
    kx_pi = kx_flat * np.pi
    ky_pi = ky_flat  * np.pi
    
    # ============================================================
    # Step 7: Radial density compensation: sqrt(kx^2 + ky^2)
    # ============================================================
    density = np.sqrt(kx_flat**2 + ky_flat**2) + 1e-8
    
    # Flatten k-space (n_ro_per_slice, RO) -> (N,)
    k_flat = k_slice.reshape(-1).cpu().numpy()
    
    # Apply density weighting (XCAT: kdata_nt = kspaceSorted_slices .* DensityComp_nt)
    k_weighted = k_flat * density
    
    # ============================================================
    # Step 8: 2D NUFFT adjoint (gridding + IFFT)
    # ============================================================
    # Convert to cupy for cufinufft
    k_weighted_cu = cp.asarray(k_weighted, dtype=cp.complex64)
    kx_cu = cp.asarray(kx_pi, dtype=cp.float32)
    ky_cu = cp.asarray(ky_pi, dtype=cp.float32)
    
    # Create cufinufft plan and execute (type=1 for adjoint: non-uniform -> uniform)
    plan = cufinufft.Plan(nufft_type=1, n_modes=img_size, eps=1e-6, dtype=np.complex64)
    plan.setpts(kx_cu, ky_cu)
    img_cu = plan.execute(k_weighted_cu)
    
    # ============================================================
    # Step 9: Return magnitude
    # ============================================================
    img = np.abs(cp.asnumpy(img_cu))
    
    return img


def ifft1d_kz_to_z(k_t, traj_t, t_frame):
    """
    Split interleaved k-space into z-slices and IFFT along kz -> z.

    Args:
        k_t: (T,S,C,RO) k-space tensor (complex) where S is interleaved readouts
        traj_t: (T,S,3,RO) trajectory tensor
        t_frame: frame index

    Returns:
        k_img_space: (T, n_ro_per_slice, C, n_slices, RO) after kz->z IFFT
        n_slices: number of z-slices detected from kz values
        n_ro_per_slice: number of readouts per slice
        kz_sort_order: indices used to sort slices by kz
    """
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
                 img_size=(128, 128), n_slices=None):
    """
    2D NUFFT adjoint reconstruction for one z-slice after kz->z IFFT.
    """
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

    img = np.abs(cp.asnumpy(img_cu))
    return img


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



