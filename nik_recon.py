# nik_recon.py
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
    """
    Build a  dataset for one frame, one z-slice, one coil
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
    n_ro_per_slice = int(S // n_slices)

    # Interleaved readouts for one slice
    indices = torch.arange(0, S, n_slices, device=traj_t.device)
    
    # per-point spoke/ro ids aligned with flattening order
    spoke_ids = torch.arange(n_ro_per_slice, device=dev_data, dtype=torch.long)[:, None].expand(n_ro_per_slice, RO)
    spoke_id_all = spoke_ids.reshape(-1)   # (N,)

    ro_ids = torch.arange(RO, device=dev_data, dtype=torch.long)[None, :].expand(n_ro_per_slice, RO)
    ro_id_all = ro_ids.reshape(-1)  # (N,)
    

    # kx, ky for those readouts
    kx = traj_t[t_fixed, indices, 0, :] / sx
    ky = traj_t[t_fixed, indices, 1, :] / sy

    ro_mid = RO // 2
    theta_sp = torch.atan2(ky[:, ro_mid], kx[:, ro_mid])  # (n_ro_per_slice,)

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
    """
    Build a dataset for one frame, one z-slice, ALL coils after kz->z IFFT.

    Same as make_fixed_frame_zslice_coil_dataset but returns data for all C
    coils concatenated, with a coil_idx tensor identifying each point's coil.

    The (kx, ky) coordinates are identical across coils (same trajectory).
    The spoke train/val split is applied identically across all coils.

    Returns:
      x_all      : (N_total, 4) float [kx, ky, z_norm, t_norm]  (repeated C times)
      y_all      : (N_total, 2) float [Re, Im]  (different per coil)
      coil_id_all: (N_total,) long — coil index 0..C-1 for each point
      spoke_id_all: (N_total,) long — spoke index (same spoke IDs repeated per coil)
      ro_id_all  : (N_total,) long — readout index
      meta       : dict with 'n_coils', 'N_per_coil', 'N_total', etc.
    """
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

    # Interleaved readouts for one slice
    indices = torch.arange(0, S, n_slices, device=traj_t.device)

    # Per-point spoke/ro ids for ONE coil (N_per_coil points)
    spoke_ids_1c = torch.arange(n_ro_per_slice, device=dev_data, dtype=torch.long)[:, None].expand(n_ro_per_slice, RO).reshape(-1)
    ro_ids_1c = torch.arange(RO, device=dev_data, dtype=torch.long)[None, :].expand(n_ro_per_slice, RO).reshape(-1)

    # kx, ky for those readouts (shared across coils)
    kx = traj_t[t_fixed, indices, 0, :] / sx
    ky = traj_t[t_fixed, indices, 1, :] / sy

    # Fixed z and t
    z_norm = (torch.tensor(z_slice_idx, device=dev_data, dtype=traj_t.dtype) / (n_slices - 1 + 1e-8)) * 2.0 - 1.0
    t_norm = (torch.tensor(t_fixed, device=dev_data, dtype=traj_t.dtype) / (T - 1 + 1e-8)) * 2.0 - 1.0
    z_col = torch.full((indices.numel(), RO), z_norm, device=dev_data, dtype=traj_t.dtype)
    t_col = torch.full((indices.numel(), RO), t_norm, device=dev_data, dtype=traj_t.dtype)

    kx_all_1c = kx.reshape(-1)
    ky_all_1c = ky.reshape(-1)
    z_all_1c = z_col.reshape(-1)
    t_all_1c = t_col.reshape(-1)

    x_1c = torch.stack([kx_all_1c, ky_all_1c, z_all_1c, t_all_1c], dim=1).float()  # (N_per_coil, 4)
    N_per_coil = x_1c.shape[0]

    # Tile coordinates C times (same trajectory for all coils)
    x_all = x_1c.repeat(C, 1)  # (N_total, 4)

    # Build coil index: [0,0,...,0, 1,1,...,1, ..., C-1,C-1,...,C-1]
    coil_id_all = torch.arange(C, device=dev_data, dtype=torch.long).repeat_interleave(N_per_coil)

    # Spoke and RO ids repeated per coil
    spoke_id_all = spoke_ids_1c.repeat(C)
    ro_id_all = ro_ids_1c.repeat(C)

    # k-space targets: concatenate all coils
    y_parts = []
    for c in range(C):
        y_c = k_img_space[t_fixed, :, c, z_slice_idx, :].reshape(-1)
        y_parts.append(torch.view_as_real(y_c).float())
    y_all = torch.cat(y_parts, dim=0)  # (N_total, 2)

    # Move to compute device
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
    """
    spoke_id_all: (N,) long, values in {0..S_kz-1} identifying which spoke each point came from.

    Returns:
      train_idx, val_idx: 1D Long tensors of point indices into x_all/y_all
      train_spokes, val_spokes: 1D Long tensors of spoke ids
    """
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
    val_mask   = torch.isin(spoke_id_all, val_spokes)

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
    """Hold out all spokes in one angular sector for evaluation.

    More challenging than random spoke holdout because the model must
    interpolate across a larger angular gap (~pi/n_sectors).

    Intended as an ADDITIONAL evaluation diagnostic, not as the primary
    training split.

    Args:
        spoke_id_all: (N,) long — spoke index per point (may repeat across coils).
        theta_sp: (n_unique_spokes,) float — angle of each spoke in (-pi, pi].
        n_sectors: divide [0, pi) into this many sectors.
        val_sector: which sector to hold out (0 to n_sectors-1).

    Returns:
        train_idx, val_idx: 1D Long tensors of point indices
        val_spokes: 1D tensor of spoke ids in the held-out sector
    """
    device = spoke_id_all.device
    theta_sp = theta_sp.to(device).float()

    # Fold angles to [0, pi) since opposing spokes are the same line
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
    """Verify that whole-spoke validation holdout is correct.

    Checks:
      1. Every point on a val spoke is in val_idx (no partial spokes)
      2. If multi-coil, same spokes held out for all coils

    Returns True if all checks pass, raises AssertionError otherwise.
    """
    val_spokes = torch.unique(spoke_id_all[val_idx])
    train_spokes = torch.unique(spoke_id_all[train_idx])

    # Check no spoke appears in both sets
    overlap = torch.isin(val_spokes, train_spokes)
    assert not overlap.any(), f"Spokes appear in both train and val: {val_spokes[overlap].tolist()}"

    # Check whole-spoke: every point with a val spoke_id should be in val_idx
    val_mask_expected = torch.isin(spoke_id_all, val_spokes)
    val_mask_actual = torch.zeros(spoke_id_all.shape[0], dtype=torch.bool, device=spoke_id_all.device)
    val_mask_actual[val_idx] = True
    mismatch = (val_mask_expected != val_mask_actual).sum().item()
    assert mismatch == 0, f"Whole-spoke violation: {mismatch} points mismatched"

    # Multi-coil check: same val spokes per coil
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
    """Run all multi-coil data loader verification checks.

    Checks:
      1. All coils share same (kx, ky) coordinates
      2. Same spokes in val for all coils
      3. Targets differ across coils
      4. Correct total number of points
    """
    # Check 1: all coils share same coordinates
    coords_ref = x_all[:N_per_coil, :2]
    for c in range(1, n_coils):
        start = c * N_per_coil
        end = (c + 1) * N_per_coil
        coords_c = x_all[start:end, :2]
        assert torch.allclose(coords_c, coords_ref, atol=1e-6), \
            f"Coil {c} has different coordinates than coil 0!"

    # Check 2: same val spokes for all coils
    val_mask = torch.zeros(x_all.shape[0], dtype=torch.bool, device=x_all.device)
    val_mask[val_idx] = True
    val_spokes_ref = torch.unique(spoke_id_all[:N_per_coil][val_mask[:N_per_coil]])
    for c in range(1, n_coils):
        start = c * N_per_coil
        end = (c + 1) * N_per_coil
        val_spokes_c = torch.unique(spoke_id_all[start:end][val_mask[start:end]])
        assert torch.equal(val_spokes_c, val_spokes_ref), \
            f"Coil {c} has different val spokes than coil 0!"

    # Check 3: targets differ across coils
    y_ref = y_all[:N_per_coil]
    for c in range(1, n_coils):
        start = c * N_per_coil
        end = (c + 1) * N_per_coil
        assert not torch.allclose(y_all[start:end], y_ref, atol=1e-8), \
            f"Coil {c} has identical targets to coil 0!"

    # Check 4: total points
    assert x_all.shape[0] == N_per_coil * n_coils, \
        f"Expected {N_per_coil * n_coils} total points, got {x_all.shape[0]}"

    return True


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
    """
    Reconstruct SOS-combined image from a multi-coil model.

    For each coil: predict k-space → NUFFT → magnitude image.
    Then combine via sum-of-squares: img_sos = sqrt(sum_c |img_c|^2).

    Also reconstructs the measured SOS image for comparison.

    Returns:
        dict with keys:
            img_sos_pred     : (H, W) SOS combined predicted image
            img_sos_measured : (H, W) SOS combined measured image
            imgs_per_coil    : list of (H, W) per-coil predicted images
    """
    device = next(model.parameters()).device
    model.eval()

    ys = float(y_scale.item()) if isinstance(y_scale, torch.Tensor) else float(y_scale)

    # Coordinates for one coil (all coils share the same trajectory)
    x_1c = x_all[:N_per_coil, :2].to(device)

    sos_pred_sq = np.zeros(img_size, dtype=np.float64)
    sos_meas_sq = np.zeros(img_size, dtype=np.float64)
    imgs_per_coil = []

    for c in range(n_coils):
        # Predict k-space for coil c
        coil_idx_tensor = torch.full((N_per_coil,), c, device=device, dtype=torch.long)
        with torch.no_grad():
            y_pred = model(x_1c, coil_idx_tensor) * ys
        k_pred = torch.complex(y_pred[:, 0], y_pred[:, 1])
        k_pred_slice = k_pred.reshape(n_ro_per_slice, RO)

        # Build predicted k_img_space for this coil
        k_img_space_pred = torch.zeros_like(k_img_space)
        k_img_space_pred[t_frame, :, c, z_slice_idx, :] = k_pred_slice

        # NUFFT recon for predicted
        img_c = nufft2d_recon(
            k_img_space_pred, traj_t,
            t_frame=t_frame, coil_idx=c,
            z_slice_idx=z_slice_idx,
            scales=scales, img_size=img_size, n_slices=n_z_slices,
        )
        imgs_per_coil.append(img_c)
        sos_pred_sq += img_c.astype(np.float64) ** 2

        # NUFFT recon for measured
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
    """
    End-to-end image-space evaluation: model → k-space → NUFFT → metrics.

    Reconstructs an image from the model's predicted k-space and compares it
    against (a) the NUFFT reconstruction from measured k-space, and optionally
    (b) the ground-truth image.

    Parameters
    ----------
    model : nn.Module
        Trained SIREN model that maps (kx, ky) → (Re, Im).
    x_all : (N, >=2) tensor – input coordinates (only first 2 cols used).
    y_scale : float or tensor – k-space normalisation scale.
    k_img_space : (T, S', C, nz, RO) tensor – measured k-space after kz→z IFFT.
    traj_t, scales, t_frame, coil_idx, z_slice_idx, n_z_slices : NUFFT params.
    n_ro_per_slice, RO : ints – shape info for reshaping predictions.
    img_size : (Nx, Ny) – output image size for NUFFT.
    gt_img : optional (T, nz, H, W) array – ground-truth image.
    ssim_win_size : int – SSIM window size.
    nrmse_norm : str – NRMSE normalisation mode.
    compute_perceptual : bool – if True, also compute perceptual metrics
        (DISTS, HaarPSI, VSI, LPIPS) via the piq library.

    Returns
    -------
    dict with keys:
        ``img_pred``                       – (Nx, Ny) predicted magnitude image.
        ``img_nufft_proxy``                – (Nx, Ny) measured NUFFT image.
        ``metrics_vs_nufft_proxy``         – {psnr_db, ssim, nrmse} vs measured recon.
        ``metrics_vs_ground_truth``        – {psnr_db, ssim, nrmse} vs GT (or None).
        ``perceptual_vs_nufft_proxy``      – perceptual metrics vs measured (or None).
        ``perceptual_vs_ground_truth``     – perceptual metrics vs GT (or None).

    NOTE: metrics_vs_nufft_proxy compares against NUFFT reconstruction,
    which itself has artifacts. These are PROXY metrics for relative
    comparison between models, NOT absolute quality measures.
    """
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

    # NOTE: metrics_vs_nufft_proxy compares against NUFFT reconstruction,
    # which itself has artifacts. These are PROXY metrics for relative
    # comparison between models, NOT absolute quality measures.
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

    # Perceptual metrics (optional, expensive — evaluation only)
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


# =========================================================================
# Cartesian evaluation utilities
# =========================================================================

def ifft1d_kz_to_z_cartesian(k_cart):
    """
    1D IFFT along kz dimension for Cartesian k-space.

    Args:
        k_cart: (T, C, kz, ky, kx) complex tensor

    Returns:
        k_z_space: (T, C, nz, ky, kx) complex tensor after kz->z IFFT
    """
    k_z_space = torch.fft.ifft(k_cart, dim=2)
    k_z_space = torch.fft.fftshift(k_z_space, dim=2)
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
    """
    Build evaluation dataset on the Cartesian grid.

    The Cartesian grid coordinates are normalized using the radial trajectory
    scales so that the coordinate systems match between training and evaluation.

    Args:
        k_cart_z: (T, C, nz, ky, kx) complex tensor (after kz->z IFFT)
        t_fixed: time frame index
        coil_fixed: coil index
        z_slice_idx: z-slice index
        scales_radial: (sx, sy, sz) from radial prepare_tensors()
        y_scale: k-space magnitude scale (from radial data)
        compute_device: device for output tensors

    Returns:
        x_cart: (N, 2) float [kx_norm, ky_norm] on Cartesian grid
        y_cart: (N, 2) float [Re, Im] normalized by y_scale
        meta_cart: dict with nky, nkx, etc.
    """
    sx, sy, _ = scales_radial
    dev = torch.device(compute_device)

    T, C, nz, nky, nkx = k_cart_z.shape
    z_slice_idx = int(max(0, min(int(z_slice_idx), nz - 1)))

    # Extract single slice: (ky, kx) complex
    k_slice = k_cart_z[t_fixed, coil_fixed, z_slice_idx, :, :]  # (nky, nkx)

    # Generate Cartesian grid coordinates in [-0.5, 0.5]
    kx_lin = torch.linspace(-0.5, 0.5, nkx)
    ky_lin = torch.linspace(-0.5, 0.5, nky)
    KY, KX = torch.meshgrid(ky_lin, kx_lin, indexing="ij")  # (nky, nkx)

    # Normalize by radial scales (same normalization as training data)
    sx_f = float(sx.item()) if torch.is_tensor(sx) else float(sx)
    sy_f = float(sy.item()) if torch.is_tensor(sy) else float(sy)
    KX_norm = KX / sx_f
    KY_norm = KY / sy_f

    # Flatten to (N, 2)
    x_cart = torch.stack([KX_norm.reshape(-1), KY_norm.reshape(-1)], dim=1).float().to(dev)

    # Flatten k-space values to (N, 2) [Re, Im]
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
