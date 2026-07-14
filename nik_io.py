"""h5 io, xcat eric data"""
import itertools
import os
import h5py
import numpy as np

PATH_K_DCE   = "/results/kspace/DCE"
PATH_TRAJ    = "/results/kspace/trajDCE"
PATH_GT_IMG  = "/results/images/GroundTruth/img"
PATH_RC_IMG  = "/results/images/Recon/img"
PATH_GT_TIM  = "/results/images/GroundTruth/timing"
PATH_RC_TIM  = "/results/images/Recon/timing"
PATH_SP             = "/results/kspace/SP"
PATH_SPOKE_TIMING_DCE = "/results/kspace/spokeTimingDCE"

# fallback path for slice profile when not saved in the radial file
SP_FALLBACK_PATH = "/scratch/rnga/vvpshenov/XCAT-ERIC/utilities/sampling/SliceProfile.mat"


def h5_tree(file_path: str, max_items: int = 250) -> None:
    items = []
    with h5py.File(file_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                items.append(("D", name, obj.shape, str(obj.dtype)))
            else:
                items.append(("G", name, None, None))
        f.visititems(visit)

    for t, name, shape, dtype in items[:max_items]:
        if t == "D":
            print(f"DATASET  {name:50s}  shape={shape}  dtype={dtype}")
        else:
            print(f"GROUP    {name}")
    if len(items) > max_items:
        print(f"... ({len(items)-max_items} more)")


def _as_complex(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.fields and ("real" in arr.dtype.fields) and ("imag" in arr.dtype.fields):
        return arr["real"] + 1j * arr["imag"]
    return arr


def h5_load(file_path: str, path_in_file: str) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        arr = f[path_in_file][()]
    return _as_complex(arr)


def h5_exists(file_path: str, path_in_file: str) -> bool:
    with h5py.File(file_path, "r") as f:
        return path_in_file in f

PATH_COIL_MAPS = "/results/kspace/coilMaps"
PATH_META_PREFIX = "/results/kspace/meta/"


def load_event(file_path: str, load_images: bool = False, load_coil_maps: bool = False):
    """radial event loader"""
    k_np  = h5_load(file_path, PATH_K_DCE)
    traj_np = h5_load(file_path, PATH_TRAJ)

    k_np = k_np.astype(np.complex64, copy=False)
    traj_np = traj_np.astype(np.float32, copy=False)

    out = {"k": k_np, "traj": traj_np}

    if load_images:
        out["gt_img"] = h5_load(file_path, PATH_GT_IMG).astype(np.float32) if h5_exists(file_path, PATH_GT_IMG) else None
        out["rc_img"] = h5_load(file_path, PATH_RC_IMG).astype(np.float32) if h5_exists(file_path, PATH_RC_IMG) else None
        out["gt_tim"] = h5_load(file_path, PATH_GT_TIM).reshape(-1) if h5_exists(file_path, PATH_GT_TIM) else None
        out["rc_tim"] = h5_load(file_path, PATH_RC_TIM).reshape(-1) if h5_exists(file_path, PATH_RC_TIM) else None

    if load_coil_maps:
        out["coil_maps"] = h5_load(file_path, PATH_COIL_MAPS).astype(np.float64) if h5_exists(file_path, PATH_COIL_MAPS) else None

    # per-spoke acquisition time in ms, DCE-sorted (shape: nROperFrame x nt). new field; older sims may not have it.
    out["spoke_timing_dce"] = (
        h5_load(file_path, PATH_SPOKE_TIMING_DCE).astype(np.float64)
        if h5_exists(file_path, PATH_SPOKE_TIMING_DCE) else None
    )

    return out


def load_cartesian_kspace(file_path: str, load_images: bool = False, load_coil_maps: bool = True):
    """cartesian event loader"""
    k_np = h5_load(file_path, PATH_K_DCE)
    k_np = k_np.astype(np.complex64, copy=False)

    out = {"k_cart": k_np}

    # metadata
    meta = {}
    with h5py.File(file_path, "r") as f:
        for key in ["matrixRL", "matrixAP", "matrixFH"]:
            path = PATH_META_PREFIX + key
            if path in f:
                meta[key] = int(f[path][()].flat[0])
    out["meta"] = meta

    if load_coil_maps:
        out["coil_maps"] = h5_load(file_path, PATH_COIL_MAPS).astype(np.float64) if h5_exists(file_path, PATH_COIL_MAPS) else None

    if load_images:
        out["gt_img"] = h5_load(file_path, PATH_GT_IMG).astype(np.float32) if h5_exists(file_path, PATH_GT_IMG) else None
        out["rc_img"] = h5_load(file_path, PATH_RC_IMG).astype(np.float32) if h5_exists(file_path, PATH_RC_IMG) else None

    return out


def _load_slice_profile(file_path: str, n_fh: int) -> np.ndarray:
    """slice profile from radial file or fallback mat, resampled to n_fh"""
    if h5_exists(file_path, PATH_SP):
        sp = h5_load(file_path, PATH_SP).reshape(-1)
    elif os.path.exists(SP_FALLBACK_PATH):
        import scipy.io
        sp = scipy.io.loadmat(SP_FALLBACK_PATH)["SP"].reshape(-1)
    else:
        raise FileNotFoundError(
            f"slice profile not in {file_path} and fallback {SP_FALLBACK_PATH} missing"
        )
    if len(sp) != n_fh:
        from scipy.signal import resample_poly
        sp = resample_poly(sp, n_fh, len(sp))
        h = n_fh // 2
        sp[h + 1:] = sp[h - 1::-1][:n_fh - h - 1]
    return sp.astype(np.float32)


def _bin_gt_to_dce(gt_img: np.ndarray, gt_tim, rc_tim, T_target: int) -> np.ndarray:
    """average gt phases into dce bin windows, uniform-count fallback"""
    if rc_tim is not None:
        t_dce = np.asarray(rc_tim).reshape(-1)
        if t_dce.size != T_target:
            # mismatch: fall back to uniform-count split
            pass
        else:
            L = float(t_dce[1] - t_dce[0]) if t_dce.size > 1 else (2.0 * float(t_dce[0]))
            edges = np.concatenate([t_dce - L / 2, t_dce[-1:] + L / 2])
            if gt_tim is not None:
                t_gt = np.asarray(gt_tim).reshape(-1)
                out = np.zeros((t_dce.size,) + gt_img.shape[1:], dtype=np.float32)
                for k in range(t_dce.size):
                    m = (t_gt >= edges[k]) & (t_gt < edges[k + 1])
                    if not m.any():
                        j = int(np.argmin(np.abs(t_gt - t_dce[k])))
                        out[k] = gt_img[j]
                    else:
                        out[k] = gt_img[m].mean(axis=0)
                return out
    # uniform-count fallback aligned to T_target
    n_phases = gt_img.shape[0]
    idx_edges = np.linspace(0, n_phases, T_target + 1).astype(int)
    return np.stack(
        [gt_img[idx_edges[k]:idx_edges[k + 1]].mean(axis=0) for k in range(T_target)]
    ).astype(np.float32)


def synthesize_cartesian_from_radial(
    radial_file: str,
    T_target: int,
    *,
    event: dict = None,
) -> dict:
    """build dense cartesian event from radial gt_img + coil_maps + sp

    matches the layout of load_cartesian_kspace:
        k_cart:    (T, C, kz, ky, kx) complex64, on the (padded) coil grid
        coil_maps: (C, kz, RL_pad, AP_pad) float64
        gt_img:    (T, kz, RL_pad, AP_pad) float32, dce-binned, padded
        meta:      {matrixRL, matrixAP, matrixFH}
    """
    if event is None:
        event = load_event(radial_file, load_images=True, load_coil_maps=True)
    gt_img    = event.get("gt_img")
    coil_maps = event.get("coil_maps")
    gt_tim    = event.get("gt_tim")
    rc_tim    = event.get("rc_tim")

    if gt_img is None or coil_maps is None:
        raise ValueError(
            "radial file lacks gt_img or coil_maps; cannot synthesize cartesian"
        )

    n_phases, n_kz, n_RL, n_AP   = gt_img.shape
    n_C,      _,    n_RL_pad, n_AP_pad = coil_maps.shape

    sp = _load_slice_profile(radial_file, n_kz)
    gt_binned = _bin_gt_to_dce(gt_img, gt_tim, rc_tim, T_target)

    pad_RL = (n_RL_pad - n_RL) // 2
    pad_AP = (n_AP_pad - n_AP) // 2
    gt_pad = np.pad(
        gt_binned,
        ((0, 0), (0, 0), (pad_RL, pad_RL), (pad_AP, pad_AP)),
    ).astype(np.float32)

    sp_b = sp.reshape(-1, 1, 1)  # (kz, 1, 1)
    k_cart = np.empty(
        (T_target, n_C, n_kz, n_AP_pad, n_RL_pad), dtype=np.complex64
    )
    for t in range(T_target):
        img = gt_pad[t]                                       # (kz, RL_pad, AP_pad)
        obj = coil_maps * img[None] * sp_b[None]              # (C, kz, RL_pad, AP_pad)
        k   = np.fft.fftshift(np.fft.fftn(obj, axes=(1, 2, 3)), axes=(1, 2, 3))
        k_cart[t] = k.transpose(0, 1, 3, 2).astype(np.complex64)

    return {
        "k_cart":    k_cart,
        "coil_maps": coil_maps,
        "gt_img":    gt_pad,
        "meta":      {"matrixRL": n_RL_pad, "matrixAP": n_AP_pad, "matrixFH": n_kz},
        "SP":        sp,
    }


