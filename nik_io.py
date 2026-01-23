# nik_io.py
import itertools
import h5py
import numpy as np

PATH_K_DCE   = "/results/kspace/DCE"
PATH_TRAJ    = "/results/kspace/trajDCE"
PATH_GT_IMG  = "/results/images/GroundTruth/img"
PATH_RC_IMG  = "/results/images/Recon/img"
PATH_GT_TIM  = "/results/images/GroundTruth/timing"
PATH_RC_TIM  = "/results/images/Recon/timing"


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

def load_event(file_path: str, load_images: bool = False):
    """
    Returns:
      k_np    : complex64 (RO,S,C,T)
      traj_np : float32  (RO,S,3,T)
      optional: gt_img, rc_img, gt_tim, rc_tim
    """
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

    return out


