"""NUFFT as a METHOD at each spoke fraction (naive gridding baseline), using the SAME
shared keep-files as CS and NIK. Static recon (all kept spokes) + pre-contrast recon
(kept spokes inside the pre-contrast window). References stay the full-spoke versions.
out: results_nufft/frac_{lab}.npy, frac_pre_{lab}.npy"""
import numpy as np, finufft, json, os
REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
D = "/scratch/rnga/vvpshenov/DCE_NIK"; NUF = f"{D}/results_nufft"
SL = 13
meta = json.load(open(f"{NUF}/meta.json")); SIGN = meta["sign"]
LABS = ["f100", "f70", "f50", "f25"]

sh = np.load(f"{REF}/shared.npz")
traj = np.asarray(sh["traj_norm"]).astype(np.complex64)
nx = int(sh["nx"]); bas = int(sh["bas"])
sl = np.load(f"{REF}/slice_{SL:02d}.npz")
kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64)
b1 = np.asarray(sl["b1"]).astype(np.complex64); ncc = kdata.shape[2]
den = np.sum(np.abs(b1) ** 2, axis=2) + 1e-12
pre_idx = np.load(f"{NUF}/pre_view_idx.npy")

def recon(idx, eps=1e-6):
    tr = traj[:, idx]; w = np.maximum(np.abs(tr), 1.0 / nx / 4.0)
    x = (SIGN * 2 * np.pi * np.real(tr)).astype(np.float64).ravel()
    y = (SIGN * 2 * np.pi * np.imag(tr)).astype(np.float64).ravel()
    acc = np.zeros((nx, nx), dtype=np.complex128)
    for c in range(ncc):
        v = (kdata[:, idx, c] * w).astype(np.complex128).ravel()
        acc += finufft.nufft2d1(x, y, v, (nx, nx), eps=eps, isign=1) * np.conj(b1[:, :, c])
    img = np.abs(acc / den); s = (nx - bas) // 2
    return img[s:s + bas, s:s + bas].astype(np.float32)

for lab in LABS:
    keep = np.load(f"{D}/spoke_masks/keep_{lab}.npy")
    kp = np.intersect1d(keep, pre_idx)
    np.save(f"{NUF}/frac_{lab}.npy", recon(keep))
    np.save(f"{NUF}/frac_pre_{lab}.npy", recon(kp))
    print(f"{lab}: {len(keep):5d} spokes total, {len(kp):4d} in pre-contrast window", flush=True)
print(f"wrote -> {NUF}/frac_*.npy")
