"""TASK 4: CS reconstruct-then-fit baseline. Stage 1: free-frame CS recon of the full 181-frame
series from the identical k-space, using a GENERIC spatial+temporal-TV prior that knows nothing
about the F0 basis (implemented by running cs_nufft's solver with an IDENTITY temporal basis, so
the 'coefficients' ARE the frames). Stage 2: fit the 3-atom F0 basis Phi to the reconstructed
pixel time-curves. This is the standard clinical two-stage pipeline, on identical measurements.
usage: python task4_csfit.py --frac f25"""
import warnings; warnings.filterwarnings("ignore")
import argparse, sys, numpy as np; sys.path.insert(0, ".")
from cs_nufft import NufftSubspace, recon as cs_recon
OUT = "/scratch/rnga/vvpshenov/DCE_NIK/results/task4_xcat_nomotion_pilot"
ap = argparse.ArgumentParser(); ap.add_argument("--frac", required=True, choices=["f100", "f25"]); a = ap.parse_args()
S = np.load(f"{OUT}/arrays/sim.npz"); Phi = S["Phi"]; b1 = S["b1"]; kx = S["kx"]; ky = S["ky"]; keep = S["keep_f25"]
F, NA, RO = kx.shape; C = b1.shape[-1]; N = RO
y = np.load(f"{OUT}/arrays/y{'100' if a.frac=='f100' else '25_in'}.npy", allow_pickle=True)
trajs = [(kx[t], ky[t]) for t in range(F)] if a.frac == "f100" else [(kx[t][keep[t]], ky[t][keep[t]]) for t in range(F)]
yl = [np.asarray(y[t]).astype(np.complex64) for t in range(F)]
# ---- Stage 1: free-frame CS recon (identity basis => per-frame images, spatial+temporal TV) ----
Ieye = np.eye(F).astype(np.complex64)
E_free = NufftSubspace(Ieye, b1, trajs, N)
x_series = cs_recon(E_free, yl, iters=40, ls=1e-3, lt=1e-3, verbose=True)     # [N,N,F] free frames
# ---- Stage 2: fit the F0 basis Phi to the reconstructed pixel time-curves ----
theta_csfit = np.einsum("rt,xyt->xyr", np.linalg.pinv(Phi), x_series)          # least-squares fit
np.savez(f"{OUT}/arrays/csfit_{a.frac}.npz", theta_csfit=theta_csfit.astype(np.complex64), x_series=x_series.astype(np.complex64))
print(f"DONE {a.frac}: CS-recon-then-fit -> {OUT}/arrays/csfit_{a.frac}.npz", flush=True)
