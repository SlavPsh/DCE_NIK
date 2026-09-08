"""TASK 4 baselines: matched direct-discrete subspace recon (cs_nufft, same F0 basis + forward)
and a NUFFT adjoint sanity recon, for f100 and f25. Fixed default regularizer (cs_nufft repo
default ls=1e-3, lt=1e-3), NOT tuned on truth. usage: python task4_baselines.py --frac f25"""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, sys, numpy as np, finufft
sys.path.insert(0, "."); from cs_nufft import NufftSubspace, recon as cs_recon, frames_from_coeff
D = "/scratch/rnga/vvpshenov/DCE_NIK"; OUT = f"{D}/results/task4_xcat_nomotion_pilot"; SIGN = -1.0
ap = argparse.ArgumentParser(); ap.add_argument("--frac", required=True, choices=["f100", "f25"]); a = ap.parse_args()
S = np.load(f"{OUT}/arrays/sim.npz"); Phi = S["Phi"]; b1 = S["b1"]; kx = S["kx"]; ky = S["ky"]; keep = S["keep_f25"]
F, NA, RO = kx.shape; C = b1.shape[-1]; N = RO
y = np.load(f"{OUT}/arrays/y{'100' if a.frac=='f100' else '25_in'}.npy", allow_pickle=True)
if a.frac == "f100":
    trajs = [(kx[t], ky[t]) for t in range(F)]
else:
    trajs = [(kx[t][keep[t]], ky[t][keep[t]]) for t in range(F)]
yl = [np.asarray(y[t]).astype(np.complex64) for t in range(F)]
E = NufftSubspace(Phi, b1, trajs, N)
# ---- direct discrete subspace (FISTA, fixed default reg) ----
theta_direct = cs_recon(E, yl, iters=40, ls=1e-3, lt=1e-3, verbose=True)
# ---- NUFFT adjoint sanity: per-frame density-comp adjoint -> dynamic -> project onto Phi ----
Icn = np.zeros((N, N, F), np.complex64)
for t in range(F):
    kxf, kyf = trajs[t]; kxr = (SIGN*2*np.pi*kxf.ravel()).astype(np.float64); kyr = (SIGN*2*np.pi*kyf.ravel()).astype(np.float64)
    w = np.maximum(np.abs(kxf.ravel()+1j*kyf.ravel()), 1e-3)                 # ramp density comp
    yc = np.asarray(yl[t]).reshape(C, -1)
    ci = np.stack([finufft.nufft2d1(kxr, kyr, (yc[c]*w).astype(np.complex128), (N,N), isign=1, eps=1e-4) for c in range(C)], -1)
    Icn[:, :, t] = np.sum(np.conj(b1)*ci, -1)/(np.sum(np.abs(b1)**2,-1)+1e-8)
theta_nufft = np.einsum("rt,xyt->xyr", np.linalg.pinv(Phi), Icn)
np.savez(f"{OUT}/arrays/direct_{a.frac}.npz", theta_direct=theta_direct.astype(np.complex64))
np.savez(f"{OUT}/arrays/nufft_{a.frac}.npz", theta_nufft=theta_nufft.astype(np.complex64), Ic=Icn.astype(np.complex64))
print(f"DONE {a.frac}: direct+nufft baselines -> {OUT}/arrays/", flush=True)
