"""is lam=0.02 effectively "no TV"? if so, tuned grasp v2 is really unregularized NUFFT-SENSE and
the CS machinery that DEFINES grasp contributes nothing to the comparison.

two measurements at 40 spokes/frame (the tuned operating point):
  1 how much of the converged objective is the TV term at each lam
  2 how different is the lam=0.02 recon from a true lam=0 recon
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, glob, json
import numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK"); sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_v2")
import xph_pipeline as P, xph_common as X
from grasp_v2_py import MCNUFFT, TVTemp

SW = f"{P.OUT}/v2_sweep"; A = f"{P.OUT}/arrays"; G = 8; ISIGN = +1
z = np.load(f"{A}/xph_slice_cache.npz"); kx, ky, kdata, b1 = z["kx"], z["ky"], z["kdata"], z["b1"]
tr = np.array(P.TRAIN_ANG); spf = len(tr); C, F, nang, RO = kdata.shape
nG = F // G
kdx = kdata[:, :nG*G][:, :, tr, :].reshape(C, nG, G*spf, RO)
kxs = kx[:nG*G][:, tr, :].reshape(nG, G*spf, RO); kys = ky[:nG*G][:, tr, :].reshape(nG, G*spf, RO)
k = (kxs + 1j*kys).transpose(2, 1, 0); y = kdx.transpose(3, 2, 0, 1)
w = np.maximum(np.abs(k), 1e-3); b1n = (b1/np.abs(b1).max()).astype(np.complex64)
E = MCNUFFT(k.astype(np.complex128), w.astype(np.float64), b1n, isign=ISIGN)
yw = (y*np.sqrt(w)[:, :, None, :]).astype(np.complex128)
x0 = E.adjoint(yw); mx = float(np.abs(x0).max()); W = TVTemp()

print(f"40 spokes/frame, {nG} frames.  max|x0| = {mx:.4g}\n")
print(f"{'lam_frac':>9} {'lam':>10} {'||Ex-y||^2':>12} {'lam*||dt x||_1':>15} {'TV share of obj':>16}")
res = {}
for f in sorted(glob.glob(f"{SW}/v2_G08*.npy")):
    bn = os.path.basename(f)[:-4]
    lf = float(bn.split("_lam")[1]) if "_lam" in bn else 0.25
    x = np.load(f).astype(np.complex128)
    s = np.vdot(np.abs(E.adjoint(yw)).ravel(), np.abs(x).ravel())/ (np.vdot(np.abs(x).ravel(), np.abs(x).ravel())+1e-30)
    x = x * s                                            # put magnitude recon back on the operator's scale
    r = E.forward(x) - yw; L2 = float(np.vdot(r.ravel(), r.ravel()).real)
    L1 = float(np.sum(np.abs(W @ x)))
    lam = lf * mx
    res[lf] = (L2, lam*L1)
    print(f"{lf:>9.3f} {lam:>10.4g} {L2:>12.4g} {lam*L1:>15.4g} {100*lam*L1/(L2+lam*L1):>15.1f}%")
print("\nif the TV share is a few percent, the temporal TV constraint is barely active and the recon is")
print("close to an unregularized data fit. that would make 'tuned grasp v2' NUFFT-SENSE, not GRASP.")
json.dump({str(k_): v for k_, v in res.items()}, open(f"{SW}/lam_meaning.json", "w"), indent=1)
print("LAM_MEANING_DONE")
