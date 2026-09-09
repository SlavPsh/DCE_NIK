"""classic GRASP (v2, feng/otazo 2014) on the XCAT phantom, matched to xph_grasp_fair's inputs.

same slice (ZI), same TRAIN_ANG 5 spokes/frame (spoke-matched to NIK), same ramp dcf, same true
coil maps, same orientation search and LS scale-to-truth. ONLY the algorithm differs:
  grasp-pro (xph_grasp_fair) = pca subspace K* + spatial&temporal TV + NUFFT-SENSE
  grasp v2  (here)           = mcnufft + temporal TV only + nlcg, NO subspace, so no K
isign resolved empirically against truth (the phantom carries its own k-space convention).
out: arrays/grasp_v2_recon.npz
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time
import numpy as np
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_v2")
import xph_pipeline as P, xph_common as X
from grasp_v2_py import MCNUFFT, TVTemp, cs_l1_nlcg

NITE, NOUTER, LAM_FRAC = 8, 3, 0.25
A = f"{P.OUT}/arrays"; os.makedirs(A, exist_ok=True)

def corr(a, b, m):
    a = a[m].ravel() - a[m].mean(); b = b[m].ravel() - b[m].mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def orient(v, nm):
    return {"id": v, "rot180": v[::-1, ::-1], "fliplr": v[:, ::-1],
            "flipud": v[::-1], "T": np.transpose(v, (1, 0, 2))}[nm]

def build(isign):
    d = X.load_slice(P.ZI)
    kx, ky, kdata, b1 = d["kx"], d["ky"], d["kdata"], d["b1"]
    tr = np.array(P.TRAIN_ANG)                                     # 5 spokes/frame, same as grasp-pro
    C, F, nang, RO = kdata.shape
    k = (kx[:, tr] + 1j * ky[:, tr]).transpose(2, 1, 0)            # [RO, 5, F] cycles
    w = np.maximum(np.abs(k), 1e-3)                                # same ramp dcf as xph_grasp_nufft
    y = kdata[:, :, tr, :].transpose(3, 2, 0, 1)                   # [RO, 5, C, F]
    b1n = (b1 / np.abs(b1).max()).astype(np.complex64)
    E = MCNUFFT(k.astype(np.complex128), w.astype(np.float64), b1n, isign=isign)
    yw = (y * np.sqrt(w)[:, :, None, :]).astype(np.complex128)
    return E, yw, (C, F, nang, RO)

if __name__ == "__main__":
    t0 = time.time()
    d = X.load_slice(P.ZI); tq = d["times"] if "times" in d else P.data()["times"]
    body = P.data()["labels"] > 0
    Tr = X.truth_at(P.ZI, tq); tm = Tr.mean(2)
    best = None
    for isign in (-1, +1):                                          # phantom k-space convention, resolved by data
        E, yw, shp = build(isign)
        x0 = E.adjoint(yw)
        c = max(corr(orient(np.abs(x0), nm).mean(2), tm, body) for nm in ("id", "rot180", "fliplr", "flipud", "T"))
        print(f"  isign={isign:+d}: adjoint best-orientation corr vs truth = {c:+.4f}", flush=True)
        if best is None or c > best[0]: best = (c, isign)
    isign = best[1]
    print(f"chosen isign={isign:+d} (corr {best[0]:+.4f}) | shapes C,F,nang,RO={shp}", flush=True)

    E, yw, _ = build(isign)
    x = E.adjoint(yw); lam = LAM_FRAC * np.abs(x).max()
    print(f"lam={lam:g}  nt={E.nt}  nspokes={E.nspokes}  grid={E.Nd}  (no subspace, no K)", flush=True)
    W = TVTemp()
    for it in range(NOUTER):
        x = cs_l1_nlcg(x, E, yw, W, lam, nite=NITE, display=True)
        print(f"  nlcg outer {it+1}/{NOUTER} done", flush=True)

    rec0 = np.abs(x)
    nm = max(["id", "rot180", "fliplr", "flipud", "T"], key=lambda n: corr(orient(rec0, n).mean(2), tm, body))
    rec = orient(rec0, nm)
    print(f"orientation best: {nm} corr {corr(rec.mean(2), tm, body):.3f}")
    s = np.sum(rec[body] * Tr[body]) / (np.sum(rec[body] ** 2) + 1e-12)   # same LS scale-to-truth as grasp-pro
    rec = rec * s
    def nrmse(a): return float(np.sqrt(np.mean((a.mean(2)[body] - tm[body]) ** 2)) / (tm[body].max() - tm[body].min()))
    pro = np.load(f"{A}/grasp_recon.npz")["rec"]
    print(f"imgNRMSE vs truth:  grasp-pro {nrmse(pro):.4f}   grasp v2 {nrmse(rec):.4f}")
    np.savez(f"{A}/grasp_v2_recon.npz", rec=rec.astype(np.float32), orient=nm, scale=float(s),
             isign=isign, spf=len(P.TRAIN_ANG), algo="classic GRASP v2, temporal TV only, no subspace")
    print(f"SAVED {A}/grasp_v2_recon.npz ({time.time()-t0:.0f}s)")
    print("XPH_GRASP_V2_DONE")
