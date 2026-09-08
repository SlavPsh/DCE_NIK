"""Lock ONE defensible GRASP-Pro config and reconstruct with it, for the honest final comparison.
Choices (all truth-blind):
  - navigator: per-frame-averaged k-centre DC magnitude (denoised; ranks the bolus into the top PCs).
  - spokes/frame = 5 (344 frames): matched to NIK's temporal resolution (no coarsening-denoise advantage).
  - K: smallest rank retaining >= VAR_THRESH of navigator variance (pre-registered; not tuned on truth).
Prints the eigenspectrum + the chosen K, reconstructs, overwrites grasp_recon.npz (old K5 backed up),
and prints sanity metrics vs truth."""
import warnings; warnings.filterwarnings("ignore")
import os, shutil, numpy as np
import xph_grasp_nufft as GN, xph_pipeline as P, xph_common as X
import precompute_ref as pr
VAR_THRESH = 0.99
A = f"{X.OUT}/arrays"

d = P.data(); kx = d["kx"]; ky = d["ky"]; kdata = d["kdata"]; b1 = d["b1"]
C, F, nang, RO = kdata.shape; tr = np.array(P.TRAIN_ANG); c0 = RO // 2
tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)

# ---- navigators + eigenspectra (truth-blind) ----
nav_avg = np.stack([np.abs(kdata[:, fr, tr, c0-2:c0+3]).mean(0).ravel() for fr in range(F)], 1)   # [5off, F]  (avg over coils+spokes? no: mean(0) over C) -> keep per-offset
# richer averaged navigator: mean over training spokes only, keep coils+offsets -> [C*5, F]
nav_avg = np.stack([np.abs(kdata[:, fr, tr, c0-2:c0+3]).mean(1).reshape(-1) for fr in range(F)], 1)  # [C*5, F]
nav_raw = np.abs(kdata[:, :, tr, c0-2:c0+3]).transpose(0, 2, 3, 1).reshape(-1, F)                    # [C*5*5, F]
def spectrum(nav, tag):
    w = np.sort(np.linalg.eigvalsh(np.cov(nav, rowvar=False)))[::-1]; cum = np.cumsum(w) / w.sum()
    ks = {th: int(np.searchsorted(cum, th) + 1) for th in (0.90, 0.95, 0.99, 0.995, 0.999)}
    print(f"[{tag}] top-16 var frac:", np.round(w[:16] / w.sum(), 4))
    print(f"[{tag}] K at thresholds:", ks, flush=True)
    return cum
cum_avg = spectrum(nav_avg, "avg-nav"); spectrum(nav_raw, "raw-nav")
Kstar = int(np.searchsorted(cum_avg, VAR_THRESH) + 1)
print(f"LOCKED navigator=avg, spf=5, K*={Kstar} (>= {VAR_THRESH:.0%} variance)", flush=True)

# ---- reconstruct at the locked config (avg navigator, spf=5, K*) ----
def build_phi_avg(K):
    w, PC = np.linalg.eigh(np.cov(nav_avg, rowvar=False)); return PC[:, np.argsort(-w)][:, :K].astype(np.complex64)
trajs = [(-kx[t, tr], -ky[t, tr]) for t in range(F)]
dcf = [np.maximum(np.abs(kx[t, tr] + 1j * ky[t, tr]), 1e-3) for t in range(F)]
Phi = build_phi_avg(Kstar); PCA = pr.TempPCASub(Phi)
E = GN.Emat_NUFFT(trajs, dcf, b1, Phi, RO)
raw = np.stack([kdata[:, t, tr, :].reshape(C, -1) for t in range(F)], 1).astype(np.complex128)
y = E.apply_dcf(raw); recon = E.H @ y
for it in range(GN.NOUTER):
    recon = pr.cs_l1_nlcg_sptv(recon, dict(E=E, y=y, PCA=PCA, TV1=pr.TV_Temp(), TV2=pr.FD1OP(),
        TVWeight1=np.abs(recon).max()*pr.Weight1, TVWeight2=np.abs(recon).max()*pr.Weight2, nite=GN.NITE))
    print(f"  nlcg outer {it+1}/{GN.NOUTER} finite={np.isfinite(recon).all()}", flush=True)
dyn = np.abs(np.asarray(PCA.H @ recon))

# ---- orient + scale to truth ----
tm = Tr.mean(2)
def orient(v, nm): return {"id": v, "rot180": v[::-1, ::-1], "fliplr": v[:, ::-1], "flipud": v[::-1], "T": np.transpose(v, (1, 0, 2)), "rot90": np.rot90(v), "rot270": np.rot90(v, 3)}[nm]
def cc(a, b): a = a[body].ravel()-a[body].mean(); b = b[body].ravel()-b[body].mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
best = max(["id", "rot180", "fliplr", "flipud", "T", "rot90", "rot270"], key=lambda nm: cc(orient(dyn, nm).mean(2), tm))
rec = orient(dyn, best); s = float((rec[body]*Tr[body]).sum()/((rec[body]**2).sum()+1e-12)); rec = rec*s

# ---- sanity metrics ----
rv = float(Tr[body].max()-Tr[body].min()); _pre = tq < 18
def _bsub(c): return c - np.median(c[_pre])
def mnr(a, b): return float(np.sqrt(np.mean((a[body]-b[body])**2))/(rv+1e-12))
truec = {nm: _bsub(np.median(Tr[R[nm]], 0)) for nm in ["aorta", "cortex", "medulla"]}
cur = {nm: float(np.linalg.norm(_bsub(np.median(rec[R[nm]], 0))-truec[nm])/np.linalg.norm(truec[nm])) for nm in truec}
ac = _bsub(np.median(rec[R["aorta"]], 0)); peak = float(ac.max())
imgn = float(np.mean([mnr(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))]))
print(f"FAIR GRASP: orient={best} K*={Kstar} | imgNRMSE {imgn:.4f} | aortaCurve {cur['aorta']:.3f} peak {peak:.3f} | cortex {cur['cortex']:.3f} medulla {cur['medulla']:.3f}", flush=True)

# ---- persist: back up K5, overwrite grasp_recon.npz so aggregate/notebook/report pick it up ----
if os.path.exists(f"{A}/grasp_recon.npz") and not os.path.exists(f"{A}/grasp_recon_K5_pernav.npz"):
    shutil.copy(f"{A}/grasp_recon.npz", f"{A}/grasp_recon_K5_pernav.npz")
np.savez(f"{A}/grasp_recon.npz", rec=rec.astype(np.float32), Phi=Phi, orient=best, scale=s,
         K=Kstar, navigator="avg", spf=5, var_thresh=VAR_THRESH)
np.savez(f"{A}/grasp_fair_recon.npz", rec=rec.astype(np.float32), Phi=Phi, K=Kstar, navigator="avg", spf=5)
print("SAVED grasp_recon.npz (fair) + grasp_fair_recon.npz; old -> grasp_recon_K5_pernav.npz"); print("FAIR_DONE")
