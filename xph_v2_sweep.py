"""spatial-vs-temporal frontier for classic GRASP v2 on the XCAT phantom.

spokes/frame is THE knob that trades the two axes: grouping G consecutive frames gives 5G
spokes/frame and F/G frames. grasp v2 was pinned at G=1 (5 spokes/frame, 344 frames), the extreme
temporal end, so its spatial score was near worst case. sweeping G traces its own frontier, so NIK
can be compared against grasp v2 AT ITS BEST rather than at an arbitrary operating point.

metric decomposition (deliberate, so the two axes stay separable):
  SPATIAL  = haarpsi/ssim/psnr vs truth AVERAGED OVER THE SAME WINDOW the frame integrates.
             generous to grasp: it does not charge grasp for temporal blur here.
  TEMPORAL = aorta curve nrmse on the FINE truth grid, coarse curve interpolated up.
             this is where coarse binning is charged, and it is charged correctly.
usage: xph_v2_sweep.py <G>     (G=1 reuses the existing grasp_v2_recon.npz)
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, json
import numpy as np
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_v2")
import xph_pipeline as P, xph_common as X
from grasp_v2_py import MCNUFFT, TVTemp, cs_l1_nlcg

A = f"{P.OUT}/arrays"; OUTD = f"{P.OUT}/v2_sweep"; os.makedirs(OUTD, exist_ok=True)
CACHE = f"{A}/xph_slice_cache.npz"
ISIGN = +1          # resolved against truth in xph_grasp_v2.py (0.9655 vs 0.9077)
NITE, NOUTER = 8, 3
# lam multiplies max|x0|. 0.25 is the published value. LOWER lam = weaker temporal TV = less bolus
# damping but noisier. swept so grasp v2 is compared at ITS best, not at a default.
LAM_FRAC = float(os.environ.get("LAM_FRAC", "0.25"))
LTAG = "" if abs(LAM_FRAC-0.25) < 1e-9 else f"_lam{LAM_FRAC:g}"

def load_cached():
    """load_slice is slow (z-fft over the whole DCE stack); cache it once for the whole sweep."""
    if not os.path.exists(CACHE):
        d = X.load_slice(P.ZI)
        np.savez(CACHE, kx=d["kx"], ky=d["ky"], kdata=d["kdata"], b1=d["b1"])
        print("wrote slice cache", flush=True)
    z = np.load(CACHE)
    return z["kx"], z["ky"], z["kdata"], z["b1"]

def rebin(kx, ky, kdata, G):
    """group G consecutive frames -> 5G spokes/frame, F//G frames. frame-major so time order holds."""
    tr = np.array(P.TRAIN_ANG); spf = len(tr)
    C, F, nang, RO = kdata.shape
    nG = F // G
    kdx = kdata[:, :nG * G][:, :, tr, :]                       # [C, nG*G, spf, RO]
    kxs = kx[:nG * G][:, tr, :]; kys = ky[:nG * G][:, tr, :]   # [nG*G, spf, RO]
    kdx = kdx.reshape(C, nG, G * spf, RO)
    kxs = kxs.reshape(nG, G * spf, RO); kys = kys.reshape(nG, G * spf, RO)
    k = (kxs + 1j * kys).transpose(2, 1, 0)                    # [RO, G*spf, nG]
    y = kdx.transpose(3, 2, 0, 1)                              # [RO, G*spf, C, nG]
    return k, y, nG, G * spf

def main():
    G = int(sys.argv[1])
    t0 = time.time()
    d = P.data(); tq = d["times"]; body = d["labels"] > 0
    Rz = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq); F = len(tq)
    nG = F // G
    tG = np.array([tq[g * G:(g + 1) * G].mean() for g in range(nG)])

    if G == 1 and abs(LAM_FRAC-0.25) < 1e-9 and os.path.exists(f"{A}/grasp_v2_recon.npz"):
        rec = np.abs(np.load(f"{A}/grasp_v2_recon.npz")["rec"])   # already oriented + scaled
        print("G=1: reusing existing grasp_v2_recon.npz", flush=True)
    else:
        kx, ky, kdata, b1 = load_cached()
        k, y, nG, nsp = rebin(kx, ky, kdata, G)
        w = np.maximum(np.abs(k), 1e-3)
        b1n = (b1 / np.abs(b1).max()).astype(np.complex64)
        E = MCNUFFT(k.astype(np.complex128), w.astype(np.float64), b1n, isign=ISIGN)
        yw = (y * np.sqrt(w)[:, :, None, :]).astype(np.complex128)
        x = E.adjoint(yw); lam = LAM_FRAC * np.abs(x).max()
        print(f"G={G}: nspokes={nsp} nt={nG} lam={lam:g}", flush=True)
        W = TVTemp()
        for it in range(NOUTER):
            x = cs_l1_nlcg(x, E, yw, W, lam, nite=NITE, display=False)
            print(f"  nlcg outer {it+1}/{NOUTER}", flush=True)
        rec0 = np.abs(x)
        # orientation is a fixed geometry fact; verify rather than re-search silently
        tmw = np.stack([Tr[:, :, g*G:(g+1)*G].mean(2) for g in range(nG)], -1).mean(2)
        def cc(a, b):
            a = a[body].ravel() - a[body].mean(); b = b[body].ravel() - b[body].mean()
            return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))
        c_id = cc(rec0.mean(2), tmw)
        if c_id < 0.5:
            print(f"  WARN orientation corr {c_id:.3f} low, re-searching", flush=True)
            cands = {"id": rec0, "rot180": rec0[::-1, ::-1], "fliplr": rec0[:, ::-1],
                     "flipud": rec0[::-1], "T": np.transpose(rec0, (1, 0, 2))}
            rec0 = cands[max(cands, key=lambda n: cc(cands[n].mean(2), tmw))]
        rec = rec0

    # ---- window-averaged truth: what the frame actually integrates ----
    TrW = np.stack([Tr[:, :, g*G:(g+1)*G].mean(2) for g in range(nG)], -1)
    s = np.sum(rec[body] * TrW[body]) / (np.sum(rec[body] ** 2) + 1e-12)
    rec = rec * s

    rv = float(TrW[body].max() - TrW[body].min())
    nrmse = float(np.mean([np.sqrt(np.mean((rec[:, :, t][body]-TrW[:, :, t][body])**2))/rv for t in range(nG)]))
    pk = float(TrW[body].max())
    mse = float(np.mean([((rec[:, :, t][body]-TrW[:, :, t][body])**2).mean() for t in range(nG)]))
    psnr = 10*np.log10(pk**2/(mse+1e-20))
    import torch
    from masked_metrics import haarpsi_masked, ssim_masked
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev)
    hs, ss = [], []
    for t in range(0, nG, max(1, nG // 40)):
        vmax = float(np.percentile(TrW[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        rt = torch.from_numpy(np.clip(TrW[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu()))
        ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))

    # ---- temporal: coarse curve interpolated onto the FINE truth grid ----
    out = dict(G=G, lam_frac=LAM_FRAC, spokes_per_frame=int(5*G), n_frames=int(nG), dt_s=float(np.diff(tq).mean()*G),
               psnr=psnr, ssim=float(np.mean(ss)), haarpsi=float(np.mean(hs)), nrmse_spatial=nrmse)
    for nm in ("aorta", "cortex", "medulla"):
        cg = rec[Rz[nm]].mean(0); ct = Tr[Rz[nm]].mean(0)
        ci = np.interp(tq, tG, cg)
        out[f"c_{nm}"] = float(np.linalg.norm(ci-ct)/(np.linalg.norm(ct)+1e-12))
    ca = np.interp(tq, tG, rec[Rz["aorta"]].mean(0)); at = Tr[Rz["aorta"]].mean(0)
    out.update(aorta_pk=float(ca.max()), aorta_pk_truth=float(at.max()),
               aorta_ttp=float(tq[np.argmax(ca)]), aorta_ttp_truth=float(tq[np.argmax(at)]))
    json.dump(out, open(f"{OUTD}/v2_G{G:02d}{LTAG}.json", "w"), indent=1)
    np.save(f"{OUTD}/v2_G{G:02d}{LTAG}.npy", rec.astype(np.float32))
    print(json.dumps(out, indent=1))
    print(f"SWEEP_G{G}_DONE ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
