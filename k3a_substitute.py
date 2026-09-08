"""K3a: data-consistency SUBSTITUTION ceiling test (no training). For the existing complex NIK
recons (sub16, free; all seeds + complex seed-avg), inject gridded MEASURED low-|k| values at the
per-frame covered cells, then score vs XCAT truth. Bounds the achievable gain from a DC step.

CONSISTENCY: substitution is done as  K_sub = K_model + m . alpha . (Gmeas_k - Gmod_k)  at covered
cells, where Gmeas/Gmod are the density-compensated adjoint gridding (finufft type-1 + ramp DCF +
SENSE) of the MEASURED data resp. the MODEL's own radial predictions, and alpha is the pure
gridding<->render convention factor calibrated from <Gmod_k,K_model> on covered low-|k| cells (same
content, different pipeline). Thus alpha.Gmod_k ~ K_model, so at covered cells K_sub ~ alpha.Gmeas_k
(true substitution) and radius=0 reproduces baseline exactly. Only MEASURED data + b1 + DCF used; no
truth enters the substitution. Native (acquisition) frame throughout; rot->truth only for scoring."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob, finufft
import xph_pipeline as P, xph_common as X, recon_asserts as RA
from masked_metrics import haarpsi_masked, ssim_masked
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1))
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"])
kx = d["kx"]; ky = d["ky"]; RO = d["b1"].shape[0]; b1 = d["b1"].astype(np.complex128); F = len(tq)
tr = np.array(P.TRAIN_ANG); rv = float(Tr[body].max()-Tr[body].min()); Ttime = float(tq.max())
den = (np.sum(np.abs(b1)**2, -1) + 1e-8)
yy, xxg = np.mgrid[0:RO, 0:RO]; r01g = np.sqrt(((xxg-RO/2)/(RO/2))**2 + ((yy-RO/2)/(RO/2))**2)

# ---------- gridding: density-compensated adjoint of per-coil radial values -> coil-combined k-space ----------
def grid_kspace(vals_fcm):                                                        # vals_fcm[t] = [C, M] complex
    Gk = np.zeros((RO, RO, F), np.complex128)
    for t in range(F):
        fxc = np.ascontiguousarray((2*np.pi*kx[t, tr]).reshape(-1).astype(np.float64))
        fyc = np.ascontiguousarray((2*np.pi*ky[t, tr]).reshape(-1).astype(np.float64))
        dcf = np.maximum(np.abs(kx[t, tr]+1j*ky[t, tr]).reshape(-1), 1e-3).astype(np.float64)  # ramp DCF
        cimg = finufft.nufft2d1(fxc, fyc, np.ascontiguousarray(vals_fcm[t]*dcf[None]), (RO, RO), isign=1, eps=1e-6)  # [C,RO,RO]
        gi = np.sum(np.conj(b1)*np.transpose(cimg, (1, 2, 0)), -1)/den             # SENSE combine
        Gk[:, :, t] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gi)))
    return Gk

meas_fcm = [np.ascontiguousarray(d["kdata"][:, t, tr, :].reshape(C, -1).astype(np.complex128)) for t in range(F)]
Gmeas = grid_kspace(meas_fcm)

# ---------- per-frame covered-cell mask (this frame's 5 spokes, +/-1 cell) ----------
def cov_mask(t):
    c = np.zeros((RO, RO), bool)
    ix = np.clip(np.round(kx[t, tr]/0.5*(RO/2)+RO/2).astype(int), 0, RO-1).ravel()
    iy = np.clip(np.round(ky[t, tr]/0.5*(RO/2)+RO/2).astype(int), 0, RO-1).ravel()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1): c[np.clip(iy+dy, 0, RO-1), np.clip(ix+dx, 0, RO-1)] = True
    return c
COV = np.stack([cov_mask(t) for t in range(F)], -1)

# ---------- model machinery ----------
@torch.no_grad()
def model_radial(model):                                                          # model's per-coil radial preds -> vals_fcm
    out = []
    for t in range(F):
        cx = torch.tensor(np.stack([2*kx[t, tr].reshape(-1), 2*ky[t, tr].reshape(-1)], 1), dtype=torch.float32, device=dev)
        tt = torch.full((cx.shape[0],), 2*tq[t]/Ttime-1, dtype=torch.float32, device=dev)
        vc = np.empty((C, cx.shape[0]), np.complex128)
        for c in range(C):
            pr = nz.denormalize(cx, model(cx, tt, torch.full((cx.shape[0],), c, dtype=torch.long, device=dev)))
            vc[c] = (pr[:, 0]+1j*pr[:, 1]).cpu().numpy()
        out.append(np.ascontiguousarray(vc))
    return out

def build_model(kind, seed, step):
    tag = (f"sub16_w768_s{seed}" if kind == "sub16" else f"free_w768_s{seed}")
    p = f"{P.OUT}/checkpoints/{tag}/ck_{step}.pt"
    if not os.path.exists(p): p = sorted(glob.glob(f"{P.OUT}/checkpoints/{tag}/ck_*.pt"))[-1]
    mt = "wire_ff_subspace" if kind == "sub16" else "wire_ff"
    m = P.make_model_g(mt, 768, P.FIX["k_sigma"], 0, C, dev, rank=16, warmstart=False)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); m.eval(); return m

@torch.no_grad()
def truth_frame_complex(model):                                                   # complex recon in TRUTH frame (rot applied)
    dyn = P.reconstruct_g(model, nz, tq, dev)                                      # native [RO,RO,F] complex
    return np.stack([rot(dyn[:, :, t]) for t in range(dyn.shape[2])], -1)          # -> truth frame (Gmeas frame)

# ---------- scoring ----------
def score_vol(recmag, name, check=False):
    s = np.sum(recmag[body]*Tr[body])/(np.sum(recmag[body]**2)+1e-12); rec = recmag*s
    if check: RA.check_recon(rec, Tr, mask=body, name=name)
    nrmse = float(np.mean([np.sqrt(np.mean((rec[:, :, t][body]-Tr[:, :, t][body])**2))/rv for t in range(F)]))
    fsub = np.arange(0, F, 8); pk = float(Tr[body].max())
    mse = float(np.mean([((rec[:, :, t][body]-Tr[:, :, t][body])**2).mean() for t in fsub]))
    psnr = 10*np.log10(pk**2/(mse+1e-20))
    mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev)
    hs, ss = [], []
    for t in fsub:
        vmax = float(np.percentile(Tr[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        rt = torch.from_numpy(np.clip(Tr[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu())); ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))
    cur = {nm: float(np.linalg.norm(rec[Rz[nm]].mean(0)-Tr[Rz[nm]].mean(0))/(np.linalg.norm(Tr[Rz[nm]].mean(0))+1e-12)) for nm in ("aorta", "cortex", "medulla")}
    ac = rec[Rz["aorta"]].mean(0); at = Tr[Rz["aorta"]].mean(0)
    def ttp(c): return float(tq[np.argmax(c)])
    def fwhm(c):
        p = np.argmax(c); h = (c[p]+c[:5].mean())/2; l = p
        while l > 0 and c[l] > h: l -= 1
        r = p
        while r < len(c)-1 and c[r] > h: r += 1
        return float(tq[r]-tq[l])
    bgring = float(np.sqrt((rec[~body & (r01g < 0.6)]**2).mean())/(np.sqrt((rec[body]**2).mean())+1e-12))  # air ripple near object
    return dict(name=name, nrmse=nrmse, psnr=psnr, haarpsi=float(np.mean(hs)), ssim=float(np.mean(ss)),
                cur=cur, aorta_ttp=ttp(ac), aorta_ttp_true=ttp(at), aorta_peak=float(ac.max()), aorta_peak_true=float(at.max()),
                aorta_fwhm=fwhm(ac), aorta_fwhm_true=fwhm(at), bgring=bgring)

# ---------- substitution in native k-space ----------
_checked = [False]
def substitute(dyn_tf, Gmod, radius, soft):                                       # dyn_tf already in truth frame (== Gmeas frame)
    Kmod = np.stack([np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(dyn_tf[:, :, t]))) for t in range(F)], -1)
    covlow = COV & (r01g[:, :, None] < radius)
    a = np.sum(np.conj(Gmod[covlow])*Kmod[covlow])/(np.sum(np.abs(Gmod[covlow])**2)+1e-30)  # convention factor (scalar)
    if not _checked[0]:                                                            # gridding<->render consistency guard
        rr = np.linalg.norm(Kmod[covlow]-a*Gmod[covlow])/(np.linalg.norm(Kmod[covlow])+1e-30)
        print(f"  [consistency] |a|={abs(a):.3e} relres(Kmod vs a.Gmod on covered r<{radius}) = {rr:.3f}  "
              f"({'OK gridding~render' if rr < 0.4 else 'WARN frames may be misaligned'})", flush=True); _checked[0] = True
    if soft:
        from scipy.ndimage import gaussian_filter
        w = np.stack([gaussian_filter(covlow[:, :, t].astype(float), 1.5) for t in range(F)], -1); w = np.clip(w, 0, 1)
    else:
        w = covlow.astype(float)
    Ksub = Kmod + w*a*(Gmeas-Gmod)
    dyn2 = np.stack([np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Ksub[:, :, t]))) for t in range(F)], -1)
    return dyn2, complex(a)

def fmt(r):
    return (f"{r['name']:22s} PSNR {r['psnr']:6.2f}  SSIM {r['ssim']:.3f}  Haar {r['haarpsi']:.3f}  "
            f"NRMSE {r['nrmse']:.4f} | aorta cNRMSE {r['cur']['aorta']:.3f} TTP {r['aorta_ttp']:.1f}/{r['aorta_ttp_true']:.1f}s "
            f"FWHM {r['aorta_fwhm']:.1f}/{r['aorta_fwhm_true']:.1f}s pk {r['aorta_peak']:.3f}/{r['aorta_peak_true']:.3f} | bgring {r['bgring']:.4f}")

# ---------- GRASP-K12 baseline (the ~6 dB gap reference) ----------
gk = np.load(f"{P.OUT}/arrays/grasp_ksweep_K12.npz")["rec"].astype(np.float32)
rG = score_vol(gk, "GRASP-K12", check=True); print("REFERENCE:", fmt(rG), flush=True)

CFG = {"sub16": [(0, 24000), (1, 40000)], "free": [(0, 40000), (1, 34000), (2, 36000)]}
if os.environ.get("K3A_QUICK"): CFG = {"sub16": [(0, 24000), (1, 40000)]}
RADII = [0.05, 0.10, 0.15]
results = {}
for kind, seeds in CFG.items():
    tfs, gmods = [], []                                                            # truth-frame complex recons + gridded model
    for s, st in seeds:
        m = build_model(kind, s, st); tfs.append(truth_frame_complex(m)); gmods.append(grid_kspace(model_radial(m)))
        print(f"  built {kind} s{s}", flush=True)
    for i, (s, st) in enumerate(seeds):
        base = score_vol(np.abs(tfs[i]), f"{kind}_s{s} base", check=(i == 0))
        print(fmt(base), flush=True); results[f"{kind}_s{s}_base"] = base
        for R in RADII:
            for soft in (False, True):
                dyn2, a = substitute(tfs[i], gmods[i], R, soft)
                r = score_vol(np.abs(dyn2), f"{kind}_s{s} r{R} {'soft' if soft else 'hard'}")
                print(fmt(r), f"| a={abs(a):.2e}", flush=True); results[f"{kind}_s{s}_r{R}_{'soft' if soft else 'hard'}"] = r
    navg = np.mean(tfs, 0); gavg = np.mean(gmods, 0)                               # complex seed-average (truth frame)
    ba = score_vol(np.abs(navg), f"{kind}_cplxavg base", check=True)
    print(fmt(ba), flush=True); results[f"{kind}_cplxavg_base"] = ba
    for R in RADII:
        dyn2, a = substitute(navg, gavg, R, True)
        r = score_vol(np.abs(dyn2), f"{kind}_cplxavg r{R} soft"); print(fmt(r), flush=True); results[f"{kind}_cplxavg_r{R}_soft"] = r

import csv
with open(f"{P.OUT}/k3a_substitute.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["name", "psnr", "ssim", "haarpsi", "nrmse", "aorta_cNRMSE", "cortex_cNRMSE",
                                    "medulla_cNRMSE", "aorta_ttp", "aorta_ttp_true", "aorta_fwhm", "aorta_fwhm_true",
                                    "aorta_peak", "aorta_peak_true", "bgring"])
    for r in [rG]+list(results.values()):
        w.writerow([r["name"], f"{r['psnr']:.3f}", f"{r['ssim']:.4f}", f"{r['haarpsi']:.4f}", f"{r['nrmse']:.5f}",
                    f"{r['cur']['aorta']:.4f}", f"{r['cur']['cortex']:.4f}", f"{r['cur']['medulla']:.4f}",
                    f"{r['aorta_ttp']:.2f}", f"{r['aorta_ttp_true']:.2f}", f"{r['aorta_fwhm']:.2f}", f"{r['aorta_fwhm_true']:.2f}",
                    f"{r['aorta_peak']:.4f}", f"{r['aorta_peak_true']:.4f}", f"{r['bgring']:.5f}"])
print("DONE_K3A")
