"""per-slice method-neutral NUFFT rulers (ramp dcf, coil combine with the saved b1, same path as every arm):
  nufft_all   every spoke (bolus blended through the anatomy; body mask source only)
  nufft_pre   spokes before contrast arrival (static anatomy, under-sampled ~3x, streaky)
  nufft_late  spokes with t > LATE_T0 (slow washout only, above nyquist on p3 and p14): primary sharpness / structure ruler at the 300 s frame
  *_gated     the same windows with respiratory soft gating from the k-centre navigator: weight = exp(-(d / s)^2) with d = navigator
              distance to the end-expiration mode and s the 40th percentile of d, so the ruler is dominated by end-expiration spokes
              without dropping below nyquist (effective spoke count printed in meta). removes the respiratory blur every arm shares.
navigator: |k-centre| per spoke summed over coils, contrast trend removed by a 15 s running median, first svd mode across coils of the residual.
out: results_nufft{_ds}_slice{Z}/{nufft_all,nufft_pre,nufft_late,nufft_pre_gated,nufft_late_gated,pre_view_idx,late_view_idx,resp_nav,resp_weight}.npy + meta.json
usage: DCE_DS=p14 python build_rulers.py 21,24,27 [LATE_T0=200 GATE_Q=40]"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, finufft, os, json
import sys; D = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, D)
import dsp
REF = dsp.REF
SLICES = [int(z) for z in sys.argv[1].split(",")] if len(sys.argv) > 1 else [18, 19, 20, 21]
LATE_T0 = float(os.environ.get("LATE_T0", 200.0)); GATE_Q = float(os.environ.get("GATE_Q", 40.0))
sh = np.load(f"{REF}/shared.npz")
traj = np.asarray(sh["traj_norm"]).astype(np.complex64); vt = np.asarray(sh["view_time"]).ravel().astype(np.float64)
TA = float(sh["TA"]); nx = int(sh["nx"]); bas = int(sh["bas"]); ts = vt * TA; dt = TA / len(vt)

def ramp_dcf(tr): return np.maximum(np.abs(tr).astype(np.float64), (1.0 / nx) / 4.0)
def crop(img, n=bas): s = (img.shape[0] - n) // 2; return img[s:s + n, s:s + n]

def resp_navigator(kdata):
    """per-spoke respiratory signal: k-centre magnitude per coil, contrast trend removed (running median over 15 s, in acquisition
    order), first svd mode across coils, sign so that the mode (end-expiration plateau) sits at the histogram peak."""
    from scipy.ndimage import median_filter
    c0 = nx // 2; o = np.argsort(ts); m = np.abs(kdata[c0 - 1:c0 + 2]).mean(0)[o]           # (nsp, ncc), 3 central samples, time order
    k = max(5, int(round(15.0 / dt)) | 1); trend = median_filter(m, size=(k, 1), mode="nearest"); r = (m - trend) / (trend + 1e-9)
    u, s, vh = np.linalg.svd(r - r.mean(0), full_matrices=False); nav_s = u[:, 0] * s[0]
    nav = np.empty_like(nav_s); nav[o] = nav_s
    hist, edges = np.histogram(nav, bins=60); mode = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    f = np.fft.rfftfreq(len(nav_s), dt); p = np.abs(np.fft.rfft(nav_s - nav_s.mean())); f0 = float(f[1:][np.argmax(p[1:])])
    return nav, float(mode), f0

for SL in SLICES:
    sl = np.load(f"{REF}/slice_{SL:02d}.npz"); kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64)
    b1 = np.asarray(sl["b1"]).astype(np.complex64); ncc = kdata.shape[2]; OUT = dsp.NUF(SL); os.makedirs(OUT, exist_ok=True)
    def nufft_recon(view_idx, sign=-1.0, eps=1e-6, wts=None):
        tr = traj[:, view_idx]; w = ramp_dcf(tr)
        if wts is not None: w = w * wts[None, :]                                           # soft gating: per-spoke weight on top of the dcf
        x = (sign * 2 * np.pi * np.real(tr)).astype(np.float64).ravel(); y = (sign * 2 * np.pi * np.imag(tr)).astype(np.float64).ravel()
        ci = np.zeros((nx, nx, ncc), dtype=np.complex128)
        for c in range(ncc):
            ci[:, :, c] = finufft.nufft2d1(x, y, (kdata[:, view_idx, c] * w).astype(np.complex128).ravel(), (nx, nx), eps=eps, isign=1)
        return np.sum(ci * np.conj(b1), 2) / (np.sum(np.abs(b1) ** 2, 2) + 1e-12)
    cs_mean = np.abs(np.asarray(sl["cs_img"])).mean(-1); allv = np.arange(traj.shape[1]); best = None
    for sign in (-1.0, 1.0):
        im = np.abs(crop(nufft_recon(allv, sign=sign))); c = float(np.corrcoef(im.ravel(), cs_mean.ravel())[0, 1])
        if best is None or c > best[1]: best = (sign, c, im)
    SIGN, corr_all, img_all = best
    c0 = nx // 2; nav = np.sqrt((np.abs(kdata[c0]) ** 2).sum(-1)); o = np.argsort(vt); nav_s = nav[o]; t_s = ts[o]
    k = 41; nav_sm = np.convolve(np.pad(nav_s, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    base = np.median(nav_sm[t_s < 30]); amp = nav_sm.max() - base; arrival = float(t_s[np.argmax(nav_sm > base + 0.15 * amp)])
    t_pre = max(20.0, arrival - 8.0); pre_idx = np.where(ts < t_pre)[0]; late_idx = np.where(ts > LATE_T0)[0]
    img_pre = np.abs(crop(nufft_recon(pre_idx, sign=SIGN))); img_late = np.abs(crop(nufft_recon(late_idx, sign=SIGN)))
    rnav, mode, f0 = resp_navigator(kdata); d = np.abs(rnav - mode); s = np.percentile(d, GATE_Q) + 1e-9; wg = np.exp(-(d / s) ** 2)
    def neff(idx): w = wg[idx]; return float(w.sum() ** 2 / (w ** 2).sum() + 1e-9)                 # effective spoke count of the weighted set
    img_pre_g = np.abs(crop(nufft_recon(pre_idx, sign=SIGN, wts=wg[pre_idx]))); img_late_g = np.abs(crop(nufft_recon(late_idx, sign=SIGN, wts=wg[late_idx])))
    for nm, im in (("nufft_all", img_all), ("nufft_pre", img_pre), ("nufft_late", img_late), ("nufft_pre_gated", img_pre_g), ("nufft_late_gated", img_late_g)): np.save(f"{OUT}/{nm}.npy", im.astype(np.float32))
    np.save(f"{OUT}/pre_view_idx.npy", pre_idx); np.save(f"{OUT}/late_view_idx.npy", late_idx); np.save(f"{OUT}/resp_nav.npy", rnav.astype(np.float32)); np.save(f"{OUT}/resp_weight.npy", wg.astype(np.float32))
    nyq = int(round(bas * np.pi / 2))
    meta = dict(sign=SIGN, corr_all_vs_cs=corr_all, arrival_s=arrival, t_pre_s=t_pre, late_t0_s=LATE_T0, n_pre_spokes=int(len(pre_idx)), n_late_spokes=int(len(late_idx)), n_all_spokes=int(len(allv)),
                nyquist_spokes=nyq, gate_q=GATE_Q, resp_freq_hz=f0, n_eff_pre_gated=neff(pre_idx), n_eff_late_gated=neff(late_idx), nx=nx, bas=bas)
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print(f"slice {SL}: sign {SIGN:+.0f} corr {corr_all:.4f} | arrival {arrival:.0f}s pre {len(pre_idx)} late {len(late_idx)} (nyquist {nyq}) | resp {f0:.2f} Hz, gated n_eff pre {neff(pre_idx):.0f} late {neff(late_idx):.0f} -> {OUT}", flush=True)
print("RULERS DONE")
