"""Noise-vs-blur diagnostic for NIK's spatial deficit. Take NIK's existing recon (no retrain), apply
post-hoc spatial (and spatiotemporal) TV denoising sweeping the weight, and re-measure SSIM/HaarPSI/PSNR
against truth (same body-masked, per-frame-then-mean metrics as xph_aggregate). If the metrics jump toward
GRASP-K12 (SSIM 0.980, HaarPSI 0.888), the content is right and NIK just lacks an image prior (fix =
regularizer); if they stay ~0.8, the edges are genuinely lost (fix = bandwidth/high-k). Truth-blind knob:
we only report the metric ceiling reachable by denoising."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import uniform_filter
import torch as _t, piq
import xph_pipeline as P, xph_common as X

def denoise_tv_chambolle(f, weight=0.1, n_iter=100, tau=0.125):
    """compact Chambolle (2004) TV denoising for 2D or 3D arrays. minimizes ||u-f||^2 + 2*weight*TV(u)."""
    f = np.asarray(f, np.float64); nd = f.ndim; p = np.zeros((nd,) + f.shape)
    for _ in range(n_iter):
        div_p = np.zeros_like(f)
        for k in range(nd):
            div_p += np.diff(p[k], axis=k, prepend=0)
        u = f - weight * div_p; g = np.stack(np.gradient(u), 0)
        norm = np.sqrt((g ** 2).sum(0))[None]; p = (p + (tau / weight) * g) / (1.0 + (tau / weight) * norm)
    div_p = np.zeros_like(f)
    for k in range(nd): div_p += np.diff(p[k], axis=k, prepend=0)
    return f - weight * div_p
A = f"{X.OUT}/arrays"; W = 768
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq)
rv = float(Tr[body].max() - Tr[body].min()); _p99 = np.percentile(Tr[body], 99) + 1e-9
_ys, _xs = np.where(body); _y0, _y1, _x0, _x1 = _ys.min(), _ys.max()+1, _xs.min(), _xs.max()+1

def mnr(a, b): return float(np.sqrt(np.mean((a[body]-b[body])**2))/(rv+1e-12))
def ssim(a, b, win=7):
    C1 = (0.01*rv)**2; C2 = (0.03*rv)**2; ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    va = uniform_filter(a*a, win)-ma**2; vb = uniform_filter(b*b, win)-mb**2; vab = uniform_filter(a*b, win)-ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2)))[body].mean())
def psnr(a, b): return float(20*np.log10(rv/(np.sqrt(np.mean((a[body]-b[body])**2))+1e-12)))
def haar(a, b):
    xa = _t.tensor(np.clip(a[_y0:_y1, _x0:_x1]/_p99, 0, 1), dtype=_t.float32)[None, None]
    xb = _t.tensor(np.clip(b[_y0:_y1, _x0:_x1]/_p99, 0, 1), dtype=_t.float32)[None, None]
    try: return float(piq.haarpsi(xa, xb, data_range=1.0).item())
    except Exception: return float("nan")
def scoreset(rec):
    rec = rec * float((rec[body]*Tr[body]).sum()/((rec[body]**2).sum()+1e-12))       # LS-scale to truth
    S = np.mean([ssim(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])
    H = np.nanmean([haar(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])
    Ps = np.mean([psnr(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])
    Nr = np.mean([mnr(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])
    return dict(SSIM=S, HaarPSI=H, PSNR=Ps, NRMSE=Nr)

def tv_spatial(v, w):     # per-frame 2D spatial TV
    o = np.empty_like(v)
    for t in range(v.shape[2]): o[:, :, t] = denoise_tv_chambolle(v[:, :, t]/_p99, weight=w) * _p99
    return o
def tv_st(v, w):          # 3D spatiotemporal TV
    return denoise_tv_chambolle(v/_p99, weight=w) * _p99

WEIGHTS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
for tag, fn in [("NIK-sub16", f"{A}/img_eval_sub16_w{W}_s0.npz"), ("NIK-free", f"{A}/img_eval_free_w{W}_s0.npz")]:
    rec0 = np.abs(np.load(fn)["rec_best"]).astype(np.float64)
    base = scoreset(rec0)
    print(f"\n=== {tag} baseline: SSIM {base['SSIM']:.3f} HaarPSI {base['HaarPSI']:.3f} PSNR {base['PSNR']:.1f} NRMSE {base['NRMSE']:.4f}", flush=True)
    print("  spatial-TV sweep:")
    best = dict(base); bestw = 0.0
    for w in WEIGHTS:
        s = base if w == 0 else scoreset(tv_spatial(rec0, w))
        print(f"    w={w:.2f}  SSIM {s['SSIM']:.3f}  HaarPSI {s['HaarPSI']:.3f}  PSNR {s['PSNR']:.1f}  NRMSE {s['NRMSE']:.4f}", flush=True)
        if s['SSIM'] > best['SSIM']: best, bestw = s, w
    print(f"  spatial-TV CEILING: SSIM {best['SSIM']:.3f} HaarPSI {best['HaarPSI']:.3f} at w={bestw} (target GRASP-K12 SSIM 0.980/HaarPSI 0.888)", flush=True)
    print("  spatiotemporal-TV sweep:")
    for w in [0.02, 0.05, 0.1]:
        s = scoreset(tv_st(rec0, w)); print(f"    w={w:.2f}  SSIM {s['SSIM']:.3f}  HaarPSI {s['HaarPSI']:.3f}  PSNR {s['PSNR']:.1f}", flush=True)
print("\nTVDIAG_DONE")
