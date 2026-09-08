"""Step 1 (measure only): robustly determine the NIK render's sub-pixel offset vs truth, using exact
Fourier shifts (no spline) + masked NCC, for sub12/sub16/free. Confirms the offset is shared and gives
the precise value the source fix must cancel. Also reports SSIM after the EXACT Fourier recentering (vs
the earlier spline estimate +0.052)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import uniform_filter, fourier_shift
import xph_pipeline as P, xph_common as X
A = f"{X.OUT}/arrays"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); rv = float(Tr[body].max()-Tr[body].min())
def lss(v): return v * float((v[body]*Tr[body]).sum()/((v[body]**2).sum()+1e-12))
def fshift(im, sy, sx): return np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(im), (sy, sx))))
def ncc(a, b): a = a[body]-a[body].mean(); b = b[body]-b[body].mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def ssim(a, b, win=7):
    C1 = (0.01*rv)**2; C2 = (0.03*rv)**2; ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    va = uniform_filter(a*a, win)-ma**2; vb = uniform_filter(b*b, win)-mb**2; vab = uniform_filter(a*b, win)-ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2)))[body].mean())
tm = Tr.mean(2)
for tag, fn in [("NIK-sub12", "img_eval_sub12_w768_s1"), ("NIK-sub16", "img_eval_sub16_w768_s0"), ("NIK-free", "img_eval_free_w768_s0")]:
    vol = lss(np.abs(np.load(f"{A}/{fn}.npz")["rec_best"]).astype(np.float64)); mm = vol.mean(2)
    # coarse then fine NCC scan of the temporal-mean image
    best = (-9, 0, 0)
    for sy in np.arange(-1.0, 1.01, 0.05):
        for sx in np.arange(-1.0, 1.01, 0.05):
            c = ncc(fshift(mm, sy, sx), tm)
            if c > best[0]: best = (c, sy, sx)
    b0, sy0, sx0 = best
    for sy in np.arange(sy0-0.06, sy0+0.061, 0.01):
        for sx in np.arange(sx0-0.06, sx0+0.061, 0.01):
            c = ncc(fshift(mm, sy, sx), tm)
            if c > best[0]: best = (c, sy, sx)
    _, sy, sx = best
    base = float(np.mean([ssim(vol[:, :, t], Tr[:, :, t]) for t in range(vol.shape[2])]))
    volc = lss(np.stack([fshift(vol[:, :, t], sy, sx) for t in range(vol.shape[2])], -1))
    corr = float(np.mean([ssim(volc[:, :, t], Tr[:, :, t]) for t in range(vol.shape[2])]))
    print(f"{tag}: render offset to cancel (sy,sx)=({sy:+.3f},{sx:+.3f}) NCC {best[0]:.4f} | SSIM base {base:.4f} -> exact-recenter {corr:.4f} (delta {corr-base:+.4f})", flush=True)
print("STEP1_MEASURE_DONE")
