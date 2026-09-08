"""D2: seed variance for sub12 (s0,s1,s2) and free (s0,s1,s2) on the corrected render. Report mean and
spread on SSIM, PSNR, aorta curve-NRMSE, aorta peak. Compare spread to the GRASP-K12 gap (SSIM ~0.05)."""
import warnings; warnings.filterwarnings("ignore")
import glob, numpy as np
from scipy.ndimage import uniform_filter
import xph_pipeline as P, xph_common as X
A = f"{X.OUT}/arrays"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
rv = float(Tr[body].max()-Tr[body].min()); _pre = tq < 18
def bsub(c): return c - np.median(c[_pre])
def ls(v): return v * float((v[body]*Tr[body]).sum()/((v[body]**2).sum()+1e-12))
def ssim(a, b, win=7):
    C1 = (0.01*rv)**2; C2 = (0.03*rv)**2; ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    va = uniform_filter(a*a, win)-ma**2; vb = uniform_filter(b*b, win)-mb**2; vab = uniform_filter(a*b, win)-ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2)))[body].mean())
def psnr(a, b): return float(20*np.log10(rv/(np.sqrt(np.mean((a[body]-b[body])**2))+1e-12)))
def pf(rec, fn): return np.mean([fn(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])
truec = bsub(np.median(Tr[R["aorta"]], 0))
def metrics(rec):
    ac = bsub(np.median(rec[R["aorta"]], 0))
    return dict(SSIM=pf(rec, ssim), PSNR=pf(rec, psnr),
                aortaCurve=float(np.linalg.norm(ac-truec)/np.linalg.norm(truec)), aortaPeak=float(ac.max()))
print(f"{'variant':10s} {'metric':11s} {'seeds':>26s} {'mean':>8s} {'spread(max-min)':>16s}")
for v, pat in [("sub12", "img_eval_sub12_w768_s*"), ("free", "img_eval_free_w768_s*")]:
    fs = sorted(glob.glob(f"{A}/{pat}.npz")); M = [metrics(ls(np.abs(np.load(f)["rec_best"]).astype(np.float64))) for f in fs]
    for k, fmt in [("SSIM", "%.4f"), ("PSNR", "%.2f"), ("aortaCurve", "%.4f"), ("aortaPeak", "%.4f")]:
        vals = [m[k] for m in M]
        print(f"{v:10s} {k:11s} {str([float(fmt%x) for x in vals]):>26s} {np.mean(vals):8.4f} {max(vals)-min(vals):16.4f}")
print(f"\n(reference: GRASP-K12 SSIM 0.980; truth aorta peak 0.772; NIK-vs-GRASP SSIM gap ~0.05)")
print("D2_SEEDS_DONE")
