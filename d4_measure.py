"""D4 measurement: for each envelope_exponent (0.5, 0.75, 1.0), over 3 seeds, report BOTH axes plus the
smooth-bias magnitude AT SOURCE. Answers: does the low-|k| bias actually reduce when the envelope
changes (|B| rms down), how much spatial is gained, and is it a Pareto slide (temporal fidelity paid for
low-|k| reweighting -- low |k| carries the contrast baseline; higher DCF power hurt dynamics in vivo)."""
import warnings; warnings.filterwarnings("ignore")
import glob, numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
import xph_pipeline as P, xph_common as X
A = f"{X.OUT}/arrays"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; b1 = d["b1"]; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq); RO = Tr.shape[0]
rv = float(Tr[body].max()-Tr[body].min()); _pre = tq < 18
yy, xx = np.meshgrid(np.arange(RO)-RO//2, np.arange(RO)-RO//2, indexing="ij"); rad = np.sqrt(xx**2+yy**2)
def bsub(c): return c - np.median(c[_pre])
def ls(v): return v * float((v[body]*Tr[body]).sum()/((v[body]**2).sum()+1e-12))
def ssim(a, b, win=7):
    C1 = (0.01*rv)**2; C2 = (0.03*rv)**2; ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    va = uniform_filter(a*a, win)-ma**2; vb = uniform_filter(b*b, win)-mb**2; vab = uniform_filter(a*b, win)-ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2)))[body].mean())
def psnr(a, b): return float(20*np.log10(rv/(np.sqrt(np.mean((a[body]-b[body])**2))+1e-12)))
def pf(rec, fn): return float(np.mean([fn(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))]))
truec = {nm: bsub(np.median(Tr[R[nm]], 0)) for nm in ["aorta", "cortex", "medulla"]}
def corr(a, c): a = a[body]-a[body].mean(); c = c[body]-c[body].mean(); return float((a*c).sum()/(np.linalg.norm(a)*np.linalg.norm(c)+1e-12))
def biasrms(rec):
    B = gaussian_filter(rec.mean(2)-Tr.mean(2), sigma=8)
    return float(np.sqrt(np.mean(B[body]**2))), corr(B, rad)
def metrics(npz):
    rec = ls(np.abs(np.load(npz)["rec_best"]).astype(np.float64)); ac = bsub(np.median(rec[R["aorta"]], 0))
    ho = float(np.load(npz)["test_best"][0]); brms, bcorr = biasrms(rec)
    m = dict(SSIM=pf(rec, ssim), PSNR=pf(rec, psnr), heldout=ho, biasrms=brms, biascorr=bcorr,
             aortaPeak=float(ac.max()), aortaCurve=float(np.linalg.norm(ac-truec["aorta"])/np.linalg.norm(truec["aorta"])))
    for nm in ["cortex", "medulla"]:
        c = bsub(np.median(rec[R[nm]], 0)); m[nm] = float(np.linalg.norm(c-truec[nm])/np.linalg.norm(truec[nm]))
    return m
def band(vals): return f"{np.mean(vals):.4f}+-{(max(vals)-min(vals))/2:.4f}"
print(f"{'env':>5s} {'nseed':>5s} {'SSIM':>16s} {'PSNR':>14s} {'|bias|rms':>12s} {'biasCorrRad':>12s} {'heldout':>12s} {'aortaCurve':>16s} {'aortaPeak':>16s} {'cortex':>14s} {'medulla':>14s}")
for code, env in [("050", 0.5), ("060", 0.6), ("065", 0.65), ("070", 0.7), ("075", 0.75), ("100", 1.0)]:
    fs = sorted(glob.glob(f"{A}/envsweep_sub16_e{code}_w768_s*.npz"))
    if not fs: print(f"{env:5.2f}  (no runs yet)"); continue
    M = [metrics(f) for f in fs]
    g = lambda k: [m[k] for m in M]
    print(f"{env:5.2f} {len(fs):5d} {band(g('SSIM')):>16s} {band(g('PSNR')):>14s} {band(g('biasrms')):>12s} {band(g('biascorr')):>12s} {band(g('heldout')):>12s} {band(g('aortaCurve')):>16s} {band(g('aortaPeak')):>16s} {band(g('cortex')):>14s} {band(g('medulla')):>14s}")
print("\nreference: GRASP-K12 SSIM 0.980 PSNR 40.7; truth aorta peak 0.772; env=0.75 is the current default.")
print("watch: SSIM/PSNR up + |bias|rms down = envelope fix works at source; but if aortaCurve/aortaPeak/cortex/medulla worsen = Pareto slide (low-|k| contrast paid for spatial).")
print("D4_MEASURE_DONE")
