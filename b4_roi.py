"""B4: ROI masks vs recon grid. ROIs are defined on the truth/label grid (RO=220); the recon carried a
+1px shift, so recon[ROI] sampled tissue offset by ~1px from truth[ROI]. Quantify how much a 1px shift
perturbs each ROI's median curve (proxy: compare median(shift(truth,1)[ROI]) vs median(truth[ROI])).
Large => the curve/PK 'wins' were affected by the shift; small => robust."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import shift as ndshift, center_of_mass
import xph_pipeline as P, xph_common as X
d = P.data(); tq = d["times"]; Tr = X.truth_at(P.ZI, tq); R = X.rois(P.ZI, d["labels"]); body = d["labels"] > 0
_pre = tq < 18
def bsub(c): return c - np.median(c[_pre])
Trs = np.stack([ndshift(Tr[:, :, t], (1, 1), order=1) for t in range(Tr.shape[2])], -1)   # truth shifted +1px (mimics the recon shift vs ROI)
print(f"{'ROI':8s} {'npix':>6s} {'centroid':>14s} {'curveNRMSE(1px)':>16s} {'peak true':>10s} {'peak 1px':>10s}")
for nm in ["aorta", "cortex", "medulla"]:
    m = R[nm]; c0 = bsub(np.median(Tr[m], 0)); c1 = bsub(np.median(Trs[m], 0))
    cn = float(np.linalg.norm(c1-c0)/(np.linalg.norm(c0)+1e-12)); cy, cx = center_of_mass(m)
    print(f"{nm:8s} {int(m.sum()):6d} ({cy:6.1f},{cx:6.1f}) {cn:16.4f} {c0.max():10.3f} {c1.max():10.3f}")
# also: fraction of body pixels whose value changes >5% of range under 1px shift (spatial sensitivity)
rng = Tr[body].max()-Tr[body].min(); tm = Tr.mean(2); tms = ndshift(tm, (1, 1), order=1)
frac = float(np.mean(np.abs(tms[body]-tm[body]) > 0.05*rng))
print(f"\nbody pixels changing >5% range under 1px shift: {frac*100:.1f}%  (edge-density proxy)")
print("B4_ROI_DONE")
