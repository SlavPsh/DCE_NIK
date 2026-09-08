import warnings; warnings.filterwarnings("ignore")
import numpy as np, finufft
import xph_pipeline as P, xph_common as X
from scipy.ndimage import uniform_filter1d
d = P.data(); RO = d["RO"]; b1 = d["b1"]; C = b1.shape[-1]; times = d["times"]; F = len(times)
kx = d["kx"]; ky = d["ky"]; kd = d["kdata"]
nb = 30; e = np.linspace(0, F, nb + 1).astype(int)
dyn = np.zeros((RO, RO, nb), np.float32); nsp = np.zeros(nb); wsum = np.zeros(nb); opang = np.zeros(nb)
for i in range(nb):
    xs = np.concatenate([kx[fr].ravel() for fr in range(e[i], e[i + 1])])
    ys = np.concatenate([ky[fr].ravel() for fr in range(e[i], e[i + 1])])
    kk = np.concatenate([kd[:, fr].reshape(C, -1) for fr in range(e[i], e[i + 1])], 1)
    w = np.maximum(np.abs(xs + 1j * ys), 1e-3)
    ci = np.stack([finufft.nufft2d1((2 * np.pi * xs).astype(np.float64), (2 * np.pi * ys).astype(np.float64), (kk[c] * w).astype(np.complex128), (RO, RO), isign=1, eps=1e-3) for c in range(C)], -1)
    dyn[:, :, i] = np.abs(np.sum(np.conj(b1) * ci, -1) / (np.sum(np.abs(b1) ** 2, -1) + 1e-8))
    nsp[i] = xs.size; wsum[i] = w.sum()
    ang = np.mod(np.arctan2(ky[e[i]:e[i + 1], :, -1], kx[e[i]:e[i + 1], :, -1]).ravel(), np.pi)
    opang[i] = np.abs(np.mean(np.exp(2j * ang)))
base = dyn[:, :, :4].mean(2); peak = dyn.max(2); ttp = np.array([times[e[i]:e[i+1]].mean() for i in range(nb)])[dyn.argmax(2)]
enh = (peak - base) / (base + 1e-6); body = d["labels"] > 0
art = body & (ttp < 34) & (ttp > 15) & (enh > np.percentile(enh[body], 92))
curve = dyn[art].mean(0)                                         # RAW 30-bin arterial curve (pre-interp)
def rip(x): return x - uniform_filter1d(x, 5)
cr = rip(curve)
print("raw 30-bin arterial curve: %d arterial px" % int(art.sum()))
print("bin-to-bin ripple std / curve mean: %.3f" % (cr.std() / curve.mean()))
print("corr(ripple, spoke-count 77/84):   %.3f" % np.corrcoef(rip(nsp), cr)[0, 1])
print("corr(ripple, dcf weight sum):      %.3f" % np.corrcoef(rip(wsum), cr)[0, 1])
print("corr(ripple, coverage anisotropy): %.3f" % np.corrcoef(rip(opang), cr)[0, 1])
# normalize each bin by its dcf weight sum (per-bin scale) -> does ripple drop?
dyn_n = dyn / (wsum / wsum.mean())[None, None, :]
curve_n = dyn_n[art].mean(0); crn = rip(curve_n)
print("ripple std AFTER per-bin dcf-normalization: %.3f -> %.3f (frac remaining %.2f)" % (
    cr.std() / curve.mean(), crn.std() / curve_n.mean(), (crn.std() / curve_n.mean()) / (cr.std() / curve.mean() + 1e-9)))
# also: correlate with a bright off-ROI structure (kidney) intensity streak proxy = kidney bin curve
kid = ((d["labels"] == 23) | (d["labels"] == 25) | (d["labels"] == 24) | (d["labels"] == 26))
kcurve = dyn[kid].mean(0); print("corr(art ripple, kidney-bin ripple): %.3f" % np.corrcoef(rip(kcurve), cr)[0, 1])
