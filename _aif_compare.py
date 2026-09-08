import warnings; warnings.filterwarnings("ignore")
import numpy as np, finufft
import xph_pipeline as P
from scipy.ndimage import gaussian_filter1d
d = P.data(); RO = d["RO"]; b1 = d["b1"]; C = b1.shape[-1]; times = d["times"]; F = len(times)
kx = d["kx"]; ky = d["ky"]; kd = d["kdata"]; den = np.sum(np.abs(b1) ** 2, -1) + 1e-8
nb = 30; e = np.linspace(0, F, nb + 1).astype(int); tb = np.array([times[e[i]:e[i + 1]].mean() for i in range(nb)])
body = d["labels"] > 0
def pipe_dcf(x, y, niter=15, eps=1e-3):
    w = np.ones(x.size, np.complex128)
    for _ in range(niter):
        g = finufft.nufft2d1(x, y, w, (RO, RO), isign=1, eps=eps); s = finufft.nufft2d2(x, y, g, isign=1, eps=eps); w = w / (np.abs(s) + 1e-9)
    return np.abs(w)
def recon(dcf_kind):
    dyn = np.zeros((RO, RO, nb), np.float32)
    for i in range(nb):
        xs = np.concatenate([kx[fr].ravel() for fr in range(e[i], e[i + 1])]); ys = np.concatenate([ky[fr].ravel() for fr in range(e[i], e[i + 1])])
        x = (2 * np.pi * xs).astype(np.float64); y = (2 * np.pi * ys).astype(np.float64)
        w = pipe_dcf(x, y) if dcf_kind == "pipe" else np.maximum(np.abs(xs + 1j * ys), 1e-3)
        kk = np.concatenate([kd[:, fr].reshape(C, -1) for fr in range(e[i], e[i + 1])], 1)
        ci = np.stack([finufft.nufft2d1(x, y, (kk[c] * w).astype(np.complex128), (RO, RO), isign=1, eps=1e-3) for c in range(C)], -1)
        dyn[:, :, i] = np.abs(np.sum(np.conj(b1) * ci, -1) / den)
    return dyn
def build(dyn, scalecorr):
    if scalecorr:                                            # divide out the fast common per-bin scale fluctuation (the measured mechanism)
        bm = dyn[body].mean(0); f = bm / (gaussian_filter1d(bm, 3) + 1e-9); dyn = dyn / f[None, None, :]
    base = dyn[:, :, :4].mean(2); peak = dyn.max(2); ttp = tb[dyn.argmax(2)]; enh = (peak - base) / (base + 1e-6)
    art = body & (ttp < 34) & (ttp > 15) & (enh > np.percentile(enh[body], 92))
    aif = np.interp(times, tb, dyn[art].mean(0)); aif = np.clip(aif - aif[:15].mean(), 0, None); aif = gaussian_filter1d(aif, 1.5); aif /= (aif.max() + 1e-9)
    tail = slice(aif.argmax() + 3, len(aif)); nt = int((np.diff(np.sign(np.diff(aif[tail]))) != 0).sum())
    return int(art.sum()), nt
dyn_r = recon("ramp"); dyn_p = recon("pipe")
print("variant                | arterial px | tail turning-points (lower=smoother; naive was 21)")
for nm, dyn, sc in [("ramp (original)", dyn_r, False), ("ramp + scale-corr", dyn_r, True), ("pipe", dyn_p, False), ("pipe + scale-corr", dyn_p, True)]:
    npx, nt = build(dyn, sc); print("%-22s | %5d       | %d" % (nm, npx, nt))
