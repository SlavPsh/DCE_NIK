"""Data-driven F0 AIF (coarse 30-bin recon, ramp dcf + per-bin scale-fluctuation correction; no truth labels).
the ripple is a per-bin GLOBAL brightness fluctuation (arterial ~ kidney corr 0.99) from imperfect ramp
density compensation under bin-varying angular sampling. fix: divide out the fast common per-bin factor
(scale-corr). Pipe dcf and sliding windows were tried and BROKE the arterial pixel selection (780 -> 43).
AIF-build only; does NOT touch the researched NIK model dcf/envelope. NOTE: partial fix; the clean fix is a
parametric AIF model (deferred)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, finufft, xph_pipeline as P, xph_common as X
from scipy.ndimage import gaussian_filter1d
d = P.data(); RO = d["RO"]; b1 = d["b1"]; C = b1.shape[-1]; times = d["times"]; F = len(times); ZI = P.ZI
den = np.sum(np.abs(b1) ** 2, -1) + 1e-8
nb = 30; e = np.linspace(0, F, nb + 1).astype(int); tb = np.array([times[e[i]:e[i + 1]].mean() for i in range(nb)])
dyn = np.zeros((RO, RO, nb), np.float32)
for i in range(nb):
    xs = np.concatenate([d["kx"][fr].ravel() for fr in range(e[i], e[i + 1])]); ys = np.concatenate([d["ky"][fr].ravel() for fr in range(e[i], e[i + 1])])
    kk = np.concatenate([d["kdata"][:, fr].reshape(C, -1) for fr in range(e[i], e[i + 1])], 1); w = np.maximum(np.abs(xs + 1j * ys), 1e-3)   # ramp dcf
    ci = np.stack([finufft.nufft2d1((2 * np.pi * xs).astype(np.float64), (2 * np.pi * ys).astype(np.float64), (kk[c] * w).astype(np.complex128), (RO, RO), isign=1, eps=1e-3) for c in range(C)], -1)
    dyn[:, :, i] = np.abs(np.sum(np.conj(b1) * ci, -1) / den)
body = d["labels"] > 0
bm = dyn[body].mean(0); f = bm / (gaussian_filter1d(bm, 3) + 1e-9); dyn = dyn / f[None, None, :]        # scale-corr: divide out fast common per-bin brightness
base = dyn[:, :, :4].mean(2); peak = dyn.max(2); ttp = tb[dyn.argmax(2)]; enh = (peak - base) / (base + 1e-6)
art = body & (ttp < 34) & (ttp > 15) & (enh > np.percentile(enh[body], 92))
aif = np.interp(times, tb, dyn[art].mean(0)); aif = np.clip(aif - aif[:15].mean(), 0, None); aif = gaussian_filter1d(aif, 2.5); aif /= (aif.max() + 1e-9)
integ = np.concatenate([[0], np.cumsum(0.5 * (aif[1:] + aif[:-1]) * np.diff(times))]); integ /= (integ.max() + 1e-9)
np.savez("aif_xph.npz", aif_frame=aif.astype(np.float32), tC=times.astype(np.float32), integ=integ.astype(np.float32), source="ramp-dcf-scalecorr", zi=ZI)
tail = slice(aif.argmax() + 3, len(aif)); nturn = int((np.diff(np.sign(np.diff(aif[tail]))) != 0).sum())
Tr = X.truth_at(ZI, times); R = X.rois(ZI, d["labels"]); ta = np.median(Tr[R["aorta"]], 0); ta = ta - ta[:15].mean()
print("z%d AIF (ramp+scalecorr): %d arterial px | AIF peak %.1fs | true aorta peak %.1fs | tail turning-points %d (naive-ramp was 21)" % (
    ZI, int(art.sum()), times[aif.argmax()], times[ta.argmax()], nturn))
