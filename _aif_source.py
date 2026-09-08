import warnings; warnings.filterwarnings("ignore")
import numpy as np
import xph_pipeline as P, xph_common as X
d = P.data(); kx = d["kx"]; ky = d["ky"]; F, nang, RO = kx.shape
# spoke angle (mod pi, radial symmetry) per frame per spoke, from outer k point
ang = np.mod(np.arctan2(ky[:, :, -1], kx[:, :, -1]), np.pi)     # (F, nang)
# aif_build binning
nb = 30; e = np.linspace(0, F, nb + 1).astype(int)
sizes = np.diff(e)
print("frames total:", F, " nbins:", nb, " frames/bin exact:", F / nb)
print("bin sizes (frames):", sizes.tolist())
print("bin sizes alternate?:", "YES" if len(set(sizes.tolist())) <= 2 and np.abs(np.diff(sizes)).max() >= 1 else "no")
# per-bin angular-coverage order parameter |mean exp(2i*angle)| (0=uniform, 1=clustered) and spoke count
op = []; nsp = []
for i in range(nb):
    a = ang[e[i]:e[i + 1]].ravel()
    op.append(np.abs(np.mean(np.exp(2j * a))))
    nsp.append(a.size)
op = np.array(op); nsp = np.array(nsp)
# AIF (aif_frame) sampled at bin centers
aif = np.load("aif_xph.npz"); af = aif["aif_frame"].astype(float); tC = aif["tC"].astype(float)
times = d["times"]; tb = np.array([times[e[i]:e[i + 1]].mean() for i in range(nb)])
af_b = np.interp(tb, tC, af)
# detrend both (remove smooth trend) to expose ripple
from scipy.ndimage import uniform_filter1d
def rip(x): return x - uniform_filter1d(x, 5)
print("\nper-bin spoke count:", nsp.tolist())
print("spoke-count parity (even/odd bin sizes -> 2-bin ripple):", (nsp % 2).tolist()[:12], "...")
print("corr(bin-size, AIF ripple):        %.3f" % np.corrcoef(rip(sizes.astype(float)), rip(af_b))[0, 1])
print("corr(coverage order-param, AIF rip): %.3f" % np.corrcoef(rip(op), rip(af_b))[0, 1])
print("corr(spoke-count, AIF ripple):      %.3f" % np.corrcoef(rip(nsp.astype(float)), rip(af_b))[0, 1])
# 2-bin (alternating) component strength in AIF ripple
r = rip(af_b); alt = np.array([(-1)**i for i in range(nb)], float)
print("AIF-ripple projection onto alternating (+-1) pattern: %.3f (|corr|)" % abs(np.corrcoef(r, alt)[0, 1]))
