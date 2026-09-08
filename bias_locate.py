"""D1: locate NIK's smooth low-|k| bias (worth ~4.2 dB). Candidates, all testable in the image domain:
(a) SENSE/coil combination -> bias field correlates with the coil sum-of-squares profile sum|b1|^2
    (peaks off-centre where coils are sensitive), (b) k-space envelope normalization (exponent 0.75)
    -> a radially-symmetric central modulation. Compute the bias field B = smooth(NIK-truth), correlate
    with SoS(b1), individual coils, and a radial ramp; measure how much of B is radially symmetric.
Show the bias field. STOP after."""
import warnings; warnings.filterwarnings("ignore")
import glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, gaussian_filter
import xph_pipeline as P, xph_common as X
A = f"{X.OUT}/arrays"; FIG = f"{X.OUT}/figures"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; b1 = d["b1"]; Tr = X.truth_at(P.ZI, tq); RO = Tr.shape[0]
rv = float(Tr[body].max()-Tr[body].min())
def ls(v): return v * float((v[body]*Tr[body]).sum()/((v[body]**2).sum()+1e-12))
def mnr(a, b): return float(np.sqrt(np.mean((a[body]-b[body])**2))/(rv+1e-12))
def per_frame(rec, fn): return np.array([fn(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])
fs = sorted(glob.glob(f"{A}/img_eval_sub12_w768_s*.npz")); recs = [ls(np.abs(np.load(f)["rec_best"]).astype(np.float64)) for f in fs]
rec = recs[int(np.argsort([per_frame(r, mnr).mean() for r in recs])[len(recs)//2])]
recTM = rec.mean(2); TrTM = Tr.mean(2)
E = recTM - TrTM; B = gaussian_filter(E, sigma=8)                       # smooth bias field
SoS = np.sum(np.abs(b1)**2, -1)                                          # SENSE denominator / coil profile
yy, xx = np.meshgrid(np.arange(RO)-RO//2, np.arange(RO)-RO//2, indexing="ij"); rr = np.sqrt(xx**2+yy**2)
def corr(a, c, m=body): a = a[m]-a[m].mean(); c = c[m]-c[m].mean(); return float((a*c).sum()/(np.linalg.norm(a)*np.linalg.norm(c)+1e-12))
# radially symmetric part of B about image centre
rint = rr.astype(int); Brad = np.zeros(RO*2)
for k in range(rint.max()+1):
    mk = (rint == k) & body
    if mk.sum(): Brad[k] = B[mk].mean()
Bsym = Brad[rint]; frac_sym = float(np.var(Bsym[body])/(np.var(B[body])+1e-12))
print("D1 bias-field correlations (over body):")
print(f"  corr(B, SoS(b1))            = {corr(B, SoS):+.3f}   <- SENSE/coil candidate")
print(f"  corr(B, radial |k| ramp rr) = {corr(B, rr):+.3f}   <- central/envelope candidate")
print(f"  frac of B variance that is radially symmetric about centre = {frac_sym:.3f}")
print(f"  corr(B, best single coil |b1_c|) = {max(corr(B, np.abs(b1[:,:,c])) for c in range(b1.shape[-1])):+.3f}")
print(f"  |B| rms over body = {np.sqrt(np.mean(B[body]**2)):.4f}  (rv {rv:.3f}); B range [{B[body].min():+.3f},{B[body].max():+.3f}]")
# figure
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
vb = np.percentile(np.abs(B[body]), 99)
for a, im, ttl, cm, vk in [(ax[0], TrTM, "truth (temporal mean)", "gray", dict(vmax=np.percentile(Tr[body], 99))),
                            (ax[1], recTM, "NIK-sub12 (temporal mean)", "gray", dict(vmax=np.percentile(Tr[body], 99))),
                            (ax[2], B, "smooth bias field B = smooth(NIK-truth)", "bwr", dict(vmin=-vb, vmax=vb)),
                            (ax[3], SoS, "coil SoS  sum|b1|^2", "viridis", {})]:
    im2 = a.imshow(im, cmap=cm, **vk); a.set_title(ttl, fontsize=9); a.axis("off"); fig.colorbar(im2, ax=a, fraction=0.046)
fig.suptitle(f"D1 smooth bias: corr(B,SoS)={corr(B,SoS):+.2f}, corr(B,radial)={corr(B,rr):+.2f}, radsym frac={frac_sym:.2f}")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_bias_locate.png", dpi=130); plt.close(fig)
print("SAVED", f"{FIG}/fig_bias_locate.png"); print("BIAS_LOCATE_DONE")
