"""STEP 2/3 gate (CPU, ROI-level). the per-voxel residual map needs a streak-free recon
(full-rank NIK); the gate itself does not. test: do kidney cortex & medulla have distinct,
smooth, supra-K5 kinetics that the CS K=5 navigator flattens?
  - anatomy/segmentation from cs_img (K=5 ok for drawing ROIs, single-frame contrast)
  - CURVES from model-free NUFFT ROI-means (streak-free by spatial averaging; independent of K5)
  - K5 residual via the per-slice navigator basis; smoothness via oscillation metric
usage: python step3_gate.py            (loops 18-21, ranks them)
out: figures/<dt>_step3_gate.png + printed verdict table."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.ndimage as ndi, sys
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
from figpath import fig as fpath
D = "/scratch/rnga/vvpshenov/DCE_NIK"; REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
TA = 375.0; SL = [18, 19, 20, 21]

def norm(t, c):
    b = c[t < 50].mean() if (t < 50).any() else c[0]
    pk = c[(t > 20) & (t < 210)].max() if ((t > 20) & (t < 210)).any() else c.max()
    return (c - b) / (pk - b + 1e-9)
def osc(t, n):                                             # oscillation = noise fraction (smooth kinetics -> small)
    sm = savgol_filter(n, 11, 3) if len(n) > 11 else n
    return float(np.std(n - sm) / (np.abs(sm).max() + 1e-9))
def ttp(t, n):
    sm = savgol_filter(n, 11, 3) if len(n) > 11 else n; fp = (t > 20) & (t < 200)
    return float(t[fp][np.argmax(sm[fp])]) if fp.any() else np.nan

rows = []; panels = []
for Z in SL:
    d = np.load(f"{D}/step2_slice{Z}.npz"); mf = d["mf"]; tmf = d["tmf"]; P = d["Phi5"]  # model-free windows + navigator basis (window grid)
    PtPi = np.linalg.inv(P.T @ P)
    def resid(c):
        c = np.atleast_2d(c); proj = (c @ P) @ PtPi @ P.T
        return float(np.linalg.norm(c - proj) / (np.linalg.norm(c - c.mean()) + 1e-9))
    cs = np.abs(np.load(f"{REF}/slice_{Z:02d}.npz")["cs_img"]).astype(np.float32)  # [192,192,342] K=5 recon
    tC = np.linspace(0, TA, cs.shape[-1])
    # --- kidney segmentation from cs_img (anatomy only) ---
    base = cs[..., tC < 50].mean(-1); enh = cs - base[..., None]
    late = enh[..., (tC > 130) & (tC < 200)].mean(-1); m = cs.mean(-1); body = m > np.quantile(m, 0.5)
    kid = body & (late > np.quantile(late[body], 0.985)); kid = ndi.binary_opening(kid, iterations=1)
    l, k = ndi.label(kid); sizes = ndi.sum(np.ones_like(l), l, range(1, k + 1)) if k else []
    kid = (l == (1 + np.argmax(sizes))) if k else kid
    # corticomedullary phase (~85s): cortex = bright rim, medulla = darker core (single-frame contrast, not K5 curve shape)
    fcm = int(np.argmin(np.abs(tC - 85))); cmp = enh[..., fcm]
    thr = np.median(cmp[kid]); cortex = kid & (cmp > thr); medulla = kid & (cmp <= thr)
    # aorta control (compact early strong enhancer)
    early = enh[..., (tC > 40) & (tC < 80)].mean(-1); ao = body & (early > np.quantile(early[body], 0.995)) & (~kid)
    ao = ndi.binary_opening(ao, iterations=1); la, ka = ndi.label(ao)
    ao = (la == (1 + np.argmax(ndi.sum(np.ones_like(la), la, range(1, ka + 1))))) if ka else ao

    def mf_curve(mask): return norm(tmf, np.array([im[mask].mean() for im in mf]))
    def cs_curve(mask): return norm(tC, np.array([cs[..., i][mask].mean() for i in range(cs.shape[-1])]))
    cC, cM, cA = mf_curve(cortex), mf_curve(medulla), mf_curve(ao)   # model-free (real kinetics)
    sC, sM = cs_curve(cortex), cs_curve(medulla)                      # CS K=5 (flattened?)
    # gate metrics
    corr_mf = float(np.corrcoef(cC, cM)[0, 1]); corr_cs = float(np.corrcoef(sC, sM)[0, 1])
    resid_C = resid(cC); resid_M = resid(cM)
    row = dict(slice=Z, nC=int(cortex.sum()), nM=int(medulla.sum()),
               ttpC=ttp(tmf, cC), ttpM=ttp(tmf, cM), corr_mf=corr_mf, corr_cs=corr_cs,
               resid_C=resid_C, resid_M=resid_M, osc_C=osc(tmf, cC), osc_M=osc(tmf, cM))
    rows.append(row); panels.append((Z, cs, enh[..., fcm], cortex, medulla, ao, tmf, cC, cM, cA, tC, sC, sM, row))

# rank slices by gate strength: distinct model-free (low corr_mf) + high residual + smooth
def gate_score(r): return (1 - r["corr_mf"]) + 0.5 * (r["resid_C"] + r["resid_M"]) - 2 * (r["osc_C"] + r["osc_M"])
best = max(rows, key=gate_score)
print(f"{'sl':>3}{'nC':>5}{'nM':>5}{'ttpC':>6}{'ttpM':>6}{'corrMF':>8}{'corrCS':>8}{'residC':>8}{'residM':>8}{'oscC':>7}{'oscM':>7}")
for r in rows:
    star = " *" if r["slice"] == best["slice"] else ""
    print(f"{r['slice']:>3}{r['nC']:>5}{r['nM']:>5}{r['ttpC']:>6.0f}{r['ttpM']:>6.0f}{r['corr_mf']:>8.2f}{r['corr_cs']:>8.2f}{r['resid_C']:>8.2f}{r['resid_M']:>8.2f}{r['osc_C']:>7.3f}{r['osc_M']:>7.3f}{star}")

# figure for the best slice
Z, cs, cmpimg, cortex, medulla, ao, tmf, cC, cM, cA, tC, sC, sM, r = [p for p in panels if p[0] == best["slice"]][0]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
a = cs[..., int(np.argmin(np.abs(tC - 85)))]
ov = np.zeros((*a.shape, 4)); ov[cortex] = [1, 0, 0, .6]; ov[medulla] = [0, .5, 1, .6]; ov[ao] = [1, 1, 0, .5]
ax[0].imshow(np.rot90(a), cmap="gray", vmax=np.percentile(a, 99.5)); ax[0].imshow(np.rot90(ov)); ax[0].axis("off")
ax[0].set_title(f"slice {Z} @85s  cortex(red)/medulla(blue)/aorta(yellow)")
ax[1].plot(tmf, cC, "r", lw=2, label=f"cortex TTP{r['ttpC']:.0f}"); ax[1].plot(tmf, cM, "b", lw=2, label=f"medulla TTP{r['ttpM']:.0f}")
ax[1].plot(tmf, cA, "0.5", lw=1, label="aorta"); ax[1].set_xlim(0, 260); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
ax[1].set_title(f"MODEL-FREE (real): corr {r['corr_mf']:.2f}, resid C{r['resid_C']:.2f}/M{r['resid_M']:.2f}")
ax[2].plot(tC, sC, "r", lw=2, label="cortex"); ax[2].plot(tC, sM, "b", lw=2, label="medulla")
ax[2].set_xlim(0, 260); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8); ax[2].set_title(f"CS K=5 (flattened): corr {r['corr_cs']:.2f}")
fig.suptitle(f"STEP 3 gate: kidney cortex vs medulla, slice {Z}  (model-free distinct? CS K=5 flat?)", fontweight="bold")
fig.tight_layout(); p = fpath("step3_gate.png"); fig.savefig(p, dpi=135)
print(f"\nbest slice {best['slice']}  |  wrote {p.split('/')[-1]}")
print("GATE read: GO needs corrMF notably < corrCS(~1.0), resid>~0.3, osc small (<~0.06), ttpC<ttpM")
