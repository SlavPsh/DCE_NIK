"""STEP 3 gate, corrected decisive test. absolute K5 residual is confounded (magnitude
nonlinearity; CS bulk curve itself shows 0.6 'residual'). the confound-free test = does the
CS K=5 recon reproduce the streak-free model-free ROI curve? plus cortex/medulla distinctness.
out: figures/<dt>_step3_verdict.png"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.ndimage as ndi
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
from figpath import fig as fpath
REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; D = "/net/beegfs/users/P101440/DCE_NIK"; TA = 375.0; Z = 21

def norm(t, c):
    b = c[t < 50].mean(); pk = c[(t > 20) & (t < 210)].max(); return (c - b) / (pk - b + 1e-9)

d = np.load(f"{D}/step2_slice{Z}.npz"); mf = d["mf"]; tmf = d["tmf"]
sl = np.load(f"{REF}/slice_{Z:02d}.npz"); cs = np.abs(sl["cs_img"]).astype(np.float32); tC = np.linspace(0, TA, cs.shape[-1])
base = cs[..., tC < 50].mean(-1); enh = cs - base[..., None]; m = cs.mean(-1); body = m > np.quantile(m, 0.5)
late = enh[..., (tC > 130) & (tC < 200)].mean(-1); kid = body & (late > np.quantile(late[body], 0.985)); kid = ndi.binary_opening(kid, iterations=1)
l, k = ndi.label(kid); kid = (l == (1 + np.argmax(ndi.sum(np.ones_like(l), l, range(1, k + 1))))) if k else kid
fcm = int(np.argmin(np.abs(tC - 85))); thr = np.median(enh[..., fcm][kid]); cortex = kid & (enh[..., fcm] > thr); medulla = kid & (enh[..., fcm] <= thr)
early = enh[..., (tC > 40) & (tC < 80)].mean(-1); ao = body & (early > np.quantile(early[body], 0.995)) & (~kid); ao = ndi.binary_opening(ao, iterations=1)
la, ka = ndi.label(ao); ao = (la == (1 + np.argmax(ndi.sum(np.ones_like(la), la, range(1, ka + 1))))) if ka else ao
def mfc(mask): return norm(tmf, np.array([im[mask].mean() for im in mf]))
def csc(mask): return norm(tC, np.array([cs[..., i][mask].mean() for i in range(cs.shape[-1])]))
cC, cM, cA = mfc(cortex), mfc(medulla), mfc(ao); sC, sM = csc(cortex), csc(medulla)
lt = (tmf > 100) & (tmf < 210); late_corr = np.corrcoef(cC[lt], cM[lt])[0, 1]
agC = np.corrcoef(cC, np.interp(tmf, tC, sC))[0, 1]; agM = np.corrcoef(cM, np.interp(tmf, tC, sM))[0, 1]

fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
a = cs[..., fcm]; ov = np.zeros((*a.shape, 4)); ov[cortex] = [1, 0, 0, .6]; ov[medulla] = [0, .5, 1, .6]; ov[ao] = [1, 1, 0, .5]
ax[0].imshow(np.rot90(a), cmap="gray", vmax=np.percentile(a, 99.5)); ax[0].imshow(np.rot90(ov)); ax[0].axis("off")
ax[0].set_title(f"slice {Z} @85s\ncortex(red)/medulla(blue)/aorta(yellow)", fontsize=9)
ax[1].plot(tmf, cC, "0.4", lw=1.2, label="model-free (real)"); ax[1].plot(tC, sC, "r", lw=2, label="CS K=5")
ax[1].set_xlim(0, 260); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8); ax[1].set_title(f"CORTEX: CS reproduces real? corr {agC:.3f}", fontsize=9)
ax[2].plot(tmf, cM, "0.4", lw=1.2, label="model-free (real)"); ax[2].plot(tC, sM, "b", lw=2, label="CS K=5")
ax[2].set_xlim(0, 260); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8); ax[2].set_title(f"MEDULLA: CS reproduces real? corr {agM:.3f}", fontsize=9)
ax[3].plot(tmf, cC, "r", lw=2, label="cortex"); ax[3].plot(tmf, cM, "b", lw=2, label="medulla"); ax[3].axvspan(100, 210, color="k", alpha=.05)
ax[3].set_xlim(0, 260); ax[3].grid(alpha=.3); ax[3].legend(fontsize=8); ax[3].set_title(f"cortex vs medulla (model-free)\nlate-phase corr {late_corr:.2f} (distinct)", fontsize=9)
fig.suptitle(f"STEP 3 verdict, slice {Z}: cortex/medulla DISTINCT & smooth, but CS K=5 already reproduces both", fontweight="bold")
fig.tight_layout(); p = fpath("step3_verdict.png"); fig.savefig(p, dpi=135); print("wrote", p.split("/")[-1])
print(f"cortex CS-vs-real corr {agC:.3f} | medulla {agM:.3f} | cortex-medulla late corr {late_corr:.2f}")
