"""GATE readout for the full-rank NIK run on kidney slice 21. the whole point: in the CORTEX
ROI, does NIK track the REAL cortical washout (~0.58) rather than CS's inflated ~0.70, and
does it do so SMOOTHLY (not by oscillating / streak-fitting)? read the two together.
out: figures/<dt>_gate_readout_slice21.png + printed verdict."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.ndimage as ndi
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
from figpath import fig as fpath
REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"; D = "/scratch/rnga/vvpshenov/DCE_NIK"
NIKD = f"{D}/results_spoke_full_slice21"; TA = 375.0; Z = 21

def norm(t, c):
    b = c[t < 50].mean(); pk = c[(t > 20) & (t < 210)].max(); return (c - b) / (pk - b + 1e-9)
def osc(t, n):                                             # noise fraction around the smooth curve
    sm = savgol_filter(n, 11, 3); return float(np.std(n - sm) / (np.abs(sm).max() + 1e-9))
def plateau(t, n): lt = (t > 100) & (t < 210); return float(np.median(n[lt]))

# --- recons ---
nik = np.abs(np.load(f"{NIKD}/nik_slice_{Z:02d}_cplx.npy")).astype(np.float32)   # [192,192,342] streak-free per-voxel
cs = np.abs(np.load(f"{REF}/slice_{Z:02d}.npz")["cs_img"]).astype(np.float32)     # [192,192,342] K=5
tN = np.linspace(0, TA, nik.shape[-1]); tC = np.linspace(0, TA, cs.shape[-1])
d = np.load(f"{D}/step2_slice{Z}.npz"); mf = d["mf"]; tmf = d["tmf"]              # model-free windows (real)

# --- segmentation from cs_img anatomy (same as step3_verdict) ---
base = cs[..., tC < 50].mean(-1); enh = cs - base[..., None]; m = cs.mean(-1); body = m > np.quantile(m, 0.5)
late = enh[..., (tC > 130) & (tC < 200)].mean(-1); kid = body & (late > np.quantile(late[body], 0.985)); kid = ndi.binary_opening(kid, iterations=1)
l, k = ndi.label(kid); kid = (l == (1 + np.argmax(ndi.sum(np.ones_like(l), l, range(1, k + 1))))) if k else kid
fcm = int(np.argmin(np.abs(tC - 85))); thr = np.median(enh[..., fcm][kid]); cortex = kid & (enh[..., fcm] > thr); medulla = kid & (enh[..., fcm] <= thr)
early = enh[..., (tC > 40) & (tC < 80)].mean(-1); ao = body & (early > np.quantile(early[body], 0.995)) & (~kid); ao = ndi.binary_opening(ao, iterations=1)
la, ka = ndi.label(ao); ao = (la == (1 + np.argmax(ndi.sum(np.ones_like(la), la, range(1, ka + 1))))) if ka else ao

def cur(vol, t, mask): return norm(t, np.array([vol[..., i][mask].mean() for i in range(vol.shape[-1])]))
def mcur(mask): return norm(tmf, np.array([im[mask].mean() for im in mf]))
# cortex
rC, cC, nC = mcur(cortex), cur(cs, tC, cortex), cur(nik, tN, cortex)
rM, cM, nM = mcur(medulla), cur(cs, tC, medulla), cur(nik, tN, medulla)
rA, nA = mcur(ao), cur(nik, tN, ao)
# plateaus (the gate)
pR, pCS, pNIK = plateau(tmf, rC), plateau(tC, cC), plateau(tN, nC)
oNIK_C, oNIK_A = osc(tN, nC), osc(tN, nA)
# nRMSE vs real (resample real -> each grid)
def nrmse(t, c, tr, cr): return float(np.sqrt(np.mean((c - np.interp(t, tr, cr)) ** 2)))
nrmse_cs_C, nrmse_nik_C = nrmse(tC, cC, tmf, rC), nrmse(tN, nC, tmf, rC)
pR_M, pCS_M, pNIK_M = plateau(tmf, rM), plateau(tC, cM), plateau(tN, nM)

print("===== GATE READOUT slice 21 full-rank NIK =====")
print(f"CORTEX late plateau:  real(model-free) {pR:.3f}  |  CS K=5 {pCS:.3f}  |  NIK full {pNIK:.3f}")
print(f"  -> NIK tracks real washout? {'YES' if abs(pNIK-pR) < abs(pNIK-pCS) and pNIK < 0.5*(pR+pCS) else 'NO'} (closer to {'real' if abs(pNIK-pR)<abs(pNIK-pCS) else 'CS'})")
print(f"CORTEX smoothness:    NIK osc {oNIK_C:.3f}  vs aorta baseline {oNIK_A:.3f}  -> {'SMOOTH' if oNIK_C < 2*oNIK_A else 'OSCILLATORY'}")
print(f"CORTEX nRMSE vs real: CS {nrmse_cs_C:.3f}  NIK {nrmse_nik_C:.3f}  -> NIK {'closes' if nrmse_nik_C < nrmse_cs_C else 'does NOT close'} the gap")
print(f"MEDULLA plateau (control): real {pR_M:.3f} CS {pCS_M:.3f} NIK {pNIK_M:.3f}  (NIK should NOT diverge much; CS reproduces medulla)")

verdict = "GO" if (abs(pNIK - pR) < abs(pNIK - pCS) and oNIK_C < 2 * oNIK_A and nrmse_nik_C < nrmse_cs_C) else "NO-GO"
print(f"\nVERDICT: {verdict}")

# --- figure ---
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = cs[..., fcm]; ov = np.zeros((*a.shape, 4)); ov[cortex] = [1, 0, 0, .6]; ov[medulla] = [0, .5, 1, .6]; ov[ao] = [1, 1, 0, .5]
ax[0].imshow(np.rot90(a), cmap="gray", vmax=np.percentile(a, 99.5)); ax[0].imshow(np.rot90(ov)); ax[0].axis("off")
ax[0].set_title(f"slice {Z} cortex/medulla/aorta", fontsize=9)
ax[1].plot(tmf, rC, "0.4", lw=1.4, label=f"real {pR:.2f}"); ax[1].plot(tC, cC, "b", lw=1.8, label=f"CS {pCS:.2f}"); ax[1].plot(tN, nC, "r", lw=1.8, label=f"NIK {pNIK:.2f}")
ax[1].axhspan(min(pR, pCS), max(pR, pCS), color="k", alpha=.05); ax[1].axvspan(100, 210, color="k", alpha=.05)
ax[1].set_xlim(0, 260); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8, title="late plateau"); ax[1].set_title(f"CORTEX gate: NIK osc {oNIK_C:.3f} (aorta {oNIK_A:.3f})", fontsize=9)
ax[2].plot(tmf, rM, "0.4", lw=1.4, label=f"real {pR_M:.2f}"); ax[2].plot(tC, cM, "b", lw=1.8, label=f"CS {pCS_M:.2f}"); ax[2].plot(tN, nM, "r", lw=1.8, label=f"NIK {pNIK_M:.2f}")
ax[2].set_xlim(0, 260); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8); ax[2].set_title("MEDULLA control (CS reproduces; NIK should agree)", fontsize=9)
fig.suptitle(f"GATE: does full-rank NIK correct CS's cortical-washout bias?  VERDICT {verdict}", fontweight="bold")
fig.tight_layout(); p = fpath("gate_readout_slice21.png"); fig.savefig(p, dpi=135); print(f"wrote {p.split('/')[-1]}")

# --- streak-free per-voxel cortical residual map this run enables (kidney bbox) ---
ys, xs = np.where(kid); pad = 12; y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, 192), max(xs.min() - pad, 0), min(xs.max() + pad, 192)
np.savez(f"{D}/gate_slice21.npz", cortex=cortex, medulla=medulla, ao=ao, rC=rC, cC=cC, nC=nC, rM=rM, cM=cM, nM=nM,
         tmf=tmf, tN=tN, tC=tC, bbox=(y0, y1, x0, x1), verdict=verdict,
         plateau=dict(real=pR, cs=pCS, nik=pNIK), osc=dict(cortex=oNIK_C, aorta=oNIK_A))
