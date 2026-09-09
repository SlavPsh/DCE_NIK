"""Reference C: MODEL-FREE bolus curve from sliding-window NUFFT (no rank cage, no prior).
Each window is a density-compensated adjoint NUFFT + SENSE combine -> streaky but UNBIASED;
the ROI mean averages the incoherent streaks down, so its expectation is the true curve.
Compare NIK and CS bolus shapes against it. out: figures/nufft_bolus.png"""
import numpy as np, finufft, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figpath import fig as fpath
REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"
D = "/net/beegfs/users/P101440/DCE_NIK"; CSD = "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs"
SL = 13
meta = json.load(open(f"{D}/results_nufft/meta.json")); SIGN = meta["sign"]

sh = np.load(f"{REF}/shared.npz")
traj = np.asarray(sh["traj_norm"]).astype(np.complex64)
vt = np.asarray(sh["view_time"]).ravel().astype(np.float64)
TA = float(sh["TA"]); nx = int(sh["nx"]); bas = int(sh["bas"])
sl = np.load(f"{REF}/slice_{SL:02d}.npz")
kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64)
b1 = np.asarray(sl["b1"]).astype(np.complex64); ncc = kdata.shape[2]
aroi = np.load(f"{D}/aorta_roi.npy")
den = np.sum(np.abs(b1) ** 2, axis=2) + 1e-12
order = np.argsort(vt)                                    # acquisition order

def win_image(idx, eps=1e-5):
    tr = traj[:, idx]; w = np.maximum(np.abs(tr), 1.0 / nx / 4.0)
    x = (SIGN * 2 * np.pi * np.real(tr)).astype(np.float64).ravel()
    y = (SIGN * 2 * np.pi * np.imag(tr)).astype(np.float64).ravel()
    acc = np.zeros((nx, nx), dtype=np.complex128)
    for c in range(ncc):
        v = (kdata[:, idx, c] * w).astype(np.complex128).ravel()
        acc += finufft.nufft2d1(x, y, v, (nx, nx), eps=eps, isign=1) * np.conj(b1[:, :, c])
    img = np.abs(acc / den)
    s = (nx - bas) // 2
    return img[s:s + bas, s:s + bas]

def sliding_curve(W, step=8):
    ts, cs_ = [], []
    for a in range(0, len(order) - W + 1, step):
        idx = order[a:a + W]
        ts.append(vt[idx].mean() * TA)
        cs_.append(float(win_image(idx)[aroi].mean()))
    return np.array(ts), np.array(cs_)

print("building model-free sliding-window curves ...", flush=True)
curves = {}
for W in (21, 41, 81):
    t, c = sliding_curve(W)
    curves[W] = (t, c)
    print(f"  W={W:3d} spokes ({W*TA/len(order):.1f}s)  {len(t)} windows", flush=True)

# --- method curves ---
nik = np.abs(np.load(f"{D}/results_spoke_nik_f100/nik_slice_13.npy")).astype(np.float32)
cs = np.abs(np.load(f"{CSD}/cs_slice13_f100.npy")).astype(np.float32)
def roi_curve(v):
    t = np.linspace(0, TA, v.shape[-1])
    return t, np.array([v[..., i][aroi].mean() for i in range(v.shape[-1])])
tn, cn = roi_curve(nik); tc, cc = roi_curve(cs)

def norm(t, c, ref_t=None):
    b = c[t < 50].mean(); pk = c[(t > 50) & (t < 200)].max()
    return (c - b) / (pk - b + 1e-9)

REFW = 41
tr_, cr_ = curves[REFW]
nr = norm(tr_, cr_); nn = norm(tn, cn); nc_ = norm(tc, cc)

def shape_stats(t, n, lab):
    fp = (t > 30) & (t < 200)
    up = np.gradient(n, t)[fp].max(); ttp = t[fp][np.argmax(n[fp])]
    half = 0.5; above = fp & (n > half)
    fwhm = (t[above].max() - t[above].min()) if above.any() else np.nan
    print(f"  {lab:26} upslope {up:7.4f}/s   TTP {ttp:6.1f}s   FWHM {fwhm:6.1f}s")
    return up, ttp, fwhm
print(f"\nbolus shape (reference = model-free NUFFT W={REFW}):")
s_ref = shape_stats(tr_, nr, f"NUFFT model-free (W={REFW})")
s_nik = shape_stats(tn, nn, "NIK")
s_cs = shape_stats(tc, nc_, "CS")
# curve RMSE vs the model-free reference (on the reference time grid)
for lab, t, n in [("NIK", tn, nn), ("CS", tc, nc_)]:
    r = np.sqrt(np.mean((np.interp(tr_, t, n) - nr) ** 2))
    print(f"  RMSE vs model-free reference: {lab:4} {r:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
for W, col in [(21, "0.75"), (41, "0.45"), (81, "0.2")]:
    t, c = curves[W]; ax[0].plot(t, norm(t, c), lw=1.2, color=col, label=f"NUFFT model-free W={W} ({W*TA/len(order):.1f}s)")
ax[0].set_title("model-free reference at 3 window widths", fontsize=10.5)
for a in ax: a.set_xlabel("time (s)"); a.set_ylabel("norm. enhancement"); a.grid(alpha=.3); a.set_xlim(0, 250)
ax[0].legend(fontsize=8)
ax[1].plot(tr_, nr, lw=2.2, color="0.35", label=f"NUFFT model-free (W={REFW}) = reference")
ax[1].plot(tn, nn, lw=1.6, color="#7c3aed", label="NIK")
ax[1].plot(tc, nc_, lw=1.6, color="#0369a1", label="CS")
ax[1].set_title("NIK / CS vs the model-free bolus", fontsize=10.5); ax[1].legend(fontsize=9)
fig.suptitle("Aorta bolus vs a model-free NUFFT reference (slice 13)", fontweight="bold")
fig.tight_layout(); fig.savefig(fpath(f"nufft_bolus.png"), dpi=145, bbox_inches="tight")
print("\nwrote figures/nufft_bolus.png")
