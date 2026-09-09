"""TASK 4 evaluation: compare NIK-F0, direct-discrete, NUFFT-sanity coefficient maps against the
KNOWN XCAT truth. Metrics: (1) coefficient-map NRMSE per component x ROI, (2) held-out (non-input)
k-space NMSE at f25 (data no method saw), (3) ROI temporal-curve NRMSE (theta@Phi vs I_true).
Fairness: ONE global complex scale per method (fit on body mask), applied to all components/ROIs.
Truth is never derived from a reconstruction. usage: python task4_evaluate.py"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, csv, numpy as np; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cs_nufft import NufftSubspace
OUT = "/net/beegfs/users/P101440/DCE_NIK/results/task4_xcat_nomotion_pilot"; FIG = f"{OUT}/figures"; os.makedirs(FIG, exist_ok=True)
S = np.load(f"{OUT}/arrays/sim.npz"); R = np.load(f"{OUT}/arrays/rois.npz")
Phi = S["Phi"]; b1 = S["b1"]; kx = S["kx"]; ky = S["ky"]; keep = S["keep_f25"]; times = S["times"]
F, NA, RO = kx.shape; C = b1.shape[-1]; N = RO
th_true = S["theta_true"]; I_true = S["I_true"]; body = S["labels"] > 0
ROIS = {k: R[k] for k in R.files}; COMP = ["AIF", "intAIF", "baseline"]
def nrmse(a, b, m): return float(np.linalg.norm(a[m] - b[m]) / (np.linalg.norm(b[m]) + 1e-12))
def gscale(th):  # single global complex scalar, fit on body, all components jointly
    return np.vdot(th[body], th_true[body]) / (np.vdot(th[body], th[body]) + 1e-12)

# ---- collect available method outputs ----
M = {}  # (method,frac) -> theta[N,N,3]
for frac in ["f100", "f25"]:
    p = f"{OUT}/arrays/direct_{frac}.npz"
    if os.path.exists(p): M[("direct", frac)] = np.load(p)["theta_direct"]
    p = f"{OUT}/arrays/nufft_{frac}.npz"
    if os.path.exists(p): M[("nufft", frac)] = np.load(p)["theta_nufft"]
    p = f"{OUT}/arrays/csfit_{frac}.npz"
    if os.path.exists(p): M[("csfit", frac)] = np.load(p)["theta_csfit"]
    import glob
    for f in sorted(glob.glob(f"{OUT}/arrays/nik_F0_{frac}_seed*.npz")):
        seed = f.split("seed")[-1].split(".")[0]; M[(f"nik_s{seed}", frac)] = np.load(f)["thetaC"]
print("methods loaded:", list(M.keys()), flush=True)

# ---- held-out non-input k-space NMSE (f25 only; the data nobody saw) ----
# forward on GPU (cufinufft) - the 4-core node starves CPU finufft; GPUSub matches it to 3e-6.
import cupy as cp; from task4_gpu_recon import GPUSub
y25_non = np.load(f"{OUT}/arrays/y25_non.npy", allow_pickle=True)
trajs_non = [(kx[t][~keep[t]], ky[t][~keep[t]]) for t in range(F)]
E_non = GPUSub(Phi, b1, trajs_non)
def heldout_nmse(th, s):
    yh = E_non.fwd(cp.asarray((th * s).astype(np.complex128))); num = 0.0; den = 0.0
    for t in range(F):
        a = cp.asnumpy(yh[t]).astype(np.complex64).ravel(); bq = np.asarray(y25_non[t]).astype(np.complex64).ravel(); num += np.sum(np.abs(a - bq)**2); den += np.sum(np.abs(bq)**2)
    return float(num / (den + 1e-12))

# ---- metrics table ----
rows = []
for (meth, frac), th in M.items():
    s = gscale(th); thc = th * s
    rec = dict(method=meth, frac=frac, global_scale_abs=float(np.abs(s)), global_scale_phase_deg=float(np.angle(s, deg=True)))
    for ci, cn in enumerate(COMP):
        m = ROIS["aorta"] if cn == "AIF" else body
        rec[f"{cn}_NRMSE"] = nrmse(np.abs(thc[..., ci]), np.abs(th_true[..., ci]), m)
    rec["complex_map_NRMSE"] = nrmse(thc, th_true, body)
    # ROI temporal-curve NRMSE
    Ih = np.einsum("xyr,tr->xyt", thc, Phi)
    for rn, rm in ROIS.items():
        if rm.sum() == 0: continue
        ct = Ih[rm].mean(0); cg = I_true[rm].mean(0); rec[f"tcurve_{rn}_NRMSE"] = float(np.linalg.norm(ct - cg) / (np.linalg.norm(cg) + 1e-12))
    if frac == "f25": rec["heldout_kspace_NMSE"] = heldout_nmse(th, s)
    rows.append(rec)
keys = sorted({k for r in rows for k in r})
with open(f"{OUT}/metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "frac"] + [k for k in keys if k not in ("method", "frac")]); w.writeheader()
    for r in rows: w.writerow(r)
print("wrote metrics.csv", flush=True)

# ================= FIGURES =================
def show(ax, im, t, vmax=None): ax.imshow(np.abs(im), cmap="gray", vmax=vmax); ax.set_title(t, fontsize=8); ax.axis("off")
# Fig 1: truth coefficient maps
fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
for i, cn in enumerate(COMP): show(ax[i], th_true[..., i], f"truth {cn}")
fig.suptitle("Fig1 known truth coefficient maps"); fig.tight_layout(); fig.savefig(f"{FIG}/fig1_truth_maps.png", dpi=110); plt.close(fig)
# Fig 2/3: per-method maps + error at f25
f25m = [k for k in M if k[1] == "f25"]
if f25m:
    fig, ax = plt.subplots(len(f25m), 3, figsize=(9, 3 * len(f25m)), squeeze=False)
    for r, key in enumerate(f25m):
        s = gscale(M[key]); thc = M[key] * s
        for i, cn in enumerate(COMP): show(ax[r, i], thc[..., i], f"{key[0]} {cn}", vmax=np.abs(th_true[..., i]).max())
    fig.suptitle("Fig2 f25 method coefficient maps (scale-matched)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig2_f25_maps.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(len(f25m), 3, figsize=(9, 3 * len(f25m)), squeeze=False)
    for r, key in enumerate(f25m):
        s = gscale(M[key]); err = (M[key] * s - th_true)
        for i, cn in enumerate(COMP): show(ax[r, i], err[..., i], f"{key[0]} {cn} err")
    fig.suptitle("Fig3 f25 error maps (method-truth)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig3_f25_errmaps.png", dpi=110); plt.close(fig)
# Fig 4: f100 method coefficient maps (sanity / near-full sampling)
f100m = [k for k in M if k[1] == "f100"]
if f100m:
    fig, ax = plt.subplots(len(f100m), 3, figsize=(9, 3 * len(f100m)), squeeze=False)
    for r, key in enumerate(f100m):
        s = gscale(M[key]); thc = M[key] * s
        for i, cn in enumerate(COMP): show(ax[r, i], thc[..., i], f"{key[0]} {cn}", vmax=np.abs(th_true[..., i]).max())
    fig.suptitle("Fig4 f100 method coefficient maps (scale-matched)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig4_f100_maps.png", dpi=110); plt.close(fig)
# Fig 10: intAIF map f100 vs f25 side-by-side per method (undersampling degradation)
pairs = sorted({k[0] for k in M})
fig, ax = plt.subplots(len(pairs), 3, figsize=(9, 3 * len(pairs)), squeeze=False)
for r, meth in enumerate(pairs):
    show(ax[r, 0], th_true[..., 1], f"truth intAIF", vmax=np.abs(th_true[..., 1]).max())
    for ci, frac in enumerate(["f100", "f25"]):
        if (meth, frac) in M:
            s = gscale(M[(meth, frac)]); show(ax[r, ci + 1], (M[(meth, frac)] * s)[..., 1], f"{meth} intAIF {frac}", vmax=np.abs(th_true[..., 1]).max())
        else: ax[r, ci + 1].axis("off")
fig.suptitle("Fig10 intAIF map: truth vs f100 vs f25 (undersampling effect)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig10_intaif_f100_f25.png", dpi=110); plt.close(fig)
# Fig 5: ROI temporal curves truth vs methods
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for j, rn in enumerate(["aorta", "cortex", "medulla"]):
    rm = ROIS[rn]; ax[j].plot(times, np.abs(I_true[rm].mean(0)), "k-", lw=2, label="truth")
    for key in f25m:
        s = gscale(M[key]); Ih = np.einsum("xyr,tr->xyt", M[key] * s, Phi); ax[j].plot(times, np.abs(Ih[rm].mean(0)), lw=1, label=key[0])
    ax[j].set_title(f"{rn} (f25)"); ax[j].set_xlabel("s")
ax[0].legend(fontsize=7); fig.suptitle("Fig5 ROI temporal curves (f25)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig5_roi_curves.png", dpi=110); plt.close(fig)
# Fig 6: metric bars
def bars(metrics, fname, title):
    labels = [f"{r['method']}/{r['frac']}" for r in rows]
    fig, ax = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.4))
    if len(metrics) == 1: ax = [ax]
    for k, mk in enumerate(metrics):
        vals = [r.get(mk, np.nan) for r in rows]; ax[k].bar(range(len(vals)), vals); ax[k].set_xticks(range(len(vals)))
        ax[k].set_xticklabels(labels, rotation=90, fontsize=6); ax[k].set_title(mk, fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(f"{FIG}/{fname}", dpi=110); plt.close(fig)
bars(["intAIF_NRMSE", "baseline_NRMSE", "AIF_NRMSE"], "fig6_coef_nrmse.png", "Fig6 coefficient-map NRMSE")
bars(["complex_map_NRMSE", "heldout_kspace_NMSE"], "fig7_complex_heldout.png", "Fig7 complex-map + held-out k-space")
# Fig 8: NUFFT sanity dynamic frames (if present)
if os.path.exists(f"{OUT}/arrays/nufft_f25.npz"):
    Ic = np.load(f"{OUT}/arrays/nufft_f25.npz")["Ic"]; fr = [5, 30, 60, 120, 180]
    fig, ax = plt.subplots(2, len(fr), figsize=(2.4 * len(fr), 5))
    for j, t in enumerate(fr):
        show(ax[0, j], I_true[..., t], f"truth t={times[t]:.0f}s"); show(ax[1, j], Ic[..., t], f"nufft t={times[t]:.0f}s")
    fig.suptitle("Fig8 NUFFT sanity dynamic vs truth (f25)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig8_nufft_dynamic.png", dpi=110); plt.close(fig)
# Fig 9: temporal basis
fig, ax = plt.subplots(1, 1, figsize=(6, 3.4))
for i, cn in enumerate(COMP): ax.plot(times, np.real(Phi[:, i]), label=cn)
ax.legend(); ax.set_xlabel("s"); ax.set_title("Fig9 F0 temporal basis Phi(t)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig9_basis.png", dpi=110); plt.close(fig)
print("wrote figures ->", FIG, flush=True)
print("EVAL_DONE", flush=True)
