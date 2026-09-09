"""TASK 0 global-offset check (no GPU). is NIK's temporal error vs the streak-free real curve
a single global scale+baseline (a normalization/scaling BUG) or ROI-dependent SHAPE (the
temporal negative stands)? uses RAW (un-normalized) ROI curves -- per-ROI peak-normalization
would hide exactly the global offset we are testing for.
recons: full-rank NIK (best temporal expressiveness) f25, slices 18,19,21. out: task0.json + fig."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, ".")
from figpath import fig as fpath
import consolidated as C
D = "/net/beegfs/users/P101440/DCE_NIK"; REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; TA = 375.0
SLICES = [18, 19, 21]; ROIS = ["aorta", "cortex", "medulla", "liver"]
NIK = {18: f"{D}/results_batch/full_sl18/nik_slice_18_cplx.npy", 19: f"{D}/results_batch/full_sl19/nik_slice_19_cplx.npy",
       21: f"{D}/results_batch/full_sl21f25/nik_slice_21_cplx.npy"}

def relresid(pred, targ):                                  # rmse normalized by target dynamic range
    return float(np.sqrt(np.mean((pred - targ) ** 2)) / (targ.max() - targ.min() + 1e-9))

out = {}; fig, axes = plt.subplots(len(SLICES), len(ROIS), figsize=(4 * len(ROIS), 3.2 * len(SLICES)))
for si, Z in enumerate(SLICES):
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; mf = np.load(f"{D}/step2_slice{Z}.npz")["mf"]; tmf = ctx["tmf"]
    mag = np.abs(np.load(NIK[Z])).astype(np.float32); tN = np.linspace(0, TA, mag.shape[-1])
    # RAW (un-normalized) ROI curves: NIK on tN, real (model-free) resampled to tN
    names = [r for r in ROIS if rois.get(r) is not None and rois[r].sum() > 0]
    Ncur = {r: np.array([mag[..., i][rois[r]].mean() for i in range(mag.shape[-1])]) for r in names}
    Rcur = {r: np.interp(tN, tmf, np.array([im[rois[r]].mean() for im in mf])) for r in names}
    # --- single GLOBAL scale a and baseline b across ALL ROIs: minimize ||a*N + b - R|| ---
    Nstack = np.concatenate([Ncur[r] for r in names]); Rstack = np.concatenate([Rcur[r] for r in names])
    Amat = np.stack([Nstack, np.ones_like(Nstack)], 1); (a_g, b_g), *_ = np.linalg.lstsq(Amat, Rstack, rcond=None)
    # scale-only and baseline-only globals (decompose multiplicative vs additive)
    a_s = float((Nstack @ Rstack) / (Nstack @ Nstack + 1e-12))                       # b=0
    b_o = float(np.mean(Rstack - Nstack))                                            # a=1
    rows = []
    for ci, r in enumerate(names):
        N, R = Ncur[r], Rcur[r]
        res_raw = relresid(N, R)
        res_glob = relresid(a_g * N + b_g, R)                                        # after ONE global a,b
        # per-ROI best scale+baseline: only SHAPE mismatch survives this
        Ar = np.stack([N, np.ones_like(N)], 1); (ai, bi), *_ = np.linalg.lstsq(Ar, R, rcond=None)
        res_roi = relresid(ai * N + bi, R)
        rows.append(dict(roi=r, res_raw=res_raw, res_global=res_glob, res_perROI_shape=res_roi, ai=float(ai), bi=float(bi)))
        ax = axes[si, ROIS.index(r)]
        ax.plot(tN, R, "0.5", lw=1.4, label="real"); ax.plot(tN, a_g * N + b_g, "r", lw=1.6, label="NIK (global a,b)")
        ax.plot(tN, ai * N + bi, "b--", lw=1, label="NIK (per-ROI a,b)")
        ax.set_xlim(0, 260); ax.set_title(f"sl{Z} {r}\nglob {res_glob:.2f} | shape {res_roi:.2f}", fontsize=8.5)
        ax.grid(alpha=.3);
        if si == 0 and r == names[0]: ax.legend(fontsize=6.5)
    for r in ROIS:
        if r not in names: axes[si, ROIS.index(r)].axis("off")
    out[Z] = dict(global_a=float(a_g), global_b=float(b_g), scale_only=a_s, baseline_only=b_o,
                  perROI_a_spread=float(np.std([x["ai"] for x in rows]) / (np.mean([x["ai"] for x in rows]) + 1e-9)), rois=rows)
fig.suptitle("TASK 0 global-offset check: real vs NIK after ONE global scale+baseline (red) vs per-ROI (blue)", fontweight="bold")
fig.tight_layout(); p = fpath("task0_offset.png"); fig.savefig(p, dpi=130)
json.dump(out, open(f"{D}/task0.json", "w"), indent=1, default=float)

print(f"{'slice':>5} {'roi':>8} {'raw':>7} {'afterGLOBAL':>12} {'afterPerROI(shape)':>19}")
for Z in SLICES:
    for r in out[Z]["rois"]:
        print(f"{Z:>5} {r['roi']:>8} {r['res_raw']:>7.2f} {r['res_global']:>12.2f} {r['res_perROI_shape']:>19.2f}")
    print(f"      global a={out[Z]['global_a']:.3g} b={out[Z]['global_b']:.3g} | per-ROI scale spread (CoV)={out[Z]['perROI_a_spread']:.2f}")
print(f"\nwrote {p.split('/')[-1]}")
print("READ: if 'afterGLOBAL' small for ALL ROIs -> global offset (bug-like). if cortex 'afterPerROI(shape)' stays large -> shape mismatch survives scale+baseline -> temporal negative STANDS.")
