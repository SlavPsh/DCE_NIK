"""TASK 2 analysis: canonical extraction (Path C) vs Path B, physical-scaling constants,
coil-consistency (common SENSE map), coil-subset stability, F0-vs-F2 fixed components.
Consumes per-coil fields from task2_query_amplitudes.py. Read-only. out: results/task2_scaling_coil_audit/."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os, sys, csv, json, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); sys.path.insert(0, ".")
import consolidated as C
D = "/scratch/rnga/vvpshenov/DCE_NIK"; OUT = f"{D}/results/task2_scaling_coil_audit"; AR = f"{OUT}/arrays"
SLICES = [18, 19, 21]; CFGS = [("F0", 0), ("F2", 2)]; FIXED = {0: "AIF", 1: "intAIF", 2: "baseline"}; eps = 1e-8
scales = json.load(open(f"{AR}/scales.json"))
def sense(a_rc_r, b1c): return np.sum(np.conj(b1c) * a_rc_r, -1) / (np.sum(np.abs(b1c) ** 2, -1) + eps)  # [X,Y]
def nrmse(a, b, m): return float(np.linalg.norm(a[m] - b[m]) / (np.linalg.norm(b[m]) + 1e-12))
def masks(Z):
    ctx = C.slice_ctx(Z); body = ctx["BODY"]; rois = ctx["rois"]; organ = np.zeros_like(body)
    for r in ["aorta", "cortex", "medulla"]: organ |= (rois.get(r) if rois.get(r) is not None else False)
    organ = ndi.binary_dilation(organ, iterations=3) & body
    return {"body": body, "aorta": rois["aorta"], "cortex": rois["cortex"], "medulla": rois["medulla"], "organ": organ}

# ---------- Part 1: canonical (C) vs cross-check (B) ----------
p1 = []
for F_lab, F in CFGS:
    for Z in SLICES:
        a_rc = np.load(f"{AR}/a_rc_F{F}_sl{Z}.npy"); b1c = np.load(f"{AR}/b1c_sl{Z}.npy"); Phi = np.load(f"{AR}/Phi_F{F}_sl{Z}.npy")
        Ic = np.load(f"{D}/results_batch/pk_f{F}_sl{Z}/nik_slice_{Z}_cplx.npy").astype(np.complex64)
        thetaC = np.stack([sense(a_rc[:, :, r, :], b1c) for r in range(a_rc.shape[2])], -1)   # [X,Y,R]
        thetaB = np.einsum("rt,xyt->xyr", np.linalg.pinv(Phi), Ic)
        body = masks(Z)["body"]
        for r in FIXED: p1.append(dict(cfg=F_lab, slice=Z, comp=FIXED[r], NRMSE_B_vs_C=nrmse(thetaC[..., r], thetaB[..., r], body)))
        np.save(f"{AR}/thetaC_F{F}_sl{Z}.npy", thetaC.astype(np.complex64))
with open(f"{OUT}/pathB_vs_C.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(p1[0].keys())); w.writeheader(); [w.writerow(r) for r in p1]

# ---------- Part 5/7: coil-consistency (weighted) per fixed r + ROI ----------
cc = []
for F_lab, F in CFGS:
    for Z in SLICES:
        a_rc = np.load(f"{AR}/a_rc_F{F}_sl{Z}.npy"); b1c = np.load(f"{AR}/b1c_sl{Z}.npy"); MK = masks(Z)
        w = np.abs(b1c)                                                # predefined weight = |S_c| (coil sensitivity)
        for r in [0, 1, 2]:
            th = sense(a_rc[:, :, r, :], b1c); ahat = b1c * th[:, :, None]  # a_hat_r,c = S_c theta_r
            num = np.sum((w * np.abs(a_rc[:, :, r, :] - ahat)) ** 2, -1)    # [X,Y] over coils
            den = np.sum((w * np.abs(a_rc[:, :, r, :])) ** 2, -1) + 1e-20
            for nm, mk in MK.items():
                cc.append(dict(cfg=F_lab, slice=Z, comp=FIXED[r], roi=nm, vox=int(mk.sum()),
                               E_coil=float(num[mk].sum() / (den[mk].sum() + 1e-20)),
                               phase_std=float(np.std(np.angle(th[mk]))), scale_meanabs=float(np.abs(th[mk]).mean())))
with open(f"{OUT}/coil_consistency.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cc[0].keys())); w.writeheader(); [w.writerow(r) for r in cc]

# ---------- Part 6: strong-coil + leave-one-coil-out stability (fixed r=1 intAIF) ----------
st = []
for F_lab, F in CFGS:
    for Z in SLICES:
        a_rc = np.load(f"{AR}/a_rc_F{F}_sl{Z}.npy"); b1c = np.load(f"{AR}/b1c_sl{Z}.npy"); ncc = b1c.shape[-1]; body = masks(Z)["organ"]
        energy = np.array([np.sum(np.abs(b1c[..., c]) ** 2) for c in range(ncc)]); strong = np.argsort(-energy)[:ncc // 2]  # predefined: top half
        def th_sub(idx, r=1): return sense(a_rc[:, :, r, idx], b1c[..., idx])
        th_all = th_sub(np.arange(ncc)); th_strong = th_sub(strong)
        loo = np.stack([th_sub(np.array([c for c in range(ncc) if c != k])) for k in range(ncc)], -1)  # [X,Y,ncc]
        st.append(dict(cfg=F_lab, slice=Z, comp="intAIF",
            strong_vs_all_NRMSE=nrmse(th_strong, th_all, body), strong_vs_all_corr=float(np.corrcoef(np.abs(th_strong)[body], np.abs(th_all)[body])[0, 1]),
            strong_scale_ratio=float(np.abs(th_strong[body]).mean() / (np.abs(th_all[body]).mean() + 1e-12)),
            LOOO_meanNRMSE=float(np.mean([nrmse(loo[..., k], th_all, body) for k in range(ncc)])),
            LOOO_map_std_over_mean=float(np.mean(np.std(np.abs(loo), -1)[body]) / (np.abs(th_all[body]).mean() + 1e-12))))
        if Z == 21:  # LOOO variability map
            var = np.std(np.abs(loo), -1)
            np.save(f"{AR}/LOOO_std_F{F}_sl21.npy", var)
with open(f"{OUT}/coil_subset_stability.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(st[0].keys())); w.writeheader(); [w.writerow(r) for r in st]

# ---------- figures ----------
# coil-resolved + SENSE + residual (F2 sl21, r=1 intAIF)
a_rc = np.load(f"{AR}/a_rc_F2_sl21.npy"); b1c = np.load(f"{AR}/b1c_sl21.npy"); th = sense(a_rc[:, :, 1, :], b1c)
ahat = b1c * th[:, :, None]; res = np.sum(np.abs(a_rc[:, :, 1, :] - ahat) ** 2, -1)
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for c in range(4):
    ax[0, c].imshow(np.rot90(np.abs(a_rc[:, :, 1, c])), cmap="magma"); ax[0, c].axis("off"); ax[0, c].set_title(f"|a_intAIF,coil{c}|", fontsize=9)
ax[1, 0].imshow(np.rot90(np.abs(th)), cmap="viridis", vmax=np.percentile(np.abs(th), 99)); ax[1, 0].axis("off"); ax[1, 0].set_title("SENSE common intAIF", fontsize=9)
ax[1, 1].imshow(np.rot90(res), cmap="inferno", vmax=np.percentile(res, 99)); ax[1, 1].axis("off"); ax[1, 1].set_title("coil-consistency residual", fontsize=9)
LOOO = np.load(f"{AR}/LOOO_std_F2_sl21.npy"); ax[1, 2].imshow(np.rot90(LOOO), cmap="inferno", vmax=np.percentile(LOOO, 99)); ax[1, 2].axis("off"); ax[1, 2].set_title("LOOO std map", fontsize=9)
th0 = sense(np.load(f"{AR}/a_rc_F0_sl21.npy")[:, :, 1, :], b1c)
ax[1, 3].imshow(np.rot90(np.abs(th0) - np.abs(th)), cmap="bwr", vmax=np.percentile(np.abs(np.abs(th0) - np.abs(th)), 99), vmin=-np.percentile(np.abs(np.abs(th0) - np.abs(th)), 99)); ax[1, 3].axis("off"); ax[1, 3].set_title("F0-F2 intAIF diff", fontsize=9)
fig.suptitle("TASK 2: coil-resolved amplitudes, SENSE common map, consistency residual, LOOO, F0-F2 (sl21)", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/coil_consistency_F2_sl21.png", dpi=120); plt.close(fig)

# summary print
print("=== Part1 Path B vs C (body NRMSE, should ~0) ===")
for r in p1[:6]: print(f"  {r['cfg']} sl{r['slice']} {r['comp']}: {r['NRMSE_B_vs_C']:.2e}")
print("\n=== Part5/7 coil-consistency E_coil (organ) F0 vs F2, per fixed comp (mean over slices) ===")
import numpy as _np
for comp in ["AIF", "intAIF", "baseline"]:
    for F_lab, _ in CFGS:
        e = _np.mean([r["E_coil"] for r in cc if r["cfg"] == F_lab and r["comp"] == comp and r["roi"] == "organ"])
        print(f"  {comp:>9} {F_lab}: E_coil(organ) {e:.3f}")
print("\n=== Part6 strong/LOOO stability (organ, intAIF) ===")
for r in st: print(f"  {r['cfg']} sl{r['slice']}: strong-vs-all NRMSE {r['strong_vs_all_NRMSE']:.3f} corr {r['strong_vs_all_corr']:.3f} | LOOO meanNRMSE {r['LOOO_meanNRMSE']:.3f}")
print("\nscales.json:", json.dumps(scales, indent=0)[:400])
print("wrote", OUT)
