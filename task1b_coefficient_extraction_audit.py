"""TASK 1B: coefficient-extraction and directness audit (read-only; no retraining/arch change).
Paths: A = current magnitude-first (control), B = complex temporal projection onto the model's
own basis, C = accessibility/identifiability of the learned per-coil amplitudes A_r,c.
Consistency (complex/magnitude temporal NRMSE), magnitude-nonlinearity, phase(t), F2 free maps.
out: results/task1b_coefficient_extraction/."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os, sys, csv, torch, scipy.ndimage as ndi
from types import SimpleNamespace
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); sys.path.insert(0, ".")
import consolidated as C
from train_grasp_nik import build_model
D = "/scratch/rnga/vvpshenov/DCE_NIK"; REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
OUT = f"{D}/results/task1b_coefficient_extraction"
for s in ["figures", "arrays", "logs", "script_snapshot"]: os.makedirs(f"{OUT}/{s}", exist_ok=True)
SLICES = [18, 19, 21]; CFGS = [("F0", 0), ("F2", 2)]
sh = np.load(f"{REF}/shared.npz"); TA = float(sh["TA"]); frame_t_norm = (2.0 * sh["frame_time"] - 1.0).astype(np.float32)
frame_t = torch.tensor(frame_t_norm); nt = len(frame_t_norm)

def rebuild(Z, F):
    ck = torch.load(f"{D}/results_batch/pk_f{F}_sl{Z}/model_slice_{Z:02d}.pt", map_location="cpu", weights_only=False)
    a = SimpleNamespace(model="wire_ff_patlak", patlak_free=F, aif_file=f"{D}/aif_slice{Z}.npz",
        coil_embed_dim=ck["coil_embed_dim"], hidden=ck["hidden"], depth=ck["depth"], w0=ck["w0"], s0=ck["s0"],
        k_freq=ck["k_freq"], k_sigma=ck["k_sigma"], t_freq=ck["t_freq"], t_sigma=ck["t_sigma"], ff_seed=ck["ff_seed"],
        phi_hidden=64, phi_depth=3, phi_w0=30.0)
    m = build_model(a, ck["ncc"]); m.load_state_dict(ck["state_dict"], strict=True); m.eval(); return m

def phiA(Z):                                                            # current 3-col basis at linspace (analysis convention)
    az = np.load(f"{D}/aif_slice{Z}.npz"); tg = np.linspace(0, TA, nt)
    aif = np.interp(tg, az["tC"], az["aif_frame"]); aif /= (aif.max() + 1e-9)
    iaif = np.concatenate([[0], np.cumsum(0.5 * (aif[1:] + aif[:-1]) * np.diff(tg))]); iaif /= (iaif.max() + 1e-9)
    return np.stack([aif, iaif, np.ones_like(aif)], 1)                  # [nt,3] real

def nrmse(a, b, m):
    a, b = a[m], b[m]; return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

rows = []; PHASE = {}
for F_lab, F in CFGS:
    for Z in SLICES:
        ctx = C.slice_ctx(Z); body = ctx["BODY"]; rois = ctx["rois"]
        organ = np.zeros_like(body)
        for r in ["aorta", "cortex", "medulla"]: organ |= (rois.get(r) if rois.get(r) is not None else False)
        organ = ndi.binary_dilation(organ, iterations=3) & body
        MASKS = {"body": body, "aorta": rois["aorta"], "cortex": rois["cortex"], "medulla": rois["medulla"], "organ": organ}
        Ic = np.load(f"{D}/results_batch/pk_f{F}_sl{Z}/nik_slice_{Z}_cplx.npy").astype(np.complex64)  # [X,Y,T]
        m = rebuild(Z, F)
        with torch.no_grad(): Pm = torch.view_as_complex(m.basis(frame_t).contiguous()).numpy()        # [nt,R] complex
        PA = phiA(Z)                                                                                    # [nt,3] real
        # --- Path A: magnitude-first onto 3 fixed cols (current) ---
        mag = np.abs(Ic); pinvA = np.linalg.pinv(PA)                                                    # [3,nt]
        thetaA = np.einsum("rt,xyt->xyr", pinvA, mag)                                                   # [X,Y,3] real (signed)
        fitA = np.einsum("xyr,tr->xyt", thetaA, PA)                                                     # magnitude fit series
        A_AIF, A_int = np.abs(thetaA[..., 0]), np.abs(thetaA[..., 1])                                   # current maps (abs)
        # --- Path B: complex projection onto model's full basis ---
        pinvB = np.linalg.pinv(Pm)                                                                      # [R,nt] complex
        thetaB = np.einsum("rt,xyt->xyr", pinvB, Ic)                                                    # [X,Y,R] complex
        fitB = np.einsum("xyr,tr->xyt", thetaB, Pm)                                                     # complex recon of series
        B_AIF, B_int = np.abs(thetaB[..., 0]), np.abs(thetaB[..., 1])                                   # direct fixed-atom maps
        # --- Part 5: magnitude nonlinearity ---
        S1 = np.abs(fitB)                                                                               # |sum theta_r Phi_r| = |I_c|
        S2 = np.abs(np.einsum("xyr,tr->xyt", np.abs(thetaB), Pm))                                       # |sum |theta_r| Phi_r|
        # --- consistency + comparison metrics per ROI ---
        for nm, mk in MASKS.items():
            rows.append(dict(cfg=F_lab, slice=Z, roi=nm, vox=int(mk.sum()),
                complexNRMSE_B=nrmse(fitB, Ic, mk),                       # ~0 expected (recon = linear factorization)
                magNRMSE_B=nrmse(np.abs(fitB), mag, mk),                  # ~0
                magNRMSE_A=nrmse(fitA, mag, mk),                          # current magnitude-fit residual
                magNL_S1_vs_S2=nrmse(S2, S1, mk),                         # |theta|-mix vs true magnitude
                AIFmap_A_vs_B_NRMSE=nrmse(A_AIF, B_AIF, mk),              # Q5: does magnitude-first change the map?
                intAIFmap_A_vs_B_NRMSE=nrmse(A_int, B_int, mk),
                AIFmap_A_vs_B_corr=float(np.corrcoef(A_AIF[mk], B_AIF[mk])[0, 1]),
                intAIFmap_A_vs_B_corr=float(np.corrcoef(A_int[mk], B_int[mk])[0, 1])))
        # save arrays
        for tag, arr in [("A_AIF", A_AIF), ("A_intAIF", A_int), ("B_AIF", B_AIF), ("B_intAIF", B_int)]:
            np.save(f"{OUT}/arrays/{F_lab}_sl{Z}_{tag}.npy", arr)
        if F == 2:
            np.save(f"{OUT}/arrays/{F_lab}_sl{Z}_free0.npy", np.abs(thetaB[..., 3]))
            np.save(f"{OUT}/arrays/{F_lab}_sl{Z}_free1.npy", np.abs(thetaB[..., 4]))
        # phase(t) per ROI (Q7)
        PHASE[(F_lab, Z)] = {nm: np.angle(np.array([Ic[..., i][MASKS[nm]].mean() for i in range(nt)])) for nm in ["aorta", "cortex", "medulla"]}
        # figures per (cfg, slice): maps A vs B + diff, residual, ROI fits
        if Z == 21:
            tsec = frame_t_norm  # for x-axis use frame index->s
            tt = (frame_t_norm + 1) / 2 * TA
            fig, ax = plt.subplots(2, 4, figsize=(17, 8))
            vmx = np.percentile(B_int, 99)
            for j, (img, ttl) in enumerate([(A_int, "Path A intAIF (mag-first)"), (B_int, "Path B intAIF (complex)"),
                                            (A_int - B_int, "A-B diff"), (np.abs(fitA - mag).mean(-1), "Path A mag residual (mean_t)")]):
                cmap = "bwr" if "diff" in ttl else ("inferno" if "residual" in ttl else "viridis")
                vv = np.percentile(np.abs(img), 99); im = ax[0, j].imshow(np.rot90(img), cmap=cmap, vmax=vv, vmin=-vv if "diff" in ttl else 0, interpolation="nearest")
                ax[0, j].axis("off"); ax[0, j].set_title(f"{F_lab} sl{Z} {ttl}", fontsize=9); fig.colorbar(im, ax=ax[0, j], fraction=.046)
            for k, nm in enumerate(["aorta", "cortex", "medulla"]):
                mk = MASKS[nm]
                ax[1, k].plot(tt, mag[mk].mean(0), "k", lw=1.4, label="|I_c| true")
                ax[1, k].plot(tt, np.abs(fitB)[mk].mean(0), "g--", lw=1, label="|Path B fit|")
                ax[1, k].plot(tt, fitA[mk].mean(0), "r:", lw=1.2, label="Path A mag-fit")
                ax[1, k].set_xlim(0, 260); ax[1, k].grid(alpha=.3); ax[1, k].legend(fontsize=7); ax[1, k].set_title(f"{nm} temporal fit", fontsize=9)
            ax[1, 3].plot(tt, PHASE[(F_lab, Z)]["aorta"], label="aorta"); ax[1, 3].plot(tt, PHASE[(F_lab, Z)]["cortex"], label="cortex")
            ax[1, 3].plot(tt, PHASE[(F_lab, Z)]["medulla"], label="medulla"); ax[1, 3].set_xlim(0, 260); ax[1, 3].grid(alpha=.3); ax[1, 3].legend(fontsize=7); ax[1, 3].set_title("phase(t) [rad]", fontsize=9)
            fig.suptitle(f"TASK 1B {F_lab} sl{Z}: extraction paths, consistency, phase", fontweight="bold")
            fig.tight_layout(); fig.savefig(f"{OUT}/figures/{F_lab}_sl{Z}_paths.png", dpi=120); plt.close(fig)

# F2 free basis + free maps figure
m = rebuild(21, 2)
with torch.no_grad(): Pm = torch.view_as_complex(m.basis(frame_t).contiguous()).numpy()
tt = (frame_t_norm + 1) / 2 * TA
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for r in [3, 4]:
    ax[0].plot(tt, Pm[:, r].real, label=f"free{r-3} Re"); ax[0].plot(tt, Pm[:, r].imag, "--", label=f"free{r-3} Im")
ax[0].plot(tt, Pm[:, 0].real, "k", lw=1, label="AIF (fixed)"); ax[0].set_xlim(0, 260); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7); ax[0].set_title("F2 basis: fixed AIF + free atoms")
f0 = np.load(f"{OUT}/arrays/F2_sl21_free0.npy"); f1 = np.load(f"{OUT}/arrays/F2_sl21_free1.npy")
ax[1].imshow(np.rot90(f0), cmap="magma", vmax=np.percentile(f0, 99)); ax[1].axis("off"); ax[1].set_title("|free0 coeff map|")
ax[2].imshow(np.rot90(f1), cmap="magma", vmax=np.percentile(f1, 99)); ax[2].axis("off"); ax[2].set_title("|free1 coeff map|")
fig.suptitle("TASK 1B: F2 free temporal basis + free-component coefficient maps (sl21)", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/F2_free_components.png", dpi=120); plt.close(fig)

with open(f"{OUT}/metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

# console summary
print("=== consistency + magnitude-nonlinearity + A-vs-B (organ & aorta rows) ===")
print(f"{'cfg':>4}{'sl':>4}{'roi':>8}{'cplxNRMSE_B':>12}{'magNRMSE_A':>11}{'magNL':>8}{'intAIF_AvsB_NRMSE':>18}{'intAIF_AvsB_corr':>17}")
for r in rows:
    if r["roi"] in ("organ", "aorta"):
        print(f"{r['cfg']:>4}{r['slice']:>4}{r['roi']:>8}{r['complexNRMSE_B']:>12.2e}{r['magNRMSE_A']:>11.3f}{r['magNL_S1_vs_S2']:>8.3f}{r['intAIFmap_A_vs_B_NRMSE']:>18.3f}{r['intAIFmap_A_vs_B_corr']:>17.3f}")
print("\nwrote", OUT)
