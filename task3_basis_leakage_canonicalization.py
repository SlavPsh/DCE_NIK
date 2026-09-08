"""TASK 3: F2 temporal-subspace leakage + coefficient canonicalization (read-only).
Reuses Task 2 canonical extraction: Phi_F2 (temporal basis on true frame_time) and thetaC_F2/F0
(SENSE-projected complex coefficient maps). Time-weighted (trapezoidal) inner product. No model
re-query, no retrain. out: results/task3_basis_leakage/."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os, sys, csv, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); sys.path.insert(0, ".")
import consolidated as C
D = "/scratch/rnga/vvpshenov/DCE_NIK"; T2 = f"{D}/results/task2_scaling_coil_audit/arrays"; OUT = f"{D}/results/task3_basis_leakage"
for s in ["figures", "arrays", "logs", "script_snapshot"]: os.makedirs(f"{OUT}/{s}", exist_ok=True)
SLICES = [18, 19, 21]
sh = np.load("/scratch/rnga/vvpshenov/grasp_pro_py/results_ref/shared.npz"); TA = float(sh["TA"])
ft_sec = np.asarray(sh["frame_time"]).ravel().astype(float) * TA                      # true frame_time in seconds
# trapezoidal quadrature weights from actual sample times (Part 2)
w = np.zeros_like(ft_sec); w[1:-1] = 0.5 * (ft_sec[2:] - ft_sec[:-2]); w[0] = 0.5 * (ft_sec[1] - ft_sec[0]); w[-1] = 0.5 * (ft_sec[-1] - ft_sec[-2])
L = np.sqrt(w)                                                                         # <a,b>_W = (La)^H(Lb)
def Wip(a, b): return a.conj().T @ (w[:, None] * b) if b.ndim > 1 else a.conj() @ (w * b)
def masks(Z):
    ctx = C.slice_ctx(Z); body = ctx["BODY"]; rois = ctx["rois"]; organ = np.zeros_like(body)
    for r in ["aorta", "cortex", "medulla"]: organ |= (rois.get(r) if rois.get(r) is not None else False)
    organ = ndi.binary_dilation(organ, iterations=3) & body
    return {"body": body, "aorta": rois["aorta"], "cortex": rois["cortex"], "medulla": rois["medulla"], "organ": organ}
def nrmse(a, b, m): return float(np.linalg.norm(a[m] - b[m]) / (np.linalg.norm(b[m]) + 1e-12))
def Wenergy(series, mk): return float(np.sum(w[None, :] * np.abs(series[mk]) ** 2))     # sum_x sum_t w_t |Y|^2

overlap, pang, mapchg, energy, invar, f0cmp = [], [], [], [], [], []
BASIS = {}
for Z in SLICES:
    Phi = np.load(f"{T2}/Phi_F2_sl{Z}.npy")                                            # [nt,5] complex
    thC = np.load(f"{T2}/thetaC_F2_sl{Z}.npy")                                         # [X,Y,5] complex
    th0 = np.load(f"{T2}/thetaC_F0_sl{Z}.npy")                                         # [X,Y,3]
    P, Ffree = Phi[:, 0:3], Phi[:, 3:5]; A_P, A_F = thC[..., 0:3], thC[..., 3:5]
    BASIS[f"sl{Z}"] = dict(AIF=Phi[:, 0], intAIF=Phi[:, 1], baseline=Phi[:, 2], free0=Phi[:, 3], free1=Phi[:, 4])
    # Part 3: weighted orthonormal bases via QR of L-scaled columns
    QpL, _ = np.linalg.qr(L[:, None] * P); QfL, _ = np.linalg.qr(L[:, None] * Ffree)   # QL^H QL = I  <=> Q^H W Q = I
    # Part 4: canonical correlations = svd(Qp^H W Qf) = svd(QpL^H QfL)
    M = QpL.conj().T @ QfL; s = np.linalg.svd(M, compute_uv=False)
    E_overlap = float(np.linalg.norm(M, "fro") ** 2 / Ffree.shape[1])
    angles = np.degrees(np.arccos(np.clip(s, 0, 1)))
    # projection of free span onto each fixed direction (weighted, normalized)
    qP = QpL / np.linalg.norm(QpL, axis=0, keepdims=True)  # already orthonormal in L-space
    proj_fixed = [float(np.linalg.norm(QpL[:, k:k+1].conj().T @ QfL)) for k in range(3)]  # energy of free span on fixed dir k
    fullBL = L[:, None] * Phi; sv = np.linalg.svd(fullBL, compute_uv=False); cond = float(sv[0] / sv[-1])
    overlap.append(dict(slice=Z, max_canon_corr=float(s.max()), E_overlap=E_overlap, cond_full_basis=cond))
    pang.append(dict(slice=Z, canon_corr_1=float(s[0]), canon_corr_2=float(s[1]), angle1_deg=float(angles[0]), angle2_deg=float(angles[1]),
                     proj_AIF=proj_fixed[0], proj_intAIF=proj_fixed[1], proj_baseline=proj_fixed[2]))
    # Part 5: weighted LS decomposition of F into fixed + orthogonal
    Cmat = np.linalg.solve(Wip(P, P), Wip(P, Ffree))                                   # [3,2]
    Fperp = Ffree - P @ Cmat
    resid_orth = float(np.linalg.norm(Wip(P, Fperp)))                                  # ~0 check
    for j in range(2):
        frac_in = float(np.sqrt((P @ Cmat[:, j]).conj() @ (w * (P @ Cmat[:, j])) / ((Ffree[:, j].conj() @ (w * Ffree[:, j])) + 1e-20)).real)
        pang[-1][f"free{j}_frac_in_fixed"] = frac_in
    pang[-1]["PtWFperp_norm"] = resid_orth
    # Part 6: canonicalize  A_P_canonical = A_P + C @ A_F
    A_P_can = A_P + np.einsum("pf,xyf->xyp", Cmat, A_F)
    # Part 7: reconstruction invariance
    Y_orig = np.einsum("tr,xyr->xyt", Phi, thC)
    Y_can = np.einsum("tp,xyp->xyt", P, A_P_can) + np.einsum("tf,xyf->xyt", Fperp, A_F)
    body = masks(Z)["body"]
    invar.append(dict(slice=Z, imgseries_NRMSE=nrmse(Y_can, Y_orig, body), max_abs_diff=float(np.abs(Y_can - Y_orig)[body].max())))
    # Part 8: fixed-map non-identifiability (orig vs canonical), per ROI
    MK = masks(Z)
    for r, nm in [(0, "AIF"), (1, "intAIF"), (2, "baseline")]:
        for rn, mk in MK.items():
            o, c = np.abs(A_P[..., r]), np.abs(A_P_can[..., r])
            mapchg.append(dict(slice=Z, comp=nm, roi=rn, NRMSE=nrmse(c, o, mk),
                               corr=float(np.corrcoef(o[mk], c[mk])[0, 1]), scale_change=float(c[mk].mean() / (o[mk].mean() + 1e-12))))
    # Part 9: component energy (span-level, unambiguous: P ⊥ Fperp)
    Efix = np.einsum("tp,xyp->xyt", P, A_P_can); Eorth = np.einsum("tf,xyf->xyt", Fperp, A_F)
    Efree_infix = np.einsum("tp,xyp->xyt", P, np.einsum("pf,xyf->xyp", Cmat, A_F))
    for rn, mk in MK.items():
        tot = Wenergy(Y_orig, mk)
        energy.append(dict(slice=Z, roi=rn, E_total=tot, E_fixed_span=Wenergy(Efix, mk), E_orth_residual=Wenergy(Eorth, mk),
                           E_free_in_fixed=Wenergy(Efree_infix, mk),
                           frac_orth_residual=float(Wenergy(Eorth, mk) / (tot + 1e-20))))
    # Part 11: F0 vs original-F2 vs canonical-F2 fixed maps
    for r, nm in [(0, "AIF"), (1, "intAIF")]:
        for rn, mk in MK.items():
            f0cmp.append(dict(slice=Z, comp=nm, roi=rn,
                F0_vs_F2orig_NRMSE=nrmse(np.abs(th0[..., r]), np.abs(A_P[..., r]), mk),
                F0_vs_F2canon_NRMSE=nrmse(np.abs(th0[..., r]), np.abs(A_P_can[..., r]), mk),
                F0_vs_F2orig_corr=float(np.corrcoef(np.abs(th0[..., r])[mk], np.abs(A_P[..., r])[mk])[0, 1]),
                F0_vs_F2canon_corr=float(np.corrcoef(np.abs(th0[..., r])[mk], np.abs(A_P_can[..., r])[mk])[0, 1])))
    # save canonical arrays + Fperp for sl21 figures
    np.save(f"{OUT}/arrays/A_P_orig_sl{Z}.npy", A_P); np.save(f"{OUT}/arrays/A_P_canonical_sl{Z}.npy", A_P_can)
    if Z == 21:
        np.save(f"{OUT}/arrays/Fperp_sl21.npy", Fperp); np.save(f"{OUT}/arrays/Cmat_sl21.npy", Cmat)
        SL21 = dict(Phi=Phi, P=P, Ffree=Ffree, Fperp=Fperp, Cmat=Cmat, A_P=A_P, A_P_can=A_P_can, A_F=A_F, Y_orig=Y_orig, MK=MK)

np.savez(f"{OUT}/basis_values.npz", frame_time_s=ft_sec, weights=w, **{f"{k}_{n}": v for k, dd in BASIS.items() for n, v in dd.items()})
for name, rowset in [("subspace_overlap", overlap), ("principal_angles", pang), ("canonical_map_changes", mapchg),
                     ("component_energy", energy), ("reconstruction_invariance", invar), ("f0_vs_f2_comparison", f0cmp)]:
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rowset[0].keys())); wr.writeheader(); [wr.writerow(r) for r in rowset]

# ---------- figures (sl21) ----------
S = SL21; tt = ft_sec
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for lab, col in [("AIF", S["P"][:, 0]), ("intAIF", S["P"][:, 1]), ("baseline", S["P"][:, 2])]: ax[0].plot(tt, col.real, label=lab)
ax[0].plot(tt, S["Ffree"][:, 0].real, "--", label="free0 Re"); ax[0].plot(tt, S["Ffree"][:, 1].real, "--", label="free1 Re")
ax[0].set_xlim(0, 260); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7); ax[0].set_title("fixed + free temporal bases (Re)")
for j in range(2):
    ax[1].plot(tt, S["Ffree"][:, j].real, label=f"free{j}"); ax[1].plot(tt, (S["P"] @ S["Cmat"])[:, j].real, ":", label=f"P·C (in-fixed) {j}"); ax[1].plot(tt, S["Fperp"][:, j].real, "--", label=f"Fperp{j}")
ax[1].set_xlim(0, 260); ax[1].grid(alpha=.3); ax[1].legend(fontsize=6.5); ax[1].set_title("free = fixed-span (P·C) + orthogonal (Fperp)")
# residual vs ROI curves
for nm in ["aorta", "cortex", "medulla"]:
    curve = np.array([S["Y_orig"][..., i][S["MK"][nm]].mean() for i in range(len(tt))]); ax[2].plot(tt, np.abs(curve) / np.abs(curve).max(), lw=1, label=nm)
for j in range(2): ax[2].plot(tt, S["Fperp"][:, j].real / (np.abs(S["Fperp"][:, j]).max() + 1e-9), "k--" if j == 0 else "k:", lw=1, label=f"Fperp{j} (norm)")
ax[2].set_xlim(0, 260); ax[2].grid(alpha=.3); ax[2].legend(fontsize=7); ax[2].set_title("orthogonal residual vs ROI curves (normalized)")
fig.suptitle("TASK 3 sl21: bases, free split (P·C + Fperp), residual-vs-ROI", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/bases_and_split.png", dpi=120); plt.close(fig)

# map changes: orig vs canonical AIF & intAIF + diff; and F0 vs F2orig vs F2canon
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for row, (r, nm) in enumerate([(0, "AIF"), (1, "intAIF")]):
    o, c = np.abs(S["A_P"][..., r]), np.abs(S["A_P_can"][..., r]); vmx = np.percentile(o, 99); d = c - o; vv = np.percentile(np.abs(d), 99)
    ax[row, 0].imshow(np.rot90(o), cmap="viridis", vmax=vmx); ax[row, 0].axis("off"); ax[row, 0].set_title(f"{nm} orig", fontsize=9)
    ax[row, 1].imshow(np.rot90(c), cmap="viridis", vmax=vmx); ax[row, 1].axis("off"); ax[row, 1].set_title(f"{nm} canonical", fontsize=9)
    ax[row, 2].imshow(np.rot90(d), cmap="bwr", vmax=vv, vmin=-vv); ax[row, 2].axis("off"); ax[row, 2].set_title(f"{nm} canon-orig", fontsize=9)
    th0 = np.load(f"{T2}/thetaC_F0_sl21.npy"); ax[row, 3].imshow(np.rot90(np.abs(th0[..., r])), cmap="viridis", vmax=vmx); ax[row, 3].axis("off"); ax[row, 3].set_title(f"{nm} F0", fontsize=9)
fig.suptitle("TASK 3 sl21: original vs canonicalized F2 fixed maps + F0 (fixed display scale/row)", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/map_changes.png", dpi=120); plt.close(fig)

print("=== Part4 subspace overlap ==="); [print("  sl%d: max_canon_corr %.3f E_overlap %.3f cond %.1f" % (r["slice"], r["max_canon_corr"], r["E_overlap"], r["cond_full_basis"])) for r in overlap]
print("=== Part5 free-in-fixed fraction + orth check ==="); [print("  sl%d: free0 %.2f free1 %.2f | P^H W Fperp %.2e" % (r["slice"], r["free0_frac_in_fixed"], r["free1_frac_in_fixed"], r["PtWFperp_norm"])) for r in pang]
print("=== Part7 reconstruction invariance ==="); [print("  sl%d: imgseries NRMSE %.2e maxabs %.2e" % (r["slice"], r["imgseries_NRMSE"], r["max_abs_diff"])) for r in invar]
print("=== Part8 fixed-map change (organ) ==="); [print("  sl%d %s: NRMSE %.3f corr %.3f scale %.2f" % (r["slice"], r["comp"], r["NRMSE"], r["corr"], r["scale_change"])) for r in mapchg if r["roi"] == "organ"]
print("=== Part9 orth-residual energy fraction (organ) ==="); [print("  sl%d: frac_orth %.3f" % (r["slice"], r["frac_orth_residual"])) for r in energy if r["roi"] == "organ"]
print("=== Part11 F0 vs F2 (organ, intAIF) ==="); [print("  sl%d: F0-vs-orig NRMSE %.3f  F0-vs-canon NRMSE %.3f" % (r["slice"], r["F0_vs_F2orig_NRMSE"], r["F0_vs_F2canon_NRMSE"])) for r in f0cmp if r["roi"] == "organ" and r["comp"] == "intAIF"]
print("wrote", OUT)
