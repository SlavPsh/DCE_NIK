#!/bin/bash
# grasp pro K5 phantom recon came out uncorrelated with truth (corr 0.12): is the library (grog) pathway itself off on the phantom? compare with grasp_lib_recon.npz (old lib G=1) and grasp_recon.npz (nufft path, K12), all 8 orientations
cd /net/beegfs/users/P101440/DCE_NIK || exit 1
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH XPH_SIM=nomotion
timeout 55 micromamba run -n torch29 python -u - <<'PY'
import numpy as np, xph_pipeline as P, xph_common as X, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); tm = Tr.mean(2); A = f"{P.OUT}/arrays"
print("truth", Tr.shape, "kdata", d["kdata"].shape)
def corr(u, v): u = u[body].ravel() - u[body].mean(); v = v[body].ravel() - v[body].mean(); return float((u * v).sum() / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
O = {"id": lambda v: v, "rot180": lambda v: v[::-1, ::-1], "fliplr": lambda v: v[:, ::-1], "flipud": lambda v: v[::-1], "T": lambda v: v.T, "rot90": lambda v: np.rot90(v), "rot270": lambda v: np.rot90(v, 3), "T_rot180": lambda v: v.T[::-1, ::-1]}
fig, ax = plt.subplots(1, 4, figsize=(16, 4)); ax[0].imshow(tm, cmap="gray"); ax[0].set_title("truth mean")
for j, (nm, f, key) in enumerate((("lib G1 K5", "grasp_lib_recon.npz", "rec"), ("nufft K12 (report)", "grasp_recon.npz", "rec"), ("new lib K5 G5", "grasp_pro_K5_G5.npz", "rec"))):
    z = np.load(f"{A}/{f}"); r = np.abs(z[key]).astype(np.float32); m = r.mean(2)
    if m.shape != tm.shape: print(nm, "shape", m.shape, "vs truth", tm.shape); m = m[:tm.shape[0], :tm.shape[1]]
    cs = {k: corr(o(m), tm) for k, o in O.items() if o(m).shape == tm.shape}; print(nm, r.shape, "K" , z["K"] if "K" in z else "?", "corr per orientation:", {k: round(v, 3) for k, v in cs.items()})
    ax[j + 1].imshow(m, cmap="gray"); ax[j + 1].set_title(nm)
for a in ax: a.axis("off")
fig.savefig("results/xcat_physical_nomotion_nik_vs_grasp/figures/diag_pro_k5.png", dpi=80, bbox_inches="tight"); print("saved diag png")
PY
