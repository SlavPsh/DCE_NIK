#!/bin/bash
#SBATCH -J gppreview
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 0:15:00
# preview of the p14 grasp-pro (f80match, k80) and grasp (n12 k80) recons: frames at 30, 60, 120, 300 s for slices 21 / 24 / 27, model-free reference for comparison
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p14
cd /net/beegfs/users/P101440/DCE_NIK; micromamba run -n torch29 python -u - <<'PY'
import sys, numpy as np; sys.path.insert(0, "."); import dsp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TS = (30, 60, 120, 300)
for Z in (21, 24, 27):
    gp = np.abs(np.load(dsp.GP_K80(Z))).astype(np.float32); gv = np.abs(np.load(dsp.GV_K80(Z))).astype(np.float32); z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0); tmf = np.asarray(z["tmf"], float)
    rows = [("model-free 31-spoke", mf, tmf), ("GRASP-Pro K5 14 spf (k80)", gp, (np.arange(gp.shape[-1]) + 0.5) * dsp.TA / gp.shape[-1]), ("GRASP 12 spf (k80)", gv, (np.arange(gv.shape[-1]) + 0.5) * dsp.TA / gv.shape[-1])]
    fig, ax = plt.subplots(3, 4, figsize=(15, 11.5))
    for i, (nm, v, t) in enumerate(rows):
        vm = np.percentile(v, 99.7)
        for j, ts in enumerate(TS):
            im = v[..., int(np.argmin(np.abs(t - ts)))]; ax[i, j].imshow(im, cmap="gray", vmin=0, vmax=vm); ax[i, j].axis("off"); ax[i, j].set_title(f"{nm}, t = {ts} s" if j == 0 else f"t = {ts} s", fontsize=9)
    fig.suptitle(f"p14 slice {Z}: reconstructions on the same k80 views", fontsize=11); fig.tight_layout(); fig.savefig(f"{dsp.FIGD}/figures/p14_grasp_preview_sl{Z}.png", dpi=110, facecolor="white"); print("saved", Z)
PY
