#!/bin/bash
# amplitude deficit: gain or dynamics? per roi: pre-contrast intensity ratio method/model-free vs late enhancement ratio (both after the eval's one global ls scale)
cd /net/beegfs/users/P101440/DCE_NIK || exit 1
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
micromamba run -n torch29 python -u - <<'PY'
import sys, os, numpy as np; sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import tofts_figs_wandb as TF; TF.IV = TF.RES + "/invivo_k80"; TF.REFSET = "k80"
for Z in (21, 18):
    S = TF.Slice(Z); arms, refs = S.load(); tmf = S.tmf; pre = tmf < 45; late = tmf > 200
    M = [("model-free", S.mfv, tmf)] + [(f"{a} s0", arms[a][0][0], arms[a][0][1]) for a in arms if 0 in arms[a]] + [(n.split(" (")[0], v, t) for n, v, t in refs]
    print(f"\nslice {Z}: ratio to model-free of (pre-contrast mean | late enhancement, baseline subtracted); gain problem if the two columns agree per roi")
    print(f"{'method':22s}" + "".join(f"{r:>22s}" for r in ("aorta", "cortex", "medulla", "liver", "body")))
    rois = dict(S.rois); rois["body"] = S.body
    ref = {}
    for r, m in rois.items():
        c = np.array([im[m].mean() for im in S.mf]); ref[r] = (c[pre].mean(), (c - np.median(c[:8]))[late].mean())
    for nm, v, t in M:
        row = f"{nm:22s}"
        for r, m in rois.items():
            c = np.interp(tmf, t, np.array([v[..., i][m].mean() for i in range(v.shape[-1])])); p = c[pre].mean(); e = (c - np.median(c[:8]))[late].mean()
            row += f"{p/ref[r][0]:11.3f}{e/ref[r][1]:11.3f}"
        print(row)
PY
