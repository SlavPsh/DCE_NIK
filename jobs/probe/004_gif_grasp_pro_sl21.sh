#!/bin/bash
# illustration gif: grasp pro, all spokes (cs_slice21_f100, 122 frames, 14 spokes/frame), slice 21, 2x upscale, time stamp
cd /net/beegfs/users/P101440/DCE_NIK || exit 1
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
OUT=results/realdata_nik_vs_cs_figures/figures/dce_invivo_slice21_grasp_pro_allspokes.gif
micromamba run -n torch29 python -u - "$OUT" <<'PY'
import sys, numpy as np
from PIL import Image, ImageDraw
out = sys.argv[1]
v = np.abs(np.load("/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs/cs_slice21_f100.npy")).astype(np.float32)   # [192,192,122]
nt = v.shape[-1]; TA = 375.0; t = (np.arange(nt) + 0.5) * TA / nt
vmax = float(np.percentile(v[v > 0], 99.7)); frames = []
for i in range(nt):
    im = np.clip(v[:, :, i] / vmax, 0, 1) * 255
    pil = Image.fromarray(im.astype(np.uint8)).resize((v.shape[1] * 2, v.shape[0] * 2), Image.LANCZOS)
    ImageDraw.Draw(pil).text((8, 6), f"t = {t[i]:5.1f} s", fill=255)
    frames.append(pil)
frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=False)
import os; print("saved", out, v.shape, f"{os.path.getsize(out)/1e6:.1f} MB")
PY
git add "$OUT" && echo "staged for the agent commit"
