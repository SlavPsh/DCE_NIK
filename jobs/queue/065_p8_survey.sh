#!/bin/bash
#SBATCH -J p8survey
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 96G
#SBATCH -t 2:00:00
# new dataset meas_p8_dce (same protocol as p3: 1708 views, TA 375 s, TR 4.66, flip 18, nx 384): slice survey (all-spoke sos nufft per slice) to choose the kidney slices
# before the per-slice precompute; plus a p3 regression check of the dataset-namespaced scripts (default DCE_DS=p3 must resolve to the existing files).
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=16
P="micromamba run -n torch29 python -u"
cd /net/beegfs/users/P101440/DCE_NIK
$P - <<'PY'
import os, sys; sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); import dsp, numpy as np
print("p3 paths:", dsp.REF, dsp.STEP2(21), dsp.AIF(21), dsp.NUF(21), dsp.ROIS(21), dsp.SUPPORT(21, 6), dsp.BASIS(21, 8), dsp.GV_K80(21), dsp.GP_K80(21))
for p in (dsp.REF + "/shared.npz", dsp.STEP2(21), dsp.AIF(21), dsp.NUF(21) + "/meta.json", dsp.ROIS(21), dsp.SUPPORT(21, 6), dsp.BASIS(21, 8), dsp.GV_K80(21), dsp.GP_K80(21)): print(("ok  " if os.path.exists(p) else "MISSING "), p)
import consolidated as C; ctx = C.slice_ctx(21); print("slice_ctx(21) ok, rois", {k: int(v.sum()) for k, v in ctx["rois"].items()})
os.environ["DCE_DS"] = "p8"; import importlib; importlib.reload(dsp); print("p8 paths:", dsp.RAW, dsp.REF, dsp.STEP2(20), dsp.AIF(20), dsp.NUF(20), dsp.GV, dsp.GP)
PY
echo "REGRESSION exit $?"
cd /net/beegfs/users/P101440/grasp_pro_py
$P survey_slices.py --file /net/beegfs/users/P101440/dce_data/orig/meas_p8_dce.dat --out /net/beegfs/users/P101440/DCE_NIK/results/realdata_nik_vs_cs_figures/figures/survey_p8.png --max-views 600
echo "SURVEY exit $?"
