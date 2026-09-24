#!/bin/bash
#SBATCH -J p14survey
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 120G
#SBATCH -t 2:30:00
# new dataset meas_topqmri_p14 (base 256, TR 5.95, flip 15, 2172 views, TA 390, 36 partitions): slice survey to choose the kidney slices; plus a p3
# import regression of the scripts after the TA / spoke-mask generalization (default DCE_DS=p3 must still resolve to the existing files)
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=16
P="micromamba run -n torch29 python -u"
cd /net/beegfs/users/P101440/DCE_NIK
$P - <<'PY'
import os, sys; sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); import dsp
print("p3 geom", dsp.GEOM, "masks", dsp.KEEP, dsp.VAL, dsp.TEST, [os.path.exists(p) for p in (dsp.KEEP, dsp.VAL, dsp.TEST)])
import consolidated as C; ctx = C.slice_ctx(21); print("slice_ctx(21) ok, TA", C.TA)
import story_figs, tofts_span_diag, tofts_iq_track, compare_runs_fig, mf_peak_check, roi_propose_invivo, roi_check_invivo; print("imports ok")
PY
echo "REGRESSION exit $?"
cd /net/beegfs/users/P101440/grasp_pro_py
$P survey_slices.py --file /net/beegfs/users/P101440/dce_data/orig/meas_topqmri_p14.dat --out /net/beegfs/users/P101440/DCE_NIK/results/realdata_nik_vs_cs_figures/figures/survey_p14.png --max-views 600
echo "SURVEY exit $?"
