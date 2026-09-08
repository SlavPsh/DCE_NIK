#!/bin/bash
#SBATCH -J gv2pipe
#SBATCH -p luna-cpu-short
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 7:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/gv2pipe_%j.log
# regenerate every reference-method product with classic grasp v2 instead of grasp-pro.
# identical inputs, only the recon algorithm differs. defaults of every script are untouched.
cd /scratch/rnga/vvpshenov/DCE_NIK
export CSD=/scratch/rnga/vvpshenov/grasp_v2/results_grasp_v2
export CSPRE=gv2
export TAG=_gv2
P=${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python}

echo "=== inputs present ==="; ls -1 $CSD | sed 's/^/  /'
# gate: all 19 matched recons must exist before regenerating anything
MISSING=0
for f in gv2_slice13_f100 gv2_slice13_f70 gv2_slice13_f50 gv2_slice13_f35 gv2_slice13_f25 \
         gv2_slice18_f100 gv2_slice18_f25 gv2_slice19_f100 gv2_slice19_f25 \
         gv2_slice20_f100 gv2_slice20_f25 gv2_slice21_f100 gv2_slice21_f25 gv2_slice21_f80match \
         gv2_slice13_p05 gv2_slice18_p05 gv2_slice19_p05 gv2_slice20_p05 gv2_slice21_p05; do
  [ -f "$CSD/$f.npy" ] || { echo "  MISSING $f.npy"; MISSING=1; }
done
[ $MISSING -eq 1 ] && { echo "ABORT: incomplete matched set, refusing to build a half-swapped notebook"; exit 1; }
echo "  all 19 matched recons present"
echo; echo "=== QC vs grasp-pro ==="
$P -u /scratch/rnga/vvpshenov/grasp_v2/qc_gv2_vs_pro.py || echo "QC FAILED"

echo; echo "=== f80match fairness + period-5 ripple check ==="
# v2 has no DCF/DCF_U compensation, so the variable 10-12 spokes/frame could leak into intensity.
# pro's ripple check does NOT carry over and has to be redone here.
$P -u /scratch/rnga/vvpshenov/grasp_v2/check_f80.py || echo "F80 CHECK FAILED"
run () { echo; echo "=== $* ==="; $P -u "$@" || echo "FAILED: $*"; }

run haarpsi_spoke.py            # -> haarpsi_spoke_gv2.json
run task_S.py                   # -> task_S_gv2.json
run task2_analysis.py           # -> task2_gv2.json
run task_realdata_figures.py    # -> fig1..fig7 _gv2
run step1_images.py
run sweep_compare.py step1
run step2_fig.py
run step3_fig.py
run step7_fig.py
run calib_curves.py
run report_outcoil.py
run build_gv2_nb.py
echo; echo DONE_GV2_PIPELINE
