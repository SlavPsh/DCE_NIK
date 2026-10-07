#!/bin/bash
#SBATCH -J gv2all
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 12:00:00
#SBATCH --array=0-1
# k100 standard (user decision 2026-10-07): grasp (v2, n12, lam 0.25) at ALL spokes for the slices that lack it: task 0 p3 18 / 19 (21 exists),
# task 1 p14 21 / 24 / 27 (sigma-independent, cs). grasp-pro is dropped from the comparisons.
set -uo pipefail
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=16
P="micromamba run -n torch29 python -u"; cd /net/beegfs/users/P101440/grasp_v2
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none}"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then DCE_DS=p3 SET=sweep NLINE=12 LAM_FRAC=0.25 SLICES=18,19 $P grasp_v2_real.py | grep -E "SAVED|cached|DONE|Error|Traceback"
else DCE_DS=p14 SET=sweep NLINE=12 LAM_FRAC=0.25 SLICES=21,24,27 $P grasp_v2_real.py | grep -E "SAVED|cached|DONE|Error|Traceback"; fi
echo "GV2ALL exit $?"; ls -la results_grasp_v2/gv2_slice1*_n12.npy results_grasp_v2_p14/gv2_slice*_n12.npy 2>/dev/null | awk '{print $5, $9}'
