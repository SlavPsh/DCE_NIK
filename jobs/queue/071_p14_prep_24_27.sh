#!/bin/bash
#SBATCH -J p14prep2
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 120G
#SBATCH -t 4:00:00
#SBATCH --array=0-1
# rerun of the two 070 tasks that could not see the raw file (node-dependent path?): slices 24, 27; prints hostname + path check first; chained tail = roi proposal on all three
set -uo pipefail
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=16 DCE_DS=p14
P="micromamba run -n torch29 python -u"; D=/net/beegfs/users/P101440/DCE_NIK; SELF=$D/jobs/queue/071_p14_prep_24_27.sh; RAW=/net/beegfs/users/P101440/dce_data/orig/meas_topqmri_p14.dat
if [ "${STAGE:-slice}" = tail ]; then cd $D; $P roi_propose_liver.py --slices 21,24,27; echo "ROI exit $?"; exit; fi
SL=(24 27); Z=${SL[$SLURM_ARRAY_TASK_ID]}; echo "host $(hostname) task $SLURM_ARRAY_TASK_ID slice $Z"; ls -la $RAW || { echo "RAW MISSING on $(hostname)"; ls -la /net/beegfs/users/P101440/dce_data/orig/ | head -3; mount | grep -i "beegfs\|dce" | head -3; exit 3; }
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=tail --array=0 -J p14prep2_tail -p defq -c 8 --mem 48G -t 1:00:00 \
    --output=$D/jobs/log/071_p14_prep_tail_%j.out --error=$D/jobs/log/071_p14_prep_tail_%j.out $SELF && echo "tail chained afterany $SLURM_ARRAY_JOB_ID"
fi
cd /net/beegfs/users/P101440/grasp_pro_py
[ -f results_ref_p14/slice_$Z.npz ] || $P precompute_ref.py --file $RAW --out results_ref_p14 --slices $Z; echo "PRECOMPUTE $Z exit $?"
cd $D; $P build_rulers.py $Z; echo "RULERS $Z exit $?"; $P step2_kidney.py $Z; echo "STEP2 $Z exit $?"; $P aif_gate.py $Z; echo "AIF $Z exit $?"
