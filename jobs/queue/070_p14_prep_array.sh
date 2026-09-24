#!/bin/bash
#SBATCH -J p14prep
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 120G
#SBATCH -t 8:00:00
#SBATCH --array=0-2
# p14 preparation as a per-slice array (replaces the sequential 067): task i -> slice (24, 27, 21)[i]: precompute (coil maps, grasp-pro all-spoke
# reference, radial bundle), nufft rulers, model-free 31-spoke series, aif gate. task 0 chains masks + liver / spleen roi proposal afterany.
set -uo pipefail
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=16 DCE_DS=p14
P="micromamba run -n torch29 python -u"; D=/net/beegfs/users/P101440/DCE_NIK; SELF=$D/jobs/queue/070_p14_prep_array.sh
if [ "${STAGE:-slice}" = tail ]; then
  cd $D; $P spoke_masks_build.py; echo "MASKS exit $?"; $P roi_propose_liver.py --slices 21,24,27; echo "ROI exit $?"; exit
fi
SL=(24 27 21); Z=${SL[$SLURM_ARRAY_TASK_ID]}
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=tail --array=0 -J p14prep_tail -p defq -c 8 --mem 48G -t 1:00:00 \
    --output=$D/jobs/log/070_p14_prep_tail_%j.out --error=$D/jobs/log/070_p14_prep_tail_%j.out $SELF && echo "tail chained afterany $SLURM_ARRAY_JOB_ID"
fi
cd /net/beegfs/users/P101440/grasp_pro_py
[ -f results_ref_p14/slice_$Z.npz ] || $P precompute_ref.py --file /net/beegfs/users/P101440/dce_data/orig/meas_topqmri_p14.dat --out results_ref_p14 --slices $Z; echo "PRECOMPUTE $Z exit $?"
cd $D
$P build_rulers.py $Z; echo "RULERS $Z exit $?"
$P step2_kidney.py $Z; echo "STEP2 $Z exit $?"
$P aif_gate.py $Z; echo "AIF $Z exit $?"
