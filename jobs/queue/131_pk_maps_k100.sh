#!/bin/bash
#SBATCH -J pkmaps
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 12:00:00
#SBATCH --array=0-1
# pk maps on the k100 standard recons (plan item 4): task 0 p3 kidney (18 / 19 / 21), task 1 p14 liver (21 / 24 / 27); every nik arm + grasp all spokes;
# extended-kety per voxel (DCE-NET), literature T10, model-free aorta aif. p14 caveat: inflow-bright aorta, maps relative between methods.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=16
P="micromamba run -n torch29 python -u"; RES=$D/results/tofts_vs_patlak
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none}"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
  R=$RES; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
  DCE_DS=p3 $P pk_maps_k100.py --slices 18,19,21 --tag p3_k100std --items "NIK-tofts8:$R/invivo_k100/tofts8_sl{Z}_s0/nik_slice_{Z}_cplx.npy,NIK-tofts8 out-coil:$R/invivo_k100_oc/tofts8_sl{Z}_s0/nik_slice_{Z}_cplx.npy,NIK-patlak:$R/invivo_k100/patlak_sl{Z}_s0/nik_slice_{Z}_cplx.npy,NIK-sub16:$R/invivo_k100/sub16_sl{Z}_s0/nik_slice_{Z}_cplx.npy,NIK-free:$R/invivo_k100/free_sl{Z}_s0/nik_slice_{Z}_cplx.npy,GRASP all spokes:$GV/gv2_slice{Z}_n12.npy"
else
  R=$RES/p14; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14
  DCE_DS=p14 $P pk_maps_k100.py --slices 21,24,27 --tag p14_k100std --items "NIK-tofts8:$R/invivo_k100/tofts8_sl{Z}_s0/nik_slice_{Z}_cplx.npy,NIK-tofts8 out-coil:$R/invivo_k100_oc/tofts8_sl{Z}_s0/nik_slice_{Z}_cplx.npy,NIK-patlak:$R/invivo_k100/patlak_sl{Z}_s0/nik_slice_{Z}_cplx.npy,NIK-sub16:$R/invivo_k100/sub16_sl{Z}_s0/nik_slice_{Z}_cplx.npy,NIK-free:$R/invivo_k100/free_sl{Z}_s0/nik_slice_{Z}_cplx.npy,GRASP all spokes:$GV/gv2_slice{Z}_n12.npy"
fi
echo "PKMAPS exit $?"
