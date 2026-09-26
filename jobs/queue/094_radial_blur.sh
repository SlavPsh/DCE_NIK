#!/bin/bash
#SBATCH -J radblur2
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 1:00:00
# sharpness vs distance from the image centre (user saw blur away from the centre in the p14 nik recons): p14 production arms on slices 21 / 24 / 27
# and the p3 production arms on 18 / 19 / 21 as the comparison, all vs cs100 and grasp
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; R=$D/results/tofts_vs_patlak
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
for Z in 21 24 27; do
  DCE_DS=p14 $P radial_blur_diag.py --slice $Z --tag _prod --items "tofts8 in-coil+prior:$R/p14/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$R/p14/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$R/p14/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16+prior:$R/p14/invivo_prod/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$R/p14/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14/cs_slice${Z}_f80match.npy"
done; echo "P14 exit $?"
for Z in 18 19 21; do
  DCE_DS=p3 $P radial_blur_diag.py --slice $Z --tag _prod --items "tofts8 in-coil+prior:$R/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$R/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$R/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:/net/beegfs/users/P101440/grasp_v2/results_grasp_v2/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs/cs_slice${Z}_f80match.npy"
done; echo "P3 exit $?"
