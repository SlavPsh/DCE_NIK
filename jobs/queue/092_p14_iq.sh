#!/bin/bash
#SBATCH -J p14iq
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 3:00:00
# p14 iq tracks + roi check only (the 091 eval chain crashed in these two scripts on p3 slice-21 files; tables and comparison figures are fine)
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p14
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/p14
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
$P roi_check_invivo.py --slices 21,24,27 --t-show 90; echo "ROICHECK exit $?"
for Z in 21 24 27; do IQ_VARIANTS="" IQ_TAG=_p14_prod_sl$Z IQ_EXTRA="tofts8 in-coil+prior:$D/$OUTR/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$OUTR/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$OUTR/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16+prior:$D/$OUTR/invivo_prod/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy" $P tofts_iq_track.py --slice $Z; done; echo "IQ exit $?"
