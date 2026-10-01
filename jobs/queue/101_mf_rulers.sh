#!/bin/bash
#SBATCH -J mfrulers2
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 4:00:00
# rerun of 100 with the angle-corrected navigator (100: 1.7 to 2.1 Hz = spoke angle, not breathing). model-free image-quality rulers (user idea 2026-10-01): late-window nufft (t > 200 s, above nyquist) as the primary sharpness / structure ruler
# at the 300 s frame, pre-contrast nufft as a second ruler, both also with respiratory soft gating from the k-centre navigator (removes the
# breathing blur every arm shares). rebuilds results_nufft*_slice<Z> (adds nufft_late, *_gated, resp_nav), draws the ruler check figure, then
# re-scores the existing production recons (no retraining): p3 063 arms on 18 / 19 / 21, p14 sigma-5 arms on 21 / 24 / 27. geometry check first:
# rulers_sl<Z>.png must show the gated rulers sharper than the ungated ones with the same anatomy, else the gating is wrong and the columns are not quoted.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
DCE_DS=p3 $P build_rulers.py 18,19,21; echo "RULERS p3 exit $?"
DCE_DS=p14 $P build_rulers.py 21,24,27; echo "RULERS p14 exit $?"
for Z in 18 19 21; do DCE_DS=p3 $P rulers_fig.py --slice $Z; done; for Z in 21 24 27; do DCE_DS=p14 $P rulers_fig.py --slice $Z; done; echo "RULERFIG exit $?"
GV3=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP3=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
for Z in 18 19 21; do
  IQ_VARIANTS="" IQ_TAG=_prod_mfruler IQ_EXTRA="tofts8 in-coil+prior:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$RES/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16+prior:$D/$RES/invivo_prod/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy" DCE_DS=p3 $P tofts_iq_track.py --slice $Z
done; echo "IQ p3 exit $?"
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14; OUTR=$RES/p14
for Z in 21 24 27; do
  IQ_VARIANTS="" IQ_TAG=_p14_ks5all_mfruler IQ_EXTRA="tofts8 in-coil+prior:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$OUTR/invivo_prod_ks5_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$OUTR/invivo_prod_ks5/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16+prior:$D/$OUTR/invivo_prod_ks5/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod_ks5/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy" DCE_DS=p14 $P tofts_iq_track.py --slice $Z
done; echo "IQ p14 exit $?"
