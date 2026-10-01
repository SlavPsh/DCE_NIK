#!/bin/bash
#SBATCH -J gifs
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 3:00:00
# dynamic gifs of the production recons, 6 slices: p3 (063 arms, sigma 2.5 = the 384-grid protocol) and p14 (099 arms, sigma 5 = the 512-grid
# protocol), tofts8 in / out coil + prior, patlak + prior, sub16 + prior, nik-free, grasp 12 spf k80, grasp-pro K5 14 spf k80. same k80 input
# (v%10<8) for every arm: train_grasp_nik --spoke-keep-file, grasp_v2_real keep_mask (KEEP80), cs_nikmatch keep_mask; b1 full-data for all.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; FIG=results/realdata_nik_vs_cs_figures/figures
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
GV3=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP3=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
for Z in 18 19 21; do
  DCE_DS=p3 $P recon_gif.py --slice $Z --out $FIG/recon_gif_sl$Z.gif --items "tofts8 in-coil+prior:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$RES/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16+prior:$D/$RES/invivo_prod/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP 12spf k80:$GV3/gv2_slice${Z}_n12_k80.npy,GRASP-Pro K5 k80:$GP3/cs_slice${Z}_f80match.npy"
done; echo "GIF p3 exit $?"
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14; OUTR=$RES/p14
for Z in 21 24 27; do
  DCE_DS=p14 $P recon_gif.py --slice $Z --out $FIG/recon_gif_p14_sl$Z.gif --items "tofts8 in-coil+prior:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$OUTR/invivo_prod_ks5_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$OUTR/invivo_prod_ks5/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16+prior:$D/$OUTR/invivo_prod_ks5/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod_ks5/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP 12spf k80:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro K5 k80:$GP/cs_slice${Z}_f80match.npy"
done; echo "GIF p14 exit $?"
ls -la $FIG/recon_gif*.gif | awk '{print $5, $9}'
