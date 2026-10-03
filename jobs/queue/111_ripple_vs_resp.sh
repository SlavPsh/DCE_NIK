#!/bin/bash
#SBATCH -J ripple
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 1:00:00
# is the sub16 curve ripple respiration (FINDINGS 8d): correlation of the high-passed roi curves with the k-centre navigator, and nrmse vs model-free
# after smoothing the recon curve with the reference's 31-spoke window. p3 slice 21 (sub16 own protocol, its 110 variants, tofts8, nik-free, grasp)
# and p14 slice 24 (sigma-5 arms). no retraining.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak
Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
DCE_DS=p3 $P ripple_vs_resp.py --slice $Z --tag _prod --items "sub16 wd3e-3+prior:$D/$RES/sub16_proto/wd3e-3_prior/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 nowarm:$D/$RES/sub16_osc/nowarm/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 w0_10:$D/$RES/sub16_osc/w0_10/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "P3 exit $?"
Z=24; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; OUTR=$RES/p14
DCE_DS=p14 $P ripple_vs_resp.py --slice $Z --tag _ks5 --items "sub16 wd3e-3+prior:$D/$OUTR/invivo_prod_ks5_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod_ks5/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "P14 exit $?"
