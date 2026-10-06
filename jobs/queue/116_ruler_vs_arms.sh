#!/bin/bash
#SBATCH -J rulerarms
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 1:00:00
# slide figures: late ruler next to the 300 s frame of every arm with HaarPSI, p3 sl21 and p14 sl24, ungated and gated ruler
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; FIG=results/realdata_nik_vs_cs_figures/figures
Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
IT="tofts8 in-coil:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil:$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak:$D/$RES/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3):$D/$RES/invivo_prod_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"
DCE_DS=p3 $P ruler_vs_arms_fig.py --slice $Z --items "$IT" --out $FIG/ruler_vs_arms_sl$Z.png; DCE_DS=p3 $P ruler_vs_arms_fig.py --slice $Z --gated 1 --items "$IT" --out $FIG/ruler_vs_arms_gated_sl$Z.png
Z=24; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14; OUTR=$RES/p14
IT="tofts8 in-coil:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil:$D/$OUTR/invivo_prod_ks5_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak:$D/$OUTR/invivo_prod_ks5/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3):$D/$OUTR/invivo_prod_ks5_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod_ks5/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"
DCE_DS=p14 $P ruler_vs_arms_fig.py --slice $Z --items "$IT" --out $FIG/ruler_vs_arms_p14_sl$Z.png; DCE_DS=p14 $P ruler_vs_arms_fig.py --slice $Z --gated 1 --items "$IT" --out $FIG/ruler_vs_arms_gated_p14_sl$Z.png
echo "RULERARMS exit $?"
