#!/bin/bash
#SBATCH -J panels2
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 2:00:00
# regenerate the 6 k100-standard comparison panels with the HaarPSI label = 300 s frame vs the late model-free nufft (the main ruler, user 2026-10-08)
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak
GV3=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GV14=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14
ITEMS() { local ds=$1 Z=$2 R GV; if [ $ds = p3 ]; then R=$D/$RES; GV=$GV3; else R=$D/$RES/p14; GV=$GV14; fi
  echo "tofts8 in-coil+prior:$R/invivo_k100/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$R/invivo_k100_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$R/invivo_k100/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3)+prior:$R/invivo_k100/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$R/invivo_k100/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP all spokes:$GV/gv2_slice${Z}_n12.npy"; }
for Z in 18 19 21; do DCE_DS=p3 $P compare_runs_fig.py --slice $Z --out $RES/figures/k100stdp3_sl$Z.png --title "p3 slice $Z, k100 standard (all spokes, every arm under its own protocol, grasp n12 at all spokes): images at 90 s, tissue zoom, roi curves; HaarPSI = 300 s frame vs the late model-free NUFFT" --items "$(ITEMS p3 $Z)"; done
for Z in 21 24 27; do DCE_DS=p14 $P compare_runs_fig.py --slice $Z --out $RES/figures/k100stdp14_sl$Z.png --title "p14 slice $Z, k100 standard (all spokes, every arm under its own protocol, grasp n12 at all spokes): images at 90 s, tissue zoom, roi curves; HaarPSI = 300 s frame vs the late model-free NUFFT" --items "$(ITEMS p14 $Z)"; done
echo "PANELS exit $?"
