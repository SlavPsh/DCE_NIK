#!/bin/bash
#SBATCH -J p14bweval
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 1:00:00
# eval of the 095 bandwidth test (its chained eval failed: gres line kept while -p defq was given); radial diag + comparison figure on p14 slice 24
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p14
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; OUTR=$RES/p14/bw_test; Z=24
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14
BASE=$D/$RES/p14/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy
ITEMS="tofts8 base (ks2.5 w62):$BASE,ks3.5:$D/$OUTR/ks3.5/nik_slice_${Z}_cplx.npy,ks5:$D/$OUTR/ks5/nik_slice_${Z}_cplx.npy,w90:$D/$OUTR/w90/nik_slice_${Z}_cplx.npy,ks3.5 w90:$D/$OUTR/ks3.5_w90/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"
$P radial_blur_diag.py --slice $Z --tag _bw --items "$ITEMS"; echo "RADIAL exit $?"
$P compare_runs_fig.py --slice $Z --out $RES/figures/p14_bw_sl$Z.png --title "p14 slice $Z, tofts8 in-coil + prior: fourier-feature sigma / wire w0 test (images at 90 s, liver zoom, roi curves)" --items "$ITEMS"; echo "FIG exit $?"
