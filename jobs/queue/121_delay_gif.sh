#!/bin/bash
#SBATCH -J dlygif
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 1:00:00
# gifs of the smooth-delay prior runs (118) next to the production tofts8 in / out coil and GRASP, p3 slice 21: full scan (4 s steps) and the
# arrival window (40 to 110 s, 1 s steps); staged for the agent commit
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; FIG=results/realdata_nik_vs_cs_figures/figures; Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
IT="tofts8 prod in-coil:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 prod out-coil:$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,delay-tv 0.1:$D/$RES/delay_prior/dly0.1/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,delay-tv 1:$D/$RES/delay_prior/dly1/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,delay-tv 10:$D/$RES/delay_prior/dly10/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP 12spf k80:$GV/gv2_slice${Z}_n12_k80.npy"
DCE_DS=p3 $P recon_gif.py --slice $Z --out $FIG/recon_gif_delay_prior_sl$Z.gif --items "$IT" --note "p3 slice $Z, smooth-delay prior at weight 0.1 / 1 / 10 vs production tofts8 and GRASP, same k80 input, window mean +-6 s"
DCE_DS=p3 $P recon_gif.py --slice $Z --t0 40 --t1 110 --dt 1 --w 2 --fps 4 --out $FIG/recon_gif_delay_prior_arrival_sl$Z.gif --items "$IT" --note "p3 slice $Z, contrast arrival 40 to 110 s, 1 s steps: smooth-delay prior 0.1 / 1 / 10 vs production tofts8 and GRASP"
echo "GIF exit $?"; ls -la $FIG/recon_gif_delay_prior*.gif | awk '{printf "%.1f MB %s\n", $5/1e6, $9}'
git add $FIG/recon_gif_delay_prior*.gif && echo staged
