#!/bin/bash
#SBATCH -J arrgifs
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 2:00:00
# presentation: contrast-arrival gifs (1 s steps, +-2 s window, 4 fps) combined and per arm for p3 slice 21 and p14 slice 24, and the clean
# rulers slide for both; files staged for the agent commit
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; FIG=results/realdata_nik_vs_cs_figures/figures
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
IT="tofts8 in-coil+prior:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$RES/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3)+prior:$D/$RES/invivo_prod_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP 12spf k80:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro K5 k80:$GP/cs_slice${Z}_f80match.npy"
DCE_DS=p3 $P recon_gif.py --slice $Z --t0 40 --t1 110 --dt 1 --w 2 --fps 4 --out $FIG/recon_gif_arrival_sl$Z.gif --items "$IT" --note "p3 slice $Z, contrast arrival 40 to 110 s, 1 s steps (window +-2 s), same k80 input for every arm, one global scale per arm"
DCE_DS=p3 $P recon_gif.py --slice $Z --t0 40 --t1 110 --dt 1 --w 2 --fps 4 --single 1 --single-tag arrival --out $FIG/recon_gifs_single --items "$IT"
DCE_DS=p3 $P rulers_slide.py --slice $Z
Z=24; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14; OUTR=$RES/p14
IT="tofts8 in-coil+prior:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$OUTR/invivo_prod_ks5_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$OUTR/invivo_prod_ks5/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3)+prior:$D/$OUTR/invivo_prod_ks5_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod_ks5/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP 12spf k80:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro K5 k80:$GP/cs_slice${Z}_f80match.npy"
DCE_DS=p14 $P recon_gif.py --slice $Z --t0 30 --t1 100 --dt 1 --w 2 --fps 4 --out $FIG/recon_gif_arrival_p14_sl$Z.gif --items "$IT" --note "p14 slice $Z, contrast arrival 30 to 100 s, 1 s steps (window +-2 s), same k80 input for every arm, one global scale per arm"
DCE_DS=p14 $P recon_gif.py --slice $Z --t0 30 --t1 100 --dt 1 --w 2 --fps 4 --single 1 --single-tag arrival --out $FIG/recon_gifs_single --items "$IT"
DCE_DS=p14 $P rulers_slide.py --slice $Z
echo "GIFS exit $?"; ls -la $FIG/recon_gif_arrival*.gif $FIG/recon_gifs_single/*_arrival.gif $FIG/rulers_slide*.png | awk '{printf "%.1f MB %s\n", $5/1e6, $9}'
git add $FIG/recon_gif_arrival*.gif $FIG/recon_gifs_single/*_arrival.gif $FIG/rulers_slide*.png && echo staged
