#!/bin/bash
#SBATCH -J k100eval
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
# k100 standard, the regeneration: waits until every k100 output of queues 123 / 124 / 125 exists, then per dataset: curve tables with corrected peaks
# (grasp at all spokes as the only cs reference), iq tracks (cs100 secondary, late / gated-late / gated-pre nufft rulers primary), comparison figures,
# gifs (combined 7 panels, singles and arrival gifs for the main slices), ruler-vs-arms slides (ungated and gated), arrival diagnostics for tofts8.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; FIG=results/realdata_nik_vs_cs_figures/figures
GV3=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GV14=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
NEED=""
for Z in 18 19 21; do for S in 0 1 2; do NEED="$NEED $RES/invivo_k100/tofts8_sl${Z}_s$S/nik_slice_${Z}_cplx.npy"; done
  for A in tofts8_sl${Z}_s0; do NEED="$NEED $RES/invivo_k100_oc/$A/nik_slice_${Z}_cplx.npy"; done
  for A in patlak sub16 free; do NEED="$NEED $RES/invivo_k100/${A}_sl${Z}_s0/nik_slice_${Z}_cplx.npy"; done; NEED="$NEED $GV3/gv2_slice${Z}_n12.npy"; done
for Z in 21 24 27; do for S in 0 1 2; do NEED="$NEED $RES/p14/invivo_k100/tofts8_sl${Z}_s$S/nik_slice_${Z}_cplx.npy"; done
  NEED="$NEED $RES/p14/invivo_k100_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
  for A in patlak sub16 free; do NEED="$NEED $RES/p14/invivo_k100/${A}_sl${Z}_s0/nik_slice_${Z}_cplx.npy"; done; NEED="$NEED $GV14/gv2_slice${Z}_n12.npy"; done
for k in $(seq 1 2800); do miss=0; for f in $NEED; do [ -f $f ] || miss=$((miss+1)); done; [ $miss = 0 ] && break; [ $((k % 30)) = 0 ] && echo "$(date '+%T') waiting, $miss outputs missing"; sleep 60; done
[ $miss = 0 ] || { echo "TIMEOUT, $miss missing:"; for f in $NEED; do [ -f $f ] || echo "  $f"; done; exit 3; }
echo "$(date '+%F %T') all inputs present"
ITEMS() {  # $1 dataset $2 slice -> items string (nik arms + grasp all spokes)
  local ds=$1 Z=$2 R GV; if [ $ds = p3 ]; then R=$D/$RES; GV=$GV3; else R=$D/$RES/p14; GV=$GV14; fi
  echo "tofts8 in-coil+prior:$R/invivo_k100/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$R/invivo_k100_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$R/invivo_k100/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3)+prior:$R/invivo_k100/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$R/invivo_k100/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP all spokes:$GV/gv2_slice${Z}_n12.npy"
}
for DS in p3 p14; do
  if [ $DS = p3 ]; then SLS="18 19 21"; IV=invivo_k100; SUF=_k100std; TAG=_k100std; ARR=21; else SLS="21 24 27"; IV=p14/invivo_k100; SUF=_p14_k100std; TAG=_p14_k100std; ARR=24; fi
  export DCE_DS=$DS
  $P tofts_eval_invivo.py --spokes k100 --slices $(echo $SLS | tr ' ' ',') --arms tofts8,patlak,sub16,free --iv-dir $IV --basis-suffix _rms1 --suffix $SUF; $P add_peak_correction.py invivo$SUF
  $P tofts_eval_invivo.py --spokes k100 --slices $(echo $SLS | tr ' ' ',') --arms tofts8 --iv-dir ${IV}_oc --basis-suffix _rms1 --suffix ${SUF}_oc; $P add_peak_correction.py invivo${SUF}_oc; echo "EVAL $DS exit $?"
  for Z in $SLS; do IT=$(ITEMS $DS $Z)
    IQ_VARIANTS="" IQ_TAG=${TAG}_sl$Z IQ_EXTRA="$IT" $P tofts_iq_track.py --slice $Z
    $P compare_runs_fig.py --slice $Z --out $RES/figures/k100std${DS}_sl$Z.png --title "$DS slice $Z, k100 standard (all spokes, every arm under its own protocol, grasp n12 at all spokes): images at 90 s, tissue zoom, roi curves" --items "$IT"
    $P recon_gif.py --slice $Z --out $FIG/recon_gif_k100_${DS}_sl$Z.gif --items "$IT"
    $P ruler_vs_arms_fig.py --slice $Z --items "$IT" --out $FIG/ruler_vs_arms_k100_${DS}_sl$Z.png; $P ruler_vs_arms_fig.py --slice $Z --gated 1 --items "$IT" --out $FIG/ruler_vs_arms_k100_gated_${DS}_sl$Z.png
  done; echo "FIGS $DS exit $?"
  IT=$(ITEMS $DS $ARR); T0=40; T1=110; [ $DS = p14 ] && { T0=30; T1=100; }
  $P recon_gif.py --slice $ARR --single 1 --single-tag k100 --out $FIG/recon_gifs_single --items "$IT"
  $P recon_gif.py --slice $ARR --t0 $T0 --t1 $T1 --dt 1 --w 2 --fps 4 --out $FIG/recon_gif_k100_arrival_${DS}_sl$ARR.gif --items "$IT"
  $P recon_gif.py --slice $ARR --t0 $T0 --t1 $T1 --dt 1 --w 2 --fps 4 --single 1 --single-tag k100_arrival --out $FIG/recon_gifs_single --items "$IT"
  R=$D/$RES; [ $DS = p14 ] && R=$D/$RES/p14; GV=$GV3; [ $DS = p14 ] && GV=$GV14
  for A in "tofts8 in-coil k100:$R/invivo_k100/tofts8_sl${ARR}_s0" "tofts8 out-coil k100:$R/invivo_k100_oc/tofts8_sl${ARR}_s0"; do nm=${A%%:*}; d=${A#*:}; tag=$(echo $nm | tr ' ' '_')
    $P arrival_artifact_diag.py --slice $ARR --tag _${DS}_$tag --model "$A" --items "$nm:$d/nik_slice_${ARR}_cplx.npy,GRASP all spokes:$GV/gv2_slice${ARR}_n12.npy"; done; echo "GIFS+DIAG $DS exit $?"
done
git add $FIG/recon_gif_k100_*.gif $FIG/recon_gifs_single/*k100*.gif && echo staged; echo "K100EVAL DONE"
