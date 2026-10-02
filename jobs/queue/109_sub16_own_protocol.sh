#!/bin/bash
#SBATCH -J sub16own
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 4:00:00
#SBATCH --array=0-4
# sub16 under ITS OWN protocol (FINDINGS 8a, memory feedback_protocol_per_model): wd 3e-3 + support prior 1, everything else as production
# (rank 16 input-coil, pca warm start 100 frames, 10k, final weights, k80; sigma 2.5 on p3, 5 on p14). 0-1 p3 slices 18 / 19 -> invivo_prod_sub16wd/
# (slice 21 = sub16_proto/wd3e-3_prior, copied into the same dir by the eval); 2-4 p14 slices 21 / 24 / 27 -> p14/invivo_prod_ks5_sub16wd/.
# task 0 chains the eval: curve tables with corrected peaks for both datasets, iq tracks vs the late ruler, comparison figures and gifs with the
# sub16 column replaced and labelled "sub16 (wd 3e-3)+prior"; the previous sub16 results stay in place.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; SELF=$D/jobs/queue/109_sub16_own_protocol.sh; FIG=results/realdata_nik_vs_cs_figures/figures
GV3=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP3=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs; REF3=/net/beegfs/users/P101440/grasp_pro_py/results_ref
GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14; REFD=/net/beegfs/users/P101440/grasp_pro_py/results_ref_p14
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = eval ]; then
  mkdir -p $RES/invivo_prod_sub16wd/sub16_sl21_s0; cp -n $RES/sub16_proto/wd3e-3_prior/sub16_sl21_s0/* $RES/invivo_prod_sub16wd/sub16_sl21_s0/ 2>/dev/null || true
  DCE_DS=p3 $P tofts_eval_invivo.py --spokes k80 --slices 18,19,21 --arms sub16 --iv-dir invivo_prod_sub16wd --basis-suffix _rms1 --suffix _prod_sub16wd; echo "EVAL p3 exit $?"
  DCE_DS=p3 $P add_peak_correction.py invivo_prod_sub16wd
  DCE_DS=p14 $P tofts_eval_invivo.py --spokes k80 --slices 21,24,27 --arms sub16 --iv-dir p14/invivo_prod_ks5_sub16wd --basis-suffix _rms1 --suffix _p14_ks5_sub16wd; echo "EVAL p14 exit $?"
  DCE_DS=p14 $P add_peak_correction.py invivo_p14_ks5_sub16wd
  for Z in 18 19 21; do
    IT="tofts8 in-coil+prior:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$RES/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3)+prior:$D/$RES/invivo_prod_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
    IQ_VARIANTS="" IQ_TAG=_prod_v2 IQ_EXTRA="$IT" DCE_DS=p3 $P tofts_iq_track.py --slice $Z
    DCE_DS=p3 $P compare_runs_fig.py --slice $Z --out $RES/figures/prod_v2_sl$Z.png --title "p3 slice $Z, k80, production protocol per arm (sub16 at its own wd 3e-3): images at 90 s, kidney zoom, roi curves" --items "$IT,GRASP:$GV3/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP3/cs_slice${Z}_f80match.npy"
    DCE_DS=p3 $P recon_gif.py --slice $Z --out $FIG/recon_gif_v2_sl$Z.gif --items "$IT,GRASP 12spf k80:$GV3/gv2_slice${Z}_n12_k80.npy,GRASP-Pro K5 k80:$GP3/cs_slice${Z}_f80match.npy"
    DCE_DS=p3 $P recon_gif.py --slice $Z --single 1 --out $FIG/recon_gifs_single --items "sub16 (wd 3e-3)+prior:$D/$RES/invivo_prod_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
  done; echo "P3 FIGS exit $?"
  OUTR=$RES/p14
  for Z in 21 24 27; do
    IT="tofts8 in-coil+prior:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$OUTR/invivo_prod_ks5_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$OUTR/invivo_prod_ks5/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3)+prior:$D/$OUTR/invivo_prod_ks5_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod_ks5/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
    IQ_VARIANTS="" IQ_TAG=_p14_ks5_v2 IQ_EXTRA="$IT" DCE_DS=p14 $P tofts_iq_track.py --slice $Z
    DCE_DS=p14 $P compare_runs_fig.py --slice $Z --out $RES/figures/p14_ks5_v2_sl$Z.png --title "p14 slice $Z, k80, production protocol per arm at sigma 5 (sub16 at its own wd 3e-3): images at 90 s, liver zoom, roi curves" --items "$IT,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"
    DCE_DS=p14 $P recon_gif.py --slice $Z --out $FIG/recon_gif_v2_p14_sl$Z.gif --items "$IT,GRASP 12spf k80:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro K5 k80:$GP/cs_slice${Z}_f80match.npy"
    DCE_DS=p14 $P recon_gif.py --slice $Z --single 1 --out $FIG/recon_gifs_single --items "sub16 (wd 3e-3)+prior:$D/$OUTR/invivo_prod_ks5_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
  done; echo "P14 FIGS exit $?"
  git add $FIG/recon_gif_v2*.gif $FIG/recon_gifs_single/*sub16_wd*.gif 2>/dev/null && echo staged
  exit
fi
i=$SLURM_ARRAY_TASK_ID
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J sub16own_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 4:00:00 --output=$D/jobs/log/109_sub16_own_protocol_eval_%j.out --error=$D/jobs/log/109_sub16_own_protocol_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
if [ $i -lt 2 ]; then
  Z=$([ $i = 0 ] && echo 18 || echo 19); OUT=$RES/invivo_prod_sub16wd/sub16_sl${Z}_s0; DS=p3; KEEP=spoke_masks/keep_f80match.npy; VAL=spoke_masks/val_k80_m8.npy; SUP=spoke_masks/support_sl${Z}_d6.npy; X=""; OD=$REF3
else
  SL=(21 24 27); Z=${SL[$((i-2))]}; OUT=$RES/p14/invivo_prod_ks5_sub16wd/sub16_sl${Z}_s0; DS=p14; KEEP=spoke_masks/keep_k80_p14.npy; VAL=spoke_masks/val_k80_m8_p14.npy; SUP=spoke_masks/support_p14_sl${Z}_d6.npy; X="--k-sigma 5"; OD=$REFD
fi
mkdir -p $OUT; echo "task $i: sub16 own protocol $DS slice $Z -> $OUT"
DCE_DS=$DS $P train_grasp_nik.py --model wire_ff_subspace --rank 16 --weight-decay 0.003 --support-weight 1 --support-mask $SUP --support-orient rot180 --coef-tv-every 32 $X \
  --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore --out-dir $OD \
  --spoke-keep-file $KEEP --spoke-heldout-file $VAL --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE sub16wd $DS sl$Z exit ${PIPESTATUS[0]}"
