#!/bin/bash
#SBATCH -J dlyprior
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 3:00:00
#SBATCH --array=0-2
# smooth-delay prior (FINDINGS 9, the one lever that follows from the mechanism): huber tv on the residual-atom coefficient maps divided by the
# aif-atom map (= per-voxel delay / dispersion), weighted by the aif-atom energy; the residual maps may be large where the aif map is large, but
# the delay must vary smoothly. tofts8 in-coil + prior, p3 slice 21, seed 0, production protocol; weights 0.1 / 1 / 10 -> results/tofts_vs_patlak/delay_prior/dly<w>/tofts8_sl21_s0.
# task 0 chains the eval: curve tables with corrected peaks, arrival diag, iq track, comparison figure vs production in / out coil and GRASP.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; OUTR=$RES/delay_prior; SELF=$D/jobs/queue/118_delay_prior.sh; Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
WS=(0.1 1 10); BASE=$D/$RES/invivo_prod/tofts8_sl${Z}_s0; OC=$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0
if [ "${STAGE:-train}" = eval ]; then
  IT="tofts8 prod in-coil:$BASE/nik_slice_${Z}_cplx.npy,tofts8 prod out-coil:$OC/nik_slice_${Z}_cplx.npy"
  for W in "${WS[@]}"; do T=dly$W
    $P tofts_eval_invivo.py --spokes k80 --slices $Z --arms tofts8 --iv-dir delay_prior/$T --basis-suffix _rms1 --suffix _dlyprior_$T; $P add_peak_correction.py invivo_dlyprior_$T
    $P arrival_artifact_diag.py --slice $Z --tag _dlyprior_$T --model "tofts8 $T:$D/$OUTR/$T/tofts8_sl${Z}_s0" --items "tofts8 $T:$D/$OUTR/$T/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 prod in-coil:$BASE/nik_slice_${Z}_cplx.npy,tofts8 prod out-coil:$OC/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"
    IT="$IT,tofts8 delay-tv $W:$D/$OUTR/$T/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy"
  done; echo "EVAL+DIAG exit $?"
  IQ_VARIANTS="" IQ_TAG=_dlyprior IQ_EXTRA="$IT" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"
  $P compare_runs_fig.py --slice $Z --out $RES/figures/delay_prior_sl$Z.png --title "p3 slice $Z, tofts8 smooth-delay prior (tv on residual / aif coefficient ratio) at weight 0.1 / 1 / 10 vs production in / out coil and GRASP" --items "$IT,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "FIG exit $?"
  exit
fi
i=$SLURM_ARRAY_TASK_ID; W=${WS[$i]}; T=dly$W
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J dlyprior_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 3:00:00 --output=$D/jobs/log/118_delay_prior_eval_%j.out --error=$D/jobs/log/118_delay_prior_eval_%j.out $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$OUTR/$T/tofts8_sl${Z}_s0; mkdir -p $OUT; echo "task $i: delay-tv weight $W -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --support-weight 1 --support-mask spoke_masks/support_sl${Z}_d6.npy --support-orient rot180 --coef-tv-every 32 --delay-tv-weight $W \
  --slices $Z --seed 0 --ff-seed 0 --steps 10000 --no-restore --weight-decay 0.01 \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $T exit ${PIPESTATUS[0]}"
