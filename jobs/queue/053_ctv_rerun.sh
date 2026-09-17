#!/bin/bash
#SBATCH -J ctvrerun
# rerun of the two coefficient-map tv variants after the OOM in 045: render chunked + checkpointed (--coef-tv-chunk), tv every 32 steps
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 5:00:00
#SBATCH --array=2-3
# regularization ideas 1, 2, 4 for the pk arms (tofts8, unit-rms atoms, slice 21, k80, seed 0, 12k steps, snapshots every 1000, no restore):
# 0 cos       = cosine lr schedule to 1e-7 at 12k (endpoint independent of the stop), wd 3e-3
# 1 cos_wd1e-2 = cosine + weight decay 1e-2
# 2 ctv0.03   = huber tv on the per-coil coefficient maps, weight 0.03, every 16 steps (plateau lr as in 040)
# 3 ctv0.3    = same, weight 0.3
# task 0 chains a cpu stage afterany: idea 3 (seed average of the 040 tofts8 seeds), amplitude tracker + image-quality tracker over the four runs and the average.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; AT=$RES/amp_track; Z=21; SELF=$D/jobs/queue/053_ctv_rerun.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = track ]; then
  echo "avg already done"
  $P tofts_amp_track.py --slice $Z --runs cos:$D/$AT/cos_sl$Z,cos_wd1e-2:$D/$AT/cos_wd1e-2_sl$Z,ctv0.03:$D/$AT/ctv0.03_sl$Z,ctv0.3:$D/$AT/ctv0.3_sl$Z; echo "AMP exit $?"
  IQ_VARIANTS=cos,cos_wd1e-2,ctv0.03,ctv0.3 IQ_TAG=_ideas2 IQ_EXTRA="NIK-tofts8 new avg3:$D/$RES/invivo_k80_rms1/tofts8_sl${Z}_avg3/nik_slice_${Z}_cplx.npy" $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"; exit
fi
i=$SLURM_ARRAY_TASK_ID
case $i in
  0) NM=cos;        EX="--lr-schedule cosine";;
  1) NM=cos_wd1e-2; EX="--lr-schedule cosine --weight-decay 0.01";;
  2) NM=ctv0.03;    EX="--coef-tv-weight 0.03 --coef-tv-every 32";;
  3) NM=ctv0.3;     EX="--coef-tv-weight 0.3 --coef-tv-every 32";;
esac
if [ "$i" = 2 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=track --array=0 -J regideas_cpu -p defq --gres="" -c 4 --mem 24G -t 2:00:00 \
    --output=$D/jobs/log/053_ctv_rerun_cpu_%j.out --error=$D/jobs/log/053_ctv_rerun_cpu_%j.out $SELF && echo "tracker chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$AT/${NM}_sl$Z; mkdir -p $OUT; echo "variant $NM -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --slices $Z --seed 0 --ff-seed 0 --steps 12000 $EX \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --snapshot-every 1000 --no-restore --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM exit ${PIPESTATUS[0]}"
