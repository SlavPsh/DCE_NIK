#!/bin/bash
#SBATCH -J iqtrade
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 3:30:00
#SBATCH --array=0-1
# middle ground between curve fidelity and image quality for tofts8 (unit-rms atoms, slice 21, k80, seed 0): 12k steps, |recon| snapshot every
# 1000 steps, no restore. 0 = wd 3e-3 (the 040 protocol, to find the best step), 1 = wd 1e-2 (stronger noise suppression at full amplitude).
# task 0 chains a cpu stage afterany: amplitude tracker + image-quality tracker over both runs -> iq_track_sl21_trade.{md,png}
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; AT=$RES/amp_track; Z=21; SELF=$D/jobs/queue/044_iq_tradeoff.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = track ]; then
  $P tofts_amp_track.py --slice $Z --runs rms1_wd3e-3:$D/$AT/rms1_wd3e-3_sl$Z,rms1_wd1e-2:$D/$AT/rms1_wd1e-2_sl$Z; echo "AMP exit $?"
  IQ_VARIANTS=rms1_wd3e-3,rms1_wd1e-2 IQ_TAG=_trade $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"; exit
fi
i=$SLURM_ARRAY_TASK_ID
if [ "$i" = 0 ]; then NM=rms1_wd3e-3; EX=""; else NM=rms1_wd1e-2; EX="--weight-decay 0.01"; fi
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=track --array=0 -J iqtrade_cpu -p defq --gres="" -c 4 --mem 24G -t 2:00:00 \
    --output=$D/jobs/log/044_iq_tradeoff_cpu_%j.out --error=$D/jobs/log/044_iq_tradeoff_cpu_%j.out $SELF && echo "tracker chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$AT/${NM}_sl$Z; mkdir -p $OUT; echo "variant $NM -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --slices $Z --seed 0 --ff-seed 0 --steps 12000 $EX \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --snapshot-every 1000 --no-restore --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM exit ${PIPESTATUS[0]}"
