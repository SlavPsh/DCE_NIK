#!/bin/bash
#SBATCH -J kprior
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 5:00:00
#SBATCH --array=0-3
# k-space consistency priors against the deterministic streak texture (045: seed-invariant air energy), tofts8 unit-rms sl21 k80 seed 0, 12k steps,
# snapshots every 1000, no restore, wd 3e-3: 0 support 0.3, 1 support 3.0 (coefficient images, energy outside / inside the body, every 32 steps);
# 2 pisco 0.1, 3 pisco 1.0 (self-supervised shift-invariant kernel consistency, 8 neighbours, batch 1024, every 8 steps).
# task 0 chains the amplitude + image-quality trackers afterany (needs 054's support masks; 054 runs on defq first).
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; AT=$RES/amp_track; Z=21; SELF=$D/jobs/queue/055_kspace_priors.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = track ]; then
  $P tofts_amp_track.py --slice $Z --runs sup0.3:$D/$AT/sup0.3_sl$Z,sup3:$D/$AT/sup3_sl$Z,pisco0.1:$D/$AT/pisco0.1_sl$Z,pisco1:$D/$AT/pisco1_sl$Z; echo "AMP exit $?"
  IQ_VARIANTS=sup0.3,sup3,pisco0.1,pisco1 IQ_TAG=_kspace $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"; exit
fi
i=$SLURM_ARRAY_TASK_ID
case $i in
  0) NM=sup0.3;   EX="--support-weight 0.3 --support-mask spoke_masks/support_sl$Z.npy --coef-tv-every 32";;
  1) NM=sup3;     EX="--support-weight 3.0 --support-mask spoke_masks/support_sl$Z.npy --coef-tv-every 32";;
  2) NM=pisco0.1; EX="--pisco-weight 0.1";;
  3) NM=pisco1;   EX="--pisco-weight 1.0";;
esac
[ -f spoke_masks/support_sl$Z.npy ] || $P support_mask.py --slices $Z --dilate 4
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=track --array=0 -J kprior_cpu -p defq --gres="" -c 4 --mem 24G -t 2:00:00 \
    --output=$D/jobs/log/055_kspace_priors_cpu_%j.out --error=$D/jobs/log/055_kspace_priors_cpu_%j.out $SELF && echo "tracker chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$AT/${NM}_sl$Z; mkdir -p $OUT; echo "variant $NM -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --slices $Z --seed 0 --ff-seed 0 --steps 12000 $EX \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --snapshot-every 1000 --no-restore --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM exit ${PIPESTATUS[0]}"
