#!/bin/bash
#SBATCH -J pisco
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 5:00:00
#SBATCH --array=0-1
# pisco rerun after the OOM in 055: per-coil checkpointed evaluation, batch 512. 0 = weight 0.1, 1 = weight 1.0; tofts8 unit-rms sl21 k80 seed 0, 12k, snapshots, no restore.
# task 0 chains the trackers afterany over sup0.3 / sup3 / pisco0.1 / pisco1.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; AT=$RES/amp_track; Z=21; SELF=$D/jobs/queue/057_pisco_rerun.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true
if [ "${STAGE:-train}" = track ]; then
  $P tofts_amp_track.py --slice $Z --runs sup0.3:$D/$AT/sup0.3_sl$Z,sup3:$D/$AT/sup3_sl$Z,pisco0.1:$D/$AT/pisco0.1_sl$Z,pisco1:$D/$AT/pisco1_sl$Z; echo "AMP exit $?"
  IQ_VARIANTS=sup0.3,sup3,pisco0.1,pisco1 IQ_TAG=_kspace $P tofts_iq_track.py --slice $Z; echo "IQ exit $?"; exit
fi
i=$SLURM_ARRAY_TASK_ID; if [ "$i" = 0 ]; then NM=pisco0.1; W=0.1; else NM=pisco1; W=1.0; fi
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=track --array=0 -J pisco_cpu -p defq --gres="" -c 4 --mem 24G -t 2:00:00 \
    --output=$D/jobs/log/057_pisco_cpu_%j.out --error=$D/jobs/log/057_pisco_cpu_%j.out $SELF && echo "tracker chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$AT/${NM}_sl$Z; rm -rf $OUT; mkdir -p $OUT; echo "variant $NM -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz --slices $Z --seed 0 --ff-seed 0 --steps 12000 --pisco-weight $W --pisco-batch 512 \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --snapshot-every 1000 --no-restore --no-compile --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM exit ${PIPESTATUS[0]}"
