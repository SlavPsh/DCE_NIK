#!/bin/bash
#SBATCH -J amptrack
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 4:30:00
#SBATCH --array=0-2
# pk-arm amplitude deficit, cause test on slice 21 at k80, tofts8: full 40k steps, final weights kept (no restore), |recon| snapshot every 2000 steps.
# variants: 0 base (lr 1e-5, wd 3e-3), 1 wd 0, 2 lr 1e-4. task 0 chains the cpu tracker afterany (roi amplitude ratio vs step + held-out mse).
# outputs: results/tofts_vs_patlak/amp_track/<variant>_sl21/ (snap_*.npy, train.log), amp_track_sl21.{md,json}, figures/amp_track_sl21.png
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; AT=$RES/amp_track; Z=21; SELF=$D/jobs/queue/031_amp_track.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = track ]; then
  $P tofts_amp_track.py --slice $Z --runs base:$D/$AT/base_sl$Z,wd0:$D/$AT/wd0_sl$Z,lr1e-4:$D/$AT/lr1e-4_sl$Z; echo "TRACK exit $?"; exit
fi

i=$SLURM_ARRAY_TASK_ID
case $i in
  0) NM=base;   EX="";;
  1) NM=wd0;    EX="--weight-decay 0";;
  2) NM=lr1e-4; EX="--lr 1e-4";;
esac
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=track --array=0 -J amptrack_cpu -p defq --gres="" -c 4 --mem 16G -t 1:00:00 \
    --output=$D/jobs/log/031_amp_track_cpu_%j.out --error=$D/jobs/log/031_amp_track_cpu_%j.out $SELF && echo "tracker chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$AT/${NM}_sl$Z; mkdir -p $OUT
echo "variant $NM -> $OUT"
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8.npz --slices $Z --seed 0 --ff-seed 0 --steps 40000 $EX \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --snapshot-every 2000 --no-restore --no-compile --resume --save-dir $OUT 2>&1 | tee $OUT/train.log
echo "$(date '+%F %T') TRAIN_DONE $NM exit ${PIPESTATUS[0]}"
