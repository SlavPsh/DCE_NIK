#!/bin/bash
#SBATCH -J pk_rms1
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 2:30:00
#SBATCH --array=0-26
# production rerun of the pk nik arms at k80 after the amplitude verdict (2026-09-16): unit-rms tofts atoms (basis_*_rms1.npz), 10k steps,
# final weights kept (no held-out restore), same k80 views / seeds / everything else as queue 005. patlak gets the same protocol (its atoms are O(1) already).
# i = slice_idx*9 + arm_idx*3 + seed; slices 18/19/21, arms patlak / tofts (rank 12) / tofts8, seeds 0-2.
# task 0 chains afterany: eval (tofts_eval_invivo, ratio columns) + wandb figures + story panels (tofts and tofts8) + span diagnostic, all on the rms1 dir.
# outputs: results/tofts_vs_patlak/invivo_k80_rms1/<arm>_sl<Z>_s<S>/, invivo_k80_rms1.{json,md}, span_diag_sl*_rms1.md, figures story_invivo_k80_sl21*_rms1.png
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; IV=$RES/invivo_k80_rms1; SELF=$D/jobs/queue/040_pk_arms_k80_rms1.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --spokes k80 --slices 18,19,21 --arms patlak,tofts,tofts8 --iv-dir invivo_k80_rms1 --basis-suffix _rms1 --suffix _k80_rms1; echo "EVAL_DONE exit $?"
  $P tofts_figs_wandb.py --iv invivo_k80_rms1 --refs k80 --suffix _k80_rms1 --slices 18,19,21; echo "FIGS exit $?"
  for Z in 21 18 19; do IV_DIR=invivo_k80_rms1 STORY_TAG=_rms1 $P tofts_span_diag.py --slice $Z --arms tofts,tofts8,patlak; done; echo "SPAN exit $?"
  IV_DIR=invivo_k80_rms1 STORY_TAG=_rms1 $P story_figs.py --only invivo --t-invivo 90 --tofts tofts; echo "STORY tofts exit $?"
  IV_DIR=invivo_k80_rms1 STORY_TAG=_rms1_t8 $P story_figs.py --only invivo --t-invivo 90 --tofts tofts8; echo "STORY tofts8 exit $?"
  exit
fi

i=$SLURM_ARRAY_TASK_ID; SL=(18 19 21); Z=${SL[$((i / 9))]}; AI=$(((i % 9) / 3)); S=$((i % 3))
$P basis_rms1.py $RES/basis_sl$Z.npz $RES/basis_sl${Z}_r8.npz
case $AI in
  0) NM=patlak; MA="--model wire_ff_patlak --aif-file $D/aif_slice$Z.npz --patlak-free 0";;
  1) NM=tofts;  MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_rms1.npz";;
  2) NM=tofts8; MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8_rms1.npz";;
esac
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J pk_rms1_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 3:00:00 --output=$D/jobs/log/040_pk_arms_k80_rms1_eval_%j.out --error=$D/jobs/log/040_pk_arms_k80_rms1_eval_%j.out \
    $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$IV/${NM}_sl${Z}_s$S; mkdir -p $OUT
echo "slice $Z arm $NM seed $S -> $OUT"
$P train_grasp_nik.py $MA --slices $Z --seed $S --ff-seed $S --steps 10000 --no-restore \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT
echo "$(date '+%F %T') TRAIN_DONE $NM sl$Z s$S exit $?"
