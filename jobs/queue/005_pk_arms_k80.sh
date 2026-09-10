#!/bin/bash
#SBATCH -J pk_k80
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 1:15:00
#SBATCH --array=0-26
# the standard input: k80 (keep v%10<8 = 1368 of 1708 views, val v%10==8, test v%10==9), identical for every method.
# pk nik arms patlak / tofts (rank rule 12) / tofts8 (forced 8) on slices 18/19/21, 3 seeds, 3k steps (every in vivo run restores step 2000).
# i = slice_idx*9 + arm_idx*3 + seed. task 0 chains eval (--spokes k80) + figures (--iv invivo_k80 --refs k80) afterany.
# outputs: results/tofts_vs_patlak/invivo_k80/<arm>_sl<Z>_s<S>/, invivo_k80.{json,md}; wandb gnik_<arm>_sl<Z>_s<S>_s<Z>, figs_invivo_k80_sl<Z>
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; IV=$RES/invivo_k80; SELF=$D/jobs/queue/005_pk_arms_k80.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --spokes k80 --slices 18,19,21 --arms patlak,tofts,tofts8; echo "EVAL_DONE exit $?"
  $P tofts_figs_wandb.py --iv invivo_k80 --refs k80 --suffix _k80 --slices 18,19,21; echo "FIGS exit $?"; exit
fi

i=$SLURM_ARRAY_TASK_ID; SL=(18 19 21); Z=${SL[$((i / 9))]}; AI=$(((i % 9) / 3)); S=$((i % 3))
case $AI in
  0) NM=patlak; MA="--model wire_ff_patlak --aif-file $D/aif_slice$Z.npz --patlak-free 0";;
  1) NM=tofts;  MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl$Z.npz";;
  2) NM=tofts8; MA="--model wire_ff_tofts --tofts-basis $RES/basis_sl${Z}_r8.npz";;
esac
if [ "$i" = 0 ]; then
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J pk_k80_eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 2:00:00 --output=$D/jobs/log/005_pk_arms_k80_eval_%j.out --error=$D/jobs/log/005_pk_arms_k80_eval_%j.out \
    $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
fi
OUT=$IV/${NM}_sl${Z}_s$S; mkdir -p $OUT
echo "slice $Z arm $NM seed $S -> $OUT"
$P train_grasp_nik.py $MA --slices $Z --seed $S --ff-seed $S --steps 3000 \
  --spoke-keep-file spoke_masks/keep_f80match.npy --spoke-heldout-file spoke_masks/val_k80_m8.npy \
  --no-compile --resume --save-dir $OUT
echo "$(date '+%F %T') TRAIN_DONE $NM sl$Z s$S exit $?"
