#!/bin/bash
#SBATCH -J tofts8
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 4:00:00
#SBATCH --array=0-2
# tofts in vivo slice 21 with a forced rank 8 basis (k-centre held-out overfit test vs the rank 12 rule)
# 3 seeds in parallel; task 0 builds the basis and chains the eval (afterany) as STAGE=eval on a 1g slice
# outputs: results/tofts_vs_patlak/basis_sl21_r8.npz, invivo/tofts8_sl21_s{0,1,2}/, invivo_r8.{json,md}; wandb dce_nik
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; Z=21; BAS=$RES/basis_sl${Z}_r8.npz; IV=$RES/invivo
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --slices $Z --arms patlak,tofts,tofts8 --suffix _r8
  echo "EVAL_DONE exit $?"; exit
fi

S=$SLURM_ARRAY_TASK_ID
if [ "$S" = 0 ]; then
  if [ ! -f $BAS ]; then
    $P nik_tofts_basis.py --aif aif_slice$Z.npz --out ${BAS%.npz}_tmp.npz --ranks 8 > $RES/basis_sl${Z}_r8.log 2>&1 && mv ${BAS%.npz}_tmp.npz $BAS
    tail -3 $RES/basis_sl${Z}_r8.log
  fi
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J tofts8eval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 1:30:00 --output=$D/jobs/log/001_tofts_sl21_rank8_eval_%j.out --error=$D/jobs/log/001_tofts_sl21_rank8_eval_%j.out \
    $D/jobs/queue/001_tofts_sl21_rank8.sh && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
else
  until [ -f $BAS ]; do sleep 20; done
fi

OUT=$IV/tofts8_sl${Z}_s$S; mkdir -p $OUT
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $BAS --slices $Z --seed $S --ff-seed $S \
  --spoke-keep-file spoke_masks/keep_f25.npy --spoke-heldout-file spoke_masks/val_f25c_m8.npy \
  --no-compile --resume --save-dir $OUT
echo "$(date '+%F %T') TRAIN_DONE s$S exit $?"
