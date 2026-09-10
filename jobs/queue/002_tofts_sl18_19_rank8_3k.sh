#!/bin/bash
#SBATCH -J tofts8b
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0:45:00
#SBATCH --array=0-5
# tofts rank 8 on slices 18 and 19, 3 seeds each, 3k steps (every in vivo run so far restored step 2000; probe 002)
# i = slice_idx*3 + seed; task 0 builds both bases, chains the eval (afterany, all three slices) as STAGE=eval
# outputs: basis_sl{18,19}_r8.npz, invivo/tofts8_sl{18,19}_s{0,1,2}/, invivo_r8.{json,md} (slices 18,19,21); wandb dce_nik
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="micromamba run -n torch29 python -u"
RES=results/tofts_vs_patlak; IV=$RES/invivo; SELF=$D/jobs/queue/002_tofts_sl18_19_rank8_3k.sh
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} stage ${STAGE:-train}"; nvidia-smi -L || true

if [ "${STAGE:-train}" = eval ]; then
  $P tofts_eval_invivo.py --slices 18,19,21 --arms patlak,tofts,tofts8 --suffix _r8
  echo "EVAL_DONE exit $?"
  $P tofts_figs_wandb.py --slices 18,19,21 --suffix _r8; echo "FIGS exit $?"; exit
fi

i=$SLURM_ARRAY_TASK_ID; SL=(18 19); Z=${SL[$((i / 3))]}; S=$((i % 3)); BAS=$RES/basis_sl${Z}_r8.npz
if [ "$i" = 0 ]; then
  for ZZ in 18 19; do B=$RES/basis_sl${ZZ}_r8.npz
    [ -f $B ] || { $P nik_tofts_basis.py --aif aif_slice$ZZ.npz --out ${B%.npz}_tmp.npz --ranks 8 > $RES/basis_sl${ZZ}_r8.log 2>&1 && mv ${B%.npz}_tmp.npz $B; grep -E 'SELECTED|^ +8 ' $RES/basis_sl${ZZ}_r8.log; }
  done
  sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID --export=ALL,STAGE=eval --array=0 -J tofts8beval \
    --gres=gpu:1g.12gb:1 -c 4 --mem 32G -t 1:30:00 --output=$D/jobs/log/002_tofts_sl18_19_rank8_3k_eval_%j.out --error=$D/jobs/log/002_tofts_sl18_19_rank8_3k_eval_%j.out \
    $SELF && echo "eval chained afterany $SLURM_ARRAY_JOB_ID"
else
  until [ -f $BAS ]; do sleep 20; done
fi

OUT=$IV/tofts8_sl${Z}_s$S; mkdir -p $OUT
$P train_grasp_nik.py --model wire_ff_tofts --tofts-basis $BAS --slices $Z --seed $S --ff-seed $S --steps 3000 \
  --spoke-keep-file spoke_masks/keep_f25.npy --spoke-heldout-file spoke_masks/val_f25c_m8.npy \
  --no-compile --resume --save-dir $OUT
echo "$(date '+%F %T') TRAIN_DONE sl$Z s$S exit $?"
